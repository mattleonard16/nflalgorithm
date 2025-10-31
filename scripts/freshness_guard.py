# File: freshness_guard.py
# Description: Helpers for recording feed freshness and enforcing staleness rules.
# Reason: Keeps pipeline executions from running on stale external data feeds.
# Relevant Files: config.py,data_pipeline.py,health_check.py

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class FeedStatus:
    name: str
    as_of: Optional[datetime]
    minutes_old: Optional[float]
    limit_minutes: Optional[int]
    source: Optional[str]


class FreshnessGuard:
    """Manage freshness metadata and enforce age limits for external feeds."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def register_feed_update(
        self,
        feed_name: str,
        as_of: Optional[datetime] = None,
        source: str = "unknown",
    ) -> None:
        payload = self._normalize_timestamp(as_of)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO freshness (feed_name, as_of, source, recorded_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(feed_name) DO UPDATE SET
                    as_of = excluded.as_of,
                    source = excluded.source,
                    recorded_at = CURRENT_TIMESTAMP
                """,
                (feed_name, payload, source),
            )
            conn.commit()

    def get_stale_feeds(self, limits: Dict[str, int]) -> List[FeedStatus]:
        statuses = self.get_status(limits)
        return [status for status in statuses if self._is_stale(status)]

    def enforce(self, limits: Dict[str, int]) -> None:
        stale = self.get_stale_feeds(limits)
        if stale:
            details = ", ".join(
                f"{item.name} ({self._describe_age(item.minutes_old)} > {item.limit_minutes}m)"
                for item in stale
            )
            raise RuntimeError(f"Stale feeds detected: {details}")

    def get_status(self, limits: Optional[Dict[str, int]] = None) -> List[FeedStatus]:
        limits = limits or {}
        data = self._fetch_all_freshness()
        statuses: List[FeedStatus] = []
        for feed_name in sorted(set(list(data.keys()) + list(limits.keys()))):
            raw_timestamp, source = data.get(feed_name, (None, None))
            timestamp = self._parse_timestamp(raw_timestamp) if raw_timestamp else None
            minutes_old = self._minutes_since(timestamp) if timestamp else None
            statuses.append(
                FeedStatus(
                    name=feed_name,
                    as_of=timestamp,
                    minutes_old=minutes_old,
                    limit_minutes=limits.get(feed_name),
                    source=source,
                )
            )
        return statuses

    def has_records(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM freshness").fetchone()
        return bool(row and row[0])

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def _fetch_all_freshness(self) -> Dict[str, Tuple[str, Optional[str]]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT feed_name, as_of, source FROM freshness").fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def _normalize_timestamp(self, value: Optional[datetime]) -> str:
        if value is None:
            value = datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _parse_timestamp(self, raw: str) -> Optional[datetime]:
        try:
            timestamp = datetime.fromisoformat(raw)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)
        except Exception:
            return None

    def _minutes_since(self, timestamp: datetime) -> float:
        delta = datetime.now(timezone.utc) - timestamp
        return max(delta.total_seconds() / 60.0, 0.0)

    def _is_stale(self, status: FeedStatus) -> bool:
        if status.limit_minutes is None:
            return False
        if status.minutes_old is None:
            return True
        return status.minutes_old > status.limit_minutes

    def _describe_age(self, minutes: Optional[float]) -> str:
        if minutes is None:
            return "no data"
        return f"{minutes:.1f}m"


