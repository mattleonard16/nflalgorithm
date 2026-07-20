# Data provenance and retention

Runtime sports data, model artifacts, reports, logs, and release evidence are not source code and
must not be committed to this repository. They can contain stale inputs, paid-provider data,
deployment identifiers, or outputs that are meaningful only for one commit and point in time.

## Source-of-truth policy

- NFL schedules, rosters, history, injuries, and context must be ingested through the canonical
  weekly refresh and persisted in the configured database.
- Production odds must come from the live provider path and pass the persisted freshness,
  event/market coverage, and sportsbook-breadth validation. A checked-in JSON snapshot is never a
  production fallback.
- Projections and decisions must be produced by the commit recorded on the run. Hand-authored or
  synthetic projection CSVs must not enter training, evaluation, or publication.
- Small deterministic test fixtures belong under `tests/fixtures/` and must document their purpose.
  They must not be presented as historical production evidence.

## Local locations

| Path | Contents | Retention |
|---|---|---|
| Configured database | Canonical inputs, projections, odds, runs, decisions, outcomes | Back up per deployment policy; rehearse restore before season |
| `data/` | Temporary imports and provider payloads | Local only; delete after database ingestion and verification |
| `logs/` | Runtime logs, metrics, and local evidence | Local only; ship production logs to monitored storage |
| `reports/` | Regenerable HTML/CSV/JSON reports | Local only; retain release evidence in the deployment artifact store |
| `archive/` | Optional local snapshots and backups | Local only; never treat as the live source of truth |
| Model/export storage | Versioned models, run reports, and exports | External durable storage keyed by commit/run and checksum |

## In-season lifecycle

1. Refresh a specific season/week and run `make doctor-season SEASON=<year> WEEK=<week>`.
2. Queue the durable run; do not use a checked-in cache or report as an input.
3. Run the post-run season check and capture the SHA-bound promotion evidence.
4. Grade completed outcomes, then evaluate the persisted pre-kickoff projections.
5. Keep raw provider payloads only as long as required by licensing, incident response, and audit
   policy. Keep their provenance, timestamps, validation metrics, and checksums with the run.
6. Remove local generated files when their durable copy and database backup have been verified.

The removed `2024_nfl_projections.csv` and `2024_nfl_rookies.csv` files were manually maintained,
unreferenced, and unsuitable as point-in-time training data. The removed Week 10 odds JSON and
generated reports/logs were also unreferenced runtime artifacts, not reproducible source fixtures.
The removed tracked archive contained only obsolete migration material, a superseded season-level
predictor, and an 8 KB SQLite file whose sole table was `test(id)`; none was runtime evidence.
