# Pipeline production-readiness gates

The queue/worker architecture is an implementation candidate, not a production-ready runtime.
Promotion requires evidence from the deployed private integration and staging environment in
addition to local tests.

Set `APP_COMMIT_SHA` to the deployed image's full 40-character Git SHA. The worker resolves this
identity before any stage runs, persists it in the completed run report, and projection evaluation
rejects rows outside that run's execution window or attributed to another commit.

| Gate | Current evidence | Remaining proof |
|---|---|---|
| Known commit SHA | Satisfied for the local candidate; the final SHA is recorded in the handoff | Use that exact SHA for staging and retain it with the release evidence |
| Integrated with `origin/main` | Satisfied locally after rebasing onto the fetched `origin/main`; the fetched main commit is an ancestor of the candidate | Recheck ancestry after the final fetch before staging promotion |
| Algorithm improvement | Production projections are scored only when persisted before kickoff and are bound to an exact SHA and season/week scope | Compare the legacy and candidate SHAs over the same completed weeks; require the chosen overall MAE improvement and cap every market regression |
| Old/new output equivalence | Deterministic inline-versus-worker report matrix passes on SQLite and MySQL | Run one real weekly legacy capture and queued shadow capture; require matching hashes |
| SQLite/MySQL matrices | Real SQLite and temporary MySQL 9.5 migration, concurrency, recovery, and equivalence checks pass locally; MySQL 8.4 CI service is configured | Require green remote CI for the committed SHA |
| No double claims | Eight concurrent claimers produce one winner on both databases | Repeat under staging worker concurrency and retain logs |
| Crash recovery/fencing | Every new claim has a binary attempt token; rolling-deploy tokenless leases are recovered safely; heartbeat exceptions, zero-row renewal, same-worker reclaims, case-equivalent IDs, stale terminal writes, and attempt-specific stage history are covered locally. The production worker exits with code 75 on lease loss, including while a non-cooperative handler is running. Stale recovery requeues only attempts whose persisted stages are all audited retry-safe | Kill a staging worker during both retry-safe and unaudited real stages; verify exactly one terminal run/card, correct retry decision, and supervisor restart |
| Live-odds fail closed | Offline/stale cache, absent cache timestamps, missing provenance, excessive age, partial event coverage, partial market coverage, insufficient sportsbook breadth, and empty snapshots stop before value/risk/agents/card; reason and metrics persist per attempt | Capture provider/cache evidence from a staging run and verify the deployed read API cannot publish a rejected snapshot |
| Private-server authorization | Black-box probe exists at `scripts/validate_deployed_pipeline_auth.py` | Run with real reader and operator identities against staging |
| Staging soak | Harness exists at `scripts/pipeline_soak.py` | Complete a bounded soak with no stuck, failed, or duplicated jobs |
| Migration/application rollback | A disposable rehearsal now proves candidate migration, predeploy-code compatibility, and byte-identical SQLite restore without touching the configured database | Run the same probe from the staging image with the deployed private API module, then rehearse the production database backup/restore procedure |
| Shadow weekly run | A same-database 2026 Week 1 attempt exercised both paths, but both stopped in `prepare_week`: the installed ingestion dependency supports seasons only through 2025 and no live-odds credential was available. Matching empty-set hashes are explicitly not equivalence evidence | Rerun on a supported pre-kickoff week with a frozen point-in-time database and live-odds credential; compare every behavior-bearing output |
| Runtime monitoring | Authenticated metrics expose queue, lease, retry, failure, and stage-duration data; the monitor emits a one-shot SHA-bound synthetic probe and can bind provider delivery confirmation | Retain the real staging delivery identifier in the release manifest |

## Authoritative local database matrix

```bash
# SQLite
python -m pytest tests/test_pipeline_database_matrix.py -v

# MySQL (dedicated disposable database)
TEST_DB_BACKEND=mysql \
TEST_DB_URL=mysql://user:password@127.0.0.1:3306/nfl_test \
python -m pytest tests/test_pipeline_database_matrix.py -v
```

The MySQL matrix runs the real migration manager and database driver against MySQL 8 or newer.
Startup rejects MySQL 5.7 and MariaDB because the queue relies on MySQL 8 locking semantics.
Mocked dialect tests do not satisfy this gate.

## Live-odds acceptance policy

Production defaults require every scheduled event and every configured event/market pair to be
covered by a snapshot no older than five minutes. Configure stricter values when necessary:

```bash
NFL_ODDS_MAX_AGE_SECONDS=300
NFL_ODDS_MIN_EVENT_COVERAGE=1.0
NFL_ODDS_MIN_MARKET_COVERAGE=1.0
NFL_ODDS_MIN_SPORTSBOOKS_PER_EVENT_MARKET=2
NFL_ODDS_REQUIRED_MARKETS=player_pass_yds,player_rush_yds,player_rec_yds
```

Only fresh `MISS` and `HIT` responses are trusted. `HIT-OFFLINE`, `STALE-ON-ERROR`, fallback
snapshots, cache entries without source timestamps, unknown or incomplete response provenance,
malformed validation metrics, incomplete coverage, and insufficient sportsbook breadth are always
rejected. A successful durable run stages its final card and promotes it only
inside the fenced completion transaction. See [Durable Pipeline State Machine](PIPELINE_STATE_MACHINE.md).

## Deployed authorization proof

The reader token must belong to an authenticated non-operator. The operator token may be a real
operator session or the dedicated pipeline control token. This command intentionally creates one
staging job.

```bash
python -m scripts.validate_deployed_pipeline_auth \
  --base-url https://staging-api.example.com \
  --reader-token "$STAGING_READER_TOKEN" \
  --operator-token "$STAGING_OPERATOR_TOKEN" \
  --candidate-sha "$CANDIDATE_SHA" \
  --season "$SEASON" --week "$WEEK" \
  --output evidence/authorization.json
```

## Shadow output comparison

Capture the legacy and queued paths from isolated databases created from the same point-in-time
input snapshot, then compare behavior-bearing rows. Timestamps are excluded; values, decisions,
risk results, and cards must match.

```bash
python -m scripts.shadow_weekly_outputs capture \
  --season "$SEASON" --week "$WEEK" --commit-sha "$LEGACY_SHA" \
  --run-report legacy-run.json --api-state legacy-api.json \
  --output evidence/legacy-shadow.json
python -m scripts.shadow_weekly_outputs capture \
  --season "$SEASON" --week "$WEEK" --commit-sha "$CANDIDATE_SHA" \
  --run-id "$QUEUED_RUN_ID" --output evidence/queued-shadow.json
python -m scripts.shadow_weekly_outputs compare \
  evidence/legacy-shadow.json evidence/queued-shadow.json \
  --output evidence/shadow-weekly-run.json
```

Comparison fails when either capture is empty or lacks freshness, projections, odds, candidates,
stage timing, checksummed artifacts, semantic artifact content, or API-visible state. Exact input
and decision timestamps must match. Run-report timestamps and durations are normalized for semantic
comparison, while their original file checksums remain recorded as integrity evidence.

## Algorithm improvement proof

From each exact checkout, evaluate its persisted pre-kickoff projections and completed run report on
identical season/week inputs, then compare the reports. The evaluator derives the checkout SHA; it
does not accept a caller-supplied label. The comparison fails if scope differs, coverage falls, the
required overall MAE improvement is missed, or any market regresses beyond the configured cap.

```bash
python -m scripts.evaluate_nfl_projections evaluate \
  --season 2025 --weeks 1 2 3 \
  --output evidence/algorithm-baseline.json
python -m scripts.evaluate_nfl_projections evaluate \
  --season 2025 --weeks 1 2 3 \
  --output evidence/algorithm-candidate.json
python -m scripts.evaluate_nfl_projections compare \
  evidence/algorithm-baseline.json evidence/algorithm-candidate.json \
  --min-improvement-pct 1 --max-market-regression-pct 5 \
  --output evidence/algorithm-evaluation.json
```

This gate measures persisted production outputs. It does not train a surrogate model from the same
week's outcomes and does not accept projections created at or after kickoff.

## Staging soak

```bash
python -m scripts.pipeline_soak \
  --base-url https://staging-api.example.com \
  --operator-token "$STAGING_OPERATOR_TOKEN" \
  --candidate-sha "$CANDIDATE_SHA" \
  --season "$SEASON" --week "$WEEK" --jobs 10 --timeout-seconds 3600 \
  --output evidence/staging-soak.json
```

Preserve the command output, service logs, metrics snapshot, database duplicate checks, and alert
delivery evidence with the release record. A local pass cannot substitute for this staging gate.

## Rollback rehearsal

The local rehearsal uses only disposable files. In staging, keep the default `api.application:app`
probe so the predeploy code must load the real private integration on the candidate schema.

```bash
python -m scripts.rehearse_pipeline_rollback \
  --predeploy-ref "$PREDEPLOY_SHA" \
  --candidate-sha "$CANDIDATE_SHA" \
  --output evidence/rollback-rehearsal.json
```

## Monitoring delivery proof

Emit exactly one synthetic warning, verify it arrived in the configured alert destination, then
bind that provider incident or notification identifier to the probe.

```bash
python -m scripts.queue_monitor --once --synthetic-alert \
  --candidate-sha "$CANDIDATE_SHA" --output evidence/monitor-probe.json
python -m scripts.queue_monitor \
  --confirm-probe evidence/monitor-probe.json \
  --delivery-id "$ALERT_DELIVERY_ID" \
  --output evidence/monitoring-delivery.json
```

## Final promotion manifest

Every evidence document must contain `passed: true` and the exact candidate SHA. The manifest also
verifies that the current `origin/main` commit is an ancestor of that candidate.

```bash
python -m scripts.pipeline_release_evidence \
  --candidate-sha "$CANDIDATE_SHA" \
  --baseline-sha "$LEGACY_SHA" \
  --algorithm-evaluation evidence/algorithm-evaluation.json \
  --database-matrix evidence/database-matrix.json \
  --staging-failure-safety evidence/staging-failure-safety.json \
  --authorization evidence/authorization.json \
  --staging-soak evidence/staging-soak.json \
  --rollback-rehearsal evidence/rollback-rehearsal.json \
  --shadow-weekly-run evidence/shadow-weekly-run.json \
  --monitoring-delivery evidence/monitoring-delivery.json \
  --output evidence/promotion-manifest.json
```
