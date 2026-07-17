# Pipeline production-readiness gates

The queue/worker architecture is an implementation candidate, not a production-ready runtime.
Promotion requires evidence from the deployed private integration and staging environment in
addition to local tests.

| Gate | Current evidence | Remaining proof |
|---|---|---|
| Known commit SHA | Satisfied for the local candidate; the final SHA is recorded in the handoff | Use that exact SHA for staging and retain it with the release evidence |
| Integrated with `origin/main` | Satisfied locally after rebasing onto the fetched `origin/main`; the fetched main commit is an ancestor of the candidate | Recheck ancestry after the final fetch before staging promotion |
| Old/new output equivalence | Deterministic inline-versus-worker report matrix passes on SQLite and MySQL | Run one real weekly legacy capture and queued shadow capture; require matching hashes |
| SQLite/MySQL matrices | Real SQLite and temporary MySQL 9.5 migration, concurrency, recovery, and equivalence checks pass locally; MySQL 8.4 CI service is configured | Require green remote CI for the committed SHA |
| No double claims | Eight concurrent claimers produce one winner on both databases | Repeat under staging worker concurrency and retain logs |
| Crash recovery/fencing | Simulated worker process termination, lease recovery, stale completion rejection, and stale stage-write rejection pass on both databases | Kill a staging worker during a real stage and verify exactly one terminal run/card |
| Live-odds fail closed | Odds failure stops before materialization and marks the run unsuccessful | Verify the deployed read API cannot publish a stale or failed-run card |
| Private-server authorization | Black-box probe exists at `scripts/validate_deployed_pipeline_auth.py` | Run with real reader and operator identities against staging |
| Staging soak | Harness exists at `scripts/pipeline_soak.py` | Complete a bounded soak with no stuck, failed, or duplicated jobs |
| Migration/application rollback | Migrations are idempotent on fresh SQLite and MySQL databases | Rehearse database backup, deploy, application rollback, and post-rollback reads in staging |
| Shadow weekly run | Deterministic snapshot/compare tool exists at `scripts/shadow_weekly_outputs.py` | Compare a real legacy weekly run with the queued candidate using identical point-in-time inputs |
| Runtime monitoring | Authenticated metrics expose queue, lease, retry, failure, and stage-duration data; the supervised runtime emits structured monitoring logs | Confirm staging ingestion, alert routing, and a synthetic alert notification |

## Authoritative local database matrix

```bash
# SQLite
python -m pytest tests/test_pipeline_database_matrix.py -v

# MySQL (dedicated disposable database)
TEST_DB_BACKEND=mysql \
TEST_DB_URL=mysql://user:password@127.0.0.1:3306/nfl_test \
python -m pytest tests/test_pipeline_database_matrix.py -v
```

The MySQL matrix runs the real migration manager and database driver. Mocked dialect tests do not
satisfy this gate.

## Deployed authorization proof

The reader token must belong to an authenticated non-operator. The operator token may be a real
operator session or the dedicated pipeline control token. This command intentionally creates one
staging job.

```bash
python -m scripts.validate_deployed_pipeline_auth \
  --base-url https://staging-api.example.com \
  --reader-token "$STAGING_READER_TOKEN" \
  --operator-token "$STAGING_OPERATOR_TOKEN" \
  --season 2026 --week 1
```

## Shadow output comparison

Capture the legacy and queued paths from isolated databases created from the same point-in-time
input snapshot, then compare behavior-bearing rows. Timestamps are excluded; values, decisions,
risk results, and cards must match.

```bash
python -m scripts.shadow_weekly_outputs capture --season 2026 --week 1 --output legacy.json
python -m scripts.shadow_weekly_outputs capture --season 2026 --week 1 --output queued.json
python -m scripts.shadow_weekly_outputs compare legacy.json queued.json
```

## Staging soak

```bash
python -m scripts.pipeline_soak \
  --base-url https://staging-api.example.com \
  --operator-token "$STAGING_OPERATOR_TOKEN" \
  --season 2026 --week 1 --jobs 10 --timeout-seconds 3600
```

Preserve the command output, service logs, metrics snapshot, database duplicate checks, and alert
delivery evidence with the release record. A local pass cannot substitute for this staging gate.
