# Plan: open the public contribution boundary

Status: proposed, not started. Nothing in this plan has been implemented.

## The problem

Six files are gitignored. An outside contributor cloned the repo, ran `make doctor`, and got a
FAIL because `api/server.py` was missing. That prompted the preflight fix in this branch, which
turns the FAIL into a warning. The warning is correct, but it treats a symptom.

The real issue is where the public/private line sits. It was drawn by history, not by a decision
about what needs protecting.

| File | Lines | Holds real edge? |
|---|---|---|
| `api/server.py` | 1,699 | No. It is CRUD over the database. |
| `models/position_specific/weekly.py` | 1,318 | Yes. This is the model. |
| `data_pipeline.py` | 1,243 | Partly. Ingest is public nflverse data; feature engineering is not. |
| `prop_integration.py` | 705 | Barely. The matching guards already live in tracked `utils/matching.py`. |
| `value_betting_engine.py` | 334 | Mostly no. Kelly sizing and no-vig removal are textbook. |
| `config.py` | 208 | No, and it is already handled. See below. |

Three costs, all of them already being paid:

1. **182 test functions cannot run in CI.** `tests/conftest.py:58` skips ten API test files whenever
   `api/server.py` is absent, which is always true in CI. That is `test_nba_api_contract.py` (85),
   `test_nba_api.py` (28), `test_export_api.py` (16), `test_pipeline_run_api.py` (13),
   `test_projections_api.py` (13), `test_api_contract.py` (7), `test_risk_api.py` (6),
   `test_record_bet_api.py` (5), `test_api_visibility.py` (5), `test_readiness.py` (4).
2. **`docs/DEPLOYMENT_MANIFEST.md` is a 186-line manual checklist** doing by hand what git does for
   free. It exists only because version control is switched off for these files. It will drift, and
   when it does the failure is silent. Its own line 88 says an older engine copy "silently changes
   every price on the card."
3. **No one can send a patch** for anything on the private side.

## What already works, and should be copied

`config.py` is not actually a problem. It is the pattern to reuse.

`config/__init__.py` (tracked) imports working defaults from `config/runtime.py`, then looks for a
gitignored top-level `config.py`. If that file exists it is loaded by path and its values win, with
`_fill_missing_settings` filling any gaps from the tracked defaults. A fresh clone runs on the
tracked defaults and never notices. A deployment drops in its private file and overrides whatever
it wants.

That is tracked-default-plus-optional-override, and it is the right shape. The rest of this plan is
applying it to the other five files.

## Phase 1: publish `api/server.py`

Move it into git. Delete `.gitignore:118`.

Dependency check, already done: the only private module it imports is `config`, and `config` already
resolves to the tracked package. Everything else it needs is tracked today (`api/auth.py`,
`api/pipeline_router.py`, `api/value_visibility.py`, `api/nba_router.py`, `utils/db.py`).

Secrets check, already done: no hardcoded credential literals, and the only URL in the file is
`http://localhost` in the CORS defaults. Run `gitleaks` in CI on the merge to confirm.

Then:

- Delete the `_private_api_tests` block from `tests/conftest.py:45-59`. Those 182 tests start running.
- Drop `--require-private-api` from `api-runtime-preflight` in the `Makefile`, and delete
  `check_private_api`'s missing-file branch from `scripts/preflight.py`. The stale-contract check
  stays; it is still worth catching.
- Delete the `api/server.py` section from `docs/DEPLOYMENT_MANIFEST.md:112-135`.
- Reply to the contributor. The warning he hit stops existing rather than becoming accurate.

Expected breakage: the 182 newly-collected tests have never run in CI. Some will fail on
environment assumptions. Budget time for that, and do not merge Phase 1 until they are green.

## Phase 2: publish `prop_integration.py`

Move it into git. Delete its `.gitignore` line.

Dependency check, already done: it imports no private module at all. Only `utils.db`,
`utils.matching`, and `utils.player_id_utils`, all tracked.

The conftest gate needs splitting first. `tests/conftest.py:42` uses
`all(path.is_file() for path in _private_algorithm_files)`, so it fences all 18 algorithm test files
if any one of the four is missing. Change it to a per-file map so publishing one module unlocks that
module's tests:

```python
_private_algorithm_tests = {
    Path(_project_root) / "prop_integration.py": ["test_prop_integration_wr.py", ...],
    Path(_project_root) / "data_pipeline.py": [...],
    ...
}
collect_ignore = [t for path, tests in _private_algorithm_tests.items()
                  if not path.is_file() for t in tests]
```

Do this split before moving the file, as its own commit, so the unlock is visible in the diff.

## Phase 3: a runnable baseline model

This is what actually lets someone contribute rather than just read.

Define the projector interface in a tracked module and ship a simple public implementation:
last-6-week EWMA per player per market, no defense adjustment, no role priors. Then apply the
`config/` pattern. `models/position_specific/__init__.py` loads the tracked baseline, checks for a
gitignored `weekly.py`, and prefers it when present.

A contributor can then run the whole pipeline end to end and get real, mediocre numbers. They can
measure a change against `make nfl-backtest`. Today they get an empty dashboard and no way to tell
whether their patch helped.

Keep private: the feature engineering in `data_pipeline.py`, the calibrated sigma tables that are
already tracked in `utils/nfl_sigma.py` stay tracked, and the trained `.joblib` artifacts stay out.

## What stays private after all three phases

`models/position_specific/weekly.py` and the feature-engineering half of `data_pipeline.py`. That is
the edge. Everything else is public.

`value_betting_engine.py` is a judgment call left open. Kelly sizing and no-vig removal are in every
betting textbook, and `utils/clv.py` already had to work around the engine being private. The only
part with a real claim to secrecy is `rank_weekly_value`'s selection logic. Revisit after Phase 3.

## What breaks, and what cannot be undone

Publishing is one-way. Once these files are pushed to a public repo they are cached and indexed, and
deleting them later does not retract them. Every phase here needs a deliberate yes before the push,
not just before the commit.

Everything else is reversible:

- Re-adding a `.gitignore` line and `git rm --cached` puts a file back on the private side for future
  clones, though not retroactively.
- The conftest and preflight changes are ordinary code.

## Verification for each phase

- `make test` green, and report the new total. Phase 1 should move it from 2,170 to roughly 2,350.
- Fresh-clone simulation: build a tree from tracked files only, `make migrate`, then `make doctor`.
  Exit 0 with no FAIL. The `fresh-clone-setup` CI job already does this.
- `gitleaks` clean on the newly tracked files.
- After Phase 3, `make nfl-backtest SEASON=2025` runs to completion on a tracked-files-only tree.

## Not doing

Splitting the private half into a separate repo with a published plugin interface. That is the
heavier version of Phase 3 and only pays off once there are contributors to serve. One outside
contributor is not that signal yet. Revisit if a second and third turn up.
