# Opening the public contribution boundary

Phases 1 and 2 are done. Phase 3 is not started.

## Why

An outside contributor cloned the repo, ran `make doctor`, and got a FAIL because `api/server.py`
was missing. The preflight fix earlier in this branch turned that into a warning. Correct, but it
treated a symptom.

The real problem was where the public/private line sat. History put it there, not a decision about
what needed protecting. Six files were gitignored. Only two of them held anything.

## What moved

| File | Lines | Now |
|---|---|---|
| `api/server.py` | 1,699 | Public. It is CRUD over the database and holds no edge. |
| `prop_integration.py` | 705 | Public. It imports no private module; the matching guards already live in tracked `utils/matching.py`. |
| `config.py` | 208 | Unchanged, and it was never a problem. See below. |
| `value_betting_engine.py` | 334 | Private for now. Kelly and no-vig are textbook, but `rank_weekly_value` is real selection logic. |
| `data_pipeline.py` | 1,243 | Private. Ingest is public nflverse data, feature engineering is not. |
| `models/position_specific/weekly.py` | 1,318 | Private. This is the model. |

Publishing those two bought three things:

- **182 API tests now run in CI.** They never had, because `tests/conftest.py` skipped ten files
  whenever `api/server.py` was absent, which in CI was always.
- **`docs/DEPLOYMENT_MANIFEST.md` lost two sections.** That file hand-checks what git does for free,
  and every entry exists only because version control is off for something. It is now shorter.
- **Contributors can send patches** for the API and the matching layer.

## The pattern to reuse

`config.py` was never the problem. It is the model for phase 3.

`config/__init__.py` is tracked. It loads defaults from `config/runtime.py`, then looks for a
gitignored `config.py` and prefers it when found, filling any gaps from the tracked defaults. A
clone runs on the defaults and never notices. A deployment drops in its file and overrides what it
wants.

## Phase 3: a runnable baseline model

This is what lets someone contribute rather than just read.

Define the projector interface in a tracked module and ship a plain public implementation:
last-6-week EWMA per player per market, no defense adjustment, no role priors. Then apply the
`config/` pattern in `models/position_specific/__init__.py`, so a private `weekly.py` overrides the
baseline when present.

A contributor could then run the pipeline end to end and measure a change with `make nfl-backtest`.
Today they get an empty dashboard and no way to tell whether their patch helped.

## What cannot be undone

Publishing is one-way. Files pushed to a public repo are cached and indexed, and deleting them later
does not retract them. Phase 3 needs a deliberate yes before its push, not just before its commit.

Everything else reverses. Re-adding a `.gitignore` line plus `git rm --cached` moves a file back to
the private side for future clones, though not retroactively.

## Verification

What phases 1 and 2 were checked against, and what phase 3 should match:

- `make test` green in the working tree.
- Full suite green in a tracked-files-only clone, built with
  `git ls-files -z | rsync -a --files-from=- --from0 ./ "$CLONE"/`. This is what CI and a contributor
  actually get. It catches a test fenced under the wrong module, which a working-tree run cannot.
- `make doctor` exits 0 in that clone with no FAIL rows.
- No hardcoded secrets in anything newly published. `api/server.py` and `prop_integration.py` were
  scanned for credentials, keys, emails, home paths, hostnames, and high-entropy strings before the
  push. The only hit was `http://localhost` in the CORS defaults.

## Not doing

Splitting the private half into its own repo behind a published plugin interface. That is the heavy
version of phase 3, and it only pays off once there are contributors to serve. One is not that
signal. Revisit if a second and third turn up.
