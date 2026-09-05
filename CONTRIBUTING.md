# Contributing

Thanks for looking at this. Bug reports, setup problems, docs fixes, and tests are all welcome,
and a first-time contributor hitting a rough edge is useful information, not noise. If setup
did not work for you, open an issue with the "Setup problem" template. That is a bug on our side.

## What is public and what is not

Three files are gitignored and never published: `data_pipeline.py`, `value_betting_engine.py`,
and `models/position_specific/weekly.py`. They hold the model and the pricing logic. A fourth,
`config.py`, is only an override; tracked defaults live in `config/runtime.py`.

Everything else is public, including the API (`api/server.py`) and the player matching
(`prop_integration.py`). Both were published on 2026-09-05.

Working without the private files:

- `make install`, `make migrate`, `make doctor`, `make test`
- `make api`, `make fullstack`
- `make ingest-nfl`, `make nfl-backtest`, `make frontend-build`

`make week-predict` and the other live projection runs need the model, so they will not work on a
clone. `make doctor` prints `WARN` for `private_modules`. That is expected, not a failure. See
docs/TROUBLESHOOTING.md for the full table.

Because CI never sees the private files, logic that CI must verify belongs in a tracked module.
`utils/clv.py` is the pattern: the math lives in a tracked file with tests, and the private
callers import it.

## Setup

```bash
cp .env.example .env
make install
make migrate
make doctor          # WARN rows for private_* are fine
make test
```

## Before opening a pull request

- `make test` passes locally.
- New behavior has a test. Bug fixes come with a test that failed before the fix.
- Python: 4 spaces, 100-character lines, type hints on new functions. Match the surrounding file.
- TypeScript: Prettier formatting, `npm run lint` in `frontend/`.
- Commit subject: `<type>(<scope>): <description>` with `feat`, `fix`, `refactor`, `docs`,
  `test`, `chore`, or `perf`.
- Do not commit `.env`, database files, or anything under the gitignored list above.

## What CI runs on your pull request

| Workflow | Job | What it checks |
|---|---|---|
| CI | `fresh-clone-setup` | Migrate an empty database and run `make doctor` on a clone with no private files. Must exit 0 with only WARN rows. Also checks that no proprietary file is tracked |
| CI | `test` | Full pytest suite on SQLite |
| CI | `pipeline-database-matrix` | Queue and lease tests against a real MySQL 8.4 |
| CI | `lint-frontend` | `npm run lint` and `npm run build` |
| Security | `pip-audit` | Known vulnerabilities in `requirements.txt` (blocking) |
| Security | `bandit` | Static security scan of tracked Python, high severity (blocking) |
| Security | `secrets` | Gitleaks plus a check that no proprietary file is tracked (blocking) |
| Security | `npm-audit` | Frontend advisories (report-only for now) |
| Security | `codeql` | GitHub code scanning for Python and TypeScript |

Dependabot opens weekly grouped update PRs for pip, npm, and GitHub Actions.

If a check fails and you cannot tell why from the log, say so in the PR. Flaky or confusing
checks are our problem to fix.

## Reporting a security issue

Do not open a public issue for a vulnerability. Email the maintainer listed in the repository
profile instead.
