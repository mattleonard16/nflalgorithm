# Contributing

Thanks for looking at this. Bug reports, setup problems, docs fixes, and tests are all welcome,
and a first-time contributor hitting a rough edge is useful information, not noise. If setup
did not work for you, open an issue with the "Setup problem" template. That is a bug on our side.

## What is public and what is not

Six files are gitignored and never published: `config.py`, `data_pipeline.py`,
`value_betting_engine.py`, `prop_integration.py`, `models/position_specific/weekly.py`, and
`api/server.py`. They hold the proprietary model and pricing code.

Everything else is public and works on its own:

- `make install`, `make migrate`, `make doctor`
- `make test` (tests that import the private files skip automatically, see `tests/conftest.py`)
- `make ingest-nfl`, `make nfl-backtest`, `make frontend-build`

Only `make api` and `make fullstack` need the private files. `make doctor` prints `WARN` for
`private_api` and `private_modules` on a public clone. That is expected and is not a failure.
See docs/TROUBLESHOOTING.md for the full table.

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
