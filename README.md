# NFL Algorithm

Predicts weekly NFL player props, prices them against sportsbook lines, and ranks the
disagreements by expected value. An NBA module runs the same shape of pipeline.

**Version**: 2.1 | **Status**: Production validation in progress | **Sports**: NFL + NBA

---

## System Overview

Each week the pipeline ingests player stats, rosters, schedules and snap counts from nflverse,
trains position-specific models on strictly earlier weeks, and projects a mean and a standard
deviation per player and market. It scrapes prop lines from The Odds API, strips the bookmaker
margin off each two-sided quote, and compares the model's probability against the book's. Bets
that clear the edge threshold get a Kelly-capped stake and land in a materialized view the
dashboards read.

After the games, `scripts/record_outcomes.py` grades every bet against real stats and records
closing line value, so each week's card is scored against both the result and the market.

Markets covered: passing yards, rushing yards, receiving yards, receptions, anytime touchdown.

Read [Performance Metrics](#performance-metrics) before trusting any of it. One week of real
results is in, and it is break-even.

---

## Quick Start

```bash
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm

cp .env.example .env
make install
make frontend-install
make migrate
make doctor             # a WARN row for private_modules is expected on a public clone

make fullstack          # worker + API + frontend at http://localhost:3000
```

`make ingest-nfl` pulls real NFL data. No `.env` edit is needed for local work: SQLite is the
default and `ODDS_API_KEY` only matters for live odds.

### Weekly Workflow

```bash
make week-predict SEASON=2026 WEEK=2       # train on prior weeks, project this one
make week-materialize SEASON=2026 WEEK=2   # build the value card
make week-grade SEASON=2026 WEEK=2         # after the games: grade bets, record CLV
make mae-gate SEASON=2026 WEEK=2           # fail if a position regressed past its ceiling
```

These are the local path. Production runs through a durable worker instead. See
[docs/OPERATIONS.md](docs/OPERATIONS.md).

---

## Architecture

FastAPI only creates durable jobs and reads materialized results. A separate worker owns the
fail-closed pipeline, bounded retries, cancellation, stage tracking and artifact registration.
Full topology in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Promotion to production is
governed by the evidence gates in [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md);
a green unit suite is not production readiness.

```
nflalgorithm/
├── models/              # Position-specific ML models
├── utils/               # Tracked math: CLV, grading, sigma, market pricing, player matching
├── sports/              # Market registry shared by NFL and NBA
├── config/              # Tracked runtime defaults
├── scripts/             # Ingest, scrape, predict, grade, backtest
├── api/                 # FastAPI service
├── frontend/            # Next.js dashboard
├── dashboard/           # Legacy Streamlit UI
├── tests/               # 2,248 tests
├── docs/                # Architecture, operations, model card, troubleshooting
├── prop_integration.py  # 3-tier player matching
└── materialized_value_view.py  # Dashboard data layer
```

Four modules are gitignored and supplied by the deployment: `data_pipeline.py`,
`value_betting_engine.py`, `models/position_specific/weekly.py`, and an optional `config.py`
override. Almost everything else works without them, and CI runs the full tracked suite.
[CONTRIBUTING.md](CONTRIBUTING.md) explains what is affected.

Math that CI must verify lives in tracked `utils/`, specifically so the private modules can call
it while the tests still run on a clean clone.

### Data Sources

| Source | Purpose | Update frequency |
|--------|---------|------------------|
| nflreadpy / nflverse | Player stats, schedules, rosters, snap counts, depth charts | Nightly |
| The Odds API | Prop lines from multiple books | Real-time |

See [docs/DATA_RETENTION.md](docs/DATA_RETENTION.md) before importing history or keeping provider
payloads. Hand-authored projection CSVs are not valid training evidence.

### How Bets Are Priced

- Yardage markets price off a gamma curve matched to the model's mean and standard deviation.
  A normal curve ran 4 to 11 percentage points long on real data because weekly yardage is
  right-skewed and bounded at zero. Anytime touchdown prices off Poisson survival.
- The bookmaker margin is removed from each two-sided quote before the comparison, so the edge
  is measured against a fair probability rather than the posted one.
- Stakes are Kelly-capped per bet and the whole card is scaled to the bankroll, so persisted
  stakes never sum past it.
- Predictions are adjusted for how a player performs against a specific defense relative to their
  own average.
- Player names are matched across feeds in three tiers (id, then name plus team, then fuzzy),
  each guarded so two different players at different positions cannot collide.

Details and the measurements behind each choice are in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

---

## Performance Metrics

### 2026 Week 1 (first graded week)

All 16 games final. Every number here is read back out of `bet_outcomes` and `clv_weekly` after
the fact, not simulated.

| Metric | Result |
|--------|--------|
| Bets placed | 453 (447 graded, 6 pushes) |
| Record | 238-209-6 |
| Profit | +0.76 units |
| ROI | +0.17% |
| Average CLV | -11.5 basis points |
| Coverage | 87 players across 16 games |

Break-even. The negative CLV says the card was priced slightly worse than where the market
closed, so the small profit came from outcomes rather than from beating the line.

| Market | Bets | Win rate | ROI |
|--------|------|----------|-----|
| passing_yards | 69 | 72.5% | +35.9% |
| receiving_yards | 242 | 50.4% | -5.4% |
| rushing_yards | 132 | 47.0% | -11.6% |
| receptions | 4 | 100.0% | +110.0% |

Split by the edge the engine computed at placement:

| Edge bucket | Bets | Win rate | ROI |
|-------------|------|----------|-----|
| 8-12% | 128 | 57.8% | +8.6% |
| 12-16% | 144 | 57.6% | +8.5% |
| 16-20% | 54 | 53.7% | +1.9% |
| 20-25% | 58 | 46.6% | -12.6% |
| 25%+ | 63 | 39.7% | -25.7% |

The ordering runs backwards from what the ranking assumes. A very large computed edge on a prop
line more often means the projection is wrong than that the book is. The confidence tiers show
the same split: HIGH took 211 bets for -15.1% ROI, MEDIUM took 236 for +13.8%. This is one week
and 447 graded bets, so it is a flag to re-check after week 2, not yet a reason to move a
threshold.

### Projection Accuracy

From the walk-forward backtest (`make nfl-backtest SEASON=2025`), which retrains the model each
week on strictly earlier data: 5,117 predictions, overall MAE 26.88 yards, bias +2.81.
Per-position worst single week was QB 59.5, RB 24.0, WR 26.0, TE 24.6, and `make mae-gate` fails
the build about 10% above each. One-sigma coverage is 68.2% after the recalibration in
`utils/nfl_sigma.py`.

Season and week totals are in `weekly_performance`. Per-bet detail is in `bet_outcomes` and
`clv_weekly`.

---

## Dashboards

![NFL Algorithm Dashboard](docs/dashboard-screenshot.png)

The React dashboard (Next.js 16, TypeScript, Tailwind, shadcn/ui, Recharts) is the current one,
served by FastAPI off the same database. `make fullstack` starts the worker, the API and the
frontend with readiness waiting and cleanup. To run them separately:

```bash
make api              # terminal 1: migrate, then serve
make pipeline-worker  # terminal 2
make frontend-dev     # terminal 3, then visit http://localhost:3000
```

Pages: Dashboard, Bets, Performance, Analytics, Backtest, System and Settings, plus the NBA set
under `/nba`. Filters cover best-line-only, minimum edge, and a simplified quick-bet view.

The original Streamlit UI still works via `make dashboard` at `http://localhost:8501`. It is
legacy and not where new work goes.

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) covers database, migration, API-key,
private-module, CORS, port and deployment failures.

---

## NBA Module

![NBA Dashboard](docs/nba-dashboard.png)

A full player-props pipeline for points, rebounds, assists and three-pointers, sharing the market
registry and dashboard shell with the NFL side. A dedicated minutes model feeds per-minute rate
projections; a stacking ensemble (gradient boosting, random forest, XGBoost, Ridge meta-learner)
produces the stat lines; opponent defense ratings, EWMA volatility, isotonic calibration, injury
redistribution and a 5,000-draw Monte Carlo for correlated portfolios sit on top. Both sides of
each line are evaluated. Walk-forward backtesting, PSI drift detection and SHAP importance run as
their own targets.

```bash
make ingest-nba       # 2024 + 2025 seasons
make nba-train        # 4 markets
make nba-predict      # today's projections
make nba-full         # ingest, predict, odds, value ranking
make nba-run          # daily production pipeline
make nba-tune         # Optuna search per market
make nba-backtest     # also: nba-drift, nba-importance, nba-calibrate
```

Dashboard at `http://localhost:3000/nba` after `make fullstack`.

---

## Configuration

SQLite is the default and needs no `.env` edit.

```env
DB_BACKEND=sqlite
SQLITE_DB_PATH=nfl_data.db
```

For production, MySQL 8.0 or newer. MySQL 5.7 and MariaDB are rejected at connection time.

```env
DB_BACKEND=mysql
DB_URL="mysql://user:pass@host:port/database"
```

Odds and logging:

```env
ODDS_API_KEY=your_odds_api_key
NFL_ODDS_MAX_AGE_SECONDS=300
NFL_ODDS_MIN_EVENT_COVERAGE=1.0
NFL_ODDS_MIN_MARKET_COVERAGE=1.0
NFL_ODDS_MIN_SPORTSBOOKS_PER_EVENT_MARKET=2
NFL_ODDS_REQUIRED_MARKETS=player_pass_yds,player_rush_yds,player_reception_yds,player_receptions
LOG_FORMAT=console  # use json in deployments
LOG_LEVEL=INFO
```

The UI and read-only API start without `ODDS_API_KEY`, but a live-odds NFL run fails closed.
`make doctor-production` additionally requires the deployment-supplied private modules and
rejects `DEMO_MODE=true`.

Check a week's inputs before queueing it, and its evidence afterwards:

```bash
make doctor-season SEASON=2026 WEEK=2
make doctor-season SEASON=2026 WEEK=2 SEASON_PHASE=post-run
```

A zero-play card is a warning, not a failure: rejecting every candidate can be correct.

Never commit `.env`, database credentials or API keys.

---

## Testing

```bash
make test          # 2,248 tests
make lint          # mypy
make format        # black + isort
make validate SEASON=2025 WEEKS="1 2 3"   # score persisted pre-kickoff projections
```

`make list-targets` prints every Make target.

---

## Roadmap

Shipped and validated: defense-vs-position adjustments, nflverse ingest, best-line
deduplication, three-tier player matching, dual SQLite/MySQL backends, no-vig edge calculation,
CLV capture, walk-forward backtesting with a per-position MAE gate, gamma pricing for yardage
markets, and the full NBA pipeline.

Shipped but not yet validated on real data:

- Game-script and usage-trend multipliers (`utils/context_factors.py`), live behind
  `NFL_FEATURE_CONTEXT_FACTORS` with no backtest measuring them yet.
- Receptions and anytime touchdown, which no backtest has covered.
- The market-mean blend in `utils/market_blend.py`, tested but not wired into the ranking path.

Open:

- Line movement tracking.
- Same-game correlation.
- Why a larger computed edge lost money in week 1. This is the highest-value open question.

---

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first: it covers which files
are private, what works without them (almost everything), and what CI runs on a pull request.

On a public clone, `make doctor` prints `WARN` for `private_modules`. That is expected. If setup
breaks, open an issue with the "Setup problem" template.

---

## License

MIT. See [LICENSE](LICENSE).
