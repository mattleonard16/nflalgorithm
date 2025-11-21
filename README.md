# NFL Algorithm - Professional Value Betting System

*Advanced NFL player performance prediction and value betting engine*

**Version**: 2.0 | **Status**: Production Ready | **Target MAE**: ≤ 3.0

## System Overview

This is a comprehensive NFL betting algorithm that combines:
- **Machine Learning Models**: Position-specific predictive models
- **Real-time Data Pipeline**: Live odds, weather, injuries, and player stats
- **Value Betting Engine**: Kelly Criterion optimization with CLV tracking
- **Professional Validation**: Cross-season backtesting and performance metrics
- **Automated Pipeline**: Scheduled data updates and model retraining

**Core Philosophy**: Achieve consistent profitability through disciplined, data-driven betting with rigorous risk management.

## Quick Start

> **Windows Users**: See [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) for Windows-specific installation instructions.

```bash
# Clone and setup
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm

# Install dependencies
make install

# Run validation
make validate

# Start optimization
make optimize

# Launch dashboard
make dashboard

# Start automated pipeline
make start
```

### Weekly Workflow (Game Weeks)

```bash
# 1. Ingest stats, odds, weather, injuries
make week-update SEASON=2023 WEEK=12

# 2. Generate weekly projections
make week-predict SEASON=2023 WEEK=12

# 3. Materialize dashboard view
make week-materialize SEASON=2023 WEEK=12

# 4. Run mini backtest replay
make mini-backtest SEASON=2023 WEEK=12

# 5. Launch Streamlit dashboard
make dashboard
```

**Quick Test**:
```bash
python cross_season_validation.py
# Should show MAE ≤ 3.0 for deployment readiness
```

**Development Setup**:
```bash
make dev-setup
# Installs all dev dependencies and pre-commit hooks
```

## Architecture

```
nflalgorithm/
├── models/position_specific/     # Position-focused ML models
├── data/                         # Data files (CSVs, projections)
├── docs/                         # Documentation files
├── scripts/                      # Utility scripts and tools
├── tests/                        # Comprehensive test suite
├── utils/                        # Utility modules (player_id_utils, etc.)
├── dashboard/                    # Streamlit monitoring
├── logs/                         # Performance tracking
├── data_pipeline.py             # Real-time data ingestion + baseline augmentation
├── value_betting_engine.py      # Value bet detection & CLV
├── prop_integration.py          # Player matching (normalized/fuzzy/ID)
├── materialized_value_view.py   # Dashboard materialization
├── optuna_optimization.py       # Hyperparameter tuning
├── cross_season_validation.py   # Professional validation
└── schema_migrations.py         # Database schema management
```

**Data Sources**:
- `data/2024_nfl_projections.csv`: Baseline season projections
- `data/2024_nfl_rookies.csv`: 2024 rookie projections (35 players)
- Real-time odds API integration
- Historical player stats database

**Data Flow**: Raw NFL data → Feature Engineering → ML Models → Predictions → Player Matching (Normalized/Fuzzy) → Value Detection → Bet Recommendations

**Player Matching**: Multi-strategy matching system:
- Exact player_id matching (highest confidence)
- Normalized name matching (ignores team mismatches)
- Fuzzy name matching (0.87 threshold for variations)
- Team/position validation and auto-correction

## Performance Metrics

| Metric | Current | Target | Status |
|--------|---------|---------|---------|
| Rushing MAE | **3.6** | ≤ 3.0 | Optimizing |
| Receiving MAE | **4.1** | ≤ 3.5 | Optimizing |
| Value Bet ROI | **15.2%** | > 12% | **ACHIEVED** |
| CLV Performance | **+2.3%** | > 0% | **ACHIEVED** |
| Player Match Rate | **28%** (19/69) | > 25% | **ACHIEVED** |
| Matched Opportunities | **91 rows** | > 50 | **ACHIEVED** |

**Validation Results**: 
- Cross-season testing (2021-2023)
- 5-fold time-series validation
- Out-of-sample performance tracking
- Kelly Criterion bet sizing
- Risk-adjusted returns

**Recent Improvements**:
- Enhanced feature engineering (+12 new features)
- Position-specific model architecture
- Advanced ensemble methods (Stacking, Voting)
- Real-time odds integration
- Professional CLV tracking
- **Player matching infrastructure (19x improvement: 1→19 players)**
- **Data quality fixes: team typo correction, position validation**
- **Rookie integration: 35 players with 2024 projections**
- **Passing yards market support with QB projections**

## Usage Examples

### 1. Run Cross-Season Validation
```python
from cross_season_validation import EnhancedCrossSeasonValidator

validator = EnhancedCrossSeasonValidator()
results = validator.run_validation()
print(f"Best MAE: {results['best_mae']:.3f}")
```

### 2. Find Value Bets
```python
from value_betting_engine import ValueBettingEngine

engine = ValueBettingEngine()
value_bets = engine.find_value_opportunities()
for bet in value_bets:
    print(f"{bet.player_name}: {bet.edge_percentage:.1f}% edge")
```

### 3. Optimize Hyperparameters
```python
from optuna_optimization import OptunaOptimizer

optimizer = OptunaOptimizer()
study = optimizer.optimize_all_models(X_train, y_train)
print(f"Best MAE: {study['best_mae']:.3f}")
```

### 4. Real-time Pipeline
```python
from data_pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.run_full_update()  # Updates all data sources
```

## Configuration

### Database Setup (MySQL)

The system supports both local SQLite (default) and remote MySQL databases (e.g., Sevalla, Kinsta).

1.  **Create Environment File**:
    Copy the example or create a new `.env` file in the project root:
    ```bash
    cp .env.example .env
    ```

2.  **Configure Credentials**:
    Edit `.env` to set your database backend and credentials.

    **For MySQL (Recommended for Production):**
    ```env
    DB_BACKEND=mysql
    DB_URL="mysql://username:password@host:port/database_name"
    ```

    **For SQLite (Local Development):**
    ```env
    DB_BACKEND=sqlite
    SQLITE_DB_PATH="nfl_data.db"
    ```

3.  **API Keys**:
    Add your required API keys to `.env`:
    ```env
    ODDS_API_KEY="your_odds_api_key"
    WEATHER_API_KEY="your_weather_api_key"
    ```

**⚠️ SECURITY WARNING**: 
*   **Never commit your `.env` file** to version control.
*   **Never expose your `DB_URL` or API keys** in public repositories, issues, or screenshots.
*   The `.env` file is already added to `.gitignore` to prevent accidental commits.

Key settings in `config.py`:
- **Target MAE**: 3.0 (rushing), 3.5 (receiving)
- **Min Edge**: 8% for value bets
- **Kelly Fraction**: 50% (conservative)
- **Update Intervals**: 5min (odds), 30min (injuries)

Caching knobs available under `config.cache`:
- `http_cache_expire_after`, `odds_cache_ttl_season`, `odds_cache_ttl_offseason`
- `weather_cache_ttl`, `weather_cache_ttl_dome`, `stale_while_revalidate_window`
- `cache_warm_enabled`, `http_cache_backend`, `http_cache_dir`
Use `make cache-warm`, `make cache-stats`, `make cache-clean` to manage cache.

## Dashboard Features

Access at `http://localhost:8501` after running `make dashboard`:

- **Live Bets**: Current value opportunities with real-time odds
- **Performance**: Model accuracy, ROI tracking, CLV analysis
- **System Status**: Pipeline health, data freshness, error logs
- **Historical**: Bet history, profit/loss, streak analysis

## Shareable Reports

Generate and export weekly value-betting reports and artifacts for easy sharing.

```bash
make report
```

Outputs are saved to `reports/`:
- `weekly_value_report.md` and `weekly_value_report.html`
- `value_bets.csv` and `value_bets.json`
- `img/top_edges.png` (and `roi_hist.png` if ROI available)
- `enhanced_dashboard.html` (share-ready HTML dashboard)
- `value_bets_enhanced.csv` (normalized columns incl. `confidence_score`)
- `quick_picks.md` (shortlist for quick action)

Launch the live dashboard:

```bash
make dashboard
```

Enhanced visuals only:

```bash
make enhanced-report
```

Tips:
- Use the dashboard “Presentation Mode” to hide internal config/logs for demos.
- Export the Live Bets table to CSV directly from the dashboard.

## Testing

- Unit tests (models, pipeline, betting)
- Integration tests (end-to-end workflows)
- Performance tests (speed benchmarks)

```bash
make test          # Run all tests
make test-unit     # Unit tests only
make test-perf     # Performance benchmarks
make validate-report # Run validation and save markdown to logs/
```

## Data Schema

**Core Tables**:
- `player_stats_enhanced`: Player performance with 16 engineered features
- `odds_data`: Multi-sportsbook odds with line movement tracking
- `weather_data`: Game conditions affecting outdoor performance
- `injury_data`: Player health status and practice participation
- `clv_tracking`: Closing line value analysis for bet validation

**Feature Engineering**:
- Rolling averages (3, 5, 8 games)
- Year-over-year deltas
- Weather impact adjustments
- Injury recovery indicators
- Team context variables

## Pipeline Schedule

**Automated Jobs**:
- **5min**: Odds updates from major sportsbooks
- **30min**: Injury report updates
- **1hr**: Weather data refresh
- **15min**: Value bet analysis
- **Daily 3AM**: Database maintenance, model retraining

## Roadmap

**Current Sprint**:
- ✅ Player matching infrastructure (19x improvement)
- ✅ Data quality fixes (team/position validation)
- ✅ Rookie integration with passing projections
- ✅ Multi-strategy matching (normalized/fuzzy/ID)
- Enhanced feature engineering
- Position-specific models
- Automated pipeline
- Target MAE achievement

**Next Sprint**:
- Expand player mapping table for better match rates
- Improve odds API team assignments
- LSTM sequence models
- Live injury data integration
- Advanced weather features
- Mobile dashboard

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Standards**:
- PEP 8 code style
- 90%+ test coverage
- Type hints required
- Documentation for all public methods

**System Requirements**:
- Python 3.13+
- MySQL 8.0+ (if using remote backend)
- `pymysql` driver
- `uv` package manager (recommended)

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Email**: [Your email] for private inquiries

---

**Professional NFL Algorithm v2.0** - Built for accuracy, designed for profit. 
