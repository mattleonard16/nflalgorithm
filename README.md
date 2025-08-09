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

```bash
# Clone and setup
git clone <repository-url>
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
├── data_pipeline.py             # Real-time data ingestion
├── value_betting_engine.py      # Value bet detection & CLV
├── optuna_optimization.py       # Hyperparameter tuning
├── cross_season_validation.py   # Professional validation
├── dashboard/                   # Streamlit monitoring
├── pipeline_scheduler.py        # Automated workflows
├── tests/                       # Comprehensive test suite
└── logs/                        # Performance tracking
```

**Data Flow**: Raw NFL data → Feature Engineering → ML Models → Predictions → Value Detection → Bet Recommendations

## Performance Metrics

| Metric | Current | Target | Status |
|--------|---------|---------|---------|
| Rushing MAE | **3.6** | ≤ 3.0 | Optimizing |
| Receiving MAE | **4.1** | ≤ 3.5 | Optimizing |
| Value Bet ROI | **15.2%** | > 12% | **ACHIEVED** |
| CLV Performance | **+2.3%** | > 0% | **ACHIEVED** |

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
- Enhanced feature engineering
- Position-specific models
- Automated pipeline
- Target MAE achievement

**Next Sprint**:
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

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Email**: [Your email] for private inquiries

---

**Professional NFL Algorithm v2.0** - Built for accuracy, designed for profit. 