# 🏈 NFL Algorithm - Professional Value Betting System

> *Advanced NFL player performance prediction and value betting engine*

**Version**: 2.1 | **Status**: ✨ Production Ready | **Target MAE**: ≤ 3.0

---

## 📊 System Overview

A comprehensive NFL betting algorithm combining:

| Component | Description |
|-----------|-------------|
| 🤖 **ML Models** | Position-specific predictive models with defense adjustments |
| 📡 **Data Pipeline** | Live odds, weather, injuries, and player stats via nflreadpy |
| 💰 **Value Engine** | Kelly Criterion optimization with CLV tracking |
| 🧪 **Validation** | Cross-season backtesting and performance metrics |
| ⚡ **Automation** | Scheduled data updates and model retraining |

**Core Philosophy**: Achieve consistent profitability through disciplined, data-driven betting with rigorous risk management.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm

# Install dependencies (auto-detects UV or venv)
make install

# Ingest real NFL data (2024 + 2025 seasons)
make ingest-nfl

# Option 1: Launch Streamlit dashboard (legacy)
make dashboard

# Option 2: Launch React dashboard (recommended)
make api              # Start FastAPI backend on :8000
make frontend-dev     # Start Next.js frontend on :3000
```

### 📅 Weekly Workflow

```bash
# 1️⃣ Generate projections for the week
make week-predict SEASON=2025 WEEK=13

# 2️⃣ Materialize value bets for dashboard
make week-materialize SEASON=2025 WEEK=13

# 3️⃣ Launch Streamlit dashboard
make dashboard

# 4️⃣ View at http://localhost:8501
```

---

## 🆕 React Dashboard (New)

The project now includes a modern React/Next.js dashboard alongside the original Streamlit version.

### Stack
- **Frontend**: Next.js 15 + TypeScript + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI serving the same SQLite database
- **Charts**: Recharts for data visualization

### Running the React Dashboard

```bash
# Terminal 1: Start the API
make api

# Terminal 2: Start the frontend
cd frontend && npm run dev

# Visit http://localhost:3000
```

### Frontend Structure
```
frontend/
├── src/app/           # Next.js pages (Dashboard, Performance, Analytics, System)
├── src/components/    # React components + shadcn/ui
└── src/lib/           # API client and TypeScript types
```

---

## 🏗️ Architecture

```
nflalgorithm/
├── 🤖 models/              # Position-specific ML models
├── 📁 data/                # CSVs, projections, odds cache
├── 📚 docs/                # Documentation
├── 🔧 scripts/             # Utility scripts and tools
├── 🧪 tests/               # Comprehensive test suite
├── 🛠️ utils/               # Helpers (player_id, defense_adjustments)
├── 📊 dashboard/           # Streamlit monitoring UI
├── 📝 logs/                # Performance tracking
├── 🔄 data_pipeline.py     # Data ingestion + feature engineering
├── 💵 value_betting_engine.py  # Value detection & CLV
├── 🔗 prop_integration.py  # Player matching system
└── 📈 materialized_value_view.py  # Dashboard data layer
```

### 🔌 Data Sources

| Source | Purpose | Update Frequency |
|--------|---------|------------------|
| **nflreadpy** | Player stats, schedules, rosters | Nightly |
| **The Odds API** | Prop lines from multiple books | Real-time |
| **nflverse** | Snap counts, depth charts, routes | 4x daily |

### 🎯 Key Features

- **🛡️ Defense Adjustments**: Predictions adjusted based on how players perform vs their own average against specific defenses
- **🔄 Multi-Source Matching**: 3-tier player matching (ID → name+team → fuzzy)
- **📉 Best Line Detection**: Auto-selects best sportsbook line per player/market
- **⚡ SQLite/MySQL**: Flexible backend for dev or production

---

## 📈 Performance Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| 🏃 Rushing MAE | **3.6** | ≤ 3.0 | 🔄 Optimizing |
| 🎯 Receiving MAE | **4.1** | ≤ 3.5 | 🔄 Optimizing |
| 💰 Value Bet ROI | **15.2%** | > 12% | ✅ Achieved |
| 📊 CLV Performance | **+2.3%** | > 0% | ✅ Achieved |
| 🔗 Player Match Rate | **28%** | > 25% | ✅ Achieved |

---

## 💻 Dashboard Features

Access at `http://localhost:8501`:

| Tab | Features |
|-----|----------|
| 🎰 **Live Bets** | Current value opportunities, best lines, recommendations |
| 📊 **Performance** | Model accuracy, ROI tracking, edge distribution |
| 📈 **CLV Analysis** | Closing line value tracking |
| ⚙️ **System** | Pipeline health, data freshness |

### 🎛️ Dashboard Controls

- **Best Line Only**: Toggle to show only best sportsbook per player/market
- **Minimum Edge %**: Filter bets by edge threshold
- **Quick Bet Mode**: Simplified view for fast decisions

---

## ⚙️ Configuration

### 🗄️ Database Setup

```bash
# Copy example config
cp .env.example .env
```

**SQLite (Local Dev):**
```env
DB_BACKEND=sqlite
SQLITE_DB_PATH=nfl_data.db
```

**MySQL (Production):**
```env
DB_BACKEND=mysql
DB_URL="mysql://user:pass@host:port/database"
```

### 🔑 API Keys

```env
ODDS_API_KEY="your_odds_api_key"
```

> ⚠️ **Security**: Never commit `.env` to version control!

---

## 🧪 Testing

```bash
make test          # 🧪 Run all tests
make lint          # 🔍 Run mypy
make format        # 🎨 Format with black + isort
make validate      # ✅ Cross-season validation
```

---

## 📦 Make Targets

| Command | Description |
|---------|-------------|
| `make install` | 📥 Smart install (UV or venv) |
| `make ingest-nfl` | 📡 Fetch real NFL data (2024+2025) |
| `make week-predict` | 🔮 Generate week projections |
| `make week-materialize` | 💾 Materialize value view |
| `make dashboard` | 📊 Launch Streamlit UI |
| `make test` | 🧪 Run test suite |
| `make report` | 📄 Generate shareable reports |

---

## 🗺️ Roadmap

### ✅ Completed
- [x] Defense vs position adjustments (relative performance)
- [x] nflreadpy integration for 2025 data
- [x] Best line deduplication
- [x] Multi-strategy player matching
- [x] SQLite/MySQL dual backend support

### 🔄 In Progress
- [ ] Game script adjustments (implied totals)
- [ ] Weather factor integration
- [ ] Injury status weighting

### 📋 Planned
- [ ] Line movement tracking
- [ ] Target share predictions
- [ ] Same-game correlation analysis
- [ ] Mobile-friendly dashboard

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Standards**: PEP 8 • Type hints • 90%+ coverage • Documented methods

---

## 📜 License

MIT License - see LICENSE file for details.

---

<div align="center">

**🏈 NFL Algorithm v2.1** 

*Built for accuracy. Designed for profit.*

</div>
