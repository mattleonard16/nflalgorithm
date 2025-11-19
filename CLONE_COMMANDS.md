# Clone Repository Commands

## Quick Clone (All Platforms)

### Standard Clone
```bash
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm
```

### Clone and Overwrite Existing Folder (Windows PowerShell)
If the folder already exists and you want to replace it:
```powershell
# Remove existing folder if it exists
Remove-Item -Recurse -Force nflalgorithm -ErrorAction SilentlyContinue

# Clone fresh
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm
```

### Clone and Overwrite Existing Folder (macOS/Linux)
```bash
# Remove existing folder if it exists
rm -rf nflalgorithm

# Clone fresh
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm
```

## Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/mattleonard16/nflalgorithm.git

# Navigate to the project directory
cd nflalgorithm

# Install dependencies (see INSTALL_WINDOWS.md for details)
uv pip install -r requirements.txt --system
```

## Windows (Command Prompt)

```cmd
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm
uv pip install -r requirements.txt --system
```

## macOS / Linux

```bash
# Clone the repository
git clone https://github.com/mattleonard16/nflalgorithm.git

# Navigate to the project directory
cd nflalgorithm

# Install dependencies
make install
# OR manually:
# uv venv --python 3.13
# uv pip install -r requirements.txt
```

## Prerequisites

Before cloning, make sure you have:
- **Git** installed - See [INSTALL_GIT_WINDOWS.md](INSTALL_GIT_WINDOWS.md) for Windows installation guide
  - Quick install: Download from https://git-scm.com/download/win
- **Python 3.13** installed ([Download Python](https://www.python.org/downloads/))
- **UV** (optional but recommended) - Install with:
  - Windows: `irm https://astral.sh/uv/install.ps1 | iex`
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## After Cloning

1. **Install dependencies** (see INSTALL_WINDOWS.md for Windows)
2. **Run database migrations**: `python scripts/run_migrations.py` (or `uv run python scripts/run_migrations.py`)
3. **Run the Week 9 project**: `python scripts/execute_week9_plan.py`
4. **Launch dashboard**: `streamlit run dashboard/main_dashboard.py --server.port 8501`

## Repository URL

```
https://github.com/mattleonard16/nflalgorithm.git
```

