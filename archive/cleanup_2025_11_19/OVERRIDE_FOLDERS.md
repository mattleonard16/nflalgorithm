# Override/Overwrite Folders Guide

## Cloning Repository - Overwrite Existing Folder

### Windows PowerShell

If you already have the `nflalgorithm` folder and want to replace it completely:

```powershell
# Remove existing folder (if it exists)
Remove-Item -Recurse -Force nflalgorithm -ErrorAction SilentlyContinue

# Clone fresh repository
git clone https://github.com/mattleonard16/nflalgorithm.git

# Navigate into the folder
cd nflalgorithm
```

### Windows Command Prompt

```cmd
# Remove existing folder (if it exists)
rmdir /s /q nflalgorithm

# Clone fresh repository
git clone https://github.com/mattleonard16/nflalgorithm.git

# Navigate into the folder
cd nflalgorithm
```

### macOS / Linux

```bash
# Remove existing folder (if it exists)
rm -rf nflalgorithm

# Clone fresh repository
git clone https://github.com/mattleonard16/nflalgorithm.git

# Navigate into the folder
cd nflalgorithm
```

## Creating Directories - Force Overwrite

### Windows PowerShell

The `-Force` flag creates directories if they don't exist, or does nothing if they already exist:

```powershell
# Create directories (won't delete existing content)
New-Item -ItemType Directory -Force -Path data,logs,models,dashboard,tests,scripts
```

**To completely replace a directory and its contents:**

```powershell
# Remove directory and all contents
Remove-Item -Recurse -Force data -ErrorAction SilentlyContinue

# Create fresh directory
New-Item -ItemType Directory -Path data
```

### macOS / Linux

```bash
# Create directories (won't delete existing content)
mkdir -p data logs models dashboard tests scripts

# To completely replace a directory:
rm -rf data
mkdir data
```

## Force Reinstall Dependencies

### Windows PowerShell

```powershell
# Uninstall all packages (optional - only if you want a clean slate)
pip uninstall -r requirements.txt -y

# Reinstall dependencies
uv pip install -r requirements.txt --system
```

### macOS / Linux

```bash
# Remove virtual environment and reinstall
rm -rf venv
make install

# Or with UV:
rm -rf .venv
uv venv --python 3.13
uv pip install -r requirements.txt
```

## Force Reset Database

If you want to start with a fresh database:

### Windows PowerShell

```powershell
# Backup existing database (optional)
Copy-Item nfl_data.db nfl_data.db.backup -ErrorAction SilentlyContinue

# Remove existing database
Remove-Item nfl_data.db -ErrorAction SilentlyContinue

# Run migrations to create fresh database
python scripts/run_migrations.py
```

### macOS / Linux

```bash
# Backup existing database (optional)
cp nfl_data.db nfl_data.db.backup 2>/dev/null || true

# Remove existing database
rm -f nfl_data.db

# Run migrations to create fresh database
python scripts/run_migrations.py
```

## Complete Fresh Start

To completely reset everything and start fresh:

### Windows PowerShell

```powershell
# Remove entire project folder
Remove-Item -Recurse -Force nflalgorithm -ErrorAction SilentlyContinue

# Clone fresh
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm

# Install dependencies
uv pip install -r requirements.txt --system

# Create directories
New-Item -ItemType Directory -Force -Path data,logs,models,dashboard,tests,scripts

# Run migrations
python scripts/run_migrations.py
```

### macOS / Linux

```bash
# Remove entire project folder
rm -rf nflalgorithm

# Clone fresh
git clone https://github.com/mattleonard16/nflalgorithm.git
cd nflalgorithm

# Install dependencies
make install

# Run migrations
python scripts/run_migrations.py
```

## Important Notes

⚠️ **Warning**: 
- `Remove-Item -Recurse -Force` and `rm -rf` will **permanently delete** folders and all their contents
- Always backup important data before deleting
- The `-Force` flag in `New-Item` does NOT delete existing content - it only creates if missing

✅ **Safe Operations**:
- `New-Item -Force` - Safe, won't delete existing content
- `mkdir -p` - Safe, won't delete existing content
- `git clone` into existing folder - Will fail safely, won't overwrite

