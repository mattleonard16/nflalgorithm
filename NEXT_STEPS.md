# Next Steps After Supabase Migration

## ✅ Current Status

Migration verification shows:
- **33 tables** in Supabase (16 NFL tables + existing tables)
- **49,515 rows** of NFL data successfully migrated
- All core database functions working (`read_dataframe`, `get_table_columns`, `get_connection`)
- Backend correctly set to `supabase`

## 🔧 Immediate Actions

### 1. Clean Up Test Tables

Remove test tables that are no longer needed:

```bash
# Connect to Supabase and drop test tables
uv run python -c "
from utils.db import execute
execute('DROP TABLE IF EXISTS test_cache')
execute('DROP TABLE IF EXISTS test_nfl_data')
print('✅ Test tables removed')
"
```

### 2. Verify Data Pipeline Works

Test that your data pipeline can read/write to Supabase:

```bash
# Check schema only (non-destructive)
uv run python data_pipeline.py --check-schema-only

# Or run a full pipeline check
uv run python -c "
from data_pipeline import DataPipeline
pipeline = DataPipeline()
print('✅ DataPipeline initialized successfully')
print(f'Backend: {pipeline.db_path}')
"
```

### 3. Run Value Betting Engine Test

Verify the betting engine works with Supabase:

```bash
uv run python -c "
from value_betting_engine import ValueBettingEngine
engine = ValueBettingEngine()
print('✅ ValueBettingEngine initialized successfully')
"
```

## 🧪 Testing & Validation

### Run Verification Script

```bash
# Run the verification script anytime
uv run python verify_supabase_migration.py
```

### Test Database Queries

```bash
# Test reading data
uv run python -c "
from utils.db import read_dataframe
df = read_dataframe('SELECT COUNT(*) as count FROM player_stats')
print(f'Player stats rows: {df.iloc[0][\"count\"]}')
"

# Test writing data (safe test)
uv run python -c "
from utils.db import execute
execute('CREATE TABLE IF NOT EXISTS test_write (id SERIAL PRIMARY KEY, test TEXT)')
execute('INSERT INTO test_write (test) VALUES (%s)', ('test_value',))
print('✅ Write test successful')
execute('DROP TABLE test_write')
"
```

## 🐛 Fix Test Suite Issues

Some tests are still using SQLite directly. Fix them:

### Fix test_constraint_handling.py

```bash
# Edit the file to fix the tmp_path issue
# Line 25 has: MigrationManager(tmp_path).run()
# Should be: MigrationManager(tmp).run()
```

### Update Tests to Use utils.db

Tests that use `sqlite3.connect()` directly should use `utils.db.get_connection()` instead.

## 📊 Monitor Performance

### Check Query Performance

```bash
# Enable query timing in PostgreSQL
uv run python -c "
from utils.db import execute
execute('SET log_min_duration_statement = 1000')  # Log queries > 1s
print('✅ Query logging enabled')
"
```

### Monitor Connection Usage

```bash
# Check active connections
uv run python -c "
from utils.db import read_dataframe
df = read_dataframe('SELECT count(*) as connections FROM pg_stat_activity WHERE datname = current_database()')
print(f'Active connections: {df.iloc[0][\"connections\"]}')
"
```

## 🔒 Security & Production Readiness

### 1. Enable Row Level Security (RLS)

In Supabase Dashboard:
1. Go to **Table Editor**
2. Select each NFL table
3. Click **RLS** tab
4. Enable RLS
5. Create policies for your application user

Or via SQL:

```bash
uv run python -c "
from utils.db import execute
tables = ['player_stats', 'value_bets', 'enhanced_value_bets', 'prop_lines']
for table in tables:
    execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
    print(f'✅ RLS enabled on {table}')
"
```

### 2. Verify Backups

In Supabase Dashboard:
- Go to **Settings** → **Database**
- Verify **Point-in-time Recovery** is enabled
- Check backup schedule

### 3. Set Up Connection Pooling (Optional)

For better performance, use Supabase's connection pooler:

```bash
# Update .env with pooler connection string
# Change port from 5432 to 6543
# SUPABASE_DB_URL=postgresql://postgres:nflalgo20032@db.rweztompmzjxypismzzf.supabase.co:6543/postgres
```

## 📈 Production Checklist

- [ ] Test tables removed
- [ ] Data pipeline verified working
- [ ] Value betting engine verified working
- [ ] Test suite updated and passing
- [ ] RLS enabled on production tables
- [ ] Backups verified
- [ ] Performance monitoring set up
- [ ] Connection pooling configured (if needed)

## 🚀 Daily Operations

### Check Database Health

```bash
# Quick health check
uv run python verify_supabase_migration.py
```

### View Recent Data

```bash
# Check recent value bets
uv run python -c "
from utils.db import read_dataframe
df = read_dataframe('SELECT * FROM value_bets ORDER BY created_at DESC LIMIT 10')
print(df.to_string())
"
```

### Monitor Table Sizes

```bash
uv run python -c "
from utils.db import read_dataframe
df = read_dataframe('''
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||\".\"||tablename)) AS size
    FROM pg_tables
    WHERE schemaname = ''public''
    ORDER BY pg_total_relation_size(schemaname||\".\"||tablename) DESC
    LIMIT 10
''')
print(df.to_string())
"
```

## 🔄 Switching Back to SQLite (If Needed)

If you need to switch back to SQLite temporarily:

```bash
# Edit .env file
# Change: DB_BACKEND=sqlite
# Remove or comment out: SUPABASE_DB_URL=...

# Or set environment variable
export DB_BACKEND=sqlite
```

## 📝 Useful Commands Summary

```bash
# Verify migration
uv run python verify_supabase_migration.py

# Test connection
uv run python test_supabase_with_data.py

# Check data counts
uv run python -c "from utils.db import read_dataframe; print(read_dataframe('SELECT COUNT(*) FROM player_stats').iloc[0][0])"

# Run tests (after fixing test issues)
make test

# Check backend
uv run python -c "from config import config; print(config.database.backend)"
```

## 🆘 Troubleshooting

### Connection Issues

```bash
# Test connection directly
uv run python -c "
from utils.db import get_connection
with get_connection() as conn:
    print('✅ Connection successful')
"
```

### Query Errors

```bash
# Check if table exists
uv run python -c "
from utils.db import table_exists
print('player_stats exists:', table_exists('player_stats'))
"
```

### Performance Issues

```bash
# Check slow queries
uv run python -c "
from utils.db import read_dataframe
df = read_dataframe('''
    SELECT query, calls, total_time, mean_time
    FROM pg_stat_statements
    ORDER BY mean_time DESC
    LIMIT 10
''')
print(df.to_string())
"
```

