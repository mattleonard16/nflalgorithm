# Next Steps: Fix Player Matching Between Projections and Odds

## 🔴 **PROBLEM**
Only 1 player matches because player_ids don't match between projections and odds. Teams may also be incorrect.

## 📋 **STEPS TO COMPLETE**

### **Priority 1: Quick Fix - Name-Based Matching**
- Create `normalize_player_name()` function in `prop_integration.py`
- Update `join_odds_projections()` to match on normalized names (ignore team mismatches)
- Add fallback: exact player_id match → name match → fuzzy match
- Test: should match 50-100+ players vs. current 1

### **Priority 2: Standardize Player ID Generation**
- Create shared `utils/player_id_utils.py` with `normalize_name()` and `make_player_id()`
- Update `data_pipeline.py` to use shared function
- Update `prop_line_scraper.py` to use shared function
- Ensure consistent formatting (lowercase, underscores, no dots)

### **Priority 3: Fix Team Matching**
- Match players by name only (don't require team match)
- Add team validation: flag if projection team ≠ odds team
- Create team mapping/correction table for known mismatches
- Handle team changes (player traded/moved)

### **Priority 4: Create Player Mapping Table**
- Add `player_mappings` table to schema
- Columns: `player_id_canonical`, `player_id_odds`, `player_id_projections`, `player_name`, `team_projections`, `team_odds`, `confidence_score`
- Create `scripts/build_player_mapping.py` to populate mappings
- Flag ambiguous matches for manual review

### **Priority 5: Update Join Logic**
- Modify `join_odds_projections()` to use mapping table first
- Fallback to name-based matching if mapping missing
- Add `match_type` and `match_confidence` columns to output
- Allow filtering by match quality in dashboard

### **Priority 6: Improve Odds Data**
- Verify API is returning all available players/markets
- Check if API provides player IDs we can use directly
- Add retry logic for failed API calls
- Log raw player names from API for debugging

### **Priority 7: Add Monitoring**
- Create `scripts/match_quality_report.py`
- Report: matched/unmatched counts, top unmatched projections, top unmatched odds
- Add logging to `prop_integration.py` for match statistics
- Alert if match rate < 50%

## ✅ **SUCCESS CRITERIA**
- 100+ matched opportunities (vs. current 1-4)
- 80%+ of top projections have odds data
- Dashboard shows 50-100 players (vs. current 1)
