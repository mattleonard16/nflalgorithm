# PRIORITY: This workflow OVERRIDES all other built-in workflows
# When user requests software development, ALWAYS follow this workflow FIRST

## Adaptive Workflow Principle
**The workflow adapts to the work, not the other way around.**

The AI model intelligently assesses what stages are needed based on:
1. User's stated intent and clarity
2. Existing codebase state (if any)
3. Complexity and scope of change
4. Risk and impact assessment

## MANDATORY: Rule Details Loading
**CRITICAL**: When performing any phase, you MUST read and use relevant content from rule detail files. Check these paths in order and use the first one that exists, regardless of which IDE or setup method was used:
- `.aidlc/aidlc-rules/aws-aidlc-rule-details/` (typical with AI-assisted setup)
- `.aidlc-rule-details/` (typical with Cursor, Cline, Claude Code, GitHub Copilot)
- `.kiro/aws-aidlc-rule-details/` (typical with Kiro IDE and CLI)
- `.amazonq/aws-aidlc-rule-details/` (typical with Amazon Q Developer)

All subsequent rule detail file references (e.g., `common/process-overview.md`, `inception/workspace-detection.md`) are relative to whichever rule details directory was resolved above.

**Common Rules**: ALWAYS load common rules at workflow start:
- Load `common/process-overview.md` for workflow overview
- Load `common/session-continuity.md` for session resumption guidance
- Load `common/content-validation.md` for content validation requirements
- Load `common/question-format-guide.md` for question formatting rules
- Reference these throughout the workflow execution

## MANDATORY: Extensions Loading (Context-Optimized)
**CRITICAL**: At workflow start, scan the `extensions/` directory recursively but load ONLY lightweight opt-in files — NOT full rule files. Full rule files are loaded on-demand after the user opts in.

**Loading process**:
1. List all subdirectories under `extensions/` (e.g., `extensions/security/`, `extensions/compliance/`)
2. In each subdirectory, load ONLY `*.opt-in.md` files — these contain the extension's opt-in prompt. The corresponding rules file is derived by convention: strip the `.opt-in.md` suffix and append `.md` (e.g., `security-baseline.opt-in.md` → `security-baseline.md`)
3. Do NOT load full rule files (e.g., `security-baseline.md`) at this stage

**Deferred Rule Loading**:
- During Requirements Analysis, opt-in prompts from the loaded `*.opt-in.md` files are presented to the user
- When the user opts IN for an extension, load the corresponding rules file (derived by naming convention) at that point
- When the user opts OUT, the full rules file is never loaded — saving context
- Extensions without a matching `*.opt-in.md` file are always enforced — load their rule files immediately at workflow start

**Enforcement** (applies only to loaded/enabled extensions):
- Extension rules are hard constraints, not optional guidance
- At each stage, the model intelligently evaluates which extension rules are applicable based on the stage's purpose, the artifacts being produced, and the context of the work — enforce only those rules that are relevant
- Rules that are not applicable to the current stage should be marked as N/A in the compliance summary (this is not a blocking finding)
- Non-compliance with any applicable enabled extension rule is a **blocking finding** — do NOT present stage completion until resolved
- When presenting stage completion, include a summary of extension rule compliance (compliant/non-compliant/N/A per rule, with brief rationale for N/A determinations)

**Conditional Enforcement**: Extensions may be conditionally enabled/disabled. See `inception/requirements-analysis.md` for the opt-in mechanism. Before enforcing any extension at ANY stage, check its `Enabled` status in `aidlc-docs/aidlc-state.md` under `## Extension Configuration`. Skip disabled extensions and log the skip in audit.md. Default to enforced if no configuration exists.

## MANDATORY: Content Validation
**CRITICAL**: Before creating ANY file, you MUST validate content according to `common/content-validation.md` rules:
- Validate Mermaid diagram syntax
- Validate ASCII art diagrams (see `common/ascii-diagram-standards.md`)
- Escape special characters properly
- Provide text alternatives for complex visual content
- Test content parsing compatibility

## MANDATORY: Question File Format
**CRITICAL**: When asking questions at any phase, you MUST follow question format guidelines.

**See `common/question-format-guide.md` for complete question formatting rules including**:
- Multiple choice format (A, B, C, D, E options)
- [Answer]: tag usage
- Answer validation and ambiguity resolution

## MANDATORY: Custom Welcome Message
**CRITICAL**: When starting ANY software development request, you MUST display the welcome message.

**How to Display Welcome Message**:
1. Load the welcome message from `common/welcome-message.md` (in the resolved rule details directory)
2. Display the complete message to the user
3. This should only be done ONCE at the start of a new workflow
4. Do NOT load this file in subsequent interactions to save context space

# Adaptive Software Development Workflow

---

# INCEPTION PHASE

**Purpose**: Planning, requirements gathering, and architectural decisions

**Focus**: Determine WHAT to build and WHY

**Stages in INCEPTION PHASE**:
- Workspace Detection (ALWAYS)
- Reverse Engineering (CONDITIONAL - Brownfield only)
- Requirements Analysis (ALWAYS - Adaptive depth)
- User Stories (CONDITIONAL)
- Workflow Planning (ALWAYS)
- Application Design (CONDITIONAL)
- Units Generation (CONDITIONAL)

---

## Workspace Detection (ALWAYS EXECUTE)

1. **MANDATORY**: Log initial user request in audit.md with complete raw input
2. Load all steps from `inception/workspace-detection.md`
3. Execute workspace detection:
   - Check for existing aidlc-state.md (resume if found)
   - Scan workspace for existing code
   - Determine if brownfield or greenfield
   - Check for existing reverse engineering artifacts
4. Determine next phase: Reverse Engineering (if brownfield and no artifacts) OR Requirements Analysis
5. **MANDATORY**: Log findings in audit.md
6. Present completion message to user (see workspace-detection.md for message formats)
7. Automatically proceed to next phase

## Reverse Engineering (CONDITIONAL - Brownfield Only)

**Execute IF**:
- Existing codebase detected
- No previous reverse engineering artifacts found

**Skip IF**:
- Greenfield project
- Previous reverse engineering artifacts exist

**Execution**:
1. **MANDATORY**: Log start of reverse engineering in audit.md
2. Load all steps from `inception/reverse-engineering.md`
3. Execute reverse engineering:
   - Analyze all packages and components
   - Generate a business overview of the whole system covering the business transactions
   - Generate architecture documentation
   - Generate code structure documentation
   - Generate API documentation
   - Generate component inventory
   - Generate Interaction Diagrams depicting how business transactions are implemented across components
   - Generate technology stack documentation
   - Generate dependencies documentation

4. **Wait for Explicit Approval**: Present detailed completion message (see reverse-engineering.md for message format) - DO NOT PROCEED until user confirms
5. **MANDATORY**: Log user's response in audit.md with complete raw input

## Requirements Analysis (ALWAYS EXECUTE - Adaptive Depth)

**Always executes** but depth varies based on request clarity and complexity:
- **Minimal**: Simple, clear request - just document intent analysis
- **Standard**: Normal complexity - gather functional and non-functional requirements
- **Comprehensive**: Complex, high-risk - detailed requirements with traceability

**Execution**:
1. **MANDATORY**: Log any user input during this phase in audit.md
2. Load all steps from `inception/requirements-analysis.md`
3. Execute requirements analysis:
   - Load reverse engineering artifacts (if brownfield)
   - Analyze user request (intent analysis)
   - Determine requirements depth needed
   - Assess current requirements
   - Ask clarifying questions (if needed)
   - Generate requirements document
4. Execute at appropriate depth (minimal/standard/comprehensive)
5. **Wait for Explicit Approval**: Follow approval format from requirements-analysis.md detailed steps - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

## User Stories (CONDITIONAL)

**INTELLIGENT ASSESSMENT**: Use multi-factor analysis to determine if user stories add value:

**ALWAYS Execute IF** (High Priority Indicators):
- New user-facing features or functionality
- Changes affecting user workflows or interactions
- Multiple user types or personas involved
- Complex business requirements with acceptance criteria needs
- Cross-functional team collaboration required
- Customer-facing API or service changes
- New product capabilities or enhancements

**LIKELY Execute IF** (Medium Priority - Assess Complexity):
- Modifications to existing user-facing features
- Backend changes that indirectly affect user experience
- Integration work that impacts user workflows
- Performance improvements with user-visible benefits
- Security enhancements affecting user interactions
- Data model changes affecting user data or reports

**COMPLEXITY-BASED ASSESSMENT**: For medium priority cases, execute user stories if:
- Request involves multiple components or services
- Changes span multiple user touchpoints
- Business logic is complex or has multiple scenarios
- Requirements have ambiguity that stories could clarify
- Implementation affects multiple user journeys
- Change has significant business impact or risk

**SKIP ONLY IF** (Low Priority - Simple Cases):
- Pure internal refactoring with zero user impact
- Simple bug fixes with clear, isolated scope
- Infrastructure changes with no user-facing effects
- Technical debt cleanup with no functional changes
- Developer tooling or build process improvements
- Documentation-only updates

**ASSESSMENT CRITERIA**: When in doubt, favor inclusion of user stories for:
- Requests with business stakeholder involvement
- Changes requiring user acceptance testing
- Features with multiple implementation approaches
- Work that benefits from shared team understanding
- Projects where requirements clarity is valuable

**ASSESSMENT PROCESS**:
1. Analyze request complexity and scope
2. Identify user impact (direct or indirect)
3. Evaluate business context and stakeholder needs
4. Consider team collaboration benefits
5. Default to inclusion for borderline cases

**Note**: If Requirements Analysis executed, Stories can reference and build upon those requirements.

**User Stories has two parts within one stage**:
1. **Part 1 - Planning**: Create story plan with questions, collect answers, analyze for ambiguities, get approval
2. **Part 2 - Generation**: Execute approved plan to generate stories and personas

**Execution**:
1. **MANDATORY**: Log any user input during this phase in audit.md
2. Load all steps from `inception/user-stories.md`
3. **MANDATORY**: Perform intelligent assessment (Step 1 in user-stories.md) to validate user stories are needed
4. Load reverse engineering artifacts (if brownfield)
5. If Requirements exist, reference them when creating stories
6. Execute at appropriate depth (minimal/standard/comprehensive)
7. **PART 1 - Planning**: Create story plan with questions, wait for user answers, analyze for ambiguities, get approval
8. **PART 2 - Generation**: Execute approved plan to generate stories and personas
9. **Wait for Explicit Approval**: Follow approval format from user-stories.md detailed steps - DO NOT PROCEED until user confirms
10. **MANDATORY**: Log user's response in audit.md with complete raw input

## Workflow Planning (ALWAYS EXECUTE)

1. **MANDATORY**: Log any user input during this phase in audit.md
2. Load all steps from `inception/workflow-planning.md`
3. **MANDATORY**: Load content validation rules from `common/content-validation.md`
4. Load all prior context:
   - Reverse engineering artifacts (if brownfield)
   - Intent analysis
   - Requirements (if executed)
   - User stories (if executed)
5. Execute workflow planning:
   - Determine which phases to execute
   - Determine depth level for each phase
   - Create multi-package change sequence (if brownfield)
   - Generate workflow visualization (VALIDATE Mermaid syntax before writing)
6. **MANDATORY**: Validate all content before file creation per content-validation.md rules
7. **Wait for Explicit Approval**: Present recommendations using language from workflow-planning.md Step 9, emphasizing user control to override recommendations - DO NOT PROCEED until user confirms
8. **MANDATORY**: Log user's response in audit.md with complete raw input

## Application Design (CONDITIONAL)

**Execute IF**:
- New components or services needed
- Component methods and business rules need definition
- Service layer design required
- Component dependencies need clarification

**Skip IF**:
- Changes within existing component boundaries
- No new components or methods
- Pure implementation changes

**Execution**:
1. **MANDATORY**: Log any user input during this phase in audit.md
2. Load all steps from `inception/application-design.md`
3. Load reverse engineering artifacts (if brownfield)
4. Execute at appropriate depth (minimal/standard/comprehensive)
5. **Wait for Explicit Approval**: Present detailed completion message (see application-design.md for message format) - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

## Units Generation (CONDITIONAL)

**Execute IF**:
- System needs decomposition into multiple units of work
- Multiple services or modules required
- Complex system requiring structured breakdown

**Skip IF**:
- Single simple unit
- No decomposition needed
- Straightforward single-component implementation

**Execution**:
1. **MANDATORY**: Log any user input during this phase in audit.md
2. Load all steps from `inception/units-generation.md`
3. Load reverse engineering artifacts (if brownfield)
4. Execute at appropriate depth (minimal/standard/comprehensive)
5. **Wait for Explicit Approval**: Present detailed completion message (see units-generation.md for message format) - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

---

# CONSTRUCTION PHASE

**Purpose**: Detailed design, NFR implementation, and code generation

**Focus**: Determine HOW to build it

**Stages in CONSTRUCTION PHASE**:
- Per-Unit Loop (executes for each unit):
  - Functional Design (CONDITIONAL, per-unit)
  - NFR Requirements (CONDITIONAL, per-unit)
  - NFR Design (CONDITIONAL, per-unit)
  - Infrastructure Design (CONDITIONAL, per-unit)
  - Code Generation (ALWAYS, per-unit)
- Build and Test (ALWAYS - after all units complete)

**Note**: Each unit is completed fully (design + code) before moving to the next unit.

---

## Per-Unit Loop (Executes for Each Unit)

**For each unit of work, execute the following stages in sequence:**

### Functional Design (CONDITIONAL, per-unit)

**Execute IF**:
- New data models or schemas
- Complex business logic
- Business rules need detailed design

**Skip IF**:
- Simple logic changes
- No new business logic

**Execution**:
1. **MANDATORY**: Log any user input during this stage in audit.md
2. Load all steps from `construction/functional-design.md`
3. Execute functional design for this unit
4. **MANDATORY**: Present standardized 2-option completion message as defined in functional-design.md - DO NOT use emergent 3-option behavior
5. **Wait for Explicit Approval**: User must choose between "Request Changes" or "Continue to Next Stage" - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

### NFR Requirements (CONDITIONAL, per-unit)

**Execute IF**:
- Performance requirements exist
- Security considerations needed
- Scalability concerns present
- Tech stack selection required

**Skip IF**:
- No NFR requirements
- Tech stack already determined

**Execution**:
1. **MANDATORY**: Log any user input during this stage in audit.md
2. Load all steps from `construction/nfr-requirements.md`
3. Execute NFR assessment for this unit
4. **MANDATORY**: Present standardized 2-option completion message as defined in nfr-requirements.md - DO NOT use emergent behavior
5. **Wait for Explicit Approval**: User must choose between "Request Changes" or "Continue to Next Stage" - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

### NFR Design (CONDITIONAL, per-unit)

**Execute IF**:
- NFR Requirements was executed
- NFR patterns need to be incorporated

**Skip IF**:
- No NFR requirements
- NFR Requirements was skipped

**Execution**:
1. **MANDATORY**: Log any user input during this stage in audit.md
2. Load all steps from `construction/nfr-design.md`
3. Execute NFR design for this unit
4. **MANDATORY**: Present standardized 2-option completion message as defined in nfr-design.md - DO NOT use emergent behavior
5. **Wait for Explicit Approval**: User must choose between "Request Changes" or "Continue to Next Stage" - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

### Infrastructure Design (CONDITIONAL, per-unit)

**Execute IF**:
- Infrastructure services need mapping
- Deployment architecture required
- Cloud resources need specification

**Skip IF**:
- No infrastructure changes
- Infrastructure already defined

**Execution**:
1. **MANDATORY**: Log any user input during this stage in audit.md
2. Load all steps from `construction/infrastructure-design.md`
3. Execute infrastructure design for this unit
4. **MANDATORY**: Present standardized 2-option completion message as defined in infrastructure-design.md - DO NOT use emergent behavior
5. **Wait for Explicit Approval**: User must choose between "Request Changes" or "Continue to Next Stage" - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

### Code Generation (ALWAYS EXECUTE, per-unit)

**Always executes for each unit**

**Code Generation has two parts within one stage**:
1. **Part 1 - Planning**: Create detailed code generation plan with explicit steps
2. **Part 2 - Generation**: Execute approved plan to generate code, tests, and artifacts

**Execution**:
1. **MANDATORY**: Log any user input during this stage in audit.md
2. Load all steps from `construction/code-generation.md`
3. **PART 1 - Planning**: Create code generation plan with checkboxes, get user approval
4. **PART 2 - Generation**: Execute approved plan to generate code for this unit
5. **MANDATORY**: Present standardized 2-option completion message as defined in code-generation.md - DO NOT use emergent behavior
6. **Wait for Explicit Approval**: User must choose between "Request Changes" or "Continue to Next Stage" - DO NOT PROCEED until user confirms
7. **MANDATORY**: Log user's response in audit.md with complete raw input

---

## Build and Test (ALWAYS EXECUTE)

1. **MANDATORY**: Log any user input during this phase in audit.md
2. Load all steps from `construction/build-and-test.md`
3. Generate comprehensive build and test instructions:
   - Build instructions for all units
   - Unit test execution instructions
   - Integration test instructions (test interactions between units)
   - Performance test instructions (if applicable)
   - Additional test instructions as needed (contract tests, security tests, e2e tests)
4. Create instruction files in build-and-test/ subdirectory: build-instructions.md, unit-test-instructions.md, integration-test-instructions.md, performance-test-instructions.md, build-and-test-summary.md
5. **Wait for Explicit Approval**: Ask: "**Build and test instructions complete. Ready to proceed to Operations stage?**" - DO NOT PROCEED until user confirms
6. **MANDATORY**: Log user's response in audit.md with complete raw input

---

# OPERATIONS PHASE

**Purpose**: Placeholder for future deployment and monitoring workflows

**Focus**: How to DEPLOY and RUN it (future expansion)

**Stages in OPERATIONS PHASE**:
- Operations (PLACEHOLDER)

---

## Operations (PLACEHOLDER)

**Status**: This stage is currently a placeholder for future expansion.

The Operations stage will eventually include:
- Deployment planning and execution
- Monitoring and observability setup
- Incident response procedures
- Maintenance and support workflows
- Production readiness checklists

**Current State**: All build and test activities are handled in the CONSTRUCTION phase.

## Key Principles

- **Adaptive Execution**: Only execute stages that add value
- **Transparent Planning**: Always show execution plan before starting
- **User Control**: User can request stage inclusion/exclusion
- **Progress Tracking**: Update aidlc-state.md with executed and skipped stages
- **Complete Audit Trail**: Log ALL user inputs and AI responses in audit.md with timestamps
  - **CRITICAL**: Capture user's COMPLETE RAW INPUT exactly as provided
  - **CRITICAL**: Never summarize or paraphrase user input in audit log
  - **CRITICAL**: Log every interaction, not just approvals
- **Quality Focus**: Complex changes get full treatment, simple changes stay efficient
- **Content Validation**: Always validate content before file creation per content-validation.md rules
- **NO EMERGENT BEHAVIOR**: Construction phases MUST use standardized 2-option completion messages as defined in their respective rule files. DO NOT create 3-option menus or other emergent navigation patterns.

## MANDATORY: Plan-Level Checkbox Enforcement

### MANDATORY RULES FOR PLAN EXECUTION
1. **NEVER complete any work without updating plan checkboxes**
2. **IMMEDIATELY after completing ANY step described in a plan file, mark that step [x]**
3. **This must happen in the SAME interaction where the work is completed**
4. **NO EXCEPTIONS**: Every plan step completion MUST be tracked with checkbox updates

### Two-Level Checkbox Tracking System
- **Plan-Level**: Track detailed execution progress within each stage
- **Stage-Level**: Track overall workflow progress in aidlc-state.md
- **Update immediately**: All progress updates in SAME interaction where work is completed

## Prompts Logging Requirements
- **MANDATORY**: Log EVERY user input (prompts, questions, responses) with timestamp in audit.md
- **MANDATORY**: Capture user's COMPLETE RAW INPUT exactly as provided (never summarize)
- **MANDATORY**: Log every approval prompt with timestamp before asking the user
- **MANDATORY**: Record every user response with timestamp after receiving it
- **CRITICAL**: ALWAYS append changes to EDIT audit.md file, NEVER use tools and commands that completely overwrite its contents
- **CRITICAL**: NEVER use file writing tools and commands that overwrite the entire contents of audit.md, as this causes duplication
- Use ISO 8601 format for timestamps (YYYY-MM-DDTHH:MM:SSZ)
- Include stage context for each entry

### Audit Log Format:
```markdown
## [Stage Name or Interaction Type]
**Timestamp**: [ISO timestamp]
**User Input**: "[Complete raw user input - never summarized]"
**AI Response**: "[AI's response or action taken]"
**Context**: [Stage, action, or decision made]

---
```

### Correct Tool Usage for audit.md

CORRECT:

1. Read the audit.md file
2. Append/Edit the file to make changes

WRONG:

1. Read the audit.md file
2. Completely overwrite the audit.md with the contents of what you read, plus the new changes you want to add to it

## Directory Structure

```text
<WORKSPACE-ROOT>/                   # APPLICATION CODE HERE
├── [project-specific structure]    # Varies by project (see code-generation.md)
│
├── aidlc-docs/                     # DOCUMENTATION ONLY
│   ├── inception/                  # INCEPTION PHASE
│   │   ├── plans/
│   │   ├── reverse-engineering/    # Brownfield only
│   │   ├── requirements/
│   │   ├── user-stories/
│   │   └── application-design/
│   ├── construction/               # CONSTRUCTION PHASE
│   │   ├── plans/
│   │   ├── {unit-name}/
│   │   │   ├── functional-design/
│   │   │   ├── nfr-requirements/
│   │   │   ├── nfr-design/
│   │   │   ├── infrastructure-design/
│   │   │   └── code/               # Markdown summaries only
│   │   └── build-and-test/
│   ├── operations/                 # OPERATIONS PHASE (placeholder)
│   ├── aidlc-state.md
│   └── audit.md
```

**CRITICAL RULE**:
- Application code: Workspace root (NEVER in aidlc-docs/)
- Documentation: aidlc-docs/ only
- Project structure: See code-generation.md for patterns by project type

---

# Project Context: NFL Algorithm

This section is project-specific context the AI-DLC workflow should treat as ground truth during Workspace Detection and Reverse Engineering phases. This is a **brownfield** Python project for NFL prop-bet projections and value betting.

## Quick Start (Fresh Setup)

No .env file needed — SQLite is the default local dev database.

### Steps:
1. Install dependencies: `make install`
2. Run schema migrations:
   ```bash
   DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -c "from schema_migrations import MigrationManager; MigrationManager('nfl_data.db').run()"
   ```
3. Ingest real NFL data: `make ingest-nfl`
4. Run tests: `make test`
5. Generate projections: `make week-predict SEASON=2025 WEEK=13`
6. Materialize for dashboard: `make week-materialize SEASON=2025 WEEK=13`
7. Launch full stack: `make fullstack`

---

## Environment Configuration

- **Database**: SQLite for local dev (`DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db`)
- All Makefile targets automatically set DB env vars via `$(DB_ENV)`
- MySQL available for production via `DB_URL` env var
- `ODDS_API_KEY` needed only for live odds scraping (not required for dev)

---

## Proprietary Files (.gitignored)

Excluded from version control. This is the complete set — verify with
`git check-ignore -v <file>` rather than assuming.

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration (database, API, model, betting settings) |
| `data_pipeline.py` | Data ingestion, feature engineering, EWMA market mu computation |
| `value_betting_engine.py` | Kelly criterion, probability calculations, value ranking |
| `prop_integration.py` | 3-tier player matching (odds to projections) |
| `models/position_specific/weekly.py` | Weekly model training and prediction |
| `api/server.py` | FastAPI REST API for frontend dashboard |

**Not proprietary, despite previous versions of this file saying otherwise**:
`materialized_value_view.py` and `scripts/record_outcomes.py` are tracked in git with long commit
histories. They are not gitignored and never were.

**Consequence for the gitignored set**: edits to those files live only on the local machine and in
no commit — a fresh clone gets whatever the deployment supplies. When a change spans a gitignored
module and a tracked one, only the tracked half reaches git. Say so explicitly instead of letting a
reviewer read the commit as the whole change.

Because CI has no access to these modules, logic that CI must verify belongs in a tracked module.
`utils/clv.py` exists for exactly this reason: `scripts/record_outcomes.py` can call it, and the
math stays testable without the private code.

---

## Key Configuration Values

From `config.py`:

- `config.model.target_mae = 3.0` (professional-grade target)
- `config.betting.min_edge_threshold = 0.08` (8% minimum edge)
- `config.betting.min_confidence = 0.75`
- `config.integration.ewma_decay = 0.65`
- WR role priors: alpha=75, secondary=55, slot=45, fringe=30
- Minimum mu floor: 15.0

---

## Common Commands

```bash
# Install
make install

# Ingest data
make ingest-nfl

# Run tests
make test

# Weekly workflow (local/manual — see docs/OPERATIONS.md for the production run)
make week-predict SEASON=2025 WEEK=13
make week-materialize SEASON=2025 WEEK=13
make week-grade SEASON=2025 WEEK=13      # grades bets, records CLV

# Quality gate — non-zero exit when a position regresses past its MAE ceiling
make mae-gate SEASON=2025 WEEK=13

# Durable production path
make migrate
make doctor
make production-run SEASON=2026 WEEK=1
make pipeline-worker

# Launch services
make api          # FastAPI on :8000
make frontend-dev # Next.js on :3000
make fullstack    # Both
make dashboard    # Streamlit on :8501
```

---

## nflverse/nflreadpy Reference

### Available Functions
```python
import nflreadpy as nfl

# Core data
nfl.load_player_stats([2025])      # Weekly player stats
nfl.load_pbp([2025])               # Play-by-play with EPA
nfl.load_snap_counts([2025])       # Snap counts
nfl.load_schedules([2025])         # Game schedule
nfl.load_rosters([2025])           # Current rosters
nfl.load_depth_charts([2025])      # Depth charts
nfl.load_ftn_charting([2025])      # Route/target data
```

### Update Cadence
- Player/team stats: Nightly after games
- Schedules: Every 5 minutes in-season
- Depth charts: Daily 07:00 UTC
- Snap counts: 4x/day
- FTN charting: Every 6 hours

### Data Notes
- Returns Polars DataFrames (convert with `.to_pandas()`)
- Uses `team` column (not `recent_team`)
- License: CC-BY-4.0 (FTN is CC-BY-SA-4.0)
- Pull Thursday AM UTC for corrected "clean" data

---

## Architecture

```
nflreadpy -> ingest_real_nfl_data.py -> player_stats_enhanced
                                              |
                                    weekly.py (train/predict)
                                              |
                                     weekly_projections
                                              |
Odds API -> prop_line_scraper.py -> weekly_odds
                                              |
                                prop_integration.py (3-tier match)
                                              |
                              value_betting_engine.py (Kelly + no-vig edge)
                                              |
                           materialized_value_view.py (dashboard layer)
                                              |
                             api/server.py -> React Dashboard

Grading loop (after results land):
  weekly_odds + materialized_value_view
        -> utils/clv.py (closing line, points + no-vig bp)
        -> scripts/record_outcomes.py
        -> bet_outcomes, clv_weekly, weekly_performance.clv_avg
```

This is the data-flow view. For the durable job/worker execution architecture that actually runs
production — FastAPI enqueues, a separate worker owns the fail-closed pipeline — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/OPERATIONS.md](docs/OPERATIONS.md).
`make week-*` targets are the local/manual path and are **not** the production publication path.

---

## Data Status

Local `nfl_data.db` as of 2026-07-24 (`player_stats_enhanced`):

| Season | Rows | Players | Weeks |
|--------|------|---------|-------|
| 2023 | 6,167 | 604 | 1–22 |
| 2024 | 6,407 | 639 | 1–22 |
| 2025 | 6,558 | 663 | 1–21 |

Week numbers run past 18 because postseason weeks are included. Re-run the verification query
below rather than trusting these counts — they drift with every ingest.

**2026 season prep (as of 2026-08-03)**: the full 2026 schedule is loaded (`games`: 272 games, all
with `kickoff_utc`), 2026 rosters are in `nfl_roster_players` (2,924 players, 32 teams), week-1
context snapshots exist, and `make week-refresh SEASON=2026 WEEK=1` has produced 1,396 week-1
projections (872 players) with `team` populated on every row. Feeds keyed to the stats year
(weekly stats, injuries, weekly rosters) are unpublished until the season starts; the ingest skips
them with a warning instead of crashing (`_is_missing_feed_error` treats nflreadpy's season-range
`ValueError` as an unpublished feed for optional seasons only — history seasons still fail loud).

**Data Source**: All data ingested via `scripts/ingest_real_nfl_data.py` using nflverse/nflreadpy.

**Known gap**: `weekly_projections.team` is empty on legacy 2025 rows (546 of 568), so evaluation
joins to `games` find no kickoff for those weeks and `make mae-gate` fails loud with
`missing_kickoff` there. The current roster-backed prediction path populates `team` (2026 W1: 0
empty of 1,396), so the gate becomes verifiable on real data once 2026 actuals land. Do not
re-run `make week-refresh` for a past 2025 week to "fix" those rows — it would overwrite pregame
evidence (and the pre-kickoff guard refuses anyway).

### Verify Data
```bash
DB_BACKEND=sqlite SQLITE_DB_PATH=nfl_data.db uv run python -c "
from utils.db import read_dataframe
print(read_dataframe('SELECT season, COUNT(*) as rows, COUNT(DISTINCT player_id) as players FROM player_stats_enhanced GROUP BY season'))
"
```

---

## Key Features

### Defense Adjustments
- Relative performance vs defense multipliers
- Applied during feature engineering in `data_pipeline.py`

### WR-Specific Enhancements
- EWMA with decay=0.65 for market mu computation
- Role-based cluster priors (alpha/secondary/slot/fringe)
- Blended weighting (55% hist, 30% targets, 15% role)

### Player Matching (3-Tier)
- Tier 1: player_id exact match
- Tier 2: name + team match
- Tier 3: name only match (WR team mismatch tolerance for trades)
- Implemented in `prop_integration.py`

### Dashboard Features
- "Best Line Only" toggle
- Multiple sportsbook comparison
- Value ranking by edge/CLV
- Real-time projection updates

---

## Testing

Run full test suite:
```bash
make test
```

Key test files:
- `tests/test_market_mu_wr.py` - EWMA and role priors
- `tests/test_prop_integration_wr.py` - 3-tier player matching
- `tests/test_nfl_projection_evaluation.py` - evaluation metrics and the per-position MAE gate
- `tests/test_clv.py` - closing line value math (points and no-vig basis points), including the
  kickoff-aware closing definition
- `tests/test_event_keys.py` - odds → game key resolution; the contract the gitignored writers honor
- `tests/test_odds_quality.py` - screening unjoinable and circular snapshots out of grading
- `tests/test_kelly_cap.py` - Kelly fraction capping
- `tests/test_value_engine_side.py` - over/under side handling
- `tests/test_weekly_pipeline.py` - end-to-end ingest → train → predict → materialize. Seeds its
  own `games` rows: odds are keyed by game, so a club with no scheduled game gets no line and
  every later assertion would pass vacuously.

`tests/conftest.py` uses `collect_ignore` to skip tests that import gitignored modules, so the
suite runs in CI without the private code. Tests for logic CI must cover therefore need to import
from tracked modules only.

---

## Notes

- Database migrations are managed by `schema_migrations.py`. `_ensure_indexes` has a **MySQL branch
  that returns before the SQLite index list** — an index added to only one branch silently does not
  exist on the other. Add to both. `materialized_value_view` also has a near-duplicate `CREATE`
  inside `_rebuild_mvv_pk_if_needed`; schema changes must land in both copies.
- Most proprietary logic is gitignored, but not all — see the Proprietary Files section for the
  exact set and why it matters for CI.
- Use `make fullstack` for complete local development environment
- Front-end dashboard is in `/frontend` (Next.js + TypeScript)
- Legacy Streamlit dashboard available via `make dashboard`
- Further docs: [ARCHITECTURE](docs/ARCHITECTURE.md) (durable job pipeline),
  [OPERATIONS](docs/OPERATIONS.md) (weekly runbook),
  [TROUBLESHOOTING](docs/TROUBLESHOOTING.md), [MODEL_CARD](docs/MODEL_CARD.md)

---

## Active 2026-Prep Punch List

A 5-agent audit identified blockers and high-impact fixes for the 2026 season. Use this as the source of work when invoking AI-DLC's Requirements Analysis on a fix item.

### Tier 0 — BLOCKERS (broken code) — ALL RESOLVED
1. [RESOLVED] Snap counts dead merge — `scripts/ingest_real_nfl_data.py:100-111`. Real snap counts now merged; no `snap_percentage=50.0` fallback.
2. [RESOLVED] Auth endpoints all TypeError — `api/server.py` vs `api/auth.py`. Signatures aligned and password hashing moved to bcrypt.
3. [RESOLVED] `user_bets` INSERT broken — column list and placeholder count now match.
4. [RESOLVED] Side hardcoded "over" — over/under both supported end to end (`side` column on `materialized_value_view`).
5. [RESOLVED] Hardcoded season/week — `prop_integration.py` now requires explicit season/week.
6. [RESOLVED] Hardcoded fields — real `age` and `game_date` are ingested rather than defaulted.

### Tier 1 — HIGH IMPACT (MAE + ROI)
7. Premium features dropped in `_CONTEXTUAL_COLS` (weekly.py:44).
8. [RESOLVED] No vig removal — `implied_probability_no_vig` now lives in `value_betting_engine.py`
   and is what `utils/clv.py` uses for probability-space CLV.
9. [RESOLVED] CLV never captured — `utils/clv.py` computes it; `scripts/record_outcomes.py` writes
   per-bet rows to `clv_weekly` and aggregates into `weekly_performance.clv_avg`. Closing line is
   now the **last snapshot at or before kickoff** when `resolve_closing_lines` is given a
   `kickoffs` frame; omitting it preserves the old `MAX(as_of)` behavior exactly. Degradation is
   per key, not wholesale — a key with no schedule row or an unparseable kickoff keeps
   `MAX(as_of)`, and a key whose every snapshot is post-kickoff yields no closing row rather than
   one graded off a stale in-game quote. A key with a single snapshot reports
   `insufficient_snapshots`, never a silent 0.

   **`weekly_odds.event_id` is now a real game key.** It previously held per-player strings
   (`2025_W22_NE_a_hooper`) and The Odds API's opaque provider ids, both of which joined to zero
   `games` rows — which is why kickoff was unreachable. `utils/event_keys.py` (tracked) mints the
   canonical nflverse form `{season}_{week:02d}_{away}_{home}`, and all three writers resolve
   through it; a row that cannot be tied to a game is dropped rather than stored under a key that
   looks joinable. `utils/odds_quality.py` screens the two disqualifiers — unjoinable keys and
   circular `SimBook` rows — out of the value/CLV path.

   **The 89 pre-existing snapshots are not backfillable and were deliberately left in place.**
   `describe_excluded` reports `{total: 89, unjoinable: 89, synthetic: 72, gradeable: 0}`. Week 10
   rows are the `alpha_receiver` test fixture; week 22 has zero scheduled games. `clv_weekly` is
   empty, so nothing was ever computed from them. Screened, not deleted.
10. No NFL walk-forward backtest — NBA has `utils/nba_backtest.py`.
11. [RESOLVED — by deletion] Universal model, no position split. Decision: the orphaned `RBModel` subclass was deleted rather than revived; `models/position_specific/weekly.py` is the single production model path. `BasePositionModel` is retained as the shared base. Revisit per-position splits as new work against weekly.py, not the old subclass.
12. [MOSTLY RESOLVED] nflreadpy sources unused — rosters, weekly rosters, schedules, depth charts,
    injuries, and pbp red-zone touches are all ingested by `scripts/ingest_real_nfl_data.py` and
    feed `games`, `nfl_roster_players`, and `nfl_player_context_snapshots`. Still unused: FTN
    charting; pbp is only mined for red-zone touches (EPA and the rest untapped).
13. Kelly cap not applied in ranking path — `materialized_value_view.py:139`.

### Tier 2 — MEDIUM (correctness/ops)
14. EWMA decay 0.65 untuned, uniform across markets.
15. Defense multiplier double-applied (`weekly.py:693-703`).
16. SQLite/MySQL parity broken (`AUTOINCREMENT`, `ON CONFLICT`).
17. SQLite no WAL, no pool.
18. Migration runner no version table, no rollback.
19. Opening vs closing line never separated.
20. Tier-3 name-only match no position guard; suffix stripping destructive.
21. Stale-line filter missing. `utils/matching.filter_stale_snapshots` exists, is tested, and is
    still **unwired**. It joins on `event_id`, which was the blocker; that join now works for new
    writes. It filters nothing on the 89 legacy rows, since none carry a joinable key.
22. CORS hard-coded localhost; no rate limits; unbounded threads.
23. No observability — no Sentry/OTel; logger has no basicConfig.

### Tier 3 — LOWER (polish)
24. Property tests for Kelly/edge/vig math.
25. [RESOLVED] CI gate on per-position MAE — `check_position_mae` in
    `scripts/evaluate_nfl_projections.py`, exposed as the `mae-gate` subcommand and `make mae-gate`.
    Ceilings: QB 18.0, RB 12.0, WR 12.0, TE 9.0; a position under 30 projections is reported as
    skipped, never silently passed. CI runs the gate's unit tests only (no projection data in CI);
    the real-data run is the Makefile target. See the Data Status note — the real-data path
    currently fails on `missing_kickoff`.
26. Perf regression budgets.
27. Stacking final estimator Ridge → LightGBM or isotonic calibration.
28. WR role priors stale.
29. Cache stale-while-revalidate no in-flight dedup.
30. [RESOLVED] `materialized_value_view` ranking index — composite `(season, week, edge_percentage)` on `idx_materialized_value_view_lookup`, created on both the SQLite and MySQL branches of `_ensure_indexes`. It supersedes the former `(season, week)` index.
