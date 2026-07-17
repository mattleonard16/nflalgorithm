# Durable Pipeline State Machine

The weekly NFL pipeline is fail-closed. FastAPI may create, cancel, retry, and
read jobs, but only a claimed worker may execute stages or publish results.

## Legal job transitions

| From | To | Authority | Conditions |
|---|---|---|---|
| `queued`, `retry_scheduled` | `running` | worker claim | available, not cancelled |
| `queued`, `retry_scheduled` | `cancelled` | operator | conditional update succeeds |
| `running` | `completed` | owning attempt | lease tuple matches and cancellation is false |
| `running` | `retry_scheduled` | owning attempt or recovery | retryable, budget remains, cancellation is false |
| `running` | `failed` | owning attempt or recovery | non-retryable or budget exhausted, cancellation is false |
| `running` | `cancelled` | owning attempt or recovery | cancellation was requested |
| `failed` | `queued` | operator retry | explicit retry only |

`completed`, `failed`, and `cancelled` are terminal. A conditional update that
does not match the expected source state is rejected; no method may infer a
transition from an unlocked, previously read job object.

## Attempt ownership and fencing

Each claim increments `attempts` and generates a random `claim_token`. Lease
ownership is the complete tuple:

```text
(job_id, worker_id, attempts, claim_token)
```

Heartbeat renewal, cancellation checks, stage writes, completion, failure,
card promotion, and artifact registration require that tuple and a currently
running job. The token is compared as binary data so case-insensitive database
collations cannot weaken fencing. Reclaims clear the prior token before a new
attempt receives a different one.

## Attempt history

`pipeline_stage_runs` is append-only by `(run_id, attempt, stage_name)`. Retry
and stale recovery never erase a prior attempt's result, error, or duration.
The run read model computes `stages_completed` from the current attempt only.
Terminal stale recovery closes an in-progress stage consistently as failed or
cancelled; retry recovery leaves history intact and starts a new attempt.

## Odds validation

The odds stage must prove all of the following before value ranking begins:

- requests were not served from offline or stale-on-error cache;
- the newest response is within `NFL_ODDS_MAX_AGE_SECONDS`;
- scheduled-event coverage meets `NFL_ODDS_MIN_EVENT_COVERAGE`;
- required event/market coverage meets `NFL_ODDS_MIN_MARKET_COVERAGE`;
- at least one complete two-sided row was persisted.

The validation reason and metrics are stored per run and attempt in
`pipeline_odds_validations` and included in the stage result exposed by the API.

## Card publication

The materialization stage writes candidates to `pipeline_card_staging`, scoped
by run and attempt. It never mutates the active dashboard table. Completion
locks the owning job, rechecks cancellation and the lease tuple, promotes the
staged card, registers the run artifact, and marks the job/run completed in one
transaction. Losing the lease or requesting cancellation rolls back every part
of publication, so a cancelled or stale attempt cannot become the active card.

