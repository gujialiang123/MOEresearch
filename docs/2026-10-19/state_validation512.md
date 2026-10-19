# Full D_validate own-state supplement

This prospective range supplement completes section 9.4 of the original
execution plan. The frozen runner observes updated validation own-state on
128 prompts; this supplement observes all 512 original D_validate prompts,
IDs 512 through 1023, separately for M_NT and M_ST. It does not change or
overwrite the frozen 18 runtime sources, five quality analysis sources, or
original run data. It does not authorize P6.

Before any original formal/validate or formal/validation_state files exist,
an operator seals `validation512/plan.json` under the independent report
directory. The plan binds the original protocol/source archive, prospective
quality draft, and actual supplementary collection/report source copies.
It freezes all IDs, two updated arms, 48 layers, hidden size 2048, original
position bins and numerical protection, FP64 population sufficient statistics,
all valid T-1 decode positions, and paired prompt bootstrap with 10000 draws
and seed 20261019. There are no adjustable range or bootstrap CLI flags.

Collection is applicable only after the original P3 gate actually triggers P4.
An explicit scientifically_not_triggered P4 and verified formal P3 identity
produce NOT_REQUIRED. Missing or pending work never produces NOT_REQUIRED.
The actual collection starts after P5/core completion and positively verified
core Slurm termination. It requires the separate exclusive shared ops handoff
claim and exact same PID/boot/start-ticks/argv/cgroup/current Slurm allocation
record. A local RAID flock adds same-node exclusion; it does not prove remote
process death. The actual allocation must have one B200, 8 CPUs, 128 GiB,
24 hours and Requeue=0, as in the existing continuation resource guards.

The only model operation is frozen StudyDriver.replay on the two arms' actual
saved observer-free formal/validate free-generation traces. No fitting or free
generation is repeated. Every replay uses the original frozen calibrator,
checks the exact rendered prompt/token IDs, all returned argmaxes, and the
actual driver output against the shadow branch. The observer passes the
actual pre-MoE-add residual to the frozen StateCollector. It observes
trace[:-1], including every valid decode forward and never an emitted EOS.
Immediate EOS is a bound empty marker, never an invented activation.

Original first-128 state archives are not reused: all 512 saved traces are
replayed. Each attempt records replay time including observer, wall time
including archival, model loading, and whether the prompt duplicates the
original first-128 coverage. These are supplemental collection costs, not
observer-free generation costs or serving latency.

Outputs are isolated at
`reports/<run>/validation512/<plan_hash>/`: source_snapshot.json,
diagnostics/, replay_checks/<prompt-arm-hash>.json, replay_checks.jsonl,
raw_completion.json, summaries/, completion.json, cost_attempts/, and a
derived manifest.json. Immutable receipts publish exclusively and bind hashes.
The verified per-prompt argmax check is durable before the collector's
commit-last NPZ/token pair. Uncommitted interrupted pairs are retained under
diagnostics/_uncommitted before replay. A committed archive lacking an actual
replay proof requires a real observer-free replay to restore that proof; no
argmax proof is inferred from expected tokens. Completed paths are skipped only
after archive, calibration, driver, trace, mask, and replay-check validation.

The fixed analyzer recomputes the original token-weighted primary and
prompt-weighted sensitivity estimands on the full 512-prompt universe, including
empty prompts. It never refits calibrations or removes undefined bootstrap
draws. A completion receipt records that collection and analysis were executed,
independently of effect direction or confidence-interval estimability. An
all-empty or undefined result can therefore be completed work while remaining
scientifically INCONCLUSIVE. COMPLETE is accepted only through the independent
full512 checker, not merely the completion file. The report replaces the
validation128 primary own-state predicate with full512 evidence and retains
all other original quality/cost/P3 and C_diag requirements.

The original final quality plan is not required to collect section 9.4 state
data. Its absence cannot justify late quality freezing or any positive claim.
The independent positive-chain report still verifies that prospective plan;
quality-final identity is not a mutable part of the state source snapshot.

CPU-only commands (do not start a model):

```
python scripts/state_study_validation512.py freeze --run-dir RUN
python scripts/state_study_validation512.py check --run-dir RUN
python scripts/state_study_validation512.py status --run-dir RUN
python scripts/state_study_validation512.py analyze --run-dir RUN
```

Actual run is dispatched only by the verified continuation helper with fixed
argv `python -u scripts/state_study_validation512.py run --run-dir RUN` (the
helper uses absolute source paths). Direct unregistered model runs are refused.
Missing plan is PLAN_PENDING; mismatched sources are INVALID_PLAN; actual
scientific errors are SCIENTIFIC_ERROR. WAITING_P4 and WAITING_CORE remain
pending; READY means registered collection may begin; RAW_COMPLETE means raw
receipts verify and only CPU analysis remains; COMPLETE means the independent
checker accepted finished work. Reports and logs are on scratch; active local
locks and process metadata are on RAID. No daemon, socket, SQLite, or new
Slurm job is created by this module.
