# Fixed-subset observer resource benchmark

The original plan section 8.4 requires separate observer time and GPU memory
accounting. The frozen runtime records calibration replay time and ordinary
observer-free generation time/peaks, but does not measure a matched diagnostic
observer-on/off marginal. This independent engineering phase fills that gap
without modifying the original 23 sources, full512 scientific sources/plan,
model traces, calibration parameters, or scientific thresholds.

The fixed subset is C_diag IDs 128,129,130,131 and scenarios A,B,C,D. Only if
the original P3/P4 actually triggered an update does it also include M_NT and
M_ST on each updated arm's own saved C_diag trajectory. Every measured replay
uses the entire saved trace, including prefill and all T-1 valid decode
forwards; emitted EOS is never consumed. There is one on/off pair per unit,
with observer-on first exactly when (sample_id+scenario_index)%2==0. Scenarios
are indexed A,B,C,D,M_NT,M_ST in that order. No short measured proxy replaces
a long trajectory. The result is a marginal engineering benchmark for this
subset, not a reconstruction of historical cost or a full-run cumulative
estimate. There is no extrapolation or serving-speed claim and no bootstrap.

Each unit first performs one observer-on replay of the first min(8,T) tokens
to warm model, shadow, collector, and calibrator kernels. This warmup is
excluded from measured on/off metrics. Both measured modes have identical
saved prompt/trace, driver, calibrators, capture_logits=False and the original
finite-logit guard. On and off must produce the same entire argmax sequence.
A,C and updated-own units must additionally reproduce the saved source trace;
B,D cross-text units are not required to reproduce the source driver's greedy
tokens. The original callback leaves the driver output unchanged. Student
callbacks also verify the actual output equals their calibrated shadow branch.

Observer-on uses the frozen StateCollector, shadow expert branches, actual
residual captured by the original pre-MoE-add hook, and the same observed
calibration identity as the core diagnostic. A-D keep full and original
min-length common-position groups separately; updated-own units keep their
full own-state groups. All 48 layers and every nonempty original position bin
must reconcile with trace[:-1]. Immediate EOS is a bound empty marker. No
whole hidden-state trajectory is retained or persisted.

Before each measured mode, collect Python garbage, empty the CUDA allocator's
unused cache, synchronize, reset peak counters, and record current allocated
and reserved bytes. A perf_counter interval encloses the entire replay and
its final CUDA synchronization. Record absolute peaks, initial baselines and
peak-minus-baseline for both allocated and reserved memory. Compute signed
on-minus-off differences without clipping negative values. Cache clearing and
baseline disclosure prevent interpreting a mere reset of retained allocator
cache counters as additional memory. Callback host wall time includes implicit
synchronizations already performed by the collector; it is not pure kernel
time and introduces no extra per-callback synchronization. Collector archival
and validation I/O occur after timing/peak collection and are reported
separately. Ordinary free-generation timing remains untouched.

Per-callback perf_counter calls and counter updates are additional benchmark
instrumentation; their small host cost is included in the paired elapsed
measurement. Archive validation checks actual commit hashes, positions and
FP64 array headers; it does not claim to scan every moment value or recompute
the original sufficient statistics.

The model loads once per registered process/allocation. A shared ops handoff
claim, exact PID/boot/start ticks/cgroup/argv and positive predecessor Slurm
terminal evidence are required before model imports. The actual allocation is
one B200, 8 CPUs,128 GiB,24 hours,Requeue=0. Core completion and actual full512
delivery (including a valid NOT_REQUIRED delivery when no update occurred)
must precede READY. Local control uses the already verified local-filesystem
RAID lock helper; no network-lock fallback or new daemon is created.

CPU `freeze` exclusively seals
`reports/<run>/observer_cost/plan.json`, binding the original23 source inventory,
sealed full512 plan and actual copies of the two new cost scripts and two new
cost documents. Outputs are isolated under
`reports/<run>/observer_cost/<plan_hash>/`. Every unit attempt retains its
intent,warmup,on/off mode records and observer-on committed stats. Only a
complete, verified pair publishes pairs/<unit_hash>.json. A half pair is
retained but never counted complete; a resumed process measures a fresh whole
pair instead of combining allocation-dependent halves. Interrupted attempts
and their unknown unfinished elapsed costs remain explicit. All immutable
files publish exclusively with hashes; source and replay identity are checked
before skipping a completed pair.

The raw completion binds all pairs. Pure CPU `analyze` writes descriptive
means/medians and all actual unit records, without a confidence interval or
scientific gate. Its completion receipt is separate from final engineering
delivery. The sole report writer links the unchanged full512 nine-question
delivery and adds the observer-cost appendix. COMPLETE requires verified data,
analysis and actual delivery bytes; missing delivery remains RAW_COMPLETE.
Missing plan is PLAN_PENDING; explicit scientific errors stop automatic
continuation. This phase does not alter scientific status or authorize P6.

```
python scripts/state_study_observer_cost.py freeze --run-dir RUN
python scripts/state_study_observer_cost.py status --run-dir RUN
python scripts/state_study_observer_cost.py check --run-dir RUN
python scripts/state_study_observer_cost.py analyze --run-dir RUN
```

The verified ops dispatcher alone runs the fixed absolute-path command
`python -u scripts/state_study_observer_cost.py run --run-dir RUN` in the
registered same PID. CPU commands do not import a model or initialize CUDA.
