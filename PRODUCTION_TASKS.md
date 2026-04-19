# Production Readiness Task List

This task list converts the current production-readiness review into concrete remediation work. The goal is to close the gaps between the documented CMLIS target architecture and the behavior of the current Python PoC in `poc/`.

## Exit Criteria

The PoC is not considered production-ready until all of the following are true:

- Real runs fail closed when required binaries, models, datasets, or baseline results are missing.
- Benchmark reports exclude failed executions and preserve enough metadata to audit every run.
- Telemetry is process-scoped and strong enough to support NUMA locality claims.
- Documented routing features are implemented in the execution path, not just represented in metadata.
- The benchmark harness enforces methodology gates instead of warning past invalid conditions.
- CI covers the real runtime path in addition to simulation mode.

## Workstreams

### 1. Fail Closed on Missing Runtime Dependencies

- [x] Make `cmlis run` return an error unless `--simulate` is explicitly set or a valid `llama.cpp` binary and model are present.
- [x] Make `cmlis ppl` return an error unless `--simulate` is explicitly set or a valid binary, model, and dataset are available.
- [x] Remove silent fallback to synthetic benchmark or perplexity results during supposed real runs.
- [x] Add explicit CLI messaging that states whether a run is simulated or real and why.
- [x] Align `engine.run()` and `ppl.run_ppl()` so they use the same binary discovery rules.

### 2. Preserve Execution Integrity in Benchmark Reports

- [x] Extend benchmark result records to include `exit_code`, stderr summary, and a machine-readable run status.
- [x] Exclude failed subprocesses from throughput and significance calculations.
- [x] Treat missing timing output as a failed measurement unless simulation mode is active.
- [x] Record whether each run used simulation, real execution, or degraded measurement logic.
- [x] Add report-level validity flags for methodology compliance.

### 3. Enforce Methodology Gates Instead of Warning Past Them

- [x] Fail the benchmark when swap is in use unless an explicit override is provided.
- [x] Fail the benchmark when repetitions are below the documented minimum for statistical validity.
- [x] Fail the benchmark when required cache-drop or NUMA-balancing controls cannot be applied on Linux benchmark runs.
- [x] Return a non-zero exit code from `cmlis bench` when benchmark validity gates fail.
- [x] Add a strict mode suitable for CI and formal benchmark collection.

### 4. Make Telemetry Evidence Defensible

- [x] Replace system-wide-only locality sampling with process-scoped measurement where possible.
- [x] Ensure telemetry attaches to the real inference PID before measurement begins.
- [x] Distinguish unavailable telemetry from zero-valued telemetry in reports.
- [x] Add explicit capture for the metrics required by the methodology: locality, cache behavior, and compute utilization.
- [x] Persist raw telemetry samples alongside summary statistics for auditability.

### 5. Implement the Documented Routing Features

- [x] Wire `kv_cache_chunks` into the real execution path or remove the claim until implemented.
- [x] Implement a real long-context KV placement strategy or document that it is not yet supported.
- [ ] Validate that expert-limiting flags map to current `llama.cpp` behavior for supported models.
- [x] Add multi-socket placement logic instead of hard-wiring all routing to NUMA node `0`.
- [x] Add run-time validation that the binding plan actually matches the intended topology.

### 6. Close the Spec-to-Code Gaps in Workloads

- [ ] Replace the placeholder `mixed` workload with the documented seeded context-switching dataset.
- [ ] Persist workload provenance, seed, and prompt source in benchmark reports.
- [ ] Add dataset validation so repeated benchmark runs are reproducible.
- [ ] Separate synthetic development workloads from methodology-grade benchmark workloads.

### 7. Strengthen PPL and Coherence Validation

- [ ] Require a naive baseline before computing PPL degradation pass/fail.
- [ ] Fail PPL evaluation when the baseline result is missing or invalid.
- [ ] Record command, exit code, dataset path, and parsing status for each PPL run.
- [ ] Add tests for baseline-missing, parse-failure, and timeout behavior.

### 8. Expand Test Coverage Beyond Simulation

- [ ] Add unit tests that assert failed subprocesses do not produce synthetic throughput.
- [ ] Add tests that assert missing binaries/models cause hard failures outside simulation mode.
- [ ] Add telemetry tests that verify PID scoping and unavailable-metric behavior.
- [ ] Add regression tests for methodology gate failures and non-zero CLI exit codes.
- [ ] Add Linux integration tests for real command composition and benchmark validity checks.

### 9. Fix Packaging and CI Assumptions

- [ ] Move runtime-required packages out of `dev` extras if they are needed by shipped CLI commands.
- [ ] Add CI jobs that exercise CLI smoke tests, not just unit tests.
- [ ] Add a Linux-only integration lane for NUMA-aware code paths.
- [ ] Keep simulation-mode CI coverage, but clearly separate it from real-runtime validation.

### 10. Update Documentation to Match Actual State

- [ ] Check the state of the project to see if we are ready to go beyond PoC.
- [ ] Distinguish implemented features from target-state architecture in README and project docs.
- [ ] Document which claims are currently simulated, partial, or unverified.
- [ ] Link this task list from the top-level README and planning docs until the gaps are closed then remove all references when they are closed.

## Suggested Delivery Order

1. Fail closed on missing dependencies and invalid subprocess results.
2. Enforce benchmark validity gates and non-zero exit behavior.
3. Fix telemetry scoping and evidence quality.
4. Implement or narrow the documented routing claims.
5. Replace placeholder workloads and strengthen PPL validation.
6. Expand CI and test coverage around real runtime behavior.
7. Re-run the documented benchmark methodology on the target Linux hardware.
