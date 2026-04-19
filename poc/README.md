# CMLIS PoC

Python orchestrator that implements the **Memory Control** and **Heuristic Routing** layers from [../SPEC.md](../SPEC.md) on top of an unmodified `llama.cpp`. Runs on Linux (full enforcement) or Windows/macOS (simulation mode for development).

## What it does

1. **Discovers hardware topology** — NUMA nodes, physical cores, L3 cache (`cmlis topo`).
2. **Classifies a prompt** as short / medium / long and picks a routing policy (`cmlis plan`).
3. **Binds a process** to one NUMA node + its local CPUs via `numactl` + `taskset` on Linux (`cmlis run`).
4. **Benchmarks three configs** — naive / NUMA-only / full CMLIS — across workloads and computes a Welch t-test (`cmlis bench`).
5. **Collects telemetry** — `numastat`-derived remote-NUMA fraction + per-CPU utilization.
6. **Simulates** the whole pipeline when `llama.cpp` or a GGUF model is not present, so the orchestration path can be validated on any machine.

## Install

```bash
cd poc
pip install -e .[dev]
```

## Usage

```bash
# Show topology
cmlis topo

# Show the routing decision + exact numactl/taskset prefix for a 2048-token prompt
cmlis plan --input-tokens 2048

# Simulated single run (no llama.cpp required)
cmlis run --simulate --input-tokens 2048 --output-tokens 128

# Simulated benchmark (short + medium workloads, all three configs)
cmlis bench --simulate --reps 10

# Real benchmark (requires llama-cli on PATH or --binary, and a GGUF model)
cmlis bench --model /models/mixtral-8x7b.Q5_K_M.gguf --reps 30
```

## Outputs

Bench reports are written to `./reports/cmlis-bench-YYYYMMDD-HHMMSS.json`, containing per-run tokens/sec, per-config stats, and significance test results (uplift %, t, p, df, pass flags against the ≥25% uplift / p<0.01 targets from [METHODOLOGY.md](../METHODOLOGY.md)).

## Scope

This PoC implements the orchestration, routing, and measurement machinery. It does **not** ship or patch `llama.cpp`, does **not** download models, and does **not** attempt dense-model sparsity (MVP Phase 2). It is the harness the full MVP would drive.

## Tests

```bash
pytest
```
