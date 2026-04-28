# CMLIS

[![CI](https://github.com/rbenzing/CMLIS/actions/workflows/ci.yml/badge.svg)](https://github.com/rbenzing/CMLIS/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/rbenzing/CMLIS/blob/main/poc/pyproject.toml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/rbenzing/CMLIS/blob/main/LICENSE)

**CPU-native Modular Language Intelligence System** — A CPU-first inference orchestration framework that validates NUMA-aware memory control and heuristic routing can deliver ≥25% throughput uplift on large models without retraining.

> 🔬 **Research PoC Status**: This is a prototype for validating architectural concepts, not production-ready inference middleware. See [production readiness gaps](#production-readiness).

---

## 📋 Table of Contents

- [What It Does](#what-it-does)
- [Current Status](#current-status)
- [Quick Start](#quick-start)
- [Project Goals](#project-goals)
- [Repository Contents](#repository-contents)
- [Platform Notes](#platform-notes)
- [Production Readiness](#production-readiness)
- [Learn More](#learn-more)

---

## What It Does

CMLIS orchestrates inference workloads on multi-socket x86 systems by:

1. **Discovering hardware topology** — NUMA nodes, physical cores, L3 cache layouts
2. **Classifying prompts** into routing tiers (short, medium, long)
3. **Binding processes** to specific NUMA nodes with CPU affinity
4. **Benchmarking configurations** across workloads with statistical validation
5. **Collecting telemetry** on locality, throughput, and resource utilization
6. **Simulating** the entire pipeline when hardware isn't available

---

## Current Status

### ✅ What Works

- **Topology discovery** — Automatic detection of sockets, NUMA nodes, physical cores, and L3 cache
- **Smart routing** — Classifies prompts and applies NUMA-aware process binding via `numactl` + `taskset`
- **Benchmarking** — Compares naive, NUMA-only, and full CMLIS configurations with Welch t-test validation
- **Cross-platform simulation** — Run the full orchestration pipeline on any machine without hardware
- **Telemetry collection** — Remote NUMA traffic fraction and per-CPU utilization tracking

### ⚠️ Known Limitations

- **KV cache placement** — `kv_cache_chunks` is currently routing metadata only; real KV placement not yet wired into `llama.cpp`
- **Long-context NUMA awareness** — Not yet implemented in the runtime path
- **Runtime isolation** — Benchmark runs rotate across NUMA nodes but don't yet launch isolated per-socket instances
- **Expert routing** — Uses `--override-kv` but model-specific validation still pending

---

## Quick Start

### Installation

```bash
cd poc
pip install -e ".[dev]"
pytest
ruff check . --fix
```

### Run Locally

```bash
cd poc

# Discover your hardware topology
cmlis topo

# Get routing decision for a 2048-token prompt
cmlis plan --input-tokens 2048

# Simulate a single run (no hardware required)
cmlis run --simulate --input-tokens 2048 --output-tokens 128

# Simulate a full benchmark (10 repetitions)
cmlis bench --simulate --reps 10

# Run real benchmark (requires llama-cli + GGUF model)
cmlis bench --model /models/mixtral-8x7b.Q5_K_M.gguf --reps 30
```

### Output

Benchmark reports are written to `./reports/cmlis-bench-YYYYMMDD-HHMMSS.json` with:
- Per-run throughput (tokens/sec)
- Per-configuration statistics
- Significance test results (uplift %, t-statistic, p-value, degrees of freedom)

---

## Project Goals

CMLIS aims to validate that memory-topology-aware orchestration can improve CPU inference **without retraining or modifying the underlying model engine**.

### Primary Success Criteria (from [SPEC.md](./SPEC.md))

| Metric | Target |
|--------|--------|
| Throughput uplift | ≥25% over naive baseline (statistically significant) |
| Mixtral 8x7B (medium-context) | ≥3.5 tok/s |
| Remote NUMA traffic | <10% during optimized runs |
| Perplexity regression | No material coherence loss |

> ⚠️ These are **target validation criteria**, not claims that the current PoC has proven them at production grade.

---

## Repository Contents

| Directory/File | Purpose |
|----------------|---------|
| [`poc/`](./poc) | Runnable Python PoC package and test suite |
| [`SPEC.md`](./SPEC.md) | System specification and success criteria |
| [`INSTALL.md`](./INSTALL.md) | Hardware and environment setup guide |
| [`METHODOLOGY.md`](./METHODOLOGY.md) | Benchmark methodology and validation rules |
| [`MODEL.md`](./MODEL.md) | Model selection and constraints |
| [`BRIEF.md`](./BRIEF.md) | Executive system overview |
| [`PRODUCTION_TASKS.md`](./PRODUCTION_TASKS.md) | Roadmap for production-readiness gaps |

---

## Platform Notes

| Platform | Support Level | Notes |
|----------|---------------|-------|
| **Linux** | ✅ Full | NUMA enforcement, CPU affinity, `numastat` telemetry |
| **Windows** | ✅ Dev/Sim | Development and simulation mode only |
| **macOS** | ✅ Dev/Sim | Development and simulation mode only |

**Reference configuration**: [`poc/configs/mixtral-dual-socket.json`](./poc/configs/mixtral-dual-socket.json)

---

## Production Readiness

This PoC has several gaps before production deployment. The complete remediation plan is documented in [`PRODUCTION_TASKS.md`](./PRODUCTION_TASKS.md).

### Key Gaps

- Real KV cache placement strategy integration with `llama.cpp`
- Long-context NUMA-aware KV management
- Per-socket runtime isolation for benchmarking
- Production-grade error handling and observability

---

## Learn More

- **PoC usage & API**: [`poc/README.md`](./poc/README.md)
- **Installation & hardware setup**: [`INSTALL.md`](./INSTALL.md)
- **System architecture brief**: [`BRIEF.md`](./BRIEF.md)
- **Benchmark methodology**: [`METHODOLOGY.md`](./METHODOLOGY.md)
- **Model selection**: [`MODEL.md`](./MODEL.md)
- **Production roadmap**: [`PRODUCTION_TASKS.md`](./PRODUCTION_TASKS.md)

---

## License

This project is licensed under the [GNU General Public License v3.0](./LICENSE).