# CMLIS

[![CI](https://github.com/rbenzing/CMLIS/actions/workflows/ci.yml/badge.svg)](https://github.com/rbenzing/CMLIS/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/rbenzing/CMLIS/blob/main/poc/pyproject.toml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/rbenzing/CMLIS/blob/main/LICENSE)

CPU-native Modular Language Intelligence System (CMLIS) is a CPU-first inference project focused on improving large-model performance on multi-socket x86 systems through NUMA-aware memory control and heuristic routing.

The repository currently contains the project specification, methodology, installation guidance, and a Python proof of concept in [`poc/`](./poc) that orchestrates topology discovery, routing decisions, benchmarking, telemetry, and simulation mode on top of unmodified `llama.cpp`.

## Current Status

This repository is currently a research PoC, not a production-ready inference layer.

- The top-level documents describe the target CMLIS architecture and evaluation criteria.
- The Python package in `poc/` is an orchestration prototype used to test routing, benchmark flow, telemetry collection, and simulation mode.
- Several production-readiness gaps remain between the documented target state and the current implementation.
- The remediation backlog for those gaps lives in [`PRODUCTION_TASKS.md`](./PRODUCTION_TASKS.md).

## What This Repo Contains

- `poc/` - the runnable Python PoC package and test suite
- `SPEC.md` - system specification and success criteria
- `INSTALL.md` - hardware and environment setup guide
- `METHODOLOGY.md` - benchmark methodology and validation rules
- `MODEL.md` - model selection and constraints
- `PRODUCTION_TASKS.md` - task list for closing production-readiness gaps

## Current PoC Capabilities

- Discovers sockets, NUMA nodes, physical cores, and L3 layout
- Classifies prompts into short, medium, and long routing tiers
- Applies Linux NUMA binding and CPU affinity for real runs
- Rotates benchmark routing across discovered NUMA nodes instead of pinning every routed run to node `0`
- Supports `--simulate` as a first-class execution path on any machine
- Benchmarks naive, NUMA-only, and full CMLIS configurations
- Records telemetry for locality and throughput analysis

## Current PoC Limitations

- `kv_cache_chunks` is planning metadata in the current PoC; it does not yet wire a real KV placement strategy into `llama.cpp`.
- Long-context NUMA-aware KV placement remains unimplemented in the current runtime path.
- Expert-limiting uses `--override-kv`, but model-specific validation against current `llama.cpp` behavior is still pending.

## Quick Start

```bash
cd poc
pip install -e ".[dev]"
pytest
ruff check . --fix
```

Try the PoC locally:

```bash
cd poc
cmlis topo
cmlis plan --input-tokens 2048
cmlis run --simulate --input-tokens 2048 --output-tokens 128
cmlis bench --simulate --reps 10
```

## Project Goals

CMLIS is aiming to validate that memory-topology-aware orchestration can improve CPU inference without retraining or modifying the underlying model engine.

The goals below remain target-state validation criteria, not claims that the current PoC has already proven them in a production-grade way.

Primary targets from [`SPEC.md`](./SPEC.md):

- at least 25% throughput uplift over a naive baseline with statistical significance
- at least 3.5 tok/s on Mixtral 8x7B medium-context workloads
- under 10% remote NUMA traffic during optimized runs
- no material coherence regression relative to baseline perplexity

## Platform Notes

- Linux is the primary target for full NUMA enforcement
- Windows and macOS are supported for development and simulation mode
- The reference benchmark configuration lives in `poc/configs/mixtral-dual-socket.json`

## More Detail

- PoC usage: [`poc/README.md`](./poc/README.md)
- Installation and hardware setup: [`INSTALL.md`](./INSTALL.md)
- System brief: [`BRIEF.md`](./BRIEF.md)
- Benchmark methodology: [`METHODOLOGY.md`](./METHODOLOGY.md)
- Production remediation plan: [`PRODUCTION_TASKS.md`](./PRODUCTION_TASKS.md)
