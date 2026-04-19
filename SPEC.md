# CMLIS System Specification

**Version:** 3.0 (Systems-Rigorous & Peer-Reviewed)  
**Date:** April 2026  
**Scope:** Memory-Aware CPU Inference Layer (Heuristic + NUMA Optimization)

---

## 1. System Architecture

The CMLIS-MVP consists of three interacting layers built entirely on existing tools.

### 1.1 Base Inference Engine
- **`llama.cpp`** (latest main branch)
- GGUF quantized execution with CPU SIMD optimizations (e.g., AVX-512)
- Supports Mixtral (MoE, native sparse) and Llama 3.1 (dense) models

### 1.2 Memory Control Layer (Core Focus)
- Strict NUMA node binding (`numactl --cpunodebind --membind`)
- Thread affinity and physical core pinning (`taskset`) to prevent cross-chiplet drift
- One isolated instance per socket on multi-socket systems
- **Telemetry Stack:** Continuous monitoring of memory traffic (`numastat`), cache misses (`perf stat`), and compute utilization (`mpstat` to isolate dequantization overhead)

### 1.3 Heuristic Routing Layer
- Purely rules-based orchestrator (no ML training overhead)
- Prompt-length and context-based decisions
- Short prompts: Aggressive limitation of active experts/modules to maximize cache residency
- Long prompts: Full context + chunked KV handling + NUMA-aware KV placement

---

## 2. Memory Model Specification

- Standard GGUF format, fully loaded into DRAM
- **Phase 1 (Primary):** Mixtral 8x7B (Q5_K_M / Q6_K)
- **Phase 2 (Secondary):** Llama 3.1 70B (Q4_K_M / Q5_K_M)
- Explicit NUMA binding applied to both model weights and KV cache segments

---

## 3. Execution Flow

1. **State Reset:** OS-level cache cleared (`echo 3 > /proc/sys/vm/drop_caches`) to ensure true DRAM fetch metrics.
2. **Routing Assessment:** Heuristic Layer classifies prompt length and context-switching requirements.
3. **Hardware Binding:** Memory Control Layer enforces strict NUMA binding and thread affinity.
4. **Execution:** Engine executes with locality-optimized scheduling.
5. **Telemetry:** Hardware counters (`perf`, `numastat`, `mpstat`) logged to time-series output.

---

## 4. Hardware Assumptions

**Minimum Baseline:** - 128 GB DDR5 RAM, 16+ core x86 CPU, Linux (Ubuntu 24.04+)
- Single-socket systems are used *only* for baseline control comparisons.

**Primary Validation Target:** - Dual-socket AMD EPYC (e.g., Genoa) or Intel Xeon Scalable (4th Gen+)
- 256+ GB RAM with multiple memory channels
- **Requirement:** L3 cache partition structure must be explicitly mapped prior to execution.

---

## 5. Performance Targets

**Primary Success Metrics (Mixtral 8x7B on dual-socket, medium context)** - **Throughput:** Statistically significant ($p < 0.01$) ≥ 20–25% improvement in tokens/sec over naive baseline.
- **Absolute Floor:** ≥ 3.5 tokens/sec sustained (2048→512 tokens).
- **Locality:** Remote NUMA access strictly < 10% of total traffic.
- **Stability:** Stable inference with < 5% variance and zero OS swapping.
- **Strict Coherence:** Perplexity (PPL) degradation on WikiText-2 remains ≤ 0.1 relative to baseline.

**Critical Caveat:** These targets are conditional on successfully mitigating random-access memory penalties and KV cache bottlenecks. If these hardware limits dominate, the MVP must transparently document the null outcome.

---

## 6. Critical Risks & Edge Cases

**Highest Priority Risks**
1. **Random Access Penalty:** Dynamic subset selection destroys sequential hardware prefetching, severely reducing effective bandwidth (GB/s).
2. **KV Cache Bottleneck:** KV cache frequently dominates memory traffic at >4k contexts; static weight binding does not solve dynamic KV placement.
3. **NUMA Binding Fragility:** Inconsistent topology enforcement across different chiplet architectures and Linux kernel versions.

**Medium Priority Risks**
4. Compute overhead masking memory gains (e.g., GGUF dequantization cycles).
5. Threading deadlocks between `llama.cpp` internal schedulers and external OS affinity wrappers.
6. Output coherence collapse resulting from overly aggressive heuristic routing on dense models.

---

## 7. Non-Goals

- Learned routing or neural policies
- Training, fine-tuning, or altering weights
- GPU acceleration or distributed network inference
- New foundational model architectures
- Production serving features (continuous batching, speculative decoding)

---

**Success Definition** Demonstrate **statistically significant, reproducible** performance gains on medium-context workloads while maintaining strict coherence limits and transparently documenting hardware constraints. The project succeeds by rigorously validating (or refuting) the engineering value of memory topology awareness on CPU hardware.