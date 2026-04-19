# CMLIS-MVP: Memory-Aware CPU LLM Inference Layer

**Version:** 1.3 (Systems-Rigorous & Peer-Reviewed)  
**Status:** Active Development

---

## 1. MVP Goal

Demonstrate that **heuristic routing combined with NUMA-aware memory control** can deliver a statistically significant improvement in the inference performance of large language models on CPU-only systems. The primary focus is optimizing memory locality and bandwidth efficiency rather than raw compute throughput.

**Core Objective:** Validate the CMLIS framework by proving that intelligent management of memory topology and selective parameter activation reduces remote memory traffic and increases tokens/sec on 128 GB+ NUMA systems—without requiring new model architectures or retraining.

**Target Performance:** A statistically significant ($p < 0.01$) improvement of ≥ 25% over naive baselines, maintaining ≥ 3.5 tokens/sec on medium-context workloads on optimized hardware.

---

## 2. Scientific Foundation

CMLIS reframes LLM inference as a **memory-conditioned system**:

**y_t = g( R(x_t, M), M_t ⊆ M, H(R(x_t)) )**

where the routing function $R(\cdot)$ selects active memory subsets to minimize data movement across the hardware bus and maximize NUMA/L3 cache locality.

This MVP tests the hypothesis that **effective local memory bandwidth, NUMA locality, and working-set management dominate CPU LLM performance** far more than core count or theoretical peak FLOPS.

---

## 3. System Architecture

The MVP consists of three interacting layers:

### 3.1 Base Inference Engine
- **`llama.cpp`** (latest main branch, GGUF format).
- Handles low-level quantized model execution and CPU SIMD instructions (e.g., AVX-512).

### 3.2 Memory Control Layer (Core Innovation)
- Enforces strict NUMA node binding (`numactl`).
- Manages thread affinity and physical core pinning (`taskset`).
- Explicitly prevents cross-socket and cross-chiplet memory traffic.
- Deploys one isolated instance per NUMA node on multi-socket systems.
- Implemented via a lightweight Python orchestrator (`psutil` + `subprocess`).

### 3.3 Heuristic Routing Layer
- Rules-based subset selection (no ML training overhead).
- **Short contexts:** Aggressively limits active experts/modules to maximize cache residency.
- **Long contexts:** Implements chunked processing and NUMA-aware KV cache placement.
- Serves as the software proxy for validating hardware-aware routing behavior.

---

## 4. Target Models

| Phase | Model                  | Type   | Quantization     | Purpose |
|-------|------------------------|--------|------------------|---------|
| 1     | **Mixtral 8x7B** | MoE    | Q5_K_M / Q6_K    | Primary: Isolates hardware topology optimization from algorithmic sparsity. |
| 2     | **Llama 3.1 70B** | Dense  | Q4_K_M / Q5_K_M  | Secondary: Tests the viability of post-hoc routing on dense models. |

---

## 5. Hardware Requirements

**Minimum Baseline**
- 128 GB DDR5 RAM
- 16+ core x86 CPU
- NVMe SSD
- Linux (Ubuntu 24.04+ recommended)

**Primary Validation Target (Multi-Socket)**
- Dual-socket AMD EPYC (e.g., 9004 Genoa) or Intel Xeon Scalable 4th Gen+.
- 256+ GB RAM.
- **Topology Requirement:** The exact L3 cache partition structure (shared vs. localized per CCX/tile) must be mapped and documented, as it heavily influences heuristic routing success.

---

## 6. Software Stack

- **Engine:** `llama.cpp` (main)
- **Models:** GGUF standard
- **Systems Profiling:** `numactl`, `taskset`, `perf`, `numastat`, `mpstat`, `turbostat`
- **Control Layer:** Python 3.11+ (Wrapper, Logging, Orchestration)

---

## 7. Experimental Methodology (Summary)

*See `METHODOLOGY.md` for full execution protocols.*

**Workloads (Standardized/Seeded)**
- Short: 512 input → 256 output
- Medium: 2048 input → 512 output
- Long: 8192 input → 256 output
- Context-Switching: Variable lengths using LMSYS Chatbot Arena subsample.
- **Repetitions:** N ≥ 30 per configuration (with forced OS cache-clearing between batches).

**Configurations Compared**
1. Naive baseline (default `llama.cpp`)
2. NUMA-optimized baseline (Affinity routing only)
3. Full CMLIS (Memory Control + Heuristic Routing)

---

## 8. Success Metrics

**Primary Success (Mixtral 8x7B on multi-socket)**
- **Throughput:** Statistically significant (≥ 25%) tokens/sec improvement over naive baseline.
- **Absolute Floor:** ≥ 3.5 tokens/sec on medium context.
- **Locality:** Remote NUMA access < 10% of total memory traffic (`numastat`).
- **Stability:** No OS-level swapping, < 5% variance across runs.
- **Strict Coherence:** Perplexity (PPL) degradation on WikiText-2 remains ≤ 0.1 relative to baseline.

**Stretch Goals:** - ≥ 40% uplift on long-context workloads via KV cache pinning.
- Viable heuristic routing on Llama 70B (Perplexity degradation ≤ 0.15).

---

## 9. Non-Goals (Out of Scope)

- Training, fine-tuning, or altering model weights.
- Learned/neural routing functions.
- Modifying standard transformer architectures.
- GPU acceleration or distributed network inference.
- Production serving features (continuous batching, speculative decoding).

---

## 10. Critical Systems Risks Acknowledged

- **Random Access Penalty:** Unpredictable cache misses negating the gains of subset routing.
- **KV Cache Bottleneck:** Memory footprint dominance during long-context generation.
- **Dequantization Overhead:** CPU cycles spent decompressing GGUF weights masking true memory bandwidth limits.
- **Post-Hoc Sparsity:** The inherent difficulty of routing dense models (Phase 2) without accuracy collapse.

---

## 11. Expected Outcomes

If successful, this MVP will empirically demonstrate that:
1. Memory topology awareness combined with lightweight software routing delivers practical, scalable gains on CPU systems.
2. 70B-class models can achieve viable interactive speeds on properly configured CPU hardware.
3. The CMLIS memory-centric framework possesses strong engineering merit, paving the way for native memory-addressed model architectures.

---

**Next Steps**
1. Document L3 cache/NUMA topology of the target hardware.
2. Run naive + NUMA baselines on available hardware to establish statistical control.
3. Implement Python Memory Control orchestrator.
4. Execute full experimental suite according to `METHODOLOGY.md`.