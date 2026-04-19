# CMLIS-MVP: Experimental Methodology

**Version:** 1.3 (Strictly Quantified & Peer-Reviewed)  
**Status:** Ready for Execution

---

## 1. Objectives

Test whether heuristic routing combined with NUMA-aware memory control delivers statistically significant improvements in tokens/sec and memory efficiency. The experiment must clearly quantify the impact of known critical systems risks: random access penalties, KV cache bottlenecks, NUMA fragility, and dequantization overhead.

---

## 2. Target Models

- **Phase 1 (Primary):** Mixtral 8x7B (Q5_K_M / Q6_K) – *Isolates hardware topology optimization from algorithmic sparsity.*
- **Phase 2 (Secondary):** Llama 3.1 70B (Q4_K_M / Q5_K_M) – *Tests the viability of post-hoc routing on dense models.*

---

## 3. Hardware Platforms

*Hardware architecture must be explicitly documented to account for cross-chiplet and mesh latency variations.*

- **Single-socket:** Baseline control.
- **Dual/multi-socket (Primary Validation Target):** e.g., AMD EPYC 9004 series (Genoa) or Intel Xeon Scalable 4th Gen (Sapphire Rapids) with ≥ 128 GB RAM. 
- **Topology Requirement:** The exact L3 cache partition structure (shared vs. localized per CCX/tile) must be mapped and documented prior to execution.

---

## 4. Workloads

To ensure the "Mixed Realistic" workload is strictly reproducible and ecologically valid, it utilizes a standardized, seeded dataset. This dataset must explicitly force context-switching (e.g., alternating between coding, mathematics, and creative writing) to rigorously test the routing overhead $R(x_t)$.

| Type            | Input Tokens | Output Tokens | Repetitions (N) | Dataset Source / Composition |
|-----------------|--------------|---------------|-----------------|------------------------------|
| Short Context   | 512          | 256           | 50              | Synthetic / Fixed Seed       |
| Medium Context  | 2048         | 512           | 50              | Synthetic / Fixed Seed       |
| Long Context    | 8192         | 256           | 30              | Synthetic / Fixed Seed       |
| Mixed Realistic | Variable     | Variable      | 50              | LMSYS Chatbot Arena (Context-Switching Subsample) |

*(Note: Repetitions (N) have been set to ≥ 30 to account for micro-architectural noise and satisfy the Central Limit Theorem for significance testing).*

---

## 5. Configurations to Test

1. **Naive Baseline:** Default `llama.cpp`
2. **NUMA-Optimized Baseline:** Binding + affinity only (`numactl`, `taskset`)
3. **Full CMLIS:** NUMA Control + Heuristic Routing

---

## 6. Success Metrics (Conditional)

**Primary Success Criteria (Mixtral on dual-socket, medium context)**
- **Throughput:** A statistically significant improvement ($p < 0.01$, using a two-sample t-test) of ≥ 20% in tokens/sec over the naive baseline.
- **Absolute Floor:** ≥ 3.5 tokens/sec absolute performance.
- **Locality:** Remote NUMA traffic < 10% of total memory accesses.
- **Stability:** No OS-level swapping; < 5% variance across runs (with 95% confidence intervals reported).
- **Strict Coherence Metric:** Perplexity (PPL) degradation on WikiText-2 must remain ≤ 0.1 relative to the naive baseline.

**Important Qualification:** Success requires that random-access and KV cache penalties do not fully negate the gains. If top risks dominate, the experiment must clearly document this null outcome.

**Stretch Goals**
- Meaningful, statistically significant gains on long-context workloads.
- Viable heuristic performance on Llama 70B (Perplexity degradation ≤ 0.15).

---

## 7. Measurement & Risk Mitigation Protocol

- **Mandatory Memory Monitoring:** `numastat -m`, `perf stat` (cache misses, bandwidth), `turbostat`.
- **Compute vs. Bandwidth Disambiguation:** `mpstat` must be run concurrently to track CPU utilization. Because quantized models (GGUF) require CPU cycles to dequantize weights before computation, dequantization compute bottlenecks must be isolated from true memory bandwidth bottlenecks.
- **Fragmentation & Cache Mitigation:** To prevent OS-level memory fragmentation and residual page caching from skewing cache miss metrics during multi-batch runs, the host system must execute `echo 3 > /proc/sys/vm/drop_caches` and `echo 1 > /proc/sys/vm/compact_memory` between every test batch.
- **Validation:** Explicit validation of NUMA binding effectiveness must occur on every run.
- **Early Testing:** Early focused testing on random-access vs. sequential behavior.
- **KV Cache Routing:** KV cache NUMA placement experiments (manual pinning where possible).

---

## 8. Timeline (4 Weeks)

- **Week 1:** Baselines + risk validation (random access profiling & KV cache bottleneck mapping).
- **Week 2:** Implement & tune heuristic routing + Memory Control wrapper.
- **Week 3:** Full test suite execution on Mixtral.
- **Week 4:** Llama 70B tests + statistical significance analysis.