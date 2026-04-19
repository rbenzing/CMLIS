# Single-Socket vs Dual/Multi-Socket Systems for Memory-Aware CPU LLM Inference

**CMLIS-MVP Context**: How NUMA topology affects 70B-class model performance and which hardware configuration best validates the memory-control hypothesis.

---

## Executive Summary

**Single-socket** systems are **easier to optimize** and deliver more predictable results, making them ideal for initial MVP validation.

**Dual/multi-socket** systems offer **higher performance ceiling** but introduce significant NUMA complexity — exactly the challenge your Memory Control Layer is designed to solve. They are the best platform to **prove** your core hypothesis.

---

## Detailed Comparison

| Category                      | Single Socket                                      | Dual / Multi-Socket                                      | Winner for CMLIS-MVP |
|-------------------------------|----------------------------------------------------|----------------------------------------------------------|----------------------|
| **Memory Bandwidth**          | High (all channels local)                          | Up to ~2× aggregate (when properly balanced)             | Dual (potential)    |
| **NUMA Complexity**           | None or minimal                                    | High (cross-socket penalties via UPI/Infinity Fabric)    | Single              |
| **Optimization Difficulty**   | Easy                                               | Hard — requires strong binding & scheduling              | Single              |
| **70B Model Fit (128–256 GB)**| Usually fits comfortably                           | Excellent fit + large KV cache headroom                  | Dual                |
| **Performance Ceiling**       | Good (3–8 t/s realistic)                           | Higher (6–15+ t/s possible with optimization)            | Dual                |
| **Cross-Node Traffic Risk**   | None                                               | High if not optimized                                    | Single              |
| **Reproducibility**           | Excellent                                          | Good only with strict controls                           | Single              |
| **Cost / Availability**       | Lower (Threadripper, single EPYC)                  | Higher (dual EPYC, dual Xeon)                            | Single              |
| **Hypothesis Testing Power**  | Solid baseline                                     | **Excellent** — clearly shows value of your layer        | **Dual**            |

---

## Pros & Cons

### Single-Socket Systems

**Pros**
- Simpler configuration — fewer variables to control
- No cross-socket memory penalties
- Easier to achieve high memory locality
- More predictable and reproducible results
- Better for consumer/prosumer hardware validation (aligns with original 128 GB target)
- Faster iteration during early MVP stages
- Lower risk of "naive run being worse than baseline"

**Cons**
- Lower total memory bandwidth and core count
- Limited headroom for very long contexts or high batch sizes
- Hits scalability ceiling earlier
- Less impressive maximum performance numbers

---

### Dual / Multi-Socket Systems

**Pros**
- Significantly higher aggregate memory bandwidth
- More total cores and RAM capacity
- Better for demonstrating real-world scaling of your Memory Control Layer
- Strong validation of your hypothesis (memory locality matters more than raw cores)
- Excellent for showing "before vs after" gains with `numactl` + affinity
- More representative of high-end enterprise CPU deployments
- Can comfortably run 70B models with 8k–32k context

**Cons**
- High risk of performance regression without proper binding
- Cross-socket communication is expensive (often 2–4× slower than local access)
- More complex tuning (NUMA node balancing, SNC modes, etc.)
- Results can be misleading if not carefully documented
- Higher cost and power consumption
- Harder to reproduce consistently

---

## Recommendation for CMLIS-MVP

**Best Strategy: Use Both**

1. **Phase 1 (Initial Validation)** — Single-socket system
   - Clean, convincing "baseline vs optimized" numbers
   - Easier to hit your 3–5 tokens/sec target
   - Good for documentation and sharing

2. **Phase 2 (Strong Proof)** — Dual-socket system
   - Demonstrate that your Memory Control Layer scales and handles real NUMA complexity
   - Show large performance deltas (e.g. 2.5–4× uplift)
   - Makes the project much more compelling

**Ideal Test Hardware**
- Single: AMD Threadripper 7960X/7970X or single-socket EPYC
- Dual: Dual EPYC 9005 series or Intel Xeon 6980P (Granite Rapids) as referenced in LMSYS blog

---

## Practical Guidance

**For Dual-Socket Success:**
- Run **one llama.cpp instance per socket** with strict binding
- Use: `numactl --cpunodebind=0 --membind=0`
- Disable automatic NUMA balancing: `echo 0 > /proc/sys/kernel/numa_balancing`
- Monitor with `numastat -m` and `perf stat`
- Consider socket-local model loading + memory pinning

**Success Metric Suggestion**
> "Achieve ≥ 2.5× throughput improvement over naive baseline on dual-socket while keeping remote memory access < 8%"