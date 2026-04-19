# Project Brief: CPU-Native Modular Language Intelligence System (CMLIS)

## 1. Overview
CMLIS is a CPU-first language model inference architecture designed to run large-scale language intelligence workloads (including 70B-class models) efficiently on commodity multi-socket CPU + RAM hardware, completely eliminating the dependency on GPUs.

The system reframes LLM inference not as dense tensor computation, but as a **memory-conditioned process** driven by hardware topology. 

## 2. Problem Statement
Modern LLMs are optimized for GPU-based dense execution, creating severe inefficiencies on CPU architectures due to:
- High memory bandwidth pressure (the primary bottleneck).
- Cross-socket interconnect latency (NUMA penalties).
- Hardware prefetcher collapse due to unpredictable random-access patterns.
- Massive KV cache memory footprints at longer contexts.

## 3. Objective
Design and validate an inference layer that:
- Enables interactive CPU inference of large language models (e.g., Mixtral 8x7B, Llama 3.1 70B).
- Operates within 128GB+ RAM multi-socket baseline systems.
- Actively minimizes cross-node memory movement per token.
- Achieves a statistically significant (≥ 25%) throughput improvement over naive dense baselines without requiring model retraining.

## 4. Core Architecture
Replace monolithic neural computation with a three-layer structured system:
1. **Base Engine:** Low-level execution and CPU SIMD operations (e.g., `llama.cpp`).
2. **OS-Level Memory Control:** Strict NUMA node binding, physical core affinity, and fragmentation management to prevent hardware bus saturation.
3. **Heuristic Routing:** Dynamic selection of parameter subsets to maximize L3 cache residency and minimize total DRAM traffic.

## 5. Expected Outcomes (MVP Phase)
- **Primary:** Achieve ≥ 3.5 tokens/sec on medium-context workloads on dual-socket x86 systems (Mixtral 8x7B baseline).
- **Secondary:** Keep remote NUMA memory traffic strictly below 10%.
- **Validation:** Prove that intelligent OS-level memory management and selective activation yield superior CPU throughput than raw algorithmic density.

## 6. Strict Constraints & Non-Goals
- **No Training:** The MVP strictly avoids model training, fine-tuning, or neural routing.
- **Hardware Profile:** Must utilize existing commodity hardware (DDR5, multi-socket x86 CPUs).
- **Conditionality:** Success is strictly dependent on overcoming the hardware penalties of random-access memory fetches and KV cache saturation.