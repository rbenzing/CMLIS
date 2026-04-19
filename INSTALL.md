# CMLIS Installation & Hardware Setup Guide

## 1. Hardware Requirements

### Minimum (single-socket baseline)
- x86-64 CPU with AVX-512, 16+ physical cores
- 128 GB DDR5 RAM (all channels populated)
- NVMe SSD (model files: 29–40 GB)
- Linux Ubuntu 22.04+ or RHEL/Rocky 8+

### Recommended (dual-socket validation target)
- Dual AMD EPYC 9004 (Genoa) or Intel Xeon 4th Gen (Sapphire Rapids)
- 256+ GB RAM across both sockets, all DIMM slots populated
- Why: cross-socket NUMA penalty is what CMLIS is designed to eliminate; dual-socket hardware is required to demonstrate the core hypothesis

## 2. Required System Packages

```bash
# Ubuntu 22.04 / 24.04
sudo apt install -y \
  numactl \
  util-linux \
  linux-tools-$(uname -r) \
  sysstat \
  cmake \
  build-essential \
  git \
  python3 \
  python3-pip \
  cpufrequtils

# RHEL / Rocky 8+
sudo dnf install -y \
  numactl \
  util-linux \
  perf \
  sysstat \
  cmake \
  gcc \
  gcc-c++ \
  git \
  python3 \
  python3-pip
```

Verify key tools:
```bash
numactl --hardware    # must show NUMA nodes
taskset --version
perf stat --version
mpstat -V
```

## 3. Linux Kernel Configuration

Run before every benchmark session (requires root):

```bash
# Disable automatic NUMA page migration (prevents kernel undoing our binding)
echo 0 > /proc/sys/kernel/numa_balancing

# Drop OS page cache for clean DRAM fetch metrics
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory

# Set CPU frequency governor to performance (prevents throttling)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu"
done
# or: sudo cpupower frequency-set -g performance

# Allow perf without root (set to 1 for user-space perf)
echo 1 > /proc/sys/kernel/perf_event_paranoid

# Disable transparent hugepages (prevents THP stalls during inference)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

Make persistent via `/etc/sysctl.d/99-cmlis.conf`:
```
kernel.numa_balancing = 0
kernel.perf_event_paranoid = 1
```

## 4. BIOS / Firmware Settings (dual-socket systems)

Configure before benchmarking. Settings vary by vendor; consult your platform's BIOS documentation.

| Setting | Recommended Value | Why |
|---------|------------------|-----|
| Sub-NUMA Clustering (SNC) / NPS mode | SNC4 / NPS4 on EPYC | Creates finer NUMA domains per CCX for tighter `numactl` binding |
| NUMA Interleaving | Disabled | Prevents hardware from mixing cross-socket traffic |
| UPI / Infinity Fabric Frequency | Maximum | Reduces cross-socket penalty when unavoidable |
| DIMM Population | All slots symmetric | Maximizes local memory bandwidth per socket |
| Hyper-Threading / SMT | Application-dependent | Disable for latency-sensitive runs; enable for throughput |

## 5. llama.cpp Build

Use the provided script (recommended):
```bash
bash scripts/setup_llama.sh
```

Or build manually:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Check for AMX support (Intel Xeon 4th Gen+)
grep -q amx_int8 /proc/cpuinfo && AMX="-DGGML_AMX=ON -DGGML_AMX_INT8=ON" || AMX=""

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_BF16=ON \
  -DGGML_AVX512_VBMI=ON \
  $AMX

cmake --build build --config Release -j$(nproc)

export LLAMA_CPP_BIN=$(pwd)/build/bin/llama-cli
echo "export LLAMA_CPP_BIN=$LLAMA_CPP_BIN" >> ~/.bashrc
```

Note: llama.cpp uses `GGML_*` CMake prefix (not `LLAMA_*`) as of 2024.

## 6. Model Acquisition

Models stored in `~/models/` by default.

### Mixtral-8x7B-Q5_K_M (Phase 1 — primary, ~29 GB)
```bash
pip install huggingface_hub
huggingface-cli download \
  bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF \
  --include "Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf" \
  --local-dir ~/models/
```

### Meta-Llama-3.1-70B-Q4_K_M (Phase 2 — dense, ~40 GB)
```bash
huggingface-cli download \
  bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
  --include "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/models/
```

Verify download:
```bash
ls -lh ~/models/*.gguf
```

## 7. CMLIS Python Installation

```bash
cd poc/
pip install -e ".[dev]"

# Smoke test
cmlis topo
cmlis bench --simulate --workloads short --configs naive,full --reps 3
```

## 8. Pre-Benchmark Validation Checklist

Run this before every real (non-simulated) bench session:

- [ ] `numactl --hardware` shows expected NUMA node count
- [ ] `cat /proc/sys/kernel/numa_balancing` returns `0`
- [ ] `free -h` shows 0 swap in use
- [ ] `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` returns `performance`
- [ ] `cat /proc/sys/kernel/perf_event_paranoid` returns `<= 1`
- [ ] `ls -lh ~/models/*.gguf` shows the model file at expected size
- [ ] `echo $LLAMA_CPP_BIN` points to a valid binary: `$LLAMA_CPP_BIN --version`
- [ ] `cmlis topo --json` shows correct `sockets` and `numa_nodes` count
- [ ] No other memory-intensive processes running: `htop`

## 9. Running the Full Benchmark

```bash
# Standard CMLIS benchmark (Mixtral Phase 1)
cmlis bench \
  --model ~/models/Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf \
  --workloads short,medium,long \
  --configs naive,numa,full \
  --reps 50 \
  --out ./reports/

# Perplexity check (coherence validation)
cmlis ppl \
  --model ~/models/Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf \
  --configs naive,full
```

### Reading the bench output

```
config   workload      n     mean   stdev    cv%     min     max  var_ok
naive    medium       50     4.21    0.18    4.3    3.89    4.62      OK
numa     medium       50     5.12    0.21    4.1    4.71    5.63      OK
full     medium       50     5.68    0.24    4.2    5.20    6.19      OK

significance (full vs naive):
  medium     uplift  34.92%  t=+18.234  p=0.0000  [PASS]
```

- **PASS** = uplift >= 25% AND p < 0.01 — primary success criterion met
- **cv%** < 5% = stable runs (SPEC §5)
- **remote_numa_fraction** in JSON report should be < 0.10
