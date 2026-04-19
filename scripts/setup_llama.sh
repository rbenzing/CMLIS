#!/usr/bin/env bash
# setup_llama.sh — Build llama.cpp for NUMA-aware CPU inference (CMLIS project).
#
# Usage:
#   bash setup_llama.sh [--help]
#
# Configurable env vars (all optional):
#   LLAMA_DIR    Where to clone/build llama.cpp  (default: ~/llama.cpp)
#   MODELS_DIR   Where to store GGUF model files (default: ~/models)
#   SKIP_BUILD   Set to 1 to skip clone+build and go straight to verification
#
# After a successful build this script:
#   - Sets LLAMA_CPP_BIN and appends the export to ~/.bashrc
#   - Prints model download instructions
#   - Verifies the CMLIS PoC by running `cmlis run --simulate`
#
# Run chmod +x scripts/setup_llama.sh on Linux to make this executable.

set -euo pipefail

# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--help" ]]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# ---------------------------------------------------------------------------
# Configuration (env vars with defaults)
# ---------------------------------------------------------------------------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
SKIP_BUILD="${SKIP_BUILD:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# poc/ lives one level above scripts/
POC_DIR="$(cd "$SCRIPT_DIR/../poc" && pwd)"

# ---------------------------------------------------------------------------
# Helper: print a step banner
# ---------------------------------------------------------------------------
step() { echo; echo "==> $*"; }

# ---------------------------------------------------------------------------
# Step 1: Check required tools
# ---------------------------------------------------------------------------
step "Checking required tools"

missing=()
for tool in git cmake make python3; do
    if ! command -v "$tool" &>/dev/null; then
        missing+=("$tool")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: the following required tools are missing: ${missing[*]}"
    echo
    echo "Install on Ubuntu/Debian:"
    echo "  sudo apt-get install -y git cmake make python3 build-essential"
    echo
    echo "Install on RHEL/Rocky 8+:"
    echo "  sudo dnf install -y git cmake make python3 gcc gcc-c++"
    exit 1
fi

echo "All required tools found."

# ---------------------------------------------------------------------------
# Step 2: Clone or update llama.cpp
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "1" ]]; then
    step "SKIP_BUILD=1 — skipping clone and build"
else
    step "Cloning llama.cpp into $LLAMA_DIR"

    if [[ -d "$LLAMA_DIR/.git" ]]; then
        echo "Repository already exists — pulling latest changes."
        git -C "$LLAMA_DIR" pull --ff-only
    else
        git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    # -----------------------------------------------------------------------
    # Step 3: Build llama.cpp (CPU-only, AVX-512, AMX)
    # -----------------------------------------------------------------------
    step "Building llama.cpp (CPU-only, AVX-512, AMX if available)"

    BUILD_DIR="$LLAMA_DIR/build"
    mkdir -p "$BUILD_DIR"

    # Detect AMX support from CPUID flags
    AMX_FLAGS=""
    if grep -q "amx_int8" /proc/cpuinfo 2>/dev/null; then
        echo "AMX detected — enabling GGML_AMX and GGML_AMX_INT8."
        AMX_FLAGS="-DGGML_AMX=ON -DGGML_AMX_INT8=ON"
    else
        echo "AMX not detected — skipping AMX flags."
    fi

    # CMake configure
    # Real llama.cpp CMake variable names (verified against the repo):
    #   GGML_AVX512        — enables AVX-512F
    #   GGML_AVX512_BF16   — enables AVX-512 BF16 instructions (Sapphire Rapids+)
    #   GGML_AVX512_VBMI   — enables AVX-512 VBMI
    #   GGML_AMX           — enables AMX tile acceleration
    #   GGML_AMX_INT8      — enables AMX INT8 kernels
    # GGML_CUDA and GGML_METAL default to OFF; explicit OFF is belt-and-braces.
    cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
        -DGGML_AVX512=ON \
        -DGGML_AVX512_BF16=ON \
        -DGGML_AVX512_VBMI=ON \
        $AMX_FLAGS

    # Build
    cmake --build "$BUILD_DIR" --config Release -j "$(nproc)"

    # -----------------------------------------------------------------------
    # Step 4: Verify the binary was produced
    # -----------------------------------------------------------------------
    step "Verifying llama-cli binary"

    LLAMA_BIN="$BUILD_DIR/bin/llama-cli"
    if [[ ! -x "$LLAMA_BIN" ]]; then
        echo "ERROR: expected binary not found at $LLAMA_BIN"
        echo "Check CMake output above for build errors."
        exit 1
    fi

    echo "Binary found: $LLAMA_BIN"

    # -----------------------------------------------------------------------
    # Step 5: Export LLAMA_CPP_BIN
    # -----------------------------------------------------------------------
    step "Exporting LLAMA_CPP_BIN"

    export LLAMA_CPP_BIN="$LLAMA_BIN"
    echo "export LLAMA_CPP_BIN=\"$LLAMA_BIN\"" >> "$HOME/.bashrc"
    echo "LLAMA_CPP_BIN exported and appended to ~/.bashrc"
    echo "Run: source ~/.bashrc  (or start a new shell) for the change to take effect."
fi  # end SKIP_BUILD

# ---------------------------------------------------------------------------
# Step 6: Model download instructions
# ---------------------------------------------------------------------------
step "Model download instructions"

mkdir -p "$MODELS_DIR"

cat <<'EOF'

Models should be stored in: ~/models/   (or the MODELS_DIR you configured)

--- Phase 1 (primary) — Mixtral-8x7B-Q5_K_M (~29 GB) ---

Option A — huggingface-cli (recommended):
  pip install huggingface_hub
  huggingface-cli download \
    bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF \
    --include "Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf" \
    --local-dir ~/models/

Option B — direct wget:
  wget -c -P ~/models/ \
    https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1-Q5_K_M.gguf

Disk space required: ~29 GB

--- Phase 2 — Meta-Llama-3.1-70B-Q4_K_M (~40 GB) ---

Option A — huggingface-cli:
  huggingface-cli download \
    bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
    --include "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf" \
    --local-dir ~/models/

Option B — direct wget:
  wget -c -P ~/models/ \
    https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf

Disk space required: ~40 GB

After downloading, pass the model path to cmlis:
  cmlis run --model ~/models/<filename>.gguf --input-tokens 2048 --output-tokens 128

EOF

# ---------------------------------------------------------------------------
# Step 7: Verify CMLIS PoC (simulation mode)
# ---------------------------------------------------------------------------
step "Verifying CMLIS PoC (cmlis run --simulate)"

if [[ ! -d "$POC_DIR" ]]; then
    echo "ERROR: poc/ directory not found at $POC_DIR"
    echo "Run this script from the project root or adjust SCRIPT_DIR."
    exit 1
fi

# Ensure cmlis is importable; install in editable mode if needed.
if ! python3 -c "import cmlis" &>/dev/null 2>&1; then
    echo "cmlis package not importable — running: pip install -e $POC_DIR"
    pip install -e "$POC_DIR" --quiet
fi

cd "$POC_DIR"
python3 -m cmlis run --simulate --output-tokens 32
STATUS=$?

if [[ $STATUS -eq 0 ]]; then
    echo
    echo "CMLIS simulation smoke test PASSED."
else
    echo
    echo "ERROR: cmlis run --simulate exited with code $STATUS"
    exit "$STATUS"
fi

step "Setup complete"
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. Download a model (see instructions above)"
echo "  3. cmlis run --model ~/models/<file>.gguf --input-tokens 2048 --output-tokens 128"
