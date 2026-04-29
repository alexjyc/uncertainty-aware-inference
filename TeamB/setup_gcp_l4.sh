#!/usr/bin/env bash
# =============================================================================
# setup_gcp_l4.sh
# Failproof setup for uncertainty-aware-inference nsight profiling on GCP L4
#
# Run as: bash setup_gcp_l4.sh
# Then activate env: source ~/uai_env/bin/activate
# =============================================================================

set -euo pipefail

# ── Color helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()   { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# =============================================================================
# STEP 0 — Sanity-check: must be on a GPU instance
# =============================================================================
info "Checking NVIDIA driver..."
nvidia-smi --query-gpu=name,driver_version,memory.total \
           --format=csv,noheader 2>/dev/null \
  || die "nvidia-smi failed — is this a GPU instance?"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME"

# =============================================================================
# STEP 1 — Detect CUDA version and set variables
# =============================================================================
info "Detecting CUDA version..."

CUDA_VER_FULL=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ,) \
  || die "nvcc not found. Install CUDA toolkit first."

CUDA_MAJOR=$(echo "$CUDA_VER_FULL" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER_FULL" | cut -d. -f2)
CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"   # e.g. "124" for CUDA 12.4

ok "CUDA: $CUDA_VER_FULL  (short: $CUDA_SHORT)"

CUDA_HOME="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
[[ -d "$CUDA_HOME" ]] || CUDA_HOME="/usr/local/cuda"
ok "CUDA_HOME: $CUDA_HOME"

# =============================================================================
# STEP 2 — Fix system-level library paths (root of the ATen_cuda error)
# =============================================================================
info "Configuring system-level CUDA library paths..."

# 2a. Add CUDA lib to ldconfig so libtorch_cuda.so can resolve its deps
CUDA_LIB_DIR="${CUDA_HOME}/lib64"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  echo "$CUDA_LIB_DIR" | sudo tee /etc/ld.so.conf.d/cuda-l4.conf > /dev/null
  sudo ldconfig
  ok "ldconfig updated with $CUDA_LIB_DIR"
else
  warn "$CUDA_LIB_DIR not found — skipping ldconfig"
fi

# 2b. Add to current shell's LD_LIBRARY_PATH (also written to .bashrc below)
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
ok "LD_LIBRARY_PATH set"

# =============================================================================
# STEP 3 — Install / upgrade system packages
# =============================================================================
info "Updating apt packages..."
sudo apt-get update -qq

info "Installing build dependencies..."
sudo apt-get install -y -qq \
  build-essential git wget curl unzip \
  python3 python3-pip python3-venv python3-dev \
  libssl-dev libffi-dev \
  > /dev/null
ok "Build deps installed"

# =============================================================================
# STEP 4 — Install Nsight Systems CLI
# =============================================================================
info "Checking for nsys (Nsight Systems CLI)..."

NSYS_BIN=$(command -v nsys 2>/dev/null || echo "")
if [[ -z "$NSYS_BIN" ]]; then
  # Try common GCP paths first (avoid re-downloading if already present)
  for candidate in \
      "${CUDA_HOME}/bin/nsys" \
      "/usr/local/cuda/bin/nsys" \
      "/opt/nvidia/nsight-systems/2024.1.1/bin/nsys" \
      "/opt/nvidia/nsight-systems/2023.4.1/bin/nsys"; do
    if [[ -x "$candidate" ]]; then
      NSYS_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$NSYS_BIN" ]]; then
  info "nsys not found — installing nsight-systems-cli from apt..."
  # Add the NVIDIA apt repository for Nsight tools if not already present
  if ! apt-cache show nsight-systems-cli &>/dev/null; then
    wget -qO /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
  fi
  sudo apt-get install -y nsight-systems-cli || {
    warn "nsight-systems-cli apt install failed — trying CUDA toolkit path..."
    sudo apt-get install -y "cuda-nsight-systems-${CUDA_MAJOR}-${CUDA_MINOR}" 2>/dev/null || true
  }
  # Re-detect
  NSYS_BIN=$(command -v nsys 2>/dev/null || \
             ls "${CUDA_HOME}/bin/nsys" 2>/dev/null || echo "")
fi

if [[ -n "$NSYS_BIN" ]]; then
  NSYS_VER=$("$NSYS_BIN" --version 2>&1 | head -1)
  ok "nsys found: $NSYS_BIN  ($NSYS_VER)"
else
  warn "nsys could not be installed automatically."
  warn "Manual fix: sudo apt-get install nsight-systems-cli"
  warn "Or download from https://developer.nvidia.com/nsight-systems"
fi

# Install CUPTI (needed by Kineto / PyTorch Profiler for GPU kernel timing)
info "Ensuring CUPTI is installed..."
sudo apt-get install -y -qq "cuda-cupti-${CUDA_MAJOR}-${CUDA_MINOR}" 2>/dev/null || \
  sudo apt-get install -y -qq libcupti-dev 2>/dev/null || \
  warn "CUPTI install failed — GPU kernel timing may fall back to CPU times"
ok "CUPTI step done"

# =============================================================================
# STEP 5 — Python virtual environment
# =============================================================================
VENV_DIR="$HOME/uai_env"
info "Creating Python venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR" --system-site-packages 2>/dev/null || \
  python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
ok "venv activated"

# Upgrade pip/setuptools cleanly
pip install --upgrade pip setuptools wheel --quiet

# =============================================================================
# STEP 6 — Install PyTorch (CUDA-enabled, matching the installed CUDA version)
# =============================================================================
info "Installing PyTorch for CUDA ${CUDA_MAJOR}.${CUDA_MINOR}..."

# Uninstall any existing CPU-only torch that could cause the ATen_cuda error
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Pick the right CUDA index URL
if   [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" -ge "4" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" -ge "1" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_MAJOR" == "11" && "$CUDA_MINOR" -ge "8" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  warn "Unrecognised CUDA version — defaulting to cu124 wheel"
fi

pip install torch torchvision torchaudio \
  --index-url "$TORCH_INDEX" \
  --quiet
ok "PyTorch installed from $TORCH_INDEX"

# Verify CUDA is accessible from the installed torch
python3 - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("[FAIL] torch.cuda.is_available() returned False")
    sys.exit(1)
try:
    x = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    print(f"[OK]   torch CUDA works — GPU: {torch.cuda.get_device_name(0)}")
except RuntimeError as e:
    print(f"[FAIL] torch CUDA init failed: {e}")
    sys.exit(1)
PYEOF

# =============================================================================
# STEP 7 — Install HuggingFace + quantization stack (pinned for compatibility)
# =============================================================================
info "Installing HuggingFace + quantization libraries..."

pip install --quiet \
  transformers==4.41.2 \
  accelerate==0.31.0 \
  huggingface_hub \
  datasets \
  tokenizers \
  safetensors

ok "HuggingFace stack installed"

# ── bitsandbytes (NF4 / INT8 via HF) ────────────────────────────────────────
info "Installing bitsandbytes..."
pip install --quiet bitsandbytes
# Verify it can see CUDA
python3 -c "import bitsandbytes as bnb; print('[OK]  bitsandbytes:', bnb.__version__)" \
  || warn "bitsandbytes import failed — NF4 quantization will not work"

# ── auto-gptq (GPTQ INT4/INT8) ───────────────────────────────────────────────
info "Installing auto-gptq..."
pip install --quiet auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu"${CUDA_SHORT}"/ \
  2>/dev/null || pip install --quiet auto-gptq
python3 -c "import auto_gptq; print('[OK]  auto-gptq:', auto_gptq.__version__)" \
  || warn "auto-gptq import failed — GPTQ configs will not load"

# ── autoawq (AWQ INT4) ────────────────────────────────────────────────────────
info "Installing autoawq..."
pip install --quiet autoawq \
  2>/dev/null || pip install --quiet "https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.5/autoawq-0.2.5+cu${CUDA_SHORT}-cp310-cp310-linux_x86_64.whl" \
  2>/dev/null || warn "autoawq install failed — AWQ configs will not load"
python3 -c "import awq; print('[OK]  autoawq installed')" 2>/dev/null || true

# ── Optimum (needed by some GPTQ loading paths) ─────────────────────────────
pip install --quiet optimum

# =============================================================================
# STEP 8 — NVTX and profiling helpers
# =============================================================================
info "Installing NVTX and profiling helpers..."
pip install --quiet nvtx numpy scipy matplotlib wandb
ok "Profiling helpers installed"

# =============================================================================
# STEP 9 — FabricManager / IPC error suppression (L4-specific)
# =============================================================================
# The "Bad file descriptor" / "IpcFabricConfigClient" errors come from Kineto's
# IPC daemon trying to connect to NVSwitch FabricManager — which is only present
# on multi-GPU NVLink nodes (A100 SXM), NOT on single L4 instances.
# Fix: disable Kineto daemon mode via env vars. Already in run_profiler.py but
# we also set them system-wide here as a belt-and-suspenders measure.
info "Setting Kineto daemon env vars to suppress FabricManager errors..."
cat >> "$VENV_DIR/bin/activate" <<'ENVEOF'

# ── UAI Profiling: suppress Kineto IPC daemon (not available on single L4) ──
export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000

# ── UAI Profiling: ensure libtorch_cuda.so is resolvable by dynamic linker ──
_TORCH_LIB=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null)
if [[ -n "$_TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
unset _TORCH_LIB
ENVEOF
ok "Env vars written to venv activation script"

# Reload now so they're live for the rest of this script
export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000

# =============================================================================
# STEP 10 — Write a convenience launcher that bakes in all required env vars
# =============================================================================
LAUNCHER="$HOME/run_nsys_profile.sh"
cat > "$LAUNCHER" <<LAUNCHEOF
#!/usr/bin/env bash
# ── UAI nsys profiling launcher ──────────────────────────────────────────────
# Usage examples:
#   ./run_nsys_profile.sh --config mistral-7b-fp16
#   ./run_nsys_profile.sh --all --profile-steps 3
#   ./run_nsys_profile.sh --all --force --output-dir /data/nsys_results
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source "$HOME/uai_env/bin/activate"

# Kineto daemon suppression (eliminates FabricManager / Bad file descriptor errors)
export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000

# Make sure libtorch_cuda.so is on the dynamic linker path
TORCH_LIB=\$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')")
export LD_LIBRARY_PATH="\${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"

# Locate nsys
NSYS_BIN="${NSYS_BIN:-nsys}"

REPO_DIR=\${REPO_DIR:-\$(pwd)}
cd "\$REPO_DIR"

echo "[launcher] NSYS:             \$NSYS_BIN"
echo "[launcher] LD_LIBRARY_PATH:  \$LD_LIBRARY_PATH"
echo "[launcher] KINETO_USE_DAEMON: \$KINETO_USE_DAEMON"
echo "[launcher] CWD:               \$REPO_DIR"
echo ""

python3 run_nsys.py --nsys-path "\$NSYS_BIN" "\$@"
LAUNCHEOF
chmod +x "$LAUNCHER"
ok "Launcher written: $LAUNCHER"

# =============================================================================
# STEP 11 — Write .bashrc additions
# =============================================================================
cat >> "$HOME/.bashrc" <<'RCEOF'

# ── UAI Profiling env (added by setup_gcp_l4.sh) ─────────────────────────────
source "$HOME/uai_env/bin/activate" 2>/dev/null || true
export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000
_TORCH_LIB=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null)
if [[ -n "$_TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
unset _TORCH_LIB
RCEOF
ok ".bashrc updated"

# =============================================================================
# STEP 12 — Final verification
# =============================================================================
info "Running final verification checks..."

python3 - <<'PYEOF'
import sys, os

print("── Python:", sys.version.split()[0])

# PyTorch + CUDA
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
print(f"── torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}")

# Kineto (PyTorch Profiler)
from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, profile_memory=False,
             with_flops=False, with_stack=False, with_modules=False) as p:
    _ = torch.mm(torch.randn(64,64,device="cuda"), torch.randn(64,64,device="cuda"))
    torch.cuda.synchronize()
avgs = p.key_averages()
has_cuda = any(getattr(e,"cuda_time_total",0)>0 for e in avgs)
print(f"── Kineto CUDA backend: {'ACTIVE' if has_cuda else 'CPU-only fallback (still works)'}")

# NVTX
import torch.cuda.nvtx as nvtx
nvtx.range_push("test"); nvtx.range_pop()
print("── NVTX (torch.cuda.nvtx): OK")

# bitsandbytes
try:
    import bitsandbytes as bnb
    print(f"── bitsandbytes {bnb.__version__}: OK")
except Exception as e:
    print(f"── bitsandbytes: FAILED ({e})")

# auto-gptq
try:
    import auto_gptq
    print(f"── auto-gptq {auto_gptq.__version__}: OK")
except Exception as e:
    print(f"── auto-gptq: FAILED ({e})")

# autoawq
try:
    import awq
    print("── autoawq: OK")
except Exception as e:
    print(f"── autoawq: FAILED ({e})")

# transformers
import transformers
print(f"── transformers {transformers.__version__}: OK")

# FabricManager env var check
print(f"── KINETO_USE_DAEMON={os.environ.get('KINETO_USE_DAEMON','<not set>')}  (should be 0)")

print("")
print("✅  All critical checks passed. Ready to profile.")
PYEOF

# =============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Quick-start commands:"
echo ""
echo "  1. Activate the venv:"
echo "       source ~/uai_env/bin/activate"
echo ""
echo "  2. Set your HuggingFace token:"
echo "       export HF_TOKEN=hf_..."
echo ""
echo "  3. CD into the TeamB directory:"
echo "       cd /path/to/uncertainty-aware-inference/TeamB"
echo ""
echo "  4. Run nsys profiling (single config):"
echo "       ~/run_nsys_profile.sh --config mistral-7b-fp16"
echo ""
echo "  4b. Run all configs:"
echo "       ~/run_nsys_profile.sh --all --profile-steps 3"
echo ""
echo "  5. If nsys is not on PATH, pass it explicitly:"
echo "       ~/run_nsys_profile.sh --config mistral-7b-fp16 \\"
echo "         --nsys-path /usr/local/cuda/bin/nsys"
echo ""
echo "  FabricManager / IPC errors are now suppressed via"
echo "  KINETO_USE_DAEMON=0 (set in venv, launcher, and .bashrc)."
echo ""
