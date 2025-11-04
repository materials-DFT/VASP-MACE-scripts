#!/usr/bin/env bash
# Palmetto HPC: Build LAMMPS with ML-IAP (MACE), Python, Kokkos CUDA, MKL
set -euo pipefail
IFS=$'\n\t'

### ===== User-tunable locations =====
PREFIX="${HOME}/apps/lammps-mace"     # install prefix
SRC_DIR="${HOME}/src"                 # sources root
PY_ENV="${HOME}/venvs/lammps-mace"    # python venv
LAMMPS_BRANCH="stable"                # or "develop"

### ===== Modules (Palmetto Core) =====
module purge
module load gcc/12.3.0
module load openmpi/4.1.6
module load cuda/12.3.0
module load intel-oneapi-mkl/2025.1.0
export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"

# Optional: avoid polluted include paths that break CUDA/Kokkos config
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH || true

### ===== Must be on a GPU node =====
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: Not on a GPU node. Use: salloc -p skygpu -t 2:00:00 -c 8 --mem=32G --gres=gpu:1" >&2
  exit 1
fi
echo "=== GPU info ==="
nvidia-smi || true

### ===== Python venv + CUDA-enabled PyTorch (Python 3.13 -> cu124) =====
if [ ! -d "${PY_ENV}" ]; then
  python3 -m venv "${PY_ENV}"
fi
# shellcheck disable=SC1090
source "${PY_ENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Use the cu124 index; pass each requirement as its own arg.
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124

# MACE + essentials
pip install "mace-torch>=0.3.5" numpy scipy pyyaml

# Quick CUDA check (fail if GPU not visible to torch)
python - <<'PY'
import torch, sys
print(f"[Torch] {torch.__version__}  CUDA? {torch.cuda.is_available()}  built_for={getattr(torch.version,'cuda',None)}")
sys.exit(0 if torch.cuda.is_available() else 1)
PY

### ===== Get LAMMPS source =====
mkdir -p "${SRC_DIR}"
cd "${SRC_DIR}"
if [ ! -d lammps ]; then
  git clone --depth=1 --branch="${LAMMPS_BRANCH}" https://github.com/lammps/lammps.git
fi
cd lammps
rm -rf build && mkdir build && cd build

### ===== Detect GPU and pick Kokkos/CUDA arch =====
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
KOKKOS_ARCH_FLAG="-D Kokkos_ARCH_TURING75=on" # default: RTX6000 (Turing 7.5)
CUDA_ARCH=75

if echo "$GPU_NAME" | grep -qi "1080"; then
  KOKKOS_ARCH_FLAG="-D Kokkos_ARCH_PASCAL61=on"
  CUDA_ARCH=61
elif echo "$GPU_NAME" | grep -qi "P100"; then
  KOKKOS_ARCH_FLAG="-D Kokkos_ARCH_PASCAL60=on"
  CUDA_ARCH=60
fi

echo "Detected GPU: ${GPU_NAME}"
echo "Using ${KOKKOS_ARCH_FLAG}, CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"

### ===== Toolchain sanity =====
command -v mpicc  >/dev/null 2>&1 || { echo "ERROR: mpicc not found"; exit 1; }
command -v mpicxx >/dev/null 2>&1 || { echo "ERROR: mpicxx not found"; exit 1; }
command -v nvcc   >/dev/null 2>&1 || { echo "ERROR: nvcc not found (check cuda/12.3.0)"; exit 1; }
[ -n "${CUDA_HOME:-}" ] || { echo "ERROR: CUDA_HOME not set"; exit 1; }
[ -n "${MKLROOT:-}" ]   || { echo "ERROR: MKLROOT not set"; exit 1; }

### ===== Configure with Kokkos nvcc_wrapper (fixes <cmath> issue) =====
NVCC_WRAPPER="$(realpath ../lib/kokkos/bin/nvcc_wrapper)"

cmake ../cmake \
  -D CMAKE_INSTALL_PREFIX="${PREFIX}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_SHARED_LIBS=on \
  -D CMAKE_C_COMPILER=mpicc \
  -D CMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -D CMAKE_CUDA_COMPILER="$(which nvcc)" \
  -D CMAKE_CXX_STANDARD=17 \
  -D PKG_PYTHON=on \
  -D Python_EXECUTABLE="$(which python)" \
  -D PKG_ML-IAP=on \
  -D PKG_KOKKOS=on \
  -D Kokkos_ENABLE_OPENMP=on \
  -D Kokkos_ENABLE_CUDA=on \
  -D Kokkos_ENABLE_CUDA_UVM=on \
  ${KOKKOS_ARCH_FLAG} \
  -D CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -D FFT=MKL \
  -D MKL_ROOT="${MKLROOT}" \
  -D CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"

### ===== Build & install =====
make -j"$(nproc)"
make install

### ===== Done =====
cat <<EOF

========================================================
LAMMPS (ML-IAP + PYTHON + KOKKOS CUDA + MKL) installed to:
  ${PREFIX}

Add to your PATH & LD_LIBRARY_PATH for this shell:
  export PATH="${PREFIX}/bin:\$PATH"
  export LD_LIBRARY_PATH="${PREFIX}/lib:\$LD_LIBRARY_PATH"

Activate your Python env before running:
  source "${PY_ENV}/bin/activate"

Minimal GPU test (needs ./mace.model in the run dir):
  cat > in.mliap.test <<'EOT'
  units           metal
  atom_style      atomic
  boundary        p p p
  lattice         fcc 4.05
  region          box block 0 2 0 2 0 2
  create_box      1 box
  create_atoms    1 box
  mass            1 26.9815
  pair_style      mliap model \${PWD}/mace.model
  pair_coeff      * *
  neighbor        1.0 bin
  neigh_modify    delay 0 every 1 check yes
  timestep        0.001
  thermo          10
  min_style       cg
  minimize        1e-8 1e-8 1000 10000
  EOT

  srun -p skygpu --gres=gpu:1 -c8 --mem=16G -t 00:10:00 \\
       lmp -in in.mliap.test

Detected GPU: ${GPU_NAME}
Kokkos flag : ${KOKKOS_ARCH_FLAG}
CUDA arch   : ${CUDA_ARCH}
Torch check : printed above
========================================================
EOF
