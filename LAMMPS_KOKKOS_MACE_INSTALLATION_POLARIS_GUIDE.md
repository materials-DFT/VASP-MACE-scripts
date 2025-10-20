# LAMMPS + KOKKOS (CUDA) + MACE Installation Guide

This guide provides automated installation of LAMMPS with GPU acceleration (KOKKOS/CUDA) and MACE machine learning potential.

## Quick Installation

### Option 1: Automated Script (Recommended)

```bash
# Download and run the installation script
./install_lammps_kokkos_mace.sh
```

### Option 2: Manual Installation

Follow the step-by-step process below if you prefer manual control.

## Prerequisites

- **Slurm-managed HPC cluster** with GPU nodes
- **Miniconda** installed (`$HOME/miniconda3`)
- **CUDA 12.2** available on GPU nodes
- **Modules**: `mpi/openmpi-x86_64`, `gcc-toolset/12`

## Manual Installation Steps

### 1. Get GPU Node Access

```bash
# Request interactive GPU session
salloc --partition=gpucluster --gres=gpu:1
srun --pty bash
```

### 2. Load Required Modules

```bash
module purge
module load mpi/openmpi-x86_64
module load gcc-toolset/12
```

### 3. Setup CUDA Environment

```bash
export CUDA_HOME=/usr/local/cuda-12.2.old
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
```

### 4. Setup Conda Environment

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n mace python=3.10 -y
conda activate mace

# Install packages
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mace-torch
conda install -y mkl mkl-include intel-openmp gcc_linux-64 gxx_linux-64
```

### 5. Download LibTorch

```bash
cd ~/src
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.2.0%2Bcu121.zip
unzip libtorch-shared-with-deps-2.2.0+cu121.zip
mv libtorch libtorch-gpu
rm libtorch-shared-with-deps-2.2.0+cu121.zip
```

### 6. Download LAMMPS

```bash
cd ~/src
git clone --branch=mace --depth=1 https://github.com/ACEsuit/lammps
```

### 7. Build LAMMPS

```bash
cd ~/src/lammps
mkdir build-gpu && cd build-gpu

# CMake configuration
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$PWD/install \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_AMPERE100=ON \
    -D CMAKE_PREFIX_PATH="$HOME/src/libtorch-gpu;$CONDA_PREFIX" \
    -D PKG_ML-MACE=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
    -D CUDAToolkit_ROOT=$CUDA_HOME \
    -D MKL_ROOT="$CONDA_PREFIX" \
    -D MKL_INCLUDE_DIR="$CONDA_PREFIX/include" \
    -D CMAKE_INSTALL_RPATH="$PWD/install/lib64;$HOME/src/libtorch-gpu/lib" \
    -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -D PKG_EXTRA-DUMP=yes \
    ../cmake

# Build and install
make -j $(nproc)
make install
```

### 8. Setup MACE Model

```bash
cd ~/src/lammps
mkdir -p mace_models && cd mace_models

# Download model
curl -L -o mace_mp0_small.model https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model

# Convert to TorchScript
python -m mace.cli.create_lammps_model mace_mp0_small.model
```

## Usage

### Test Installation

```bash
cd ~/src/lammps/build-gpu

# Set environment
export LD_LIBRARY_PATH="$HOME/src/libtorch-gpu/lib:$CONDA_PREFIX/lib:$PWD/install/lib64:$LD_LIBRARY_PATH"

# Run test
srun --partition=gpucluster --gres=gpu:1 $PWD/install/bin/lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.mace_gpu
```

### Submit Batch Job

```bash
cd ~/src/lammps/build-gpu
sbatch mace_gpu_test.sh
```

### Interactive Session

```bash
salloc --partition=gpucluster --gres=gpu:1
srun --pty bash
```

## Key Features

- ✅ **GPU Acceleration**: KOKKOS with CUDA support
- ✅ **MACE ML Potential**: Machine learning interatomic potential
- ✅ **EXTRA-DUMP Package**: Extended XYZ file support
- ✅ **NVIDIA A100 Compatible**: Optimized for Ampere architecture
- ✅ **Automated Testing**: Built-in test files and scripts

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure you're on a GPU node with `nvidia-smi`
2. **Build fails**: Check that all modules are loaded and conda environment is active
3. **Runtime errors**: Verify `LD_LIBRARY_PATH` includes all required libraries

### Verification

```bash
# Check packages
$PWD/install/bin/lmp -h | grep -E 'KOKKOS|MACE|EXTRA-DUMP'

# Expected output:
# -kokkos on/off ...          : turn KOKKOS mode on or off (-k)
# KOKKOS package API: CUDA Serial
# EXTRA-DUMP KOKKOS ML-MACE
```

## Files Created

- **LAMMPS executable**: `~/src/lammps/build-gpu/install/bin/lmp`
- **Test input**: `~/src/lammps/build-gpu/in.mace_gpu`
- **Test data**: `~/src/lammps/build-gpu/data.carbon`
- **Batch script**: `~/src/lammps/build-gpu/mace_gpu_test.sh`
- **MACE model**: `~/src/lammps/mace_models/mace_mp0_small.model-lammps.pt`

## Performance

Expected performance on NVIDIA A100:
- **GPU memory usage**: ~4-8 MB for small systems
- **Acceleration**: 10-100x speedup over CPU for large systems
- **CUDA device type**: `torch::kCUDA` (verified)

---

For questions or issues, check the LAMMPS documentation or MACE repository.
