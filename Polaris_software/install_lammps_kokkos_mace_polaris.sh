#!/bin/bash
# LAMMPS + KOKKOS (CUDA) + MACE Installation Script
# This script installs LAMMPS with GPU acceleration and MACE ML potential
# Compatible with Slurm-managed HPC clusters

set -euo pipefail

# Configuration
INSTALL_DIR="$HOME/src"
LAMMPS_DIR="$INSTALL_DIR/lammps"
BUILD_DIR="$LAMMPS_DIR/build-gpu"
CONDA_ENV="mace"
PYTHON_VERSION="3.10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if we're on a GPU node
check_gpu_node() {
    log "Checking GPU node access..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        log "GPU detected successfully"
    else
        warn "nvidia-smi not found. Make sure you're on a GPU node or have CUDA available."
    fi
}

# Load required modules
load_modules() {
    log "Loading required modules..."
    module purge || true
    module load mpi/openmpi-x86_64 || error "Failed to load MPI module"
    module load gcc-toolset/12 || error "Failed to load GCC module"
    log "Modules loaded successfully"
}

# Setup CUDA environment
setup_cuda() {
    log "Setting up CUDA environment..."
    export CUDA_HOME=/usr/local/cuda-12.2.old
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    
    if command -v nvcc &> /dev/null; then
        nvcc --version
        log "CUDA environment set up successfully"
    else
        error "nvcc not found. CUDA may not be properly installed."
    fi
}

# Setup conda environment
setup_conda() {
    log "Setting up conda environment..."
    
    # Initialize conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        error "Conda not found. Please install Miniconda first."
    fi
    
    # Create or activate environment
    if conda env list | grep -q "^$CONDA_ENV "; then
        log "Activating existing conda environment: $CONDA_ENV"
        conda activate $CONDA_ENV
    else
        log "Creating new conda environment: $CONDA_ENV"
        conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
        conda activate $CONDA_ENV
    fi
    
    # Install Python packages
    log "Installing Python packages..."
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install mace-torch
    conda install -y mkl mkl-include intel-openmp gcc_linux-64 gxx_linux-64
    
    log "Python packages installed successfully"
}

# Download LibTorch
download_libtorch() {
    log "Downloading GPU-enabled LibTorch..."
    cd $INSTALL_DIR
    
    if [ ! -d "libtorch-gpu" ]; then
        wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.2.0%2Bcu121.zip
        unzip libtorch-shared-with-deps-2.2.0+cu121.zip
        mv libtorch libtorch-gpu
        rm libtorch-shared-with-deps-2.2.0+cu121.zip
        log "LibTorch downloaded and extracted"
    else
        log "LibTorch already exists, skipping download"
    fi
}

# Download LAMMPS
download_lammps() {
    log "Downloading LAMMPS with MACE package..."
    cd $INSTALL_DIR
    
    if [ ! -d "lammps" ]; then
        # Try git first, fallback to wget if git fails
        if command -v git &> /dev/null && git clone --branch=mace --depth=1 https://github.com/ACEsuit/lammps; then
            log "LAMMPS cloned via git"
        else
            warn "Git failed, downloading via wget..."
            wget https://github.com/ACEsuit/lammps/archive/refs/heads/mace.zip
            unzip mace.zip
            mv lammps-mace lammps
            rm mace.zip
            log "LAMMPS downloaded via wget"
        fi
    else
        log "LAMMPS already exists, skipping download"
    fi
}

# Build LAMMPS
build_lammps() {
    log "Building LAMMPS with KOKKOS and MACE..."
    cd $LAMMPS_DIR
    
    # Create build directory
    mkdir -p build-gpu
    cd build-gpu
    
    # Clean previous build if exists
    rm -rf CMakeCache.txt CMakeFiles
    
    # Set up environment
    export NVCC_WRAPPER_DEFAULT_COMPILER=$(which mpicxx 2>/dev/null || which g++)
    
    # CMake configuration
    log "Configuring CMake..."
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
        -D CMAKE_PREFIX_PATH="$INSTALL_DIR/libtorch-gpu;$CONDA_PREFIX" \
        -D PKG_ML-MACE=ON \
        -D CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
        -D CUDAToolkit_ROOT=$CUDA_HOME \
        -D MKL_ROOT="$CONDA_PREFIX" \
        -D MKL_INCLUDE_DIR="$CONDA_PREFIX/include" \
        -D CMAKE_INSTALL_RPATH="$PWD/install/lib64;$INSTALL_DIR/libtorch-gpu/lib" \
        -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -D PKG_EXTRA-DUMP=yes \
        ../cmake
    
    # Build
    log "Building LAMMPS (this may take a while)..."
    make -j $(nproc)
    
    # Install
    log "Installing LAMMPS..."
    make install
    
    log "LAMMPS build completed successfully!"
}

# Download and prepare MACE model
setup_mace_model() {
    log "Setting up MACE model..."
    cd $LAMMPS_DIR
    
    mkdir -p mace_models
    cd mace_models
    
    if [ ! -f "mace_mp0_small.model" ]; then
        log "Downloading MACE model..."
        curl -L -o mace_mp0_small.model https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model
    fi
    
    if [ ! -f "mace_mp0_small.model-lammps.pt" ]; then
        log "Converting MACE model to TorchScript..."
        python -m mace.cli.create_lammps_model mace_mp0_small.model
    fi
    
    log "MACE model ready: mace_mp0_small.model-lammps.pt"
}

# Create test files
create_test_files() {
    log "Creating test files..."
    cd $BUILD_DIR
    
    # Create test data file
    cat > data.carbon << 'DATA_EOF'
LAMMPS data file

4 atoms
1 atom types

0.0 3.567 xlo xhi
0.0 3.567 ylo yhi  
0.0 3.567 zlo zhi

Masses

1 12.011

Atoms

1 1 0.0 0.0 0.0
2 1 1.7835 1.7835 0.0
3 1 1.7835 0.0 1.7835
4 1 0.0 1.7835 1.7835
DATA_EOF

    # Create LAMMPS input file
    cat > in.mace_gpu << 'INPUT_EOF'
# LAMMPS input file for MACE potential with GPU acceleration

units metal
atom_style atomic
atom_modify map yes

# Data file
read_data data.carbon

# Pair style with MACE potential
pair_style mace no_domain_decomposition
pair_coeff * * /Users/924322630/src/lammps/mace_models/mace_mp0_small.model-lammps.pt C

# Just compute forces once
thermo 1
run 0

# Print some information
print "System info:"
print "Number of atoms: $(atoms)"
print "Potential energy: $(pe)"
print "Temperature: $(temp)"
INPUT_EOF

    # Create sbatch script
    cat > mace_gpu_test.sh << 'SBATCH_EOF'
#!/bin/bash
#SBATCH -p gpucluster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J lmp-mace-gpu
#SBATCH --time=00:30:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
set -euo pipefail

# ---- user settings ----
export INPUT_FILE="$HOME/src/lammps/build-gpu/in.mace_gpu"
export DATA_FILE="$HOME/src/lammps/build-gpu/data.carbon"
export MODEL_FILE="$HOME/src/lammps/mace_models/mace_mp0_small.model-lammps.pt"
export LAMMPS_PREFIX="$HOME/src/lammps/build-gpu/install"
export LIBTORCH_DIR="$HOME/src/libtorch-gpu"

# -----------------------
module purge
module load mpi/openmpi-x86_64
module load gcc-toolset/12

#Silence OpenMPI warnings:
export OMPI_MCA_btl="^openib"

# CUDA environment
export CUDA_HOME=/usr/local/cuda-12.2.old
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mace

# Runtime libs: MKL + LAMMPS + LibTorch
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LAMMPS_PREFIX/lib64:$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L | tr -d '\n' || true)"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Using INPUT: $INPUT_FILE"
echo "Using MODEL: $MODEL_FILE"

# Run with GPU acceleration
mpirun -np 1 "$LAMMPS_PREFIX/bin/lmp" -k on g 1 -sf kk -pk kokkos neigh half newton on -var model "$MODEL_FILE" -var datafile "$DATA_FILE" -in "$INPUT_FILE"
SBATCH_EOF

    chmod +x mace_gpu_test.sh
    log "Test files created successfully"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    cd $BUILD_DIR
    
    # Set up environment for verification
    export LD_LIBRARY_PATH="$INSTALL_DIR/libtorch-gpu/lib:$CONDA_PREFIX/lib:$PWD/install/lib64:$LD_LIBRARY_PATH"
    
    # Check LAMMPS help
    log "Checking LAMMPS packages..."
    $PWD/install/bin/lmp -h | grep -E 'KOKKOS|MACE|EXTRA-DUMP' || warn "Some packages may not be properly installed"
    
    log "Installation verification completed!"
}

# Print usage instructions
print_usage() {
    log "Installation completed successfully!"
    echo ""
    echo -e "${BLUE}Usage Instructions:${NC}"
    echo ""
    echo "1. Test the installation:"
    echo "   cd $BUILD_DIR"
    echo "   srun --partition=gpucluster --gres=gpu:1 $PWD/install/bin/lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -in in.mace_gpu"
    echo ""
    echo "2. Submit a batch job:"
    echo "   cd $BUILD_DIR"
    echo "   sbatch mace_gpu_test.sh"
    echo ""
    echo "3. Interactive GPU session:"
    echo "   salloc --partition=gpucluster --gres=gpu:1"
    echo "   srun --pty bash"
    echo ""
    echo -e "${BLUE}Key files created:${NC}"
    echo "  - LAMMPS executable: $BUILD_DIR/install/bin/lmp"
    echo "  - Test input: $BUILD_DIR/in.mace_gpu"
    echo "  - Test data: $BUILD_DIR/data.carbon"
    echo "  - Batch script: $BUILD_DIR/mace_gpu_test.sh"
    echo "  - MACE model: $LAMMPS_DIR/mace_models/mace_mp0_small.model-lammps.pt"
    echo ""
    echo -e "${GREEN}Happy computing with LAMMPS + KOKKOS + MACE!${NC}"
}

# Main installation function
main() {
    log "Starting LAMMPS + KOKKOS + MACE installation..."
    log "Installation directory: $INSTALL_DIR"
    
    # Create installation directory
    mkdir -p $INSTALL_DIR
    cd $INSTALL_DIR
    
    # Run installation steps
    check_gpu_node
    load_modules
    setup_cuda
    setup_conda
    download_libtorch
    download_lammps
    build_lammps
    setup_mace_model
    create_test_files
    verify_installation
    print_usage
    
    log "Installation completed successfully!"
}

# Run main function
main "$@"
