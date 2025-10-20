#!/bin/bash
#SBATCH --job-name=compile_vasp_gpu_final
#SBATCH --partition=gpucluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00

# =============================================================================
# VASP GPU Compilation Script
# Based on successful troubleshooting and GPU detection fix
# =============================================================================

# Install NVIDIA HPC SDK, if not installed already
wget https://developer.download.nvidia.com/hpc-sdk/25.9/nvhpc_2025_259_Linux_x86_64_cuda_multi.tar.gz
tar xpzf nvhpc_2025_259_Linux_x86_64_cuda_multi.tar.gz
nvhpc_2025_259_Linux_x86_64_cuda_multi/install

echo "=========================================="
echo " VASP GPU Compilation Script"
echo " Based on successful troubleshooting"
echo "=========================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo ""

# =============================================================================
# Environment Setup
# =============================================================================

echo "=== Setting up environment ==="

# Clean environment
module purge

# Load required modules
module load gcc-toolset/12

# Set up NVIDIA HPC SDK environment for GPU support
export PATH=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/compilers/bin:$PATH
export PATH=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/comm_libs/mpi/bin:$PATH

echo "NVIDIA HPC SDK compilers added to PATH"

# Set up HPCX (CUDA-aware MPI) environment - matching VASP compilation version
echo "Setting up HPCX CUDA-aware MPI environment..."
source /Users/924322630/nvidia_hpc_sdk/nvhpc_2025_259_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/25.9/comm_libs/12.9/hpcx/hpcx-2.24/hpcx-init-ompi.sh
hpcx_load

echo "HPCX environment loaded"

# Add NVIDIA HPC SDK libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/compilers/lib:/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/compilers/extras/qd/lib:$LD_LIBRARY_PATH

# Set up CUDA environment
export CUDA_HOME=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA environment configured"

# =============================================================================
# Verify Environment
# =============================================================================

echo ""
echo "=== Verifying environment ==="

# Check compilers
echo "Checking compilers:"
echo "nvfortran: $(which nvfortran)"
echo "mpif90: $(which mpif90)"
echo "nvcc: $(which nvcc)"

# Check CUDA version
echo ""
echo "CUDA version:"
nvcc --version

# Check GPU availability
echo ""
echo "GPU availability:"
nvidia-smi

# =============================================================================
# VASP Compilation
# =============================================================================

echo ""
echo "=== Starting VASP compilation ==="

# Go to VASP directory
cd /Users/924322630/vasp6/vasp.6.4.3

echo "Working directory: $(pwd)"

# Check if makefile.include.gpu exists
if [ ! -f "makefile.include.gpu" ]; then
    echo "ERROR: makefile.include.gpu not found!"
    echo "Please ensure you have the correct VASP GPU makefile."
    exit 1
fi

echo "Found makefile.include.gpu"

# Modify makefile.include.gpu to use CUDA 12.9 (matching system)
echo "Configuring makefile for CUDA 12.9..."

# Create a backup of the original makefile
cp makefile.include.gpu makefile.include.gpu.backup

# Update CUDA version to 12.9 in the makefile
sed -i 's/cuda13.0/cuda12.9/g' makefile.include.gpu

echo "Updated makefile to use CUDA 12.9"

# Copy GPU makefile to makefile.include
cp makefile.include.gpu makefile.include

echo "makefile.include configured for GPU compilation"

# Verify makefile configuration
echo ""
echo "=== Verifying makefile configuration ==="
echo "Compiler settings:"
grep "CC.*=" makefile.include
grep "FC.*=" makefile.include
grep "FCL.*=" makefile.include

echo ""
echo "CUDA settings:"
grep -i "cuda" makefile.include

# Clean previous builds
echo ""
echo "=== Cleaning previous builds ==="
rm -rf build/std/*

# Compile VASP
echo ""
echo "=== Starting VASP compilation ==="
echo "This may take 15-30 minutes..."

# Start compilation
make std

# Check compilation result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " VASP GPU compilation completed successfully!"
    echo "=========================================="
    
    # Check if binary was created
    if [ -f "bin/vasp_std" ]; then
        echo "Binary created: bin/vasp_std"
        ls -la bin/vasp_std
        
        # Check binary size and timestamp
        echo ""
        echo "Binary details:"
        echo "Size: $(du -h bin/vasp_std | cut -f1)"
        echo "Date: $(stat -c %y bin/vasp_std)"
        
        # Verify GPU libraries are linked
        echo ""
        echo "=== Checking GPU library dependencies ==="
        echo "OpenACC libraries:"
        ldd bin/vasp_std | grep -i acc || echo "No OpenACC libraries found"
        
        echo ""
        echo "CUDA libraries:"
        ldd bin/vasp_std | grep -i cuda || echo "No CUDA libraries found"
        
        echo ""
        echo "MPI libraries:"
        ldd bin/vasp_std | grep -i mpi | head -3
        
    else
        echo "ERROR: Binary not created!"
        exit 1
    fi
    
else
    echo ""
    echo "=========================================="
    echo " VASP GPU compilation FAILED!"
    echo "=========================================="
    echo "Check the compilation output above for errors."
    exit 1
fi

echo ""
echo "=========================================="
echo " Compilation Summary"
echo "=========================================="
echo "VASP version: 6.4.3"
echo "CUDA version: 12.9"
echo "Compute capability: cc80 (A100)"
echo "MPI: HPCX 12.9"
echo "OpenACC: Enabled"
echo "Binary location: /Users/924322630/vasp6/vasp.6.4.3/bin/vasp_std"
echo ""
echo "To run VASP with GPU acceleration:"
echo "1. Use the corrected submit.vasp6.sh script"
echo "2. Ensure --gres=gpu:1 is in your SLURM script"
echo "3. DO NOT manually set CUDA_VISIBLE_DEVICES"
echo "4. Let SLURM handle GPU assignment automatically"
echo ""
echo "Compilation completed at: $(date)"
echo "=========================================="
