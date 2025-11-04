# VASP GPU Compilation and Setup Guide

## Overview
This guide documents the complete process of compiling and running VASP with GPU acceleration on a cluster with NVIDIA A100 GPUs, NVIDIA HPC SDK, and SLURM.

## System Configuration
- **VASP Version**: 6.4.3
- **NVIDIA HPC SDK**: 25.9
- **CUDA Version**: 12.9
- **GPU**: NVIDIA A100 80GB PCIe (Compute Capability 8.0)
- **MPI**: HPCX 12.9 (CUDA-aware)
- **Cluster**: SLURM with GPU partitions

## Key Lessons Learned

### 1. Critical Issue: SLURM GPU Assignment Conflicts
**Problem**: Manually setting `CUDA_VISIBLE_DEVICES=0` and `ACC_DEVICE_NUM=0` in submission scripts conflicts with SLURM's dynamic GPU assignment.

**Solution**: Let SLURM handle GPU assignment automatically.

### 2. CUDA Version Matching
**Problem**: VASP compiled with CUDA 13.0 but system has CUDA 12.9.

**Solution**: Compile VASP with CUDA 12.9 to match system CUDA version.

### 3. HPCX Version Matching
**Problem**: Using HPCX 13.0 when VASP was compiled with HPCX 12.9.

**Solution**: Use HPCX 12.9 to match VASP compilation.

## Compilation Process

### Step 1: Environment Setup
```bash
# Load required modules
module purge
module load gcc-toolset/12

# Set up NVIDIA HPC SDK
export PATH=/path/to/nvidia_hpc_sdk/compilers/bin:$PATH
export PATH=/path/to/nvidia_hpc_sdk/comm_libs/mpi/bin:$PATH

# Set up HPCX CUDA-aware MPI
source /path/to/hpcx-init-ompi.sh
hpcx_load

# Set up CUDA
export CUDA_HOME=/path/to/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Step 2: Makefile Configuration
The makefile must be configured with:
- **CUDA Version**: 12.9 (matching system)
- **Compute Capability**: cc80 (for A100 GPUs)
- **OpenACC**: Enabled with `-acc -gpu=cc80,cuda12.9`
- **MPI**: HPCX 12.9 paths

### Step 3: Compilation
```bash
cd /path/to/vasp6/vasp.6.4.3
cp makefile.include.gpu makefile.include
# Ensure makefile uses CUDA 12.9
sed -i 's/cuda13.0/cuda12.9/g' makefile.include
make std
```

## Runtime Configuration

### Corrected Submission Script Settings
```bash
#!/bin/bash
#SBATCH --gres=gpu:1

# Environment setup (same as compilation)
module purge
module load gcc-toolset/12
# ... (NVIDIA HPC SDK setup)

# CRITICAL: Let SLURM handle GPU assignment
# DO NOT set these manually:
# export CUDA_VISIBLE_DEVICES=0  # SLURM sets this
# export ACC_DEVICE_NUM=0        # Causes conflicts

# Correct OpenACC settings
export ACC_DEVICE_TYPE=nvidia
export ACC_NOTIFY=1
```

### What NOT to Do
❌ **Don't manually set CUDA_VISIBLE_DEVICES**
❌ **Don't manually set ACC_DEVICE_NUM**
❌ **Don't use mismatched CUDA versions**
❌ **Don't use mismatched HPCX versions**

### What TO Do
✅ **Let SLURM handle GPU assignment**
✅ **Use ACC_DEVICE_TYPE=nvidia**
✅ **Match CUDA versions between compilation and runtime**
✅ **Use HPCX version matching compilation**

## Troubleshooting

### Issue: "0 GPUs detected"
**Cause**: SLURM assigned GPU 1, but OpenACC was configured for GPU 0.
**Solution**: Remove manual GPU device assignments from submission script.

### Issue: Library not found errors
**Cause**: Incorrect LD_LIBRARY_PATH or missing NVIDIA HPC SDK libraries.
**Solution**: Ensure all NVIDIA HPC SDK paths are in LD_LIBRARY_PATH.

### Issue: MPI initialization failures
**Cause**: Using wrong HPCX version or conflicting MPI installations.
**Solution**: Use HPCX version matching VASP compilation.

## Verification

### Successful GPU Detection
Look for this in VASP output:
```
OpenACC runtime initialized ...    1 GPUs detected
```

### Environment Variables to Check
```bash
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ACC_DEVICE_TYPE: $ACC_DEVICE_TYPE"
```

### nvidia-smi Output
Should show GPUs available and CUDA version 12.9.

## Files Created

1. **`compile_vasp_gpu_final.sh`** - Complete compilation script
2. **`submit.vasp6.sh`** - Corrected submission script
3. **`VASP_GPU_COMPILATION_GUIDE.md`** - This documentation

## Summary

The key to successful VASP GPU acceleration is:
1. **Matching versions** (CUDA, HPCX) between compilation and runtime
2. **Letting SLURM handle GPU assignment** instead of manual configuration
3. **Proper environment setup** with all necessary library paths
4. **Correct OpenACC configuration** using `ACC_DEVICE_TYPE=nvidia`

This approach ensures VASP can successfully detect and use GPUs for acceleration.
