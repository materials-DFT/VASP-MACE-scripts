#!/usr/bin/env python3
"""
VASP INCAR Parallelization Optimizer

A comprehensive script to analyze VASP systems and automatically optimize
NCORE, NPAR, KPAR, and NSIM parameters in INCAR files.

Usage:
    python3 optimize_vasp_incar.py <directory_path> [options]

Features:
- Recursively finds and analyzes all VASP systems in a directory
- Reads POSCAR to determine system size (number of atoms)
- Reads KPOINTS to determine k-point grid density
- Calculates optimal parallelization parameters based on system characteristics
- Supports multiple cluster configurations
- Validates mathematical relationships (NCORE × NPAR = total_cores)
- Provides detailed analysis and recommendations

Author: AI Assistant
Version: 1.0
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SystemInfo:
    """Container for system analysis information."""
    path: str
    atoms: int
    kpoints: int
    kpoint_grid: Tuple[int, int, int]
    current_ncore: Optional[int] = None
    current_npar: Optional[int] = None
    current_kpar: Optional[int] = None
    current_nsim: Optional[int] = None
    optimal_ncore: Optional[int] = None
    optimal_npar: Optional[int] = None
    optimal_kpar: Optional[int] = None
    optimal_nsim: Optional[int] = None


@dataclass
class ClusterConfig:
    """Cluster configuration parameters."""
    name: str
    total_cores: int
    cores_per_node: int
    nodes: int
    max_ncore: int  # Maximum recommended NCORE value


class VASPAnalyzer:
    """Main class for analyzing and optimizing VASP INCAR files."""
    
    def __init__(self, cluster_config: ClusterConfig, verbose: bool = False):
        self.cluster_config = cluster_config
        self.verbose = verbose
        self.systems_analyzed = []
        self.systems_updated = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def find_vasp_directories(self, root_dir: str) -> List[str]:
        """Recursively find directories containing VASP input files."""
        vasp_dirs = []
        root_path = Path(root_dir)
        
        if not root_path.exists():
            self.log(f"Directory {root_dir} does not exist", "ERROR")
            return vasp_dirs
        
        # Find directories containing POSCAR and KPOINTS
        for poscar_file in root_path.rglob("POSCAR"):
            dir_path = poscar_file.parent
            kpoints_file = dir_path / "KPOINTS"
            incar_file = dir_path / "INCAR"
            
            if kpoints_file.exists() and incar_file.exists():
                vasp_dirs.append(str(dir_path))
                self.log(f"Found VASP directory: {dir_path}")
        
        return sorted(vasp_dirs)
    
    def analyze_poscar(self, poscar_path: str) -> int:
        """Analyze POSCAR to determine number of atoms."""
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # Clean lines and remove empty ones
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            if len(clean_lines) < 7:
                raise ValueError("POSCAR file too short")
            
            # Skip title line (line 1)
            # Skip scaling factor (line 2) - could be a number or "Direct"/"Cartesian"
            # Skip lattice vectors (lines 3-5)
            # Element names (line 6)
            # Atom counts (line 7)
            
            # Find the atom count line by looking for a line with only integers
            atom_line_idx = None
            for i, line in enumerate(clean_lines):
                # Check if line contains only integers (atom counts)
                if re.match(r'^\s*\d+(\s+\d+)*\s*$', line):
                    # Make sure it's not the scaling factor by checking context
                    if i > 4:  # Should be after lattice vectors
                        atom_line_idx = i
                        break
            
            if atom_line_idx is None:
                raise ValueError("Could not find atom count line in POSCAR")
            
            atom_counts = [int(x) for x in clean_lines[atom_line_idx].split()]
            total_atoms = sum(atom_counts)
            
            self.log(f"POSCAR analysis: {total_atoms} atoms ({atom_counts})")
            return total_atoms
            
        except Exception as e:
            self.log(f"Error analyzing POSCAR {poscar_path}: {e}", "ERROR")
            return 0
    
    def analyze_kpoints(self, kpoints_path: str) -> Tuple[int, Tuple[int, int, int]]:
        """Analyze KPOINTS to determine k-point grid."""
        try:
            with open(kpoints_path, 'r') as f:
                lines = f.readlines()
            
            # Find Monkhorst-Pack grid line
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('monkhorst'):
                    # Next line should contain the grid
                    if i + 1 < len(lines):
                        grid_line = lines[i + 1].strip()
                        grid = [int(x) for x in grid_line.split()]
                        if len(grid) >= 3:
                            kpoints = grid[0] * grid[1] * grid[2]
                            self.log(f"KPOINTS analysis: {kpoints} k-points ({grid[0]}×{grid[1]}×{grid[2]})")
                            return kpoints, (grid[0], grid[1], grid[2])
            
            # Fallback: look for any line with 3 integers
            for line in lines:
                if re.match(r'^\s*\d+\s+\d+\s+\d+', line.strip()):
                    grid = [int(x) for x in line.split()[:3]]
                    kpoints = grid[0] * grid[1] * grid[2]
                    self.log(f"KPOINTS analysis (fallback): {kpoints} k-points ({grid[0]}×{grid[1]}×{grid[2]})")
                    return kpoints, (grid[0], grid[1], grid[2])
            
            raise ValueError("Could not find k-point grid in KPOINTS")
            
        except Exception as e:
            self.log(f"Error analyzing KPOINTS {kpoints_path}: {e}", "ERROR")
            return 0, (1, 1, 1)
    
    def analyze_incar(self, incar_path: str) -> Dict[str, Optional[int]]:
        """Analyze current INCAR parameters."""
        params = {
            'ncore': None,
            'npar': None,
            'kpar': None,
            'nsim': None
        }
        
        try:
            with open(incar_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip().upper()
                if line.startswith('#') or not line:
                    continue
                
                for param in params.keys():
                    pattern = rf'^{param}\s*=\s*(\d+)'
                    match = re.search(pattern, line)
                    if match:
                        params[param] = int(match.group(1))
            
            self.log(f"INCAR analysis: NCORE={params['ncore']}, NPAR={params['npar']}, "
                    f"KPAR={params['kpar']}, NSIM={params['nsim']}")
            
        except Exception as e:
            self.log(f"Error analyzing INCAR {incar_path}: {e}", "ERROR")
        
        return params
    
    def calculate_optimal_parameters(self, atoms: int, kpoints: int) -> Dict[str, int]:
        """Calculate optimal parallelization parameters based on system characteristics."""
        total_cores = self.cluster_config.total_cores
        
        # Determine system size category
        if atoms <= 50:
            system_size = "small"
        elif atoms <= 100:
            system_size = "medium"
        else:
            system_size = "large"
        
        # Calculate optimal parameters based on system size and k-point density
        if system_size == "small":
            # Small systems: prioritize k-point parallelization
            if kpoints >= 64:
                # Many k-points: high KPAR, low NPAR
                optimal_kpar = min(8, kpoints // 8)  # Aim for ~8 k-points per group
                optimal_ncore = min(8, total_cores // 2)
                optimal_npar = total_cores // optimal_ncore
            else:
                # Few k-points: balanced approach
                optimal_kpar = min(4, kpoints // 4)
                optimal_ncore = min(4, total_cores // 4)
                optimal_npar = total_cores // optimal_ncore
                
        elif system_size == "medium":
            # Medium systems: balanced approach
            if kpoints >= 64:
                optimal_kpar = min(8, kpoints // 8)
                optimal_ncore = min(4, total_cores // 4)
                optimal_npar = total_cores // optimal_ncore
            else:
                optimal_kpar = min(4, kpoints // 4)
                optimal_ncore = min(4, total_cores // 4)
                optimal_npar = total_cores // optimal_ncore
                
        else:  # large systems
            # Large systems: prioritize band parallelization
            if kpoints >= 64:
                optimal_kpar = min(4, kpoints // 16)
                optimal_ncore = min(2, total_cores // 8)
                optimal_npar = total_cores // optimal_ncore
            else:
                optimal_kpar = min(2, kpoints // 9)
                optimal_ncore = min(2, total_cores // 8)
                optimal_npar = total_cores // optimal_ncore
        
        # Ensure NSIM is always set to 4 for optimal band parallelization
        optimal_nsim = 4
        
        # Validate that NCORE × NPAR = total_cores
        if optimal_ncore * optimal_npar != total_cores:
            # Adjust NPAR to match total cores
            optimal_npar = total_cores // optimal_ncore
            if optimal_ncore * optimal_npar != total_cores:
                self.log(f"Warning: Could not achieve exact core matching for {atoms} atoms", "WARNING")
        
        # Ensure KPAR doesn't exceed k-points
        optimal_kpar = min(optimal_kpar, kpoints)
        
        # Ensure all parameters are reasonable
        optimal_ncore = max(1, min(optimal_ncore, self.cluster_config.max_ncore))
        optimal_npar = max(1, optimal_npar)
        optimal_kpar = max(1, optimal_kpar)
        
        return {
            'ncore': optimal_ncore,
            'npar': optimal_npar,
            'kpar': optimal_kpar,
            'nsim': optimal_nsim
        }
    
    
    def update_incar(self, incar_path: str, system_info: SystemInfo) -> bool:
        """Update INCAR file with optimized parameters."""
        try:
            # Read current INCAR
            with open(incar_path, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            parallelization_section = False
            in_parallelization = False
            
            for line in lines:
                original_line = line
                line_upper = line.strip().upper()
                
                # Detect parallelization section
                if 'PARALLELIZATION' in line_upper and 'FLAGS' in line_upper:
                    parallelization_section = True
                    in_parallelization = True
                    updated_lines.append(line)
                    continue
                
                # Check if we're still in parallelization section
                if in_parallelization and line.strip() and not line.startswith('#'):
                    if not any(param in line_upper for param in ['NCORE', 'NPAR', 'KPAR', 'NSIM', 'LPLANE', 'LSCALU']):
                        in_parallelization = False
                
                # Update parameters
                if in_parallelization:
                    if 'NCORE' in line_upper and ('=' in line_upper or '#' in line_upper):
                        line = f"NCORE = {system_info.optimal_ncore}\n"
                    elif 'NPAR' in line_upper and ('=' in line_upper or '#' in line_upper):
                        line = f"NPAR = {system_info.optimal_npar}\n"
                    elif 'KPAR' in line_upper and ('=' in line_upper or '#' in line_upper):
                        line = f"KPAR = {system_info.optimal_kpar}\n"
                    elif 'NSIM' in line_upper and ('=' in line_upper or '#' in line_upper):
                        line = f"NSIM = {system_info.optimal_nsim}\n"
                
                updated_lines.append(line)
            
            # If no parallelization section found, add one
            if not parallelization_section:
                updated_lines.append("\n# Optimized parallelization parameters\n")
                updated_lines.append(f"NCORE = {system_info.optimal_ncore}\n")
                updated_lines.append(f"NPAR = {system_info.optimal_npar}\n")
                updated_lines.append(f"KPAR = {system_info.optimal_kpar}\n")
                updated_lines.append(f"NSIM = {system_info.optimal_nsim}\n")
                updated_lines.append("LPLANE = .TRUE.\n")
                updated_lines.append("LSCALU = .FALSE.\n")
            
            # Write updated INCAR
            with open(incar_path, 'w') as f:
                f.writelines(updated_lines)
            
            self.log(f"Updated INCAR: {incar_path}")
            return True
            
        except Exception as e:
            self.log(f"Error updating INCAR {incar_path}: {e}", "ERROR")
            return False
    
    def analyze_system(self, vasp_dir: str) -> Optional[SystemInfo]:
        """Analyze a single VASP system."""
        poscar_path = os.path.join(vasp_dir, "POSCAR")
        kpoints_path = os.path.join(vasp_dir, "KPOINTS")
        incar_path = os.path.join(vasp_dir, "INCAR")
        
        # Analyze system components
        atoms = self.analyze_poscar(poscar_path)
        kpoints, kpoint_grid = self.analyze_kpoints(kpoints_path)
        current_params = self.analyze_incar(incar_path)
        
        if atoms == 0 or kpoints == 0:
            self.log(f"Skipping incomplete system: {vasp_dir}", "WARNING")
            return None
        
        # Calculate optimal parameters
        optimal_params = self.calculate_optimal_parameters(atoms, kpoints)
        
        system_info = SystemInfo(
            path=vasp_dir,
            atoms=atoms,
            kpoints=kpoints,
            kpoint_grid=kpoint_grid,
            current_ncore=current_params['ncore'],
            current_npar=current_params['npar'],
            current_kpar=current_params['kpar'],
            current_nsim=current_params['nsim'],
            optimal_ncore=optimal_params['ncore'],
            optimal_npar=optimal_params['npar'],
            optimal_kpar=optimal_params['kpar'],
            optimal_nsim=optimal_params['nsim']
        )
        
        self.systems_analyzed.append(system_info)
        return system_info
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("VASP INCAR OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Cluster Configuration: {self.cluster_config.name}")
        report.append(f"Total Cores: {self.cluster_config.total_cores}")
        report.append(f"Cores per Node: {self.cluster_config.cores_per_node}")
        report.append(f"Nodes: {self.cluster_config.nodes}")
        report.append("")
        
        report.append(f"Systems Analyzed: {len(self.systems_analyzed)}")
        report.append(f"Systems Updated: {len(self.systems_updated)}")
        report.append("")
        
        # Group systems by size
        small_systems = [s for s in self.systems_analyzed if s.atoms <= 50]
        medium_systems = [s for s in self.systems_analyzed if 50 < s.atoms <= 100]
        large_systems = [s for s in self.systems_analyzed if s.atoms > 100]
        
        report.append(f"Small Systems (≤50 atoms): {len(small_systems)}")
        report.append(f"Medium Systems (51-100 atoms): {len(medium_systems)}")
        report.append(f"Large Systems (>100 atoms): {len(large_systems)}")
        report.append("")
        
        # Detailed system analysis
        report.append("DETAILED SYSTEM ANALYSIS")
        report.append("-" * 80)
        
        for system in self.systems_analyzed:
            report.append(f"\nSystem: {os.path.basename(system.path)}")
            report.append(f"  Path: {system.path}")
            report.append(f"  Atoms: {system.atoms}")
            report.append(f"  K-points: {system.kpoints} ({system.kpoint_grid[0]}×{system.kpoint_grid[1]}×{system.kpoint_grid[2]})")
            
            report.append(f"  Current Parameters:")
            report.append(f"    NCORE: {system.current_ncore or 'Not set'}")
            report.append(f"    NPAR:  {system.current_npar or 'Not set'}")
            report.append(f"    KPAR:  {system.current_kpar or 'Not set'}")
            report.append(f"    NSIM:  {system.current_nsim or 'Not set'}")
            
            report.append(f"  Optimized Parameters:")
            report.append(f"    NCORE: {system.optimal_ncore}")
            report.append(f"    NPAR:  {system.optimal_npar}")
            report.append(f"    KPAR:  {system.optimal_kpar}")
            report.append(f"    NSIM:  {system.optimal_nsim}")
            
            # Validation
            total_cores_used = system.optimal_ncore * system.optimal_npar
            if total_cores_used == self.cluster_config.total_cores:
                report.append(f"  ✓ Core usage: {total_cores_used}/{self.cluster_config.total_cores} (optimal)")
            else:
                report.append(f"  ⚠ Core usage: {total_cores_used}/{self.cluster_config.total_cores} (suboptimal)")
        
        report.append("\n" + "=" * 80)
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("=" * 80)
        
        if small_systems:
            report.append(f"\nSmall Systems ({len(small_systems)} systems):")
            report.append("  - Prioritize k-point parallelization (high KPAR)")
            report.append("  - Use moderate NCORE values (4-8)")
            report.append("  - Suitable for systems with many k-points")
        
        if medium_systems:
            report.append(f"\nMedium Systems ({len(medium_systems)} systems):")
            report.append("  - Balanced k-point and band parallelization")
            report.append("  - Moderate NCORE values (2-4)")
            report.append("  - Good performance for most applications")
        
        if large_systems:
            report.append(f"\nLarge Systems ({len(large_systems)} systems):")
            report.append("  - Prioritize band parallelization (high NPAR)")
            report.append("  - Lower NCORE values (1-2)")
            report.append("  - Optimal for systems with many bands")
        
        report.append(f"\nGeneral Recommendations:")
        report.append("  - Always use NSIM = 4 for optimal band parallelization")
        report.append("  - Ensure NCORE × NPAR = total available cores")
        report.append("  - Adjust KPAR based on k-point density")
        report.append("  - Test performance with different parameter combinations")
        
        return "\n".join(report)
    
    def run_optimization(self, root_dir: str, update_files: bool = True) -> bool:
        """Main optimization routine."""
        self.log(f"Starting VASP INCAR optimization in: {root_dir}")
        
        # Find all VASP directories
        vasp_dirs = self.find_vasp_directories(root_dir)
        
        if not vasp_dirs:
            self.log("No VASP directories found", "ERROR")
            return False
        
        self.log(f"Found {len(vasp_dirs)} VASP directories")
        
        # Analyze each system
        for vasp_dir in vasp_dirs:
            self.log(f"Analyzing system: {vasp_dir}")
            system_info = self.analyze_system(vasp_dir)
            
            if system_info is None:
                continue
            
            if update_files:
                incar_path = os.path.join(vasp_dir, "INCAR")
                
                # Update INCAR
                if self.update_incar(incar_path, system_info):
                    self.systems_updated.append(system_info)
                    self.log(f"Successfully updated: {incar_path}")
                else:
                    self.log(f"Failed to update: {incar_path}", "ERROR")
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        # Save report to file
        report_path = os.path.join(root_dir, "vasp_optimization_report.txt")
        try:
            with open(report_path, 'w') as f:
                f.write(report)
            self.log(f"Report saved to: {report_path}")
        except Exception as e:
            self.log(f"Error saving report: {e}", "ERROR")
        
        return len(self.systems_updated) > 0


def get_cluster_configs() -> Dict[str, ClusterConfig]:
    """Get predefined cluster configurations."""
    return {
        "default": ClusterConfig("Default", 16, 8, 1, 8),
        "small": ClusterConfig("Small", 8, 8, 1, 4),
        "medium": ClusterConfig("Medium", 32, 8, 4, 8),
        "large": ClusterConfig("Large", 64, 8, 8, 8),
        "hpc": ClusterConfig("HPC", 128, 16, 8, 16),
        "custom": ClusterConfig("Custom", 16, 8, 1, 8)  # Will be updated based on user input
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize VASP INCAR parallelization parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 optimize_vasp_incar.py /path/to/vasp/calculations
  python3 optimize_vasp_incar.py /path/to/calculations --cluster medium --dry-run
  python3 optimize_vasp_incar.py /path/to/calculations --cores 32 --verbose
        """
    )
    
    parser.add_argument("directory", help="Root directory containing VASP calculations")
    parser.add_argument("--cluster", choices=["default", "small", "medium", "large", "hpc", "custom"],
                       default="default", help="Predefined cluster configuration")
    parser.add_argument("--cores", type=int, help="Total number of cores (overrides cluster config)")
    parser.add_argument("--cores-per-node", type=int, help="Cores per node (overrides cluster config)")
    parser.add_argument("--nodes", type=int, help="Number of nodes (overrides cluster config)")
    parser.add_argument("--max-ncore", type=int, help="Maximum NCORE value")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't update files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Get cluster configuration
    configs = get_cluster_configs()
    cluster_config = configs[args.cluster]
    
    # Override with custom values if provided
    if args.cores:
        cluster_config.total_cores = args.cores
    if args.cores_per_node:
        cluster_config.cores_per_node = args.cores_per_node
    if args.nodes:
        cluster_config.nodes = args.nodes
    if args.max_ncore:
        cluster_config.max_ncore = args.max_ncore
    
    # Create analyzer
    analyzer = VASPAnalyzer(cluster_config, verbose=args.verbose)
    
    # Run optimization
    success = analyzer.run_optimization(
        args.directory,
        update_files=not args.dry_run
    )
    
    if success:
        print(f"\n✓ Optimization completed successfully!")
        if args.dry_run:
            print("  (Dry run - no files were modified)")
        else:
            print(f"  Updated {len(analyzer.systems_updated)} INCAR files")
    else:
        print(f"\n✗ Optimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
