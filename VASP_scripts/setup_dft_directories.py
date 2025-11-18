#!/usr/bin/env python3
"""
Script to organize POSCAR files into directories for DFT calculations.
For files like POSCAR.alpha_oms6_111_111_8.92b3d235-d5e9-4086-99f6-97dda8b0ecc6
Creates directory: alpha_oms6_111_111_8
Moves and renames file to: alpha_oms6_111_111_8/POSCAR
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def extract_interface_name(filename):
    """
    Extract interface name from POSCAR filename.
    
    Example: POSCAR.alpha_oms6_111_111_8.92b3d235-d5e9-4086-99f6-97dda8b0ecc6
    Returns: alpha_oms6_111_111_8
    """
    # Remove 'POSCAR.' prefix
    if not filename.startswith('POSCAR.'):
        return None
    
    # Extract everything after 'POSCAR.' and before the last dot (UUID)
    match = re.match(r'^POSCAR\.(.+)\.[^.]*$', filename)
    if match:
        return match.group(1)
    return None


def parse_poscar(poscar_path):
    """
    Parse POSCAR file to extract system name, elements, and atom counts.
    
    POSCAR format:
    Line 1: System name (or element names)
    Line 2: Scaling factor
    Lines 3-5: Lattice vectors
    Line 6: Element names
    Line 7: Atom counts
    
    Returns:
        tuple: (system_name, elements, atom_counts)
    """
    with open(poscar_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # System name is first line
    system_name = lines[0]
    
    # Skip scaling factor (line 2) and lattice vectors (lines 3-5)
    # Start searching from line 6 (index 5)
    elements = None
    atom_counts = None
    
    # Find element names line (should be after lattice vectors)
    # Look for a line that contains only element symbols
    for i in range(5, min(len(lines), 10)):  # Check lines 6-10
        line = lines[i]
        # Check if line contains only element symbols (uppercase letter followed by optional lowercase)
        if re.match(r'^[A-Z][a-z]?(\s+[A-Z][a-z]?)*\s*$', line):
            elements = line.split()
            # Next line should have atom counts (integers only)
            if i + 1 < len(lines):
                try:
                    atom_counts = [int(x) for x in lines[i + 1].split()]
                    break
                except ValueError:
                    # If next line doesn't contain integers, continue searching
                    continue
    
    if not elements or not atom_counts:
        raise ValueError(f"Could not parse POSCAR file: {poscar_path}")
    
    return system_name, elements, atom_counts


def generate_magmom(elements, atom_counts):
    """
    Generate MAGMOM string based on elements and atom counts.
    K gets 0.0, Mn gets 3.0, O gets 0.0
    
    Args:
        elements: List of element symbols
        atom_counts: List of atom counts for each element
    
    Returns:
        str: MAGMOM string
    """
    magmom_parts = []
    
    for element, count in zip(elements, atom_counts):
        if element == 'Mn':
            magmom = 3.0
        else:
            magmom = 0.0
        
        magmom_parts.append(f"{count}*{magmom:.1f}")
    
    return ' '.join(magmom_parts)


def write_incar(incar_path, system_name, magmom):
    """
    Write INCAR file with the provided template.
    
    Args:
        incar_path: Path to write INCAR file
        system_name: System name from POSCAR
        magmom: MAGMOM string
    """
    incar_content = f"""System = {system_name}

Starting parameters for this run:
ISTART = 0
ICHARG = 2

Electronic Relaxtion:
PREC = Normal
ENCUT = 520
NELMIN = 6
NELM = 500
EDIFF = 1E-6
LREAL = .FALSE.
ISPIN = 2
MAGMOM = {magmom}
#LNONCOLLINEAR = .TRUE.
ALGO = Fast
#LDAU = .TRUE.
#LDAUTYPE = 4
#LDAUL = -1 -1 2 -1
#LDAUU = 0 0 5.5 0 
#LDAUJ = 0 0 0 0 
#LMAXMIX = 4
#AMIX = 0.2
#BMIX = 0.0001
#LHFCALC = .TRUE.
#GGA = PE
METAGGA = SCAN
LMIXTAU=.TRUE.
LASPH=.TRUE.
LDIAG=.TRUE.

Ionic Molecular Dynamics:
# NSW = 1000  max number of geometry steps
IBRION = -1  ionic relax: 0-MD, 1-quasi-Newton, 2-CG, 3-Damped MD
# EDIFFG =  -5E-02  force (eV/A) stopping-criterion for geometry steps
# ISIF   =  3  (1:force=y stress=trace only ions=y shape=n volume=n)
# POTIM = 0.5 initial time step for geo-opt (increase for soft sys)

DOS related values:
LORBIT =     10        
# NEDOS  =     1501
# EMIN   =    -20.0
# EMAX   =     10.0
ISMEAR =  0  (-1-Fermi, 1-Methfessel/Paxton)
SIGMA  =  0.05   broadening in eV

Parallelization flags:
NCORE = 4
NSIM = 4
LPLANE = .TRUE.
LSCALU = .FALSE.
KPAR = 1
# LSCALAPACK = .FALSE.
# ISYM = 0 
# ADDGRID = .TRUE. 

# Space-saving flags (appended by script)
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""
    
    with open(incar_path, 'w') as f:
        f.write(incar_content)


def write_kpoints(kpoints_path):
    """
    Write KPOINTS file with Gamma point.
    
    Args:
        kpoints_path: Path to write KPOINTS file
    """
    kpoints_content = """Gamma KPOINTS
0
Monkhorst-Pack
1 1 1
0 0 0
"""
    
    with open(kpoints_path, 'w') as f:
        f.write(kpoints_content)


def regenerate_files_for_existing_dirs(target_dir):
    """
    Regenerate INCAR, KPOINTS files for existing directories that contain POSCAR.
    
    Args:
        target_dir: Path to directory containing interface directories
    """
    target_path = Path(target_dir).resolve()
    
    # Check if directory exists
    if not target_path.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist")
        return 1
    
    # Find all directories containing POSCAR files recursively
    poscar_dirs = []
    for poscar_file in target_path.rglob('POSCAR'):
        # Only process POSCAR files that are directly in their directory (not POSCAR.*.*)
        if poscar_file.name == 'POSCAR' and poscar_file.parent != target_path:
            poscar_dirs.append(poscar_file.parent)
    
    if not poscar_dirs:
        print(f"No directories with POSCAR files found in '{target_dir}'")
        return 0
    
    count = 0
    
    for interface_dir in poscar_dirs:
        poscar_file = interface_dir / 'POSCAR'
        
        # Parse POSCAR to get system info
        try:
            poscar_system_name, elements, atom_counts = parse_poscar(poscar_file)
            magmom = generate_magmom(elements, atom_counts)
            # Generate system formula from elements and counts
            system_name = ''.join(f"{elem}{count}" for elem, count in zip(elements, atom_counts))
        except Exception as e:
            print(f"Warning: Could not parse POSCAR '{poscar_file}': {e}, skipping...")
            continue
        
        # Regenerate INCAR file
        if system_name and magmom:
            incar_path = interface_dir / 'INCAR'
            write_incar(incar_path, system_name, magmom)
        
        # Regenerate KPOINTS file
        kpoints_path = interface_dir / 'KPOINTS'
        write_kpoints(kpoints_path)
        
        # Show relative path from target directory
        rel_path = interface_dir.relative_to(target_path)
        print(f"Regenerated: {rel_path}")
        if system_name:
            print(f"  Updated INCAR and KPOINTS (System: {system_name}, MAGMOM: {magmom})")
        count += 1
    
    print(f"\nDone! Regenerated files for {count} directories.")
    return 0


def setup_dft_directories(target_dir, regenerate=False):
    """
    Organize POSCAR files into directories based on interface names.
    
    Args:
        target_dir: Path to directory containing POSCAR files
        regenerate: If True, regenerate files for existing directories instead of processing new POSCAR files
    """
    if regenerate:
        return regenerate_files_for_existing_dirs(target_dir)
    
    target_path = Path(target_dir).resolve()
    
    # Check if directory exists
    if not target_path.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist")
        return 1
    
    # Change to target directory
    os.chdir(target_path)
    
    # Find all POSCAR files matching the pattern POSCAR.*.* recursively
    poscar_files = list(target_path.rglob('POSCAR.*.*'))
    
    if not poscar_files:
        print(f"No POSCAR files matching pattern 'POSCAR.*.*' found in '{target_dir}' (searched recursively)")
        return 0
    
    count = 0
    
    # Process each POSCAR file
    for poscar_file in poscar_files:
        filename = poscar_file.name
        
        # Extract interface name
        interface_name = extract_interface_name(filename)
        
        if not interface_name:
            print(f"Warning: Could not extract interface name from '{filename}', skipping...")
            continue
        
        # Create directory in the same location as the POSCAR file
        interface_dir = poscar_file.parent / interface_name
        interface_dir.mkdir(exist_ok=True)
        
        # Move and rename the file
        destination = interface_dir / 'POSCAR'
        shutil.move(str(poscar_file), str(destination))
        
        # Parse POSCAR to get system info
        try:
            poscar_system_name, elements, atom_counts = parse_poscar(destination)
            magmom = generate_magmom(elements, atom_counts)
            # Generate system formula from elements and counts
            system_name = ''.join(f"{elem}{count}" for elem, count in zip(elements, atom_counts))
        except Exception as e:
            print(f"Warning: Could not parse POSCAR '{destination}': {e}, skipping INCAR/KPOINTS generation")
            system_name = None
            magmom = None
        
        # Write INCAR file
        if system_name and magmom:
            incar_path = interface_dir / 'INCAR'
            write_incar(incar_path, system_name, magmom)
        
        # Write KPOINTS file
        kpoints_path = interface_dir / 'KPOINTS'
        write_kpoints(kpoints_path)
        
        # Show relative path from target directory
        rel_path = poscar_file.relative_to(target_path)
        print(f"Processed: {rel_path} -> {rel_path.parent / interface_name / 'POSCAR'}")
        if system_name:
            print(f"  Created INCAR and KPOINTS (System: {system_name}, MAGMOM: {magmom})")
        count += 1
    
    print(f"\nDone! Processed {count} POSCAR files.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Organize POSCAR files into directories for DFT calculations (searches recursively)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s ./interfaces/phase_oms6
  %(prog)s ./interfaces/phase_oms6/oms6_1k/unique_poscars
        """
    )
    parser.add_argument(
        'directory',
        help='Path to directory containing POSCAR files (searched recursively)'
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help='Regenerate INCAR and KPOINTS files for existing directories containing POSCAR files. '
             'Use this to update files (e.g., fix blank lines in INCAR) without reprocessing POSCAR files.'
    )
    
    args = parser.parse_args()
    
    exit_code = setup_dft_directories(args.directory, regenerate=args.regenerate)
    exit(exit_code)


if __name__ == '__main__':
    main()

