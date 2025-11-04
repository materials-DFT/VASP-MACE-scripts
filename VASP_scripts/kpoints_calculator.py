#!/usr/bin/env python3
import numpy as np
import os
import argparse

def read_poscar(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    lattice = np.array([[float(x) for x in line.strip().split()] for line in lines[2:5]])
    lattice *= scale
    return lattice

def compute_kpoint_grid(lattice, density_constant=25):
    lengths = np.linalg.norm(lattice, axis=1)
    kpoints = [max(1, int(density_constant / l + 0.5)) for l in lengths]
    return kpoints, lengths

def recommend_mesh_type(lengths, gamma_threshold=15.0):
    if np.min(lengths) > gamma_threshold:
        return "Gamma"
    else:
        return "Monkhorst-Pack"

def write_kpoints_file(kpoints, mesh_type, directory):
    filepath = os.path.join(directory, "KPOINTS")
    with open(filepath, 'w') as f:
        f.write("Automatically generated KPOINTS\n")
        f.write("0\n")
        f.write(f"{mesh_type}\n")
        f.write(f"{kpoints[0]} {kpoints[1]} {kpoints[2]}\n")
        f.write("0 0 0\n")
    print(f"KPOINTS file written to {filepath}")

def analyze_poscar(path, density_constant):
    lattice = read_poscar(path)
    kpoints, lengths = compute_kpoint_grid(lattice, density_constant)
    mesh_type = recommend_mesh_type(lengths)
    return {
        "path": path,
        "lengths": lengths,
        "kpoints": kpoints,
        "mesh_type": mesh_type
    }

def find_poscars(root_path):
    poscar_files = []
    if os.path.isfile(root_path):
        if os.path.basename(root_path).lower() == "poscar":
            poscar_files.append(os.path.abspath(root_path))
    else:
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if fname.lower() == "poscar":
                    poscar_files.append(os.path.abspath(os.path.join(dirpath, fname)))
    return poscar_files

def main():
    parser = argparse.ArgumentParser(description="Recommend VASP KPOINTS parameters from POSCAR files.")
    parser.add_argument("path", type=str, help="POSCAR file or directory to search.")
    parser.add_argument("--k", type=float, default=25, help="K-point density constant (default: 25 Å⁻¹).")
    parser.add_argument("--gamma-threshold", type=float, default=15.0,
                        help="Minimum cell length (Å) above which Gamma mesh is recommended (default: 15 Å).")
    args = parser.parse_args()

    poscars = find_poscars(args.path)
    if not poscars:
        print("No POSCAR files found.")
        return

    # Ask once if user wants to overwrite all KPOINTS files
    overwrite_all = False
    while True:
        user_input = input(f"Found {len(poscars)} POSCAR files. Overwrite all KPOINTS files with recommendations? (y/N): ").strip().lower()
        if user_input in ("y", "yes"):
            overwrite_all = True
            break
        elif user_input in ("n", "no", ""):
            overwrite_all = False
            break
        else:
            print("Please answer y (yes) or n (no).")

    for poscar in poscars:
        try:
            data = analyze_poscar(poscar, args.k)
            directory = os.path.dirname(poscar)
            print(f"Structure: {poscar}")
            print(f"  Lattice lengths (Å): {data['lengths'][0]:.3f}, {data['lengths'][1]:.3f}, {data['lengths'][2]:.3f}")
            print(f"  Recommended mesh: {data['kpoints'][0]} {data['kpoints'][1]} {data['kpoints'][2]}")
            print(f"  Recommended type: {data['mesh_type']}")
            
            if overwrite_all:
                write_kpoints_file(data['kpoints'], data['mesh_type'], directory)
            else:
                print("Skipping writing KPOINTS file.")
            print("-" * 60)
        except Exception as e:
            print(f"Error reading {poscar}: {e}")

if __name__ == "__main__":
    main()
