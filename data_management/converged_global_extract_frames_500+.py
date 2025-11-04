#!/usr/bin/env python3
import os
import re
import sys
import argparse
import gc
import numpy as np
from collections import defaultdict
from ase.io.vasp import read_vasp_out
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator

# ---------- OUTCAR -> output.xyz (memory-safe) ----------
def extract_converged_frames(directory):
    """
    Create/overwrite 'output.xyz' in `directory` with only frames whose
    ionic step reached EDIFF convergence. Writes incrementally to avoid RAM blowups.
    """
    outcar_path = os.path.join(directory, "OUTCAR")
    output_xyz = os.path.join(directory, "output.xyz")
    if not os.path.exists(outcar_path):
        return False

    # First pass: find converged step indices
    converged_steps = []
    step_index = -1
    try:
        with open(outcar_path, 'r', errors='ignore') as f:
            for line in f:
                if re.match(r"\s*POSITION\s+TOTAL-FORCE", line):
                    step_index += 1
                elif "aborting loop because EDIFF is reached" in line:
                    # record the step that just finished
                    converged_steps.append(step_index)
    except Exception as e:
        print(f"[WARN] Error scanning {outcar_path}: {e}")
        return False

    if not converged_steps:
        return False
    keep = set(converged_steps)

    # Second pass: iterate frames lazily and write matching ones
    # Remove existing output to start clean
    try:
        if os.path.exists(output_xyz):
            os.remove(output_xyz)
    except Exception:
        pass

    wrote = 0
    try:
        # The iterator yields Atoms one-by-one; we write immediately
        for i, atoms in enumerate(read_vasp_out(outcar_path, index=":")):
            if i in keep:
                # Attach single-point calc so energy/forces are stored in ExtXYZ
                try:
                    energy = atoms.get_potential_energy()
                except Exception:
                    energy = None
                try:
                    forces = atoms.get_forces()
                except Exception:
                    forces = None
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                write(output_xyz, atoms, format="extxyz", append=True)
                wrote += 1
            # Proactively free
            del atoms
            if i % 200 == 0:
                gc.collect()
    except Exception as e:
        print(f"[WARN] Error reading/writing frames for {outcar_path}: {e}")

    if wrote > 0:
        print(f"[OK] {directory}: wrote {wrote} converged frames → {output_xyz}")
        return True
    return False

# ---------- TEBEG ----------
def find_tebeg(outcar_path):
    try:
        with open(outcar_path, 'r', errors='ignore') as file:
            for line in file:
                m = re.search(r'TEBEG\s*=\s*([\d\.E+-]+)', line)
                if m:
                    return float(m.group(1))
    except Exception as e:
        print(f"[WARN] Error reading {outcar_path}: {e}")
    return None

# ---------- Walk multiple roots ----------
def collect_xyz_and_temperatures(root_dirs):
    tebeg_counts = defaultdict(int)
    file_paths = []

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"[WARN] Skipping non-directory: {root_dir}")
            continue

        for dirpath, _, _ in os.walk(root_dir):
            outcar_path = os.path.join(dirpath, "OUTCAR")
            output_xyz = os.path.join(dirpath, "output.xyz")

            if not os.path.exists(output_xyz):
                ok = extract_converged_frames(dirpath)
                if not ok:
                    continue

            if os.path.exists(outcar_path):
                t = find_tebeg(outcar_path)
                if t is not None:
                    tebeg_counts[t] += 1
                    file_paths.append((output_xyz, t))
    return tebeg_counts, file_paths

# ---------- Allocation ----------
def allocate_frames(temperature_data, total_frames=2000):
    zero_k_count = temperature_data.get(0.0, 0)
    zero_k_frames = zero_k_count * 1
    remaining = max(total_frames - zero_k_frames, 0)

    allocation = {}
    non_zero = {k: v for k, v in temperature_data.items() if k != 0.0}
    if non_zero:
        items = sorted(non_zero.items(), key=lambda x: x[0])
        temps = np.array([t for t, _ in items], dtype=float)
        runs = np.array([n for _, n in items], dtype=int)

        weights = (np.sqrt(np.maximum(temps, 0.0)) / np.sqrt(300.0)) * runs
        sw = weights.sum()
        if sw > 0:
            frames_per_temp = np.round(weights / sw * remaining).astype(int)
            with np.errstate(divide='ignore', invalid='ignore'):
                per_run = np.where(runs > 0,
                                   np.round(frames_per_temp / runs).astype(int),
                                   0)
        else:
            per_run = np.zeros_like(runs, dtype=int)

        for (t, _), fr in zip(items, per_run):
            allocation[t] = max(int(fr), 0)

    if zero_k_count > 0:
        allocation[0.0] = 1
    return allocation

# ---------- Helper for float keys ----------
def alocation_get(mapping, key, default):
    if key in mapping:
        return mapping[key]
    if isinstance(key, float):
        for k in mapping.keys():
            try:
                if abs(float(k) - key) < 1e-9:
                    return mapping[k]
            except Exception:
                continue
    return default

# ---------- Stream-sample XYZ after burn-in (two passes) ----------
def extract_and_write(xyz_file, num_frames, output_file, burn_in=500):
    """
    Two-pass streaming sampler:
      1) Count frames (by 'Lattice' in comment line) to know N.
      2) Choose evenly spaced indices in [burn_in, N-1], then stream again and copy only those frames.

    Never loads the whole file in memory.
    """
    if num_frames <= 0:
        return 0

    # Pass 1: count frames by scanning line-by-line
    total = 0
    try:
        with open(xyz_file, 'r') as f:
            for line in f:
                if 'Lattice' in line:
                    total += 1
    except Exception as e:
        print(f"[WARN] Cannot scan {xyz_file}: {e}")
        return 0

    if total <= burn_in:
        print(f"[SKIP] {xyz_file}: has <= {burn_in} frames ({total}).")
        return 0

    avail = total - burn_in
    take = int(min(num_frames, avail))
    # Which frame indices (0-based among all frames) do we want?
    chosen = set(burn_in + i for i in np.linspace(0, avail - 1, take, dtype=int))

    # Pass 2: stream again and copy only chosen frames
    appended = 0
    try:
        with open(xyz_file, 'r') as src, open(output_file, 'a') as dst:
            current_frame_idx = -1
            while True:
                # Read one XYZ frame: nat, comment, nat coord lines
                nat_line = src.readline()
                if not nat_line:
                    break  # EOF
                nat_line_stripped = nat_line.strip()
                if not nat_line_stripped:
                    # Skip stray blank lines
                    continue
                try:
                    nat = int(nat_line_stripped)
                except ValueError:
                    # If the file has stray text, keep scanning
                    continue

                comment = src.readline()
                if not comment:
                    break  # malformed EOF

                # A frame boundary is recognized by comment containing 'Lattice'
                if 'Lattice' in comment:
                    current_frame_idx += 1

                # Read nat coordinate lines
                coords = []
                for _ in range(nat):
                    line = src.readline()
                    if not line:
                        coords = []
                        break
                    coords.append(line)
                if len(coords) != nat:
                    break  # malformed EOF

                if current_frame_idx in chosen:
                    # Modify header/coords text on the fly
                    mod_comment = re.sub(r'\bforces\b', 'REF_forces', comment)
                    mod_comment = re.sub(r'\benergy\b', 'REF_energy', mod_comment)
                    # Write frame
                    dst.write(f"{nat}\n")
                    dst.write(mod_comment)
                    for c in coords:
                        # (coordinates lines rarely contain these keys, but keep symmetry)
                        c = re.sub(r'\bforces\b', 'REF_forces', c)
                        c = re.sub(r'\benergy\b', 'REF_energy', c)
                        dst.write(c)
                    appended += 1

                # Periodic GC for long files
                if appended % 200 == 0:
                    gc.collect()

    except Exception as e:
        print(f"[WARN] Error sampling {xyz_file}: {e}")

    return appended

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Extract converged frames from VASP OUTCARs across one or more roots, "
                    "then downsample per TEBEG into a single ExtXYZ."
    )
    parser.add_argument("roots", nargs="+", help="One or more root directories to scan.")
    parser.add_argument("--final-output", default="final_frames.xyz",
                        help="Combined ExtXYZ output path (default: final_frames.xyz).")
    parser.add_argument("--total-frames", type=int, default=2000,
                        help="Target total frames across all runs (default: 2000).")
    parser.add_argument("--burn-in", type=int, default=500,
                        help="Frames to skip at the start of each run (default: 500).")
    args = parser.parse_args()

    tebeg_counts, file_paths = collect_xyz_and_temperatures(args.roots)
    if not file_paths:
        print("No usable output.xyz + TEBEG pairs found. Nothing to do.")
        sys.exit(0)

    allocation = allocate_frames(tebeg_counts, total_frames=args.total_frames)

    # Fresh combined file
    try:
        if os.path.exists(args.final_output):
            os.remove(args.final_output)
    except Exception:
        pass

    total_appended = 0
    for xyz_file, temp in file_paths:
        per_run_frames = int(alocation_get(allocation, temp, 0))
        if per_run_frames > 0:
            appended = extract_and_write(xyz_file, per_run_frames, args.final_output, burn_in=args.burn_in)
            total_appended += appended
            print(f"[APPEND] {xyz_file}: requested {per_run_frames}, appended {appended}")

    print(f"\nTotal frames appended: {total_appended}")
    print(f"All frames saved to {args.final_output}")

if __name__ == "__main__":
    main()
