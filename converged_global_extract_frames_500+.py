#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
converged_global_extract_frames_500+.py

Quiet-by-default refactor that preserves your one-argument usage:
    python converged_global_extract_frames_500+.py <root_dir>

Outputs (in CWD):
  - final_frames.xyz
  - extraction_log.csv

Behavior:
  1) For each run directory with an OUTCAR, create output.xyz containing only
     converged ionic steps (if missing).
  2) Allocate frames across temperatures with a simple √T weighting.
  3) Merge selected frames into final_frames.xyz using ASE-only IO.
  4) Log true energies to extraction_log.csv.
  5) Quiet by default; use --verbose for progress details.

Optional flags (all optional):
  --total-frames N     (default: 2000)
  --skip-first N       (default: 500)
  --validate           (write a sanity report and print one-line stats)
  --verbose            (print progress)

Requires: ase, numpy, pandas
"""

import os
import re
import sys
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from ase.io.vasp import read_vasp_out
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

def vprint(verbose, *a, **k):
    if verbose:
        print(*a, **k)

def parse_converged_step_indices(outcar_path: str):
    """Collect ionic step indices that reached EDIFF convergence."""
    converged = []
    step = -1
    try:
        with open(outcar_path, 'r', errors='ignore') as f:
            for line in f:
                if re.search(r'^\s*POSITION\s+TOTAL-FORCE', line):
                    step += 1
                elif 'aborting loop because EDIFF is reached' in line:
                    if step >= 0:
                        converged.append(step)
    except Exception:
        pass
    return converged

def extract_converged_frames_to_xyz(directory: str, out_xyz_name: str = "output.xyz", verbose=False) -> bool:
    """
    From a directory containing OUTCAR, read ionic steps.
    Save only converged steps into out_xyz_name (ASE Atoms with SinglePointCalculator).
    """
    outcar_path = os.path.join(directory, "OUTCAR")
    output_xyz = os.path.join(directory, out_xyz_name)
    if not os.path.exists(outcar_path):
        return False

    conv_indices = parse_converged_step_indices(outcar_path)
    if not conv_indices:
        return False

    try:
        all_frames = read_vasp_out(outcar_path, index=":")
    except Exception:
        return False

    selected = []
    for i, atoms in enumerate(all_frames):
        if i in conv_indices:
            # energy / forces
            try:
                energy = atoms.get_potential_energy()
            except Exception:
                energy = atoms.info.get('energy', None)
            try:
                forces = atoms.get_forces()
            except Exception:
                forces = None
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            selected.append(atoms)

    if not selected:
        return False

    try:
        write(output_xyz, selected, format="extxyz")
        vprint(verbose, f"[ok] {directory}: wrote {len(selected)} → output.xyz")
        return True
    except Exception:
        return False

def find_tebeg(outcar_path: str):
    """Return TEBEG (float) if present; else None."""
    try:
        with open(outcar_path, 'r', errors='ignore') as f:
            for line in f:
                m = re.search(r'\bTEBEG\s*=\s*([0-9.+-Ee]+)', line)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        return None
    except Exception:
        pass
    return None

def infer_temperature_from_path(path: str):
    """Infer temperature from tokens like '900K' or 'T900' in the directory path."""
    tokens = re.split(r'[\\/]', path)
    for t in tokens:
        m = re.match(r'([0-9]+)\s*[kK]\b', t)
        if m:
            return float(m.group(1))
        m2 = re.match(r'^[tT]([0-9]+)$', t)
        if m2:
            return float(m2.group(1))
    return None

def count_extxyz_frames(path: str) -> int:
    """Lightweight frame counter for extxyz files."""
    n = 0
    try:
        with open(path, 'r', errors='ignore') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    nat = int(line.strip())
                except ValueError:
                    continue
                # skip comment
                _ = f.readline()
                # skip atoms
                for _ in range(nat):
                    _ = f.readline()
                n += 1
    except Exception:
        return 0
    return n

def collect_xyz_files_with_temperatures(root_dir: str, out_xyz_name: str = "output.xyz", verbose=False):
    """
    Ensure per-dir output.xyz exists; return list of (xyz_path, T, nframes).
    Temperature preference: TEBEG from OUTCAR, else path token.
    """
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        outcar_path = os.path.join(dirpath, "OUTCAR")
        if not os.path.exists(outcar_path):
            continue
        xyz_path = os.path.join(dirpath, out_xyz_name)

        if not os.path.exists(xyz_path):
            _ = extract_converged_frames_to_xyz(dirpath, out_xyz_name=out_xyz_name, verbose=verbose)
        if not os.path.exists(xyz_path):
            continue

        nframes = count_extxyz_frames(xyz_path)
        if nframes <= 0:
            continue

        T = find_tebeg(outcar_path)
        if T is None:
            T = infer_temperature_from_path(dirpath)
        if T is None:
            T = 0.0

        results.append((xyz_path, float(T), nframes))

    return results

def allocate_frames_across_temperatures(files, total_target, skip_first=0):
    """
    files: list of (xyz_path, temp, nframes).
    Allocate by √(T/300) * available_count; return [(path, T, indices), ...].
    """
    by_temp = defaultdict(list)
    for path, T, n in files:
        avail = max(0, n - skip_first)
        if avail > 0:
            by_temp[T].append((path, n, avail))

    total_avail = sum(av for lst in by_temp.values() for _, _, av in lst)
    if total_avail == 0:
        return []

    total = min(total_target, total_avail)
    temps = sorted(by_temp.keys())

    # Weights per temp
    weights = []
    for T in temps:
        frames_at_T = sum(av for _, _, av in by_temp[T])
        w = (math.sqrt(T / 300.0) if T > 0 else 0.1) * frames_at_T
        weights.append(w)
    wsum = sum(weights) if sum(weights) > 0 else float(len(temps))

    raw = [w / wsum * total for w in weights]
    rounded = [int(round(x)) for x in raw]

    # Adjust to exact 'total'
    diff = total - sum(rounded)
    fracs = [x - int(x) for x in raw]
    order = np.argsort(fracs)[::-1]
    i = 0
    while diff != 0 and i < len(order):
        idx = order[i]
        if diff > 0:
            rounded[idx] += 1
            diff -= 1
        else:
            if rounded[idx] > 0:
                rounded[idx] -= 1
                diff += 1
        i += 1

    selections = []
    for T, need in zip(temps, rounded):
        if need <= 0:
            continue
        files_T = by_temp[T]
        avail_T = sum(av for _, _, av in files_T)
        if avail_T == 0:
            continue
        for path, n, av in files_T:
            take = int(round(need * (av / avail_T)))
            take = min(take, av)
            if take <= 0:
                continue
            # roughly even spread from [skip_first, n-1]
            span = n - skip_first
            if span <= 0:
                continue
            if take == av:
                idxs = list(range(skip_first, n))
            else:
                step = span / float(take)
                idxs = sorted({ int(skip_first + round(k * step)) for k in range(take)
                                if skip_first + round(k * step) < n })
                while len(idxs) < take and (skip_first + len(idxs)) < n:
                    idxs.append(skip_first + len(idxs))
                idxs = idxs[:take]
            selections.append((path, T, idxs))

    # Flatten then re-group, finally cap to total
    flat = []
    for path, T, idxs in selections:
        for idx in idxs:
            flat.append((path, T, idx))
    flat = flat[:total]

    grouped = defaultdict(lambda: (0.0, []))
    for path, T, idx in flat:
        grouped[path] = (T, grouped[path][1] + [idx])

    out = []
    for path, (T, idxs) in grouped.items():
        out.append((path, T, sorted(set(idxs))))
    return out

def merge_selected_frames(selections, final_output, log_path, verbose=False):
    """Write final extxyz and a CSV log of true energies."""
    if os.path.exists(final_output):
        os.remove(final_output)

    rows = []
    final_idx = -1
    for xyz_path, temp, idxs in selections:
        for i in idxs:
            atoms = read(xyz_path, index=i)
            # Ensure energy/forces are attached for extxyz
            e = None
            try:
                e = atoms.get_potential_energy()
            except Exception:
                e = atoms.info.get('energy', None)
            f = None
            try:
                f = atoms.get_forces()
            except Exception:
                pass
            if e is not None or f is not None:
                atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)

            write(final_output, atoms, format='extxyz', append=True)
            final_idx += 1

            # Log the actual energy we just wrote
            try:
                e_true = atoms.get_potential_energy()
            except Exception:
                e_true = None

            rows.append({
                "source_xyz_path": xyz_path,
                "temperature": temp,
                "frame_index_in_source": i,
                "final_output_frame_index": final_idx,
                "energy_eV": e_true
            })

    df = pd.DataFrame(rows)
    df.to_csv(log_path, index=False)
    vprint(verbose, f"[ok] merged {final_idx+1} frames")
    return df

def validate_final_file(final_output):
    """Return quick stats and list of positive-energy frames."""
    suspects = []
    n = 0
    e_min = e_max = None
    e_sum = 0.0
    while True:
        try:
            atoms = read(final_output, index=n)
        except Exception:
            break
        n += 1
        e = None
        try:
            e = atoms.get_potential_energy()
        except Exception:
            e = atoms.info.get('energy', None)
        if e is not None:
            e_min = e if e_min is None else min(e_min, e)
            e_max = e if e_max is None else max(e_max, e)
            e_sum += e
            if e > 0.0:
                suspects.append((n-1, e))
    return {
        "n_frames": n,
        "e_min": e_min,
        "e_max": e_max,
        "e_mean": (e_sum / n) if n > 0 else None,
        "n_positive": len(suspects)
    }, suspects

def main():
    # Quiet-by-default CLI mirroring your original usage
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("root_dir", nargs='?')
    p.add_argument("--total-frames", type=int, default=2000)
    p.add_argument("--skip-first", type=int, default=500)
    p.add_argument("--final-output", default="final_frames.xyz")
    p.add_argument("--log", default="extraction_log.csv")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("-h", "--help", action="help", help="show this help message and exit")
    args = p.parse_args()

    if not args.root_dir:
        print("Usage: python converged_global_extract_frames_500+.py <root_dir> [--total-frames N] [--skip-first N] [--validate] [--verbose]")
        sys.exit(1)

    # 1) Collect (or build) per-dir output.xyz with temperatures
    files = collect_xyz_files_with_temperatures(args.root_dir, verbose=args.verbose)
    if not files:
        print("[err] no usable output.xyz files found under root_dir")
        sys.exit(2)

    # 2) Allocate frames across temperatures
    selections = allocate_frames_across_temperatures(files, total_target=args.total_frames, skip_first=args.skip_first)
    if not selections:
        print("[err] allocation produced no selections")
        sys.exit(3)

    # 3) Merge into final_outputs
    _ = merge_selected_frames(selections, args.final_output, args.log, verbose=args.verbose)

    # 4) Optional validation (prints one-line summary; still quiet otherwise)
    if args.validate:
        stats, suspects = validate_final_file(args.final_output)
        print(f"[validate] n={stats['n_frames']}  Emin={stats['e_min']:.6f}  Emax={stats['e_max']:.6f}  "
              f"Emean={stats['e_mean']:.6f}  positives={stats['n_positive']}")
        if stats['n_positive'] > 0 and args.verbose:
            for i, e in suspects[:10]:
                print(f"  suspect frame {i}: {e:.6f} eV")

if __name__ == "__main__":
    main()
