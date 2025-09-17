#!/usr/bin/env python3
import argparse
import os
import re
import sys
from math import pi
import numpy as np

from ase.io import read
from ase import Atoms

# ---------- Physical constants (SI) ----------
KB_J_PER_K = 1.380649e-23         # Boltzmann constant [J/K]
AMU_KG     = 1.66053906660e-27     # atomic mass unit [kg]
ANG_TO_M   = 1.0e-10               # 1 Å in meters

# ---------- Core physics ----------
def nose_mass_amuA2(temperature_K: float, ndof: int, t0_fs: float, a1_len_A: float) -> float:
    """
    Compute Nosé mass Q in [amu * Å^2] for a given:
      - temperature_K : target temperature in Kelvin
      - ndof          : degrees of freedom (free Cartesian components)
      - t0_fs         : thermostat characteristic oscillation time in femtoseconds
      - a1_len_A      : length of the first lattice vector |a1| in Å

    Formula (consistent with your original script’s structure):
        Q_SI (J·s^2) = (t0 / (2π))^2 * 2 * ndof * kB * T
        Q_[amu·Å^2]  = Q_SI / (amu * Å^2)
    """
    t0_s = t0_fs * 1e-15  # fs -> s
    q_si = (t0_s / (2.0 * pi))**2 * (2.0 * ndof) * KB_J_PER_K * temperature_K
    q_amuA2 = q_si / (AMU_KG * (a1_len_A * ANG_TO_M)**2)
    return q_amuA2


def cnt_dof(atoms: Atoms) -> int:
    """
    Count number of free Cartesian components.
    - If there are constraints (FixAtoms/FixScaled/FixedPlane/FixedLine recognized by ASE),
      account for them.
    - If no constraints, return 3N - 3 (removing pure translation, typical for PBC MD).
    """
    if atoms.constraints:
        from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5, axis=1)
                if mask.sum() != 1:
                    raise RuntimeError(
                        'VASP requires FixedPlane direction parallel to a cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5, axis=1)
                if mask.sum() != 1:
                    raise RuntimeError(
                        'VASP requires FixedLine direction parallel to a cell axis')
                sflags[constr.a] = ~mask
        return int((~sflags).sum())
    else:
        return len(atoms) * 3 - 3


# ---------- INCAR helpers ----------
_RE_KEYVAL = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|:)?\s*(.*?)\s*(?:[!#;].*)?$')

def read_incar(path: str) -> dict:
    """
    Read INCAR-like key=val lines into a dict of strings.
    Keeps only the last occurrence of a key.
    Ignores comments starting with !, #, or ; at end of line.
    """
    kv = {}
    if not os.path.isfile(path):
        return kv
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = _RE_KEYVAL.match(line)
            if m:
                key = m.group(1).upper()
                val = m.group(2)
                kv[key] = val
    return kv


def update_or_insert_smass(incar_path: str, smass_value: float) -> str:
    """
    Overwrite existing SMASS line (preserving trailing comment, if any) or append if missing.
    Returns 'updated' or 'inserted'.
    """
    lines = []
    if os.path.isfile(incar_path):
        with open(incar_path, 'r', errors='ignore') as f:
            lines = f.readlines()

    smass_pattern = re.compile(r'^(\s*SMASS\s*(?:=|:)?\s*)(\S*)(\s*(?:[!#;].*)?)$', re.IGNORECASE)
    written = False
    new_lines = []
    for ln in lines:
        m = smass_pattern.match(ln)
        if m and not written:
            prefix, _old, trailing = m.groups()
            new_lines.append(f"{prefix}{smass_value:.8f}{trailing}\n" if not ln.endswith("\n") else f"{prefix}{smass_value:.8f}{trailing}")
            written = True
        else:
            new_lines.append(ln)

    if written:
        with open(incar_path, 'w') as f:
            f.writelines(new_lines)
        return 'updated'

    # Append if not present
    with open(incar_path, 'a') as f:
        f.write(f"SMASS = {smass_value:.8f}\n")
    return 'inserted'


# ---------- CLI & traversal ----------
def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Recursively compute and set SMASS in INCAR from POSCAR + TEBEG.")
    ap.add_argument("root", nargs="?", default=".",
                    help="Root directory to search (default: current directory).")
    ap.add_argument("-u", "--unit", choices=["fs", "cm-1"], default="fs",
                    help="Time/frequency unit for thermostat input: 'fs' for time (default), 'cm-1' for wavenumber.")
    ap.add_argument("-f", "--frequency", type=float, default=40.0,
                    help="If -u fs: the oscillation time t0 in fs (default 40). "
                         "If -u cm-1: the frequency in cm^-1 to be converted to t0.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report, but do not write changes to INCAR.")
    return ap.parse_args(argv)


def wavenumber_to_period_fs(nu_cm1: float) -> float:
    # Your original: t0[fs] = 1000 * 33.3564095198152 / nu[cm^-1]
    THzToCm = 33.3564095198152
    return 1000.0 * THzToCm / nu_cm1


def compute_smass_for_dir(d: str, t0_fs: float):
    """
    For a given directory d, if POSCAR and INCAR+TEBEG exist, compute SMASS.
    Returns tuple (status, info_dict) where status in {"ok", "skip"}.
    info_dict includes keys for printing/reporting.
    """
    poscar = os.path.join(d, "POSCAR")
    incar  = os.path.join(d, "INCAR")
    info = {"dir": d, "reason": "", "smass": None, "ndof": None, "a1": None, "T": None}

    if not (os.path.isfile(poscar) and os.path.isfile(incar)):
        info["reason"] = "missing POSCAR/INCAR"
        return "skip", info

    # Read TEBEG
    kv = read_incar(incar)
    if "TEBEG" not in kv:
        info["reason"] = "no TEBEG in INCAR"
        return "skip", info
    try:
        T = float(kv["TEBEG"].split()[0])
    except Exception:
        info["reason"] = f"cannot parse TEBEG='{kv['TEBEG']}'"
        return "skip", info

    # Read POSCAR
    try:
        atoms = read(poscar)
    except Exception as e:
        info["reason"] = f"failed to read POSCAR: {e}"
        return "skip", info

    # Lattice a1 length and DoF
    a1_len_A = float(np.linalg.norm(atoms.cell.array, axis=1)[0])
    ndof = cnt_dof(atoms)

    # Compute SMASS
    smass = nose_mass_amuA2(T, ndof, t0_fs, a1_len_A)

    info.update({"smass": smass, "ndof": ndof, "a1": a1_len_A, "T": T})
    return "ok", info


def main():
    args = parse_args(sys.argv[1:])

    # Determine t0 in fs from user's unit/frequency
    if args.unit == "cm-1":
        if args.frequency <= 0:
            print("ERROR: frequency in cm^-1 must be > 0", file=sys.stderr)
            sys.exit(2)
        t0_fs = wavenumber_to_period_fs(args.frequency)
    else:
        if args.frequency <= 0:
            print("ERROR: t0 in fs must be > 0", file=sys.stderr)
            sys.exit(2)
        t0_fs = args.frequency

    root = os.path.abspath(args.root)

    n_total = n_written = 0
    for dirpath, dirnames, filenames in os.walk(root):
        status, info = compute_smass_for_dir(dirpath, t0_fs)
        if status != "ok":
            # Uncomment if you want to see why each dir is skipped:
            # print(f"[skip] {dirpath} — {info['reason']}")
            continue

        n_total += 1
        smass = info["smass"]
        T     = info["T"]
        ndof  = info["ndof"]
        a1    = info["a1"]

        if args.dry_run:
            print(f"[dry-run] {dirpath}\n"
                  f"  TEBEG={T:g} K, DoF={ndof}, |a1|={a1:.6f} Å, SMASS={smass:.8f} (amu·Å²)")
        else:
            action = update_or_insert_smass(os.path.join(dirpath, "INCAR"), smass)
            n_written += 1
            print(f"[{action}] {dirpath}\n"
                  f"  wrote SMASS={smass:.8f} (amu·Å²)  "
                  f"(TEBEG={T:g} K, DoF={ndof}, |a1|={a1:.6f} Å)")

    if not args.dry_run:
        print(f"\nDone. Updated/inserted SMASS in {n_written} directories "
              f"(visited {n_total} eligible folders under {root}).")
    else:
        print(f"\nDry run complete. Would update {n_total} directories under {root}.")


if __name__ == "__main__":
    main()
