#!/usr/bin/env python3
"""
Recursively update INCAR files with MAGMOM (for K, Mn, O) and NELECT,
using atom counts from POSCAR and ZVALs parsed from POTCAR.

Usage:
    python set_magmom_nelect.py /path/to/top/dir

Notes:
- MAGMOM is written as:   MAGMOM = {nK}*0.0 {nMn}*3.0 {nO}*0.0
- NELECT is computed as:  NELECT = sum(n_i * ZVAL_i) - nK
- Expects standard VASP POTCAR blocks that include 'TITEL' and 'ZVAL' lines.
- Leaves other INCAR content untouched.
"""

import argparse
import os
import re
import sys
from typing import Dict, Tuple

try:
    from pymatgen.core import Structure
except Exception:
    Structure = None


# Edit here if you want different per-species target moments:
MOMENTS = {"K": 0.0, "Mn": 3.0, "O": 0.0}

# Filenames to look for (change if yours differ)
POSCAR_NAME = "POSCAR"
POTCAR_NAME = "POTCAR"
INCAR_NAME  = "INCAR"


def parse_args():
    ap = argparse.ArgumentParser(description="Update INCAR MAGMOM and NELECT per subdirectory.")
    ap.add_argument("topdir", help="Top-level directory to search recursively.")
    ap.add_argument("--dry-run", action="store_true", help="Show planned changes without modifying files.")
    return ap.parse_args()


def find_element_symbol_from_titel_line(line: str) -> str:
    """
    Extract an element symbol from a POTCAR 'TITEL' line.
    Example lines:
      'TITEL  = PAW_PBE K 06Sep2000'
      'TITEL  = PAW_PBE Mn_pv 06Sep2000'
      'TITEL  = PAW_PBE O 08Apr2002'
    Heuristic: first token matching pattern [A-Z][a-z]? (optionally with suffixes like '_pv')
    We'll strip any suffix after an underscore.
    """
    tokens = line.strip().split()
    # skip the "TITEL" "=" and functional tokens; just find the first plausible element symbol
    for tok in tokens:
        # candidate like "Mn_pv" -> "Mn"
        base = tok.split("_", 1)[0]
        if re.fullmatch(r"[A-Z][a-z]?", base):
            return base
    raise ValueError(f"Could not extract element symbol from TITEL line: {line.strip()}")


def parse_potcar_zvals(potcar_path: str) -> Dict[str, float]:
    """
    Parse POTCAR to map element symbol -> ZVAL (float).
    Handles concatenated POTCARs by scanning TITEL and subsequent ZVAL lines.
    """
    zvals: Dict[str, float] = {}
    current_el = None
    with open(potcar_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip().startswith("TITEL"):
                try:
                    current_el = find_element_symbol_from_titel_line(line)
                except ValueError:
                    current_el = None
            elif "ZVAL" in line and current_el:
                # lines can look like: "   ZVAL   =  9.000    mass and valence"
                m = re.search(r"ZVAL\s*=\s*([0-9]+(?:\.[0-9]*)?)", line)
                if m:
                    zvals[current_el] = float(m.group(1))
                current_el = None  # reset to avoid accidental carryover
    if not zvals:
        raise RuntimeError(f"No ZVAL entries found in {potcar_path}")
    return zvals

def counts_from_poscar(poscar_path: str) -> Dict[str, int]:
    """
    Get per-element counts from POSCAR. Uses pymatgen if available.
    Falls back to a minimal parser that relies on the element line (VASP 5+ format).
    """
    if Structure is not None:
        s = Structure.from_file(poscar_path)
        # get_el_amt_dict returns {str: float} in current pymatgen
        raw = s.composition.get_el_amt_dict()
        counts: Dict[str, int] = {}
        for k, amount in raw.items():
            # k may be a str (usual) or an Element (older code paths)
            sym = k.symbol if hasattr(k, "symbol") else str(k)
            counts[sym] = int(round(amount))
        return counts

    # --- fallback parser for POSCAR v5+ ---
    with open(poscar_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) < 8:
        raise RuntimeError(f"POSCAR seems too short: {poscar_path}")

    symbols_line = lines[5].split()
    numbers_line = lines[6].split()
    if not symbols_line or not numbers_line or len(symbols_line) != len(numbers_line):
        raise RuntimeError(f"Cannot parse element symbols/counts in {poscar_path}")

    counts = {}
    for sym, num in zip(symbols_line, numbers_line):
        counts[sym] = counts.get(sym, 0) + int(num)
    return counts

def build_magmom_string(counts: Dict[str, int]) -> str:
    """
    Build MAGMOM string in the order K, Mn, O with the moments from MOMENTS.
    Missing species are treated as zero count.
    """
    nK  = counts.get("K", 0)
    nMn = counts.get("Mn", 0)
    nO  = counts.get("O", 0)

    parts = []
    if nK  > 0: parts.append(f"{nK}*{MOMENTS['K']}")
    else:       parts.append(f"0*{MOMENTS['K']}")
    if nMn > 0: parts.append(f"{nMn}*{MOMENTS['Mn']}")
    else:       parts.append(f"0*{MOMENTS['Mn']}")
    if nO  > 0: parts.append(f"{nO}*{MOMENTS['O']}")
    else:       parts.append(f"0*{MOMENTS['O']}")

    return " ".join(parts), (nK, nMn, nO)


def compute_nelect(counts: Dict[str, int], zvals: Dict[str, float]) -> float:
    """
    NELECT = sum(n_i * ZVAL_i) - n_K
    """
    total = 0.0
    for el, n in counts.items():
        if el not in zvals:
            raise RuntimeError(f"Element {el} not found in POTCAR ZVALs.")
        total += n * zvals[el]
    total -= counts.get("K", 0)
    return total


def replace_or_append_incar_param(incar_text: str, key: str, value: str) -> Tuple[str, bool]:
    """
    Replace a line beginning with 'key' (case-insensitive, ignoring leading spaces)
    with 'key = value'. If not found, append at the end.
    Returns (new_text, replaced_flag).
    """
    pattern = re.compile(rf"^[ \t]*{re.escape(key)}\s*=.*$", re.IGNORECASE | re.MULTILINE)
    replacement_line = f"{key} = {value}"
    if pattern.search(incar_text):
        new_text = pattern.sub(replacement_line, incar_text, count=1)
        return new_text, True
    else:
        # Append with a newline if needed
        sep = "" if incar_text.endswith("\n") else "\n"
        new_text = incar_text + f"{sep}{replacement_line}\n"
        return new_text, False


def process_directory(dirpath: str, dry_run: bool = False):
    poscar = os.path.join(dirpath, POSCAR_NAME)
    potcar = os.path.join(dirpath, POTCAR_NAME)
    incar  = os.path.join(dirpath, INCAR_NAME)

    if not os.path.isfile(poscar):
        return
    if not os.path.isfile(incar):
        print(f"[skip] No INCAR in: {dirpath}")
        return
    if not os.path.isfile(potcar):
        print(f"[skip] No POTCAR in: {dirpath}")
        return

    try:
        counts = counts_from_poscar(poscar)
        magmom_str, (nK, nMn, nO) = build_magmom_string(counts)
        zvals = parse_potcar_zvals(potcar)
        nelect = compute_nelect(counts, zvals)
    except Exception as e:
        print(f"[error] {dirpath}: {e}")
        return

    # Read/modify INCAR
    try:
        with open(incar, "r", errors="ignore") as f:
            incar_text = f.read()
    except Exception as e:
        print(f"[error] Cannot read INCAR in {dirpath}: {e}")
        return

    new_text, repl_mag = replace_or_append_incar_param(incar_text, "MAGMOM", magmom_str)
    new_text, repl_ne  = replace_or_append_incar_param(new_text, "NELECT", f"{nelect:.6f}")

    action = "UPDATED" if (new_text != incar_text) else "UNCHANGED"
    print(f"[{action}] {dirpath}")
    print(f"  Counts: K={nK}, Mn={nMn}, O={nO}")
    # Show ZVALs used for K, Mn, O if present
    z_k  = zvals.get("K", None)
    z_mn = zvals.get("Mn", None)
    z_o  = zvals.get("O", None)
    z_str = ", ".join([f"K:{z_k}" if z_k is not None else "K:NA",
                       f"Mn:{z_mn}" if z_mn is not None else "Mn:NA",
                       f"O:{z_o}" if z_o is not None else "O:NA"])
    print(f"  ZVALs: {z_str}")
    print(f"  MAGMOM -> MAGMOM = {magmom_str}  ({'replaced' if repl_mag else 'appended'})")
    print(f"  NELECT -> NELECT = {nelect:.6f}   ({'replaced' if repl_ne else 'appended'})")

    if not dry_run and new_text != incar_text:
        try:
            with open(incar, "w") as f:
                f.write(new_text)
        except Exception as e:
            print(f"[error] Cannot write INCAR in {dirpath}: {e}")


def main():
    args = parse_args()
    top = os.path.abspath(args.topdir)
    if not os.path.isdir(top):
        print(f"Not a directory: {top}", file=sys.stderr)
        sys.exit(1)

    print(f"[scan] Walking: {top}")
    for root, dirs, files in os.walk(top):
        if POSCAR_NAME in files:
            process_directory(root, dry_run=args.dry_run)

    print("[done]")


if __name__ == "__main__":
    main()
