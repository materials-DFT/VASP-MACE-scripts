#!/usr/bin/env python3
"""
VASP NPT Molecular Dynamics Parameter Optimizer
===============================================

Generates sensible NPT (isothermal–isobaric) AIMD parameters for VASP with a
physically motivated Langevin thermostat setup that respects atomic masses and temperature.

Highlights
----------
1. Langevin gamma per species:
   gamma_i(T) = gamma_ref * (m_ref / m_i)^alpha * (T/300)^beta
   - Ensures lighter atoms are more strongly damped than heavier ones.
   - Clamps to [gamma_min, gamma_max] and can optionally smooth to enforce
     monotonicity vs mass (useful if rounding/clamping creates inversions).

2. Safe POTIM heuristic that decreases with temperature and with the lightest mass.

3. PMASS and LANGEVIN_GAMMA_L based on system size, with conservative bounds.

4. Preserves POSCAR order for LANGEVIN_GAMMA (critical for VASP).

Author: AI Assistant
Updated: 2025-10-30
"""

import os
import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------- Data Classes ---------------------------- #

@dataclass
class AtomicSpecies:
    symbol: str
    atomic_number: int
    atomic_mass: float
    count: int

@dataclass
class NPTParameters:
    # Core MD parameters
    ibrion: int = 0
    mdalgo: int = 3     # Langevin
    isif: int = 3
    nsw: int = 10000
    potim: float = 1.0  # fs

    # Temperature control
    tebeg: Optional[float] = None
    teend: Optional[float] = None

    # Langevin thermostat
    langevin_gamma: List[float] = None      # per species, POSCAR order
    langevin_gamma_l: float = 5.0           # lattice friction

    # Barostat
    pmass: float = 1000.0

    # Other
    ediffg: float = -5e-2
    isym: int = 0

# ---------------------------- Optimizer ---------------------------- #

class VASPNPTOptimizer:
    """
    VASP NPT parameter optimizer with mass/temperature-aware Langevin gammas,
    safe POTIM heuristics, and sensible PMASS/LANGEVIN_GAMMA_L.
    """

    # Atomic data: Z, mass (IUPAC), no gamma baked in
    ATOMIC_DATA: Dict[str, Dict[str, float]] = {
        'H':  {'Z': 1,  'mass': 1.008},
        'He': {'Z': 2,  'mass': 4.0026},
        'Li': {'Z': 3,  'mass': 6.94},
        'Be': {'Z': 4,  'mass': 9.0122},
        'B':  {'Z': 5,  'mass': 10.81},
        'C':  {'Z': 6,  'mass': 12.011},
        'N':  {'Z': 7,  'mass': 14.007},
        'O':  {'Z': 8,  'mass': 15.999},
        'F':  {'Z': 9,  'mass': 18.998},
        'Ne': {'Z': 10, 'mass': 20.180},
        'Na': {'Z': 11, 'mass': 22.990},
        'Mg': {'Z': 12, 'mass': 24.305},
        'Al': {'Z': 13, 'mass': 26.982},
        'Si': {'Z': 14, 'mass': 28.085},
        'P':  {'Z': 15, 'mass': 30.974},
        'S':  {'Z': 16, 'mass': 32.06},
        'Cl': {'Z': 17, 'mass': 35.45},
        'Ar': {'Z': 18, 'mass': 39.948},
        'K':  {'Z': 19, 'mass': 39.0983},
        'Ca': {'Z': 20, 'mass': 40.078},
        'Sc': {'Z': 21, 'mass': 44.9559},
        'Ti': {'Z': 22, 'mass': 47.867},
        'V':  {'Z': 23, 'mass': 50.9415},
        'Cr': {'Z': 24, 'mass': 51.9961},
        'Mn': {'Z': 25, 'mass': 54.938},
        'Fe': {'Z': 26, 'mass': 55.845},
        'Co': {'Z': 27, 'mass': 58.933},
        'Ni': {'Z': 28, 'mass': 58.693},
        'Cu': {'Z': 29, 'mass': 63.546},
        'Zn': {'Z': 30, 'mass': 65.38},
        'Ga': {'Z': 31, 'mass': 69.723},
        'Ge': {'Z': 32, 'mass': 72.63},
        'As': {'Z': 33, 'mass': 74.9216},
        'Se': {'Z': 34, 'mass': 78.971},
        'Br': {'Z': 35, 'mass': 79.904},
        'Kr': {'Z': 36, 'mass': 83.798},
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ---------------------------- IO Helpers ---------------------------- #

    def log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            print(f"[{level}] {msg}")

    def read_poscar(self, poscar_path: str) -> Tuple[List[AtomicSpecies], np.ndarray]:
        """
        POSCAR (VASP5):
          L1: comment
          L2: scale
          L3-5: lattice vectors
          L6: species symbols
          L7: counts
          (Selective dynamics line optional)
        """
        with open(poscar_path, 'r') as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]
        if len(lines) < 7:
            raise ValueError("POSCAR is too short / malformed.")

        lattice = np.array([
            [float(x) for x in lines[2].split()],
            [float(x) for x in lines[3].split()],
            [float(x) for x in lines[4].split()]
        ], dtype=float)

        species_symbols = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        if len(species_symbols) != len(counts):
            raise ValueError("Mismatch between species symbols and counts in POSCAR.")

        species_list: List[AtomicSpecies] = []
        for sym, cnt in zip(species_symbols, counts):
            if sym not in self.ATOMIC_DATA:
                self.log(f"Unknown species '{sym}' – using fallback mass=20, Z=0", "WARN")
                Z = 0; mass = 20.0
            else:
                Z = int(self.ATOMIC_DATA[sym]['Z'])
                mass = float(self.ATOMIC_DATA[sym]['mass'])
            species_list.append(AtomicSpecies(symbol=sym, atomic_number=Z, atomic_mass=mass, count=cnt))

        return species_list, lattice

    def read_incar(self, incar_path: str) -> Dict[str, Any]:
        """
        Very loose INCAR reader:
        - Keeps raw strings for complex/list values
        - Parses bare ints/floats when the RHS is a single token
        """
        params: Dict[str, Any] = {}
        if not os.path.exists(incar_path):
            return params
        with open(incar_path, 'r') as f:
            for ln in f:
                line = ln.strip()
                if not line or line.startswith(("#", "!")):
                    continue
                if '=' not in line:
                    continue
                key, val = [x.strip() for x in line.split('=', 1)]
                # If multi-token, keep string; else try int/float
                tokens = val.split()
                if len(tokens) == 1:
                    v = tokens[0]
                    try:
                        if any(c in v for c in ('.', 'E', 'e')):
                            params[key] = float(v)
                        else:
                            params[key] = int(v)
                    except ValueError:
                        # keep as string (e.g., 'Normal', 'Fast', 'None', 'T', 'F')
                        params[key] = v
                else:
                    params[key] = val
        return params

    # ---------------------------- Physics Heuristics ---------------------------- #

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _compute_langevin_gammas(
        self,
        species_list: List[AtomicSpecies],
        temperature: float,
        gamma_ref: float = 2.0,      # ps^-1 at 300 K for m_ref
        alpha: float = 0.5,          # mass scaling exponent (0.5–1 recommended)
        beta: float = 0.5,           # temperature scaling exponent (0–1)
        gamma_min: float = 0.5,      # ps^-1
        gamma_max: float = 20.0,     # ps^-1
        enforce_monotone: bool = True
    ) -> List[float]:
        """
        Compute per-species γ that (i) increases for lighter atoms and (ii) grows with T.
        Uses m_ref as composition-weighted average mass so mixed systems scale naturally.
        """
        total_atoms = sum(s.count for s in species_list)
        avg_mass = sum(s.atomic_mass * s.count for s in species_list) / max(1, total_atoms)
        m_ref = avg_mass  # composition-aware reference mass

        temp_factor = (max(1e-6, temperature) / 300.0) ** beta

        # Raw gammas, POSCAR order
        raw = []
        for s in species_list:
            g = gamma_ref * (m_ref / s.atomic_mass) ** alpha * temp_factor
            g = self._clamp(g, gamma_min, gamma_max)
            raw.append(g)

        if not enforce_monotone:
            return raw

        # Enforce monotonic decrease with mass: if mass_i > mass_j then gamma_i <= gamma_j
        # Simple single pass smoothing in POSCAR order, using neighboring mass order
        masses = [s.atomic_mass for s in species_list]
        gammas = raw[:]
        # Bubble down any inversions
        for _ in range(len(gammas)):
            changed = False
            for i in range(1, len(gammas)):
                # if mass increases but gamma increases too, fix by averaging
                if masses[i] > masses[i-1] and gammas[i] > gammas[i-1]:
                    new_val = 0.5 * (gammas[i] + gammas[i-1])
                    gammas[i] = min(gammas[i-1], new_val)
                    changed = True
            if not changed:
                break
        # Final clamp
        gammas = [self._clamp(g, gamma_min, gamma_max) for g in gammas]
        return gammas

    def _compute_potim(
        self,
        species_list: List[AtomicSpecies],
        temperature: float,
        base_fs: float = 1.0,
        min_fs: float = 0.2,
        max_fs: float = 2.0
    ) -> float:
        """
        Conservative POTIM heuristic:
          - Decrease with temperature: ~ (300/T)^0.5
          - Decrease with lightest mass: ~ sqrt(m_min / 16) so systems with O/H go smaller
        """
        m_min = min(s.atomic_mass for s in species_list)
        t_scale = (300.0 / max(1e-6, temperature)) ** 0.5
        m_scale = np.sqrt(m_min / 16.0)   # 16 ~ O; if H present, this further reduces
        potim = base_fs * t_scale * m_scale
        return self._clamp(potim, min_fs, max_fs)

    def _compute_pmass(
        self,
        natoms: int,
        temperature: float
    ) -> float:
        """
        PMASS heuristic (psuedo-piston mass, VASP units):
          - Scale with system size and temperature.
          - Typical safe range for solids: 1000 – 5000.
        """
        size_factor = self._clamp(natoms / 50.0, 0.5, 3.0)
        t_factor = self._clamp(temperature / 300.0, 0.5, 2.0)
        pmass = 1000.0 * size_factor * t_factor
        return float(self._clamp(pmass, 800.0, 5000.0))

    def _compute_gamma_l(
        self,
        natoms: int
    ) -> float:
        """
        Lattice friction (LANGEVIN_GAMMA_L) heuristic:
          - Small systems get lower values; large systems a bit higher.
          - Keep modest to avoid strangling cell dynamics.
        """
        size_factor = self._clamp(natoms / 50.0, 0.5, 3.0)
        gamma_l = 3.0 * size_factor   # around 1.5 – 9.0
        return float(self._clamp(gamma_l, 1.0, 10.0))

    # ---------------------------- Core API ---------------------------- #

    def calculate_optimal_parameters(
        self,
        species_list: List[AtomicSpecies],
        temperature: float,
        lattice: np.ndarray,
        existing_params: Dict[str, Any],
        gamma_ref: float = 2.0,
        alpha: float = 0.6,
        beta: float = 0.5,
        gamma_min: float = 0.5,
        gamma_max: float = 15.0,
        enforce_monotone: bool = True
    ) -> NPTParameters:
        """
        Compute optimized parameters. The defaults aim for **realistic dynamics**,
        not hyper-aggressive damping. Increase gamma_ref or beta for faster thermalization.
        """
        total_atoms = sum(s.count for s in species_list)

        # POTIM
        potim = self._compute_potim(species_list, temperature)

        # Langevin gammas per species (POSCAR order)
        langevin_gamma = self._compute_langevin_gammas(
            species_list=species_list,
            temperature=temperature,
            gamma_ref=gamma_ref,
            alpha=alpha,
            beta=beta,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            enforce_monotone=enforce_monotone
        )

        # PMASS and lattice gamma
        pmass = self._compute_pmass(total_atoms, temperature)
        gamma_l = self._compute_gamma_l(total_atoms)

        # NSW heuristic: give enough steps to equilibrate at size/temperature
        size_factor = self._clamp(total_atoms / 50.0, 0.5, 3.0)
        nsw = int(self._clamp(10000 * size_factor, 5000, 50000))

        npt = NPTParameters(
            ibrion=0,
            mdalgo=3,
            isif=3,
            nsw=nsw,
            potim=potim,
            tebeg=existing_params.get('TEBEG', temperature),
            teend=existing_params.get('TEEND', temperature),
            langevin_gamma=langevin_gamma,
            langevin_gamma_l=gamma_l,
            pmass=pmass,
            ediffg=float(existing_params.get('EDIFFG', -5e-2)),
            isym=int(existing_params.get('ISYM', 0))
        )
        return npt

    # ---------------------------- Write INCAR ---------------------------- #

    def write_incar(
        self,
        incar_path: str,
        npt_params: NPTParameters,
        existing_params: Dict[str, Any],
        backup: bool = False
    ):
        if backup and os.path.exists(incar_path):
            os.replace(incar_path, incar_path + ".backup")
            self.log(f"Backup created at {incar_path}.backup")

        # Merge
        all_params = dict(existing_params)
        all_params.update({
            'IBRION': npt_params.ibrion,
            'MDALGO': npt_params.mdalgo,
            'ISIF': npt_params.isif,
            'NSW': npt_params.nsw,
            'POTIM': f"{npt_params.potim:.3f}",
            'LANGEVIN_GAMMA': ' '.join(f"{g:.3f}" for g in npt_params.langevin_gamma),
            'LANGEVIN_GAMMA_L': f"{npt_params.langevin_gamma_l:.3f}",
            'PMASS': f"{npt_params.pmass:.0f}",
            'EDIFFG': npt_params.ediffg,
            'ISYM': npt_params.isym
        })
        if npt_params.tebeg is not None:
            all_params['TEBEG'] = npt_params.tebeg
        if npt_params.teend is not None:
            all_params['TEEND'] = npt_params.teend

        # Write
        with open(incar_path, 'w') as f:
            f.write("System = Optimized NPT Molecular Dynamics\n")
            f.write("# File generated by VASPNPTOptimizer (mass- and T-aware Langevin)\n\n")

            # Starting
            start_keys = ['ISTART', 'ICHARG']
            has_start = any(k in all_params for k in start_keys)
            if has_start:
                f.write("Starting parameters:\n")
                for k in start_keys:
                    if k in all_params:
                        f.write(f"{k} = {all_params[k]}\n")
                f.write("\n")

            # Electronic
            f.write("Electronic Relaxation:\n")
            for k in ['PREC', 'ENCUT', 'NELMIN', 'NELM', 'EDIFF', 'LREAL',
                      'ISPIN', 'MAGMOM', 'ALGO', 'METAGGA', 'LMIXTAU', 'LASPH', 'LDIAG',
                      'ISMEAR', 'SIGMA', 'LORBIT']:
                if k in all_params:
                    f.write(f"{k} = {all_params[k]}\n")
            f.write("\n")

            # MD Block
            f.write("Ionic Molecular Dynamics (NPT, Langevin):\n")
            for k in ['NSW', 'IBRION', 'EDIFFG', 'ISIF', 'POTIM', 'ISYM',
                      'MDALGO', 'LANGEVIN_GAMMA', 'LANGEVIN_GAMMA_L', 'PMASS']:
                if k in all_params:
                    f.write(f"{k} = {all_params[k]}\n")
            f.write("\n")

            # Temperature
            if ('TEBEG' in all_params) or ('TEEND' in all_params):
                f.write("Temperature Control:\n")
                if 'TEBEG' in all_params: f.write(f"TEBEG = {all_params['TEBEG']}\n")
                if 'TEEND' in all_params: f.write(f"TEEND = {all_params['TEEND']}\n")
                f.write("\n")

            # Parallel/IO if present
            for header, keys in [
                ("Parallelization:", ['NCORE', 'NSIM', 'KPAR', 'LPLANE', 'LSCALU']),
                ("IO Control:", ['LWAVE', 'LCHARG'])
            ]:
                if any(k in all_params for k in keys):
                    f.write(f"{header}\n")
                    for k in keys:
                        if k in all_params:
                            f.write(f"{k} = {all_params[k]}\n")
                    f.write("\n")

    # ---------------------------- Directory Processing ---------------------------- #

    def process_directory(self, directory: str, backup: bool = False, dry_run: bool = False,
                          gamma_ref: float = 2.0, alpha: float = 0.6, beta: float = 0.5,
                          gamma_min: float = 0.5, gamma_max: float = 15.0,
                          enforce_monotone: bool = True) -> Dict[str, Any]:
        res = {
            'processed': False,
            'poscar_found': False,
            'incar_found': False,
            'temperature_set': False,
            'species_count': 0,
            'total_atoms': 0,
            'optimized_params': None,
            'errors': []
        }

        poscar = os.path.join(directory, "POSCAR")
        incar = os.path.join(directory, "INCAR")

        if not os.path.exists(poscar):
            res['errors'].append("POSCAR not found")
            return res
        res['poscar_found'] = True

        if not os.path.exists(incar):
            res['errors'].append("INCAR not found")
            return res
        res['incar_found'] = True

        try:
            species_list, lattice = self.read_poscar(poscar)
            res['species_count'] = len(species_list)
            res['total_atoms'] = sum(s.count for s in species_list)

            existing = self.read_incar(incar)
            if 'TEBEG' not in existing:
                res['errors'].append("TEBEG not set in INCAR")
                return res
            temperature = float(existing['TEBEG'])
            res['temperature_set'] = True

            npt = self.calculate_optimal_parameters(
                species_list, temperature, lattice, existing,
                gamma_ref=gamma_ref, alpha=alpha, beta=beta,
                gamma_min=gamma_min, gamma_max=gamma_max,
                enforce_monotone=enforce_monotone
            )
            res['optimized_params'] = npt

            if not dry_run:
                self.write_incar(incar, npt, existing, backup)
            res['processed'] = True

        except Exception as e:
            res['errors'].append(str(e))

        return res

    def process_recursive(self, root: str, backup: bool = False, dry_run: bool = False,
                          gamma_ref: float = 2.0, alpha: float = 0.6, beta: float = 0.5,
                          gamma_min: float = 0.5, gamma_max: float = 15.0,
                          enforce_monotone: bool = True) -> Dict[str, Any]:
        self.log(f"Scanning: {root}")
        if dry_run:
            self.log("DRY RUN: no files will be modified")

        total = {
            'directories_processed': 0,
            'directories_skipped': 0,
            'total_species': 0,
            'total_atoms': 0,
            'errors': [],
            'directory_results': {}
        }

        for r, _, files in os.walk(root):
            if 'POSCAR' in files and 'INCAR' in files:
                self.log(f"Processing {r}")
                result = self.process_directory(
                    r, backup, dry_run,
                    gamma_ref, alpha, beta, gamma_min, gamma_max, enforce_monotone
                )
                total['directory_results'][r] = result
                if result['processed']:
                    total['directories_processed'] += 1
                    total['total_species'] += result['species_count']
                    total['total_atoms'] += result['total_atoms']
                    self.log(f"✓ {r}")
                else:
                    total['directories_skipped'] += 1
                    errs = "; ".join(result['errors']) if result['errors'] else "unknown error"
                    self.log(f"✗ {r} - {errs}", "WARN")
                    for e in result['errors']:
                        total['errors'].append(f"{r}: {e}")
        return total

    @staticmethod
    def print_summary(results: Dict[str, Any]):
        print("\n" + "="*80)
        print("VASP NPT OPTIMIZER - SUMMARY")
        print("="*80)
        print(f"Directories processed: {results['directories_processed']}")
        print(f"Directories skipped  : {results['directories_skipped']}")
        print(f"Total species        : {results['total_species']}")
        print(f"Total atoms          : {results['total_atoms']}")
        if results['errors']:
            print("\nErrors:")
            for e in results['errors'][:12]:
                print(f"  - {e}")
            if len(results['errors']) > 12:
                print(f"  ... plus {len(results['errors'])-12} more")
        print("="*80)

# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Optimize VASP NPT (Langevin) parameters with mass- and T-aware gammas."
    )
    ap.add_argument("directory", help="Root directory to scan recursively.")
    ap.add_argument("--dry-run", action="store_true", help="Analyze only, do not modify INCAR.")
    ap.add_argument("--backup", action="store_true", help="Backup existing INCAR as INCAR.backup.")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging.")

    # Expert knobs (defaults chosen for realistic dynamics)
    ap.add_argument("--gamma-ref", type=float, default=2.0,
                    help="Reference gamma (ps^-1) at 300 K for m_ref (avg mass). Default 2.0")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Mass exponent: gamma ~ (m_ref/m)^alpha. Default 0.6")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Temperature exponent: gamma ~ (T/300)^beta. Default 0.5")
    ap.add_argument("--gamma-min", type=float, default=0.5,
                    help="Lower clamp for species gamma. Default 0.5")
    ap.add_argument("--gamma-max", type=float, default=15.0,
                    help="Upper clamp for species gamma. Default 15.0")
    ap.add_argument("--no-monotone", action="store_true",
                    help="Do not enforce gamma decreasing with mass.")

    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: directory '{args.directory}' not found.")
        sys.exit(1)

    opt = VASPNPTOptimizer(verbose=not args.quiet)
    try:
        results = opt.process_recursive(
            args.directory,
            backup=args.backup,
            dry_run=args.dry_run,
            gamma_ref=args.gamma_ref,
            alpha=args.alpha,
            beta=args.beta,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            enforce_monotone=(not args.no_monotone)
        )
        opt.print_summary(results)
        # Non-fatal even with errors if dry-run; fail otherwise if any directory failed after attempting write
        if results['errors'] and not args.dry_run:
            sys.exit(2)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
