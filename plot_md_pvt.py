#!/usr/bin/env python3
"""
VASP OUTCAR quick plotter: Temperature, Volume, and Pressure vs Step.

- Python 3.13.5
- Opens an interactive Matplotlib window over X11 if possible (no files written).
- Auto-hardens against black-window issues by preferring TkAgg and setting a few
  safe env vars for remote X11 (QT_X11_NO_MITSHM, LIBGL_ALWAYS_INDIRECT).
- Clean ASCII fallback with --ascii or if GUI init fails.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

# -------------------------- Regex patterns --------------------------
FNUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

# Temperature (instantaneous)
T_PATTERNS = (
    re.compile(r"\bT\s*=\s*(%s)\b" % FNUM, re.IGNORECASE),                   # ... T= 302.4
    re.compile(r"\btemperature[^=\n]*=\s*(%s)\s*K\b" % FNUM, re.IGNORECASE), # temperature = 500 K
    re.compile(r"\btemperature.*?(%s)\s*K\b" % FNUM, re.IGNORECASE),         # ... temperature ... 500 K
)
T_SKIP = tuple(t.lower() for t in ("TEBEG", "TEIN", "SMASS", "TAV", "TCHAIN", "POTIM"))

# Volume
VOL_RE = re.compile(r"volume of cell\s*:\s*(%s)\b" % FNUM, re.IGNORECASE)

# Pressure
P_RE = re.compile(r"\btotal\s+pressure\b\s*=\s*(%s)\s*kB\b" % FNUM, re.IGNORECASE)


# -------------------------- Parsing (single pass) --------------------------
def parse_tvp(outcar: Path) -> tuple[list[float], list[float], list[float]]:
    temps: list[float] = []
    vols:  list[float] = []
    pres:  list[float] = []
    try:
        with outcar.open("r", errors="ignore") as f:
            for line in f:
                low = line.lower()

                # Temperature
                if "t=" in low:
                    m = T_PATTERNS[0].search(line)
                    if m:
                        try:
                            temps.append(float(m.group(1)))
                        except ValueError:
                            pass
                if "temperature" in low and not any(tok in low for tok in T_SKIP):
                    m = T_PATTERNS[1].search(line) or T_PATTERNS[2].search(line)
                    if m:
                        try:
                            temps.append(float(m.group(1)))
                        except ValueError:
                            pass

                # Volume
                mv = VOL_RE.search(line)
                if mv:
                    try:
                        vols.append(float(mv.group(1)))
                    except ValueError:
                        pass

                # Pressure
                mp = P_RE.search(line)
                if mp:
                    try:
                        pres.append(float(mp.group(1)))
                    except ValueError:
                        pass
    except FileNotFoundError:
        print("OUTCAR not found in current directory (or at given path).")
        sys.exit(1)

    return temps, vols, pres


# -------------------------- ASCII plotting --------------------------
def ascii_plot(y: list[float], *, width: int = 90, height: int = 16,
               title: str = "", y_label: str = "") -> None:
    if not y:
        print(f"\n== {title} ==\n[no data]")
        return

    try:
        term_w = shutil.get_terminal_size((width, 24)).columns
        width = max(40, min(term_w - 4, width))
    except Exception:
        pass

    n = len(y)
    if n > width:
        step = n / float(width)
        idxs = [int(i * step) for i in range(width)]
        data = [y[i] for i in idxs]
    else:
        data = y
        width = n

    ymin, ymax = min(data), max(data)
    yr = ymax - ymin if ymax > ymin else 1.0
    scale = (height - 1) / yr
    rows = [[" "] * width for _ in range(height)]

    print(f"\n== {title} ==")
    if y_label:
        print(y_label)
    print(f"max: {ymax:.6f} | min: {ymin:.6f}")

    for x, v in enumerate(data):
        r = height - 1 - int(round((v - ymin) * scale))
        r = max(0, min(height - 1, r))
        rows[r][x] = "█"

    for r in rows:
        print("".join(r))

    avg = sum(y) / len(y)
    print(f"Samples: {len(y)} | Avg: {avg:.6f} | Min: {ymin:.6f} | Max: {ymax:.6f}")


# -------------------------- GUI setup & plotting --------------------------
def init_matplotlib_backend(preferred: str | None = None) -> str:
    """
    Initialize a robust backend for remote X11 use.
    Preference order:
      1) explicit 'preferred' if provided (e.g., 'TkAgg', 'Qt5Agg')
      2) TkAgg (very reliable over XQuartz/X11)
      3) Qt5Agg (with env hardening)
    Returns the backend name actually in use. Raises on failure.
    """
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY not set")

    # Harden environment against black X11 windows (Qt/GL over SSH).
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")       # avoid MIT-SHM
    os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "1")  # force indirect GL
    os.environ.setdefault("LIBGL_DRI3_DISABLE", "1")     # dodge DRI3 issues

    import matplotlib  # type: ignore

    tried: list[str] = []
    candidates: list[str] = []

    if preferred:
        candidates.append(preferred)
    candidates.extend(b for b in ("TkAgg", "Qt5Agg") if b not in candidates)

    last_err: Exception | None = None
    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as _  # noqa
            return backend
        except Exception as e:  # keep trying
            last_err = e
            tried.append(backend)

    raise RuntimeError(f"Failed to init any GUI backend (tried {tried}). Last error: {last_err}")

def gui_plot(series: list[tuple[str, list[float]]], backend: str | None = None) -> None:
    backend_used = init_matplotlib_backend(backend)
    import matplotlib.pyplot as plt  # type: ignore

    series = [(label, y) for (label, y) in series if y]
    if not series:
        raise RuntimeError("No data to plot")

    nrows = len(series)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 2.8 * nrows), sharex=False)
    if nrows == 1:
        axes = [axes]  # normalize

    for ax, (label, y) in zip(axes, series):
        x = list(range(1, len(y) + 1))
        ax.plot(x, y, linewidth=1.5, label=label.split(" (", 1)[0])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"VASP MD: Temperature / Volume / Pressure vs Step\n[{backend_used}]", y=0.98, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# -------------------------- CLI --------------------------
def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Plot Temperature, Volume, and Pressure from VASP OUTCAR (X11 GUI if available; ASCII otherwise)."
    )
    p.add_argument("outcar", nargs="?", default="OUTCAR", help="Path to OUTCAR (default: ./OUTCAR)")
    p.add_argument("--ascii", action="store_true", help="Force ASCII output (no GUI).")
    p.add_argument("--width", type=int, default=90, help="ASCII width (columns)")
    p.add_argument("--height", type=int, default=16, help="ASCII height (rows) per plot")
    p.add_argument("--backend", choices=["TkAgg", "Qt5Agg"], help="Force a specific Matplotlib GUI backend.")
    args = p.parse_args(list(argv) if argv is not None else None)

    outcar = Path(args.outcar)
    temps, vols, pres = parse_tvp(outcar)
    if not any((temps, vols, pres)):
        print("No temperature, volume, or pressure data found in OUTCAR.")
        sys.exit(1)

    if not args.ascii:
        try:
            gui_plot([
                ("Temperature (K)", temps),
                ("Volume (Å³)",   vols),
                ("Pressure (kB)", pres),
            ], backend=args.backend)
            return
        except Exception as e:
            print(f"GUI plotting not available ({e}). Falling back to ASCII.\n"
                  f"Tip: use ssh -Y, and ensure Tk/Qt are installed on the remote Python.")

    # ASCII fallback (or forced)
    ascii_plot(temps, width=args.width, height=args.height, title="Temperature vs Step", y_label="Temperature (K)")
    ascii_plot(vols,  width=args.width, height=args.height, title="Volume vs Step",      y_label="Volume (Å^3)")
    ascii_plot(pres,  width=args.width, height=args.height, title="Pressure vs Step",    y_label="Pressure (kB)")


if __name__ == "__main__":
    main()
