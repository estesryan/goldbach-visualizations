"""Settings and image generation for the parity visualization.

Run from the repository root:

    python -m extras.parity.generate

  ->  images/extras/parity-landscape.png       3200 x 1800 (16:9)
  ->  images/extras/parity-preview-4x5.png     2160 x 2700 (4:5)

Runs a few quick checks on the data it is about to draw and saves nothing if they
fail. The independent verification is the test suite: python -m pytest -m parity.
"""
import os
import sys
from math import isqrt

import matplotlib.pyplot as plt
import numpy as np

from goldbach import number_theory as nt
from goldbach.style import apply_style

from .data import parity_data
from .layout import EXPORT_WIDTH_PX, FIGURE_SIZES
from .render import build_figure

# ================================================================ settings
PARITY_NS = (999_000, 999_998, 1_000_000)   # even N; the M and S panel shows the last
SPLIT_LEVELS = (7, 31)                      # sieve levels whose survivors the sign-split panel shows
OUT_DIR = "images/extras"


def sanity_checks(d):
    """Quick consistency checks on the data being drawn, as [(description, ok)].
    The independent checks are the tests."""
    checks = []
    for N, sw in d.sweeps.items():
        checks += [
            (f"N = {N:,}: levels are the primes up to the largest prime ≤ √N",
             list(sw.zs) == nt.primes_between(2, isqrt(N) + 1)),
            (f"N = {N:,}: M(z) never increases with z", bool(np.all(np.diff(sw.M) <= 0))),
            (f"N = {N:,}: |S(z)| ≤ M(z) at every level", bool(np.all(np.abs(sw.S) <= sw.M))),
            (f"N = {N:,}: S(z) = M(z) at the last level", sw.S[-1] == sw.M[-1]),
        ]
    checks.append((f"N = {d.featured:,}: caption counts add up",
                   d.goldbach_total - d.goldbach_small == d.sweeps[d.featured].M[-1]))
    sw = d.sweeps[d.featured]
    i = list(sw.zs).index(d.meet_level)
    checks.append((f"N = {d.featured:,}: S(z) = M(z) from z = {d.meet_level} on, and not just before",
                   bool(np.all(sw.S[i:] == sw.M[i:])) and (i == 0 or sw.S[i - 1] != sw.M[i - 1])))
    for c in d.class_splits:
        if c.kind == "survivors":
            j = list(sw.zs).index(c.z)
            checks.append((f"survivors at z = {c.z}: split adds up to M(z) and S(z)",
                           c.plus + c.minus == sw.M[j] and c.plus - c.minus == sw.S[j]))
    gold = [c for c in d.class_splits if c.kind == "goldbach"][0]
    checks.append(("every Goldbach pair has sign +1", gold.minus == 0 and gold.plus == d.goldbach_total))
    return checks


def save(name, data):
    """Render one layout and save it."""
    W = FIGURE_SIZES[name][0]
    fig, _ = build_figure(name, data)
    out = f"{OUT_DIR}/parity-{name.replace('_', '-')}.png"
    fig.savefig(out, dpi=EXPORT_WIDTH_PX[name] / W)
    plt.close(fig)
    print(f"saved {out}")


def main():
    for stream in (sys.stdout, sys.stderr):     # messages contain ≤ and √
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    apply_style()
    data = parity_data(PARITY_NS, SPLIT_LEVELS)
    bad = [desc for desc, ok in sanity_checks(data) if not ok]
    if bad:
        raise SystemExit("data check(s) failed; nothing saved:\n  " + "\n  ".join(bad))
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in EXPORT_WIDTH_PX:
        save(name, data)


if __name__ == "__main__":
    main()
