"""
Optional extra, beyond the poster: what the sieve can't see.

For even N and sieve level z, a pair (a, N - a) with 2 <= a <= N/2 survives if
neither a nor N - a has a prime factor <= z. The figure plots, against z,

    M(z) = number of surviving pairs
    S(z) = sum over the survivors of λ(a)·λ(N - a),   λ(n) = (-1)^Ω(n),

and splits groups of pairs (all pairs, survivors at SPLIT_LEVELS, Goldbach pairs)
by the sign of λ(a)·λ(N - a).

The mathematics is in goldbach/number_theory.py and visualization_data.py; the
drawing is in render.py, with layout.py. This script runs a few quick checks on
the data it is about to draw and saves nothing if they fail. The independent
verification is the test suite: python -m pytest -m parity. Nothing here proves
Goldbach's conjecture.

Run from the repository root:  python extras/generate_parity_visualization.py
  ->  images/extras/parity-landscape.png       3200 x 1800 (16:9)
  ->  images/extras/parity-linkedin-4x5.png    2160 x 2700 (4:5)
"""
import os
import sys
from math import isqrt

# Run as a script, this file's folder comes first on sys.path; the goldbach package is one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from goldbach import number_theory as nt  # noqa: E402
from goldbach.layout import PARITY_FIGURE_SIZES  # noqa: E402
from goldbach.render import build_parity_figure  # noqa: E402
from goldbach.style import apply_style  # noqa: E402
from goldbach.visualization_data import parity_data  # noqa: E402

# ================================================================ settings
PARITY_NS = (999_000, 999_998, 1_000_000)   # even N; the M and S panel shows the last
SPLIT_LEVELS = (7, 31)                      # sieve levels whose survivors the sign-split panel shows
OUT_DIR = "images/extras"
# name -> full-size px width; figure sizes are in layout.py
EXPORTS = {
    "landscape": 3200,
    "linkedin_4x5": 2160,
}


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
    W = PARITY_FIGURE_SIZES[name][0]
    fig, _ = build_parity_figure(name, data)
    out = f"{OUT_DIR}/parity-{name.replace('_', '-')}.png"
    fig.savefig(out, dpi=EXPORTS[name] / W)
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
    for name in EXPORTS:
        save(name, data)


if __name__ == "__main__":
    main()
