"""Settings and image generation for the first-survivor experiment.

Run from the repository root:

    python -m extras.first_survivor_experiment.generate            # framework and main figures
    python -m extras.first_survivor_experiment.generate --blocks   # and the exact-block appendix

  ->  images/extras/first-survivor-framework.png      3200 x 1800 (16:9)
  ->  images/extras/first-survivor-landscape.png      3200 x 1800 (16:9)
  ->  images/extras/first-survivor-linkedin-4x5.png   2160 x 2700 (4:5)
  ->  images/extras/first-survivor-blocks.png         2000 x 1600 (with --blocks)

Runs quick checks on the data it is about to draw and saves nothing if they fail.
The independent verification is the test suite: python -m pytest -m first_survivor.
"""
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt

from goldbach.style import apply_style

from .data import exact_blocks, first_survivor_data
from .framework import SIZE as FRAMEWORK_SIZE
from .framework import WIDTH_PX as FRAMEWORK_WIDTH_PX
from .framework import build_framework_figure, framework_data
from .layout import BLOCKS_SIZE, BLOCKS_WIDTH_PX, EXPORT_WIDTH_PX, FIGURE_SIZES
from .render import build_blocks_figure, build_figure

# ================================================================ settings
EXAMPLE_N = 999_008             # worked target: first survivor near the centre, a composite survivor at W
N_MAX = 2 ** 20                 # every even N with 6 <= N < N_MAX; a power of 2, so dyadic bins are complete
BIN_START_EXP = 8               # dyadic bins [2^k, 2^(k+1)) from k = 8, above the small exceptions
FIT_CUTOFF_EXPS = (14, 17, 20)  # nested fits over the bins inside N < 2^K
SMALL_MAX = 288                 # panel B's linear view: even N through the q = 13 block (17² = 289)
CENTRE_ZOOM = 120               # panel A: offsets 0..CENTRE_ZOOM near the centre
EDGE_ZOOM = 90                  # panel A: offsets within EDGE_ZOOM of W
# framework figure: a target small enough to draw every offset, one block, and a run of blocks
FRAMEWORK_N = 272               # q = 13, W = 33, first survivor at 27, a composite survivor at d = W
FRAMEWORK_BLOCK_Q = 23          # panel B: the block 23² < N < 29²
FRAMEWORK_SPAN = (11, 31)       # panel C: the blocks from q = 11 through q = 31
# --blocks: q = the largest prime <= each target; every even N in each block is computed
BLOCK_TARGETS = (1100, 1420, 1830, 2360, 3050, 3930, 5070, 6540,
                 8430, 10870, 14020, 18080, 23310, 30060, 38770, 50000)
OUT_DIR = "images/extras"


def load():
    return first_survivor_data(EXAMPLE_N, N_MAX, BIN_START_EXP, FIT_CUTOFF_EXPS, CENTRE_ZOOM, EDGE_ZOOM)


def sanity_checks(d):
    """Quick consistency checks on the data being drawn, as [(description, ok)].
    The independent checks are the tests."""
    sw, e = d.sweep, d.example
    big = sw.N >= d.min_N
    inside = sw.lam_sieve < sw.W
    return [
        ("every target has a first survivor and a nearest prime pair",
         bool((sw.lam_sieve >= 0).all() and (sw.lam_prime >= 0).all())),
        (f"N >= {d.min_N}: λ_sieve < W", bool(inside[big].all())),
        ("λ_sieve < W implies λ_sieve = λ_prime", bool((sw.lam_sieve[inside] == sw.lam_prime[inside]).all())),
        ("no composite survivor below W", bool(((sw.first_composite < 0) | (sw.first_composite >= sw.W)).all())),
        ("exceptions split into composite-first and prime-first",
         len(d.composite_first) + len(d.prime_outside) == len(d.exceptions)),
        ("worked example: first survivor inside W", e.lam_sieve < e.W),
        ("worked example: every composite survivor at or beyond W", bool((e.composite >= e.W).all())),
        ("worked example: a composite survivor is in the boundary zoom", bool(e.edge_composite.any())),
        ("dyadic bins are complete", 2 ** (int(d.bin_exps[-1]) + 1) == d.n_max),
        ("fits are nested", [f.n_bins for f in d.fits] == sorted(f.n_bins for f in d.fits)),
    ]


def save_framework(out_dir=OUT_DIR):
    """Draw and save the framework figure; returns its path."""
    fd = framework_data(FRAMEWORK_N, FRAMEWORK_BLOCK_Q, FRAMEWORK_SPAN)
    fig, _ = build_framework_figure(fd)
    out = f"{out_dir}/first-survivor-framework.png"
    save(fig, out, FRAMEWORK_WIDTH_PX / FRAMEWORK_SIZE[0])
    return out


def save(fig, out, dpi):
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"saved {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Where the first survivor lands.")
    ap.add_argument("--blocks", action="store_true",
                    help="also draw the exact prime-square blocks beyond the main range")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):     # messages contain ≥ and λ
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    apply_style()
    t = time.perf_counter()
    data = load()
    print(f"computed {len(data.sweep.N):,} targets in {time.perf_counter() - t:.1f} s")
    bad = [desc for desc, ok in sanity_checks(data) if not ok]
    blocks = None
    if args.blocks:
        t = time.perf_counter()
        blocks = exact_blocks(BLOCK_TARGETS)
        print(f"computed {len(blocks)} exact blocks, {sum(b.n_targets for b in blocks):,} targets, "
              f"in {time.perf_counter() - t:.1f} s")
        if not all(b.n_targets > 0 and b.q ** 2 >= N_MAX for b in blocks):
            bad.append("every selected block has targets and lies beyond the main range")
    if bad:
        raise SystemExit("data check(s) failed; nothing saved:\n  " + "\n  ".join(bad))
    os.makedirs(OUT_DIR, exist_ok=True)
    save_framework()
    for name, px in EXPORT_WIDTH_PX.items():
        fig, _ = build_figure(name, data, SMALL_MAX)
        save(fig, f"{OUT_DIR}/first-survivor-{name.replace('_', '-')}.png", px / FIGURE_SIZES[name][0])
    if blocks is not None:
        fig, _ = build_blocks_figure(blocks, data)
        save(fig, f"{OUT_DIR}/first-survivor-blocks.png", BLOCKS_WIDTH_PX / BLOCKS_SIZE[0])


if __name__ == "__main__":
    main()
