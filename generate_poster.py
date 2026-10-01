"""
Seven views of primes and Goldbach's conjecture.

Every plotted point, count and number-bearing label is computed from the
settings below. Change EXAMPLE_N and re-run to redraw everything.

The mathematics is in goldbach/number_theory.py and visualization_data.py; the
drawing is in render.py, with style.py and layout.py. The tests in tests/ check
the mathematics and that the figures show the computed data (python -m pytest).
Nothing here proves Goldbach's conjecture.

Run:  python generate_poster.py
  ->  images/poster-landscape.png              3600 x 2400 (3:2)
  ->  images/poster-preview-4x5.png            2160 x 2700 (4:5)
"""
import os

import matplotlib.pyplot as plt

from goldbach.layout import FIGURE_SIZES
from goldbach.visualization_data import build_visualization_data
from goldbach.render import build_figure
from goldbach.style import apply_style

# ================================================================ settings
EXAMPLE_N = 100           # even number used as the worked example (panels 2, 5, 6)
WHEEL_MODULUS = 30        # modulus of the prime wheel (panels 1, 2)
CRT_MODULI = (3, 5, 7)    # moduli of the residue space (panels 4, 5)
HEATMAP_MAX = 1000        # largest N in the heatmap (panel 7)
SIEVE_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]   # sieve levels z (panel 7)
COMET_MAX = 2000          # largest N in the Goldbach comet (panel 5)
# name -> (output path, px width); figure sizes are in layout.py
EXPORTS = {
    "landscape": ("images/poster-landscape.png", 3600),
    "preview_4x5": ("images/poster-preview-4x5.png", 2160),
}


def save(name, data):
    """Render one layout and save its PNG."""
    out, width = EXPORTS[name]
    fig, _ = build_figure(name, data)
    fig.savefig(out, dpi=width / FIGURE_SIZES[name][0])
    plt.close(fig)
    print(f"saved {out}")


def main():
    apply_style()
    data = build_visualization_data(EXAMPLE_N, WHEEL_MODULUS, CRT_MODULI,
                                    HEATMAP_MAX, SIEVE_PRIMES, COMET_MAX)
    os.makedirs("images", exist_ok=True)
    for name in EXPORTS:
        save(name, data)


if __name__ == "__main__":
    main()
