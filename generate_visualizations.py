"""
Seven views of primes and Goldbach's conjecture.

Every plotted point, count and number-bearing label is computed from the
settings below. Change N_EX and re-run to redraw everything.

Checks that run before each image is saved:
  * CHECKS, filled while drawing: labels agree with the data.
  * Colour audit: every coloured artist uses PALETTE, and every artist tagged
    with a semantic role uses that role's colour.
  * Contrast audit: all text is at least MIN_TEXT_CONTRAST against its background.
  * verify(): recomputes the data separately (trial-division primality,
    brute-force Gaussian/Eisenstein irreducibility, brute-force sieving),
    reads the values back out of the drawn artists and text, and compares.
Any failed check stops the run before that layout is saved.

The checks confirm the figure matches the numbers. Nothing here proves
Goldbach's conjecture.

Run:  python generate_visualizations.py
  ->  images/landscape.png                     3600 x 2400 (3:2)
  ->  images/linkedin-4x5.png                  2160 x 2700 (4:5)
  ->  images/*-preview.png                     1200 / 1080 px wide
  ->  verification_report.txt
"""
from collections import Counter, defaultdict
from math import gcd, isqrt
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_hex, to_rgba, to_rgb
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.text import Text

# ================================================================ settings
N_EX = 100            # even number used as the worked example (panels 2, 5, 6)
WHEEL = 30            # modulus of the prime wheel (panels 1, 2)
CRT = (3, 5, 7)       # moduli of the residue space (panels 4, 5)
HEAT_MAX = 1000       # largest N in the heatmap (panel 7)
SIEVE_ZS = [2, 3, 5, 7, 11, 13, 17, 19, 23]   # sieve levels z (panel 7)
COMET_MAX = 2000      # largest N in the Goldbach comet (panel 5)
LIMIT = 10_000        # sieve bound; is_prime() raises above this
# name -> (figure inches, full-size px width, preview px width)
EXPORTS = {
    "landscape": ((18.0, 12.0), 3600, 1200),    # 3:2
    "linkedin_4x5": ((16.2, 20.25), 2160, 1080),  # 4:5, LinkedIn portrait 1080 x 1350
}
MIN_TEXT_CONTRAST = 4.5     # WCAG AA. MUTED is also checked against 7:1.

# ================================================================ colours
# All colours are defined here. audit() fails on any colour not in PALETTE.
PALETTE = {
    "BG":        "#0f0e0d",   # page background
    "PANEL_BG":  "#171513",   # panel fill
    "BORDER":    "#2c2825",   # panel borders, spines
    "GRID":      "#2a2622",   # gridlines, wheel spokes
    "TEXT":      "#f1ebe2",   # primary text
    "MUTED":     "#aaa39c",   # secondary text (7.3:1 on PANEL_BG)
    "EXCLUDED":  "#4f4943",   # composite / excluded classes, background lattice
    "ACCENT":    "#ff6b35",   # prime / admissible / surviving
    "ACCENT_2":  "#7fb3c8",   # partner q = N - p
    "HIGHLIGHT": "#fff3d6",   # Goldbach pair links
    "WARNING":   "#c9a54c",   # small-prime exceptions
}

# Role -> palette key. Artists carry their role as gid; audit() checks that
# each role is drawn in its own colour.
SEMANTIC = {
    "prime": "ACCENT", "admissible": "ACCENT", "surviving": "ACCENT",
    "partner": "ACCENT_2",
    "excluded": "EXCLUDED", "lattice": "EXCLUDED",
    "pair": "HIGHLIGHT",
    "exception": "WARNING",
    "fixed": "TEXT",
}


def C(role):
    """Colour for a semantic role or a palette key."""
    return PALETTE[SEMANTIC.get(role, role)]


def mix(a, b, t):
    """Blend a -> b by t. Only used to build the colormap."""
    ra, rb = np.array(to_rgba(a)), np.array(to_rgba(b))
    return tuple((1 - t) * ra + t * rb)


# Sequential colormap for counts: PANEL_BG -> dim ACCENT -> ACCENT -> HIGHLIGHT.
# Masked (zero) cells show as BG.
BRAND_CMAP = LinearSegmentedColormap.from_list("brand", [
    PALETTE["PANEL_BG"], mix(PALETTE["PANEL_BG"], PALETTE["ACCENT"], 0.35),
    PALETTE["ACCENT"], PALETTE["HIGHLIGHT"]])
BRAND_CMAP.set_bad(PALETTE["BG"])

# ================================================================ typography
FONT_BODY = ["Poppins", "DejaVu Sans"]          # DejaVu covers ≡, ↦, φ, ω per glyph
FONT_HEAD = ["Lora", "DejaVu Serif"]
FS = {"title": 22, "head": 17, "num": 12, "sub": 11, "body": 11, "small": 10, "tick": 9.5}
plt.rcParams.update({
    "font.family": FONT_BODY, "font.size": FS["body"],
    "text.color": PALETTE["TEXT"], "axes.labelcolor": PALETTE["MUTED"],
    "xtick.color": PALETTE["MUTED"], "ytick.color": PALETTE["MUTED"],
    "xtick.labelcolor": PALETTE["MUTED"], "ytick.labelcolor": PALETTE["MUTED"],
    "axes.edgecolor": PALETTE["BORDER"], "axes.facecolor": PALETTE["PANEL_BG"],
    "figure.facecolor": PALETTE["BG"], "savefig.facecolor": PALETTE["BG"],
    "axes.labelsize": FS["small"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "grid.color": PALETTE["GRID"], "lines.color": PALETTE["TEXT"],
    "patch.edgecolor": PALETTE["BORDER"], "patch.facecolor": PALETTE["PANEL_BG"],
    "legend.edgecolor": PALETTE["BORDER"], "legend.facecolor": PALETTE["PANEL_BG"],
    "legend.labelcolor": PALETTE["TEXT"], "legend.fontsize": FS["small"],
})

# ================================================================ number theory
sieve = np.ones(LIMIT + 1, dtype=bool)
sieve[:2] = False
for i in range(2, isqrt(LIMIT) + 1):
    if sieve[i]:
        sieve[i * i::i] = False
PRIMES = np.nonzero(sieve)[0]


def is_prime(n):
    n = abs(int(n))
    if n > LIMIT:
        raise ValueError(f"{n} exceeds LIMIT")
    return bool(sieve[n])


def goldbach_pairs(N):
    """Pairs (p, N - p) with p <= N - p and both prime."""
    return [(int(p), int(N - p)) for p in PRIMES if p <= N // 2 and is_prime(N - p)]


def two_squares(n):
    """(a, b) with a <= b and a² + b² = n, or None."""
    for a in range(isqrt(n // 2) + 1):
        b = isqrt(n - a * a)
        if a * a + b * b == n:
            return a, b
    return None


WHEEL_PRIMES = [p for p in range(2, WHEEL + 1) if WHEEL % p == 0 and is_prime(p)]
COPRIME = [r for r in range(WHEEL) if gcd(r, WHEEL) == 1]
M105 = int(np.prod(CRT))
CHECKS = []


# ================================================================ layout engine
# Drawing code is written in landscape figure fractions. For other layouts each
# panel's content is mapped into that layout's panel rectangle. The header keeps
# a fixed height (HDR_IN inches); the content area below it is rescaled.
LAND = EXPORTS["landscape"][0]
LAND_RECTS = {1: (0.008, 0.655, 0.326, 0.305), 2: (0.339, 0.655, 0.326, 0.305),
              3: (0.670, 0.655, 0.322, 0.305), 4: (0.008, 0.335, 0.40, 0.312),
              5: (0.413, 0.335, 0.579, 0.312), 6: (0.008, 0.008, 0.44, 0.319),
              7: (0.453, 0.008, 0.539, 0.319)}
HDR_IN = 0.9            # height of panel heading + subtitle zone, inches


def _in(r, size):
    return (r[0] * size[0], r[1] * size[1], r[2] * size[0], r[3] * size[1])


def portrait_rects(W, H):
    """Portrait panel rects in inches. Rows: 1-2, 3-4, 5 (full width, tallest), 6-7."""
    m, g, top = 0.144, 0.09, 0.48
    hero = 5.5
    row = (H - top - m - hero - 3 * g) / 3
    full = W - 2 * m
    half = (full - g) / 2
    r = {}
    y = H - top - row
    r[1], r[2] = (m, y, half, row), (m + half + g, y, half, row)
    y -= g + row
    r[3], r[4] = (m, y, half, row), (m + half + g, y, half, row)
    y -= g + hero
    r[5] = (m, y, full, hero)
    y -= g + row
    w6 = 7.6
    r[6], r[7] = (m, y, w6, row), (m + w6 + g, y, full - w6 - g, row)
    return r


LAYOUTS = {
    "landscape": {k: _in(v, LAND) for k, v in LAND_RECTS.items()},
    "linkedin_4x5": portrait_rects(*EXPORTS["linkedin_4x5"][0]),
}

fig = None
FIG_IN = LAND
CUR = {}
REF = {}
CHECKS = []


def set_panel(num):
    CUR["land"] = _in(LAND_RECTS[num], LAND)
    CUR["new"] = LAYOUT[num]


# M maps a landscape point and MS a landscape size into the current panel.
# FA, FT, FL, FR add axes, text, a line and a rectangle at landscape coords.
def M(x, y):
    lx, ly, lw, lh = CUR["land"]
    nx, ny, nw, nh = CUR["new"]
    sx, sy = nw / lw, (nh - HDR_IN) / (lh - HDR_IN)
    X = nx + (x * LAND[0] - lx) * sx
    Y = ny + nh - (HDR_IN + ((ly + lh) - y * LAND[1] - HDR_IN) * sy)
    return X / FIG_IN[0], Y / FIG_IN[1]


def MS(w, h):
    lx, ly, lw, lh = CUR["land"]
    nx, ny, nw, nh = CUR["new"]
    sx, sy = nw / lw, (nh - HDR_IN) / (lh - HDR_IN)
    return w * LAND[0] * sx / FIG_IN[0], h * LAND[1] * sy / FIG_IN[1]


def FA(rect, **kw):
    x, y = M(rect[0], rect[1]); w, h = MS(rect[2], rect[3])
    return fig.add_axes([x, y, w, h], **kw)


def FT(x, y, s, **kw):
    X, Y = M(x, y)
    return fig.text(X, Y, s, **kw)


def FL(xs, ys, **kw):
    pts = [M(x, y) for x, y in zip(xs, ys)]
    ln = Line2D([p[0] for p in pts], [p[1] for p in pts], transform=fig.transFigure, **kw)
    fig.add_artist(ln)
    return ln


def FR(xy, w, h, **kw):
    X, Y = M(*xy); W, H = MS(w, h)
    rc = Rectangle((X, Y), W, H, transform=fig.transFigure, **kw)
    fig.patches.append(rc)
    return rc


def contrast(a, b):
    """WCAG contrast ratio of two colours."""
    def lum(c):
        v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def panel(num, title, sub, *_landscape_rect):
    set_panel(num)
    x, y, w, h = CUR["new"]
    W, H = FIG_IN
    fig.patches.append(Rectangle((x / W, y / H), w / W, h / H, transform=fig.transFigure,
                                 fc=C("PANEL_BG"), ec=C("BORDER"), lw=0.8, zorder=-10))
    top = (y + h) / H
    fig.text((x + 0.162) / W, top - 0.24 / H, f"{num:02d}", fontsize=FS["num"], color=C("ACCENT"),
             va="center")
    fig.text((x + 0.54) / W, top - 0.24 / H, title, fontsize=FS["head"], family=FONT_HEAD, va="center")
    if sub:
        fig.text((x + 0.162) / W, top - 0.516 / H, sub, fontsize=FS["sub"], va="top",
                 color=C("MUTED"), linespacing=1.4)


def style(ax, grid=False):
    ax.set_facecolor(C("PANEL_BG"))
    for s in ax.spines.values():
        s.set_color(C("BORDER")); s.set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.6, color=C("BORDER"))
    if grid:
        ax.grid(True, lw=0.5, color=C("GRID"))


def wheel(ax, label_extra=()):
    """Residue wheel mod WHEEL. Coprime classes large, the rest small and grey."""
    ang = np.pi / 2 - 2 * np.pi * np.arange(WHEEL) / WHEEL
    xs, ys = np.cos(ang), np.sin(ang)
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=C("EXCLUDED"), lw=0.8, zorder=1))
    for k in range(WHEEL):
        good = k in COPRIME
        if good:
            ax.plot([0, xs[k]], [0, ys[k]], color=C("GRID"), lw=0.8, zorder=0)
        ax.scatter(xs[k], ys[k], s=190 if good else 16, c=C("prime" if good else "excluded"),
                   lw=0, zorder=3, gid="prime" if good else "excluded")
        if good:
            ax.text(1.19 * xs[k], 1.19 * ys[k], str(k), ha="center", va="center",
                    fontsize=FS["body"] + 1, color=C("TEXT"))
        elif k in label_extra:
            ax.text(1.19 * xs[k], 1.19 * ys[k], str(k), ha="center", va="center",
                    fontsize=FS["small"], color=C("exception"), gid="exception")
    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal"); ax.axis("off")
    return xs, ys


def dot_label(x, y, role, text, marker="o", hollow=False, size=9):
    """Legend marker and label at landscape figure coords."""
    kw = dict(mfc="none", mec=C(role), mew=1.6) if hollow else dict(mfc=C(role), mec="none")
    FL([x], [y], marker=marker, ms=size, ls="", gid=role, **kw)
    FT(x + 0.011, y, text, fontsize=FS["small"], va="center")


R1, R2, R3, R4, R5, R6, R7 = (LAND_RECTS[k] for k in range(1, 8))


def draw():
    pass
    # ================================================================ PANEL 1
    wp = "·".join(map(str, WHEEL_PRIMES))
    panel(1, f"The prime wheel (mod {WHEEL})",
          f"Every prime > {max(WHEEL_PRIMES)} is coprime to {WHEEL} = {wp}, so it lies in\n"
          f"one of φ({WHEEL}) = {len(COPRIME)} classes.", *R1)
    ax = FA([0.012, 0.668, 0.175, 0.21])
    xs, ys = wheel(ax, label_extra=WHEEL_PRIMES)
    for r in WHEEL_PRIMES:
        ax.scatter(xs[r], ys[r], s=130, facecolor="none", ec=C("exception"), lw=1.3, zorder=4,
                   gid="exception")
    big = [int(p) for p in PRIMES if max(WHEEL_PRIMES) < p < 1000]
    counts = Counter(p % WHEEL for p in big)
    axb = FA([0.222, 0.70, 0.095, 0.155]); style(axb); REF["bar"] = axb
    vals = [counts[r] for r in COPRIME]
    axb.barh(range(len(COPRIME)), vals, color=C("prime"), height=0.62, gid="prime")
    axb.set_yticks(range(len(COPRIME))); axb.set_yticklabels(COPRIME)
    axb.invert_yaxis(); axb.set_xticks([])
    for s in ("top", "right", "bottom"):
        axb.spines[s].set_visible(False)
    for i, v in enumerate(vals):
        axb.text(v + 0.6, i, str(v), va="center", fontsize=FS["tick"], color=C("MUTED"))
    axb.set_xlim(0, max(vals) * 1.3)
    FT(0.205, 0.875, f"Primes {big[0]}–{big[-1]} per class", fontsize=FS["small"],
             color=C("MUTED"))
    dot_label(0.205, 0.684, "exception", f"{', '.join(map(str, WHEEL_PRIMES))}: the sieving primes",
              hollow=True)
    CHECKS.append(("p1 coprime classes", WHEEL != 30 or COPRIME == [1, 7, 11, 13, 17, 19, 23, 29]))
    CHECKS.append(("p1 all primes > 5 counted", sum(vals) == len(big)))

    # ================================================================ PANEL 2
    pairs = goldbach_pairs(N_EX)
    small_cut = max(WHEEL_PRIMES)
    main = [(p, q) for p, q in pairs if p > small_cut]
    exc = [(p, q) for p, q in pairs if p <= small_cut]
    s = N_EX % WHEEL
    # Unordered residue types {r, s - r} with both classes coprime to WHEEL.
    types = sorted({tuple(sorted((r, (s - r) % WHEEL))) for r in COPRIME if (s - r) % WHEEL in COPRIME})
    by_type = defaultdict(list)
    for p, q in main:
        by_type[tuple(sorted((p % WHEEL, q % WHEEL)))].append((p, q))
    panel(2, "Goldbach pairs on the wheel",
          f"N = {N_EX} ≡ {s} (mod {WHEEL}), so p ≡ r forces q ≡ {s} − r.\n"
          f"With p, q > {small_cut}, only {len(types)} residue types are possible.", *R2)
    ax = FA([0.343, 0.668, 0.175, 0.21]); REF["wheel2"] = ax
    xs, ys = wheel(ax, label_extra=sorted({p % WHEEL for p, _ in exc} | {q % WHEEL for _, q in exc}
                                          - set(COPRIME)))
    for (a, b), lst in by_type.items():
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=C("pair"), lw=1.8 * len(lst) + 0.6,
                solid_capstyle="round", zorder=2, gid="pair")
    for p, q in exc:
        a, b = p % WHEEL, q % WHEEL
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=C("exception"), lw=1.3, ls=(0, (2, 2)),
                zorder=2, gid="exception")
    y0 = 0.862
    for t in types:
        lst = by_type.get(t, [])
        FT(0.528, y0, f"{t[0]} ↔ {t[1]}", fontsize=FS["head"] - 1, family=FONT_HEAD,
                 color=C("pair"), va="top", gid="pair")
        FT(0.610, y0 - 0.002, f"× {len(lst)}", fontsize=FS["body"], color=C("MUTED"), va="top")
        FT(0.528, y0 - 0.029, "   ".join(f"{p}+{q}" for p, q in lst) or "none",
                 fontsize=FS["small"], color=C("TEXT"), va="top")
        y0 -= 0.07
    if exc:
        FT(0.528, y0, "   ".join(f"{p} + {q}" for p, q in exc), fontsize=FS["body"],
                 color=C("exception"), va="top", gid="exception")
        FT(0.528, y0 - 0.025, f"small-prime exception:\n{', '.join(str(p) for p, _ in exc)} "
                 f"is removed by the wheel", fontsize=FS["small"], color=C("MUTED"), va="top",
                 linespacing=1.3)
    CHECKS.append(("p2 main pairs typed", all(tuple(sorted((p % WHEEL, q % WHEEL))) in types for p, q in main)))
    if N_EX == 100:
        CHECKS.append(("p2 N=100 pairs", pairs == [(3, 97), (11, 89), (17, 83), (29, 71), (41, 59), (47, 53)]))
        CHECKS.append(("p2 N=100 types", types == [(11, 29), (17, 23)]))

    # ================================================================ PANEL 3
    panel(3, "Gaussian and Eisenstein primes",
          "Dots are prime elements z. Style shows how the rational\n"
          "prime p below z factors: split, inert or ramified.", *R3)
    R = 10


    def classify_gauss(a, b):
        # Off-axis with prime norm: split, or ramified if the norm is 2.
        # On an axis: inert iff |a + b| is a prime ≡ 3 (mod 4).
        n = a * a + b * b
        if a and b and is_prime(n):
            return "ram" if n == 2 else "split"
        if (a == 0) != (b == 0) and is_prime(a + b) and abs(a + b) % 4 == 3:
            return "inert"
        return None


    def classify_eis(a, b):          # z = a + b·ω,  N(z) = a² − ab + b²
        n = a * a - a * b + b * b
        if is_prime(n):
            return "ram" if n == 3 else "split"
        # Inert: z = unit · r for a rational prime r ≡ 2 (mod 3), so N(z) = r².
        r = isqrt(n)
        if n and r * r == n and is_prime(r) and r % 3 == 2 and a % r == 0 and b % r == 0:
            return "inert"
        return None


    def gauss_pts():
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                yield a, b, a, b


    def eis_pts():
        for a in range(-2 * R - 2, 2 * R + 3):
            for b in range(-2 * R - 2, 2 * R + 3):
                x, y = a - b / 2, b * np.sqrt(3) / 2
                if abs(x) <= R and abs(y) <= R:
                    yield a, b, x, y


    def lattice_plot(rect, pts_fn, classify, title):
        ax = FA(rect); style(ax); REF.setdefault("lattices", []).append(ax)
        groups = {"split": [], "inert": [], "ram": []}
        allp = []
        for a, b, x, y in pts_fn():
            allp.append((x, y))
            c = classify(a, b)
            if c:
                groups[c].append((x, y))
        allp = np.array(allp)
        ax.scatter(allp[:, 0], allp[:, 1], s=1.5, c=C("lattice"), lw=0, gid="lattice")
        g = np.array(groups["split"])
        ax.scatter(g[:, 0], g[:, 1], s=9, c=C("prime"), lw=0, alpha=0.75, gid="prime")
        g = np.array(groups["inert"])
        ax.scatter(g[:, 0], g[:, 1], s=26, facecolor="none", ec=C("prime"), lw=1.2, gid="prime")
        g = np.array(groups["ram"])
        ax.scatter(g[:, 0], g[:, 1], s=46, c=C("exception"), lw=0, zorder=3, gid="exception")
        ax.set_aspect("equal"); ax.set_xlim(-R - .8, R + .8); ax.set_ylim(-R - .8, R + .8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=FS["body"], color=C("TEXT"), pad=4)
        return groups


    gg = lattice_plot([0.690, 0.712, 0.125, 0.14], gauss_pts, classify_gauss, "Gaussian  ℤ[i]")
    ee = lattice_plot([0.845, 0.712, 0.125, 0.14], eis_pts, classify_eis, "Eisenstein  ℤ[ω]")
    dot_label(0.680, 0.694, "prime", "split    p ≡ 1 (mod 4) · p ≡ 1 (mod 3)", size=6)
    dot_label(0.680, 0.679, "prime", "inert    p ≡ 3 (mod 4) · p ≡ 2 (mod 3)", hollow=True, size=8)
    dot_label(0.680, 0.664, "exception", "ramified   above 2 · above 3", size=8)
    CHECKS.append(("p3 4 Gaussian primes above 2", len(gg["ram"]) == 4))
    CHECKS.append(("p3 6 Eisenstein primes above 3", len(ee["ram"]) == 6))

    # ================================================================ PANEL 4
    mstr = ", ".join(map(str, CRT))
    ncop = int(np.prod([m - 1 for m in CRT]))
    panel(4, f"Residue space (mod {mstr})",
          f"By CRT, n ↦ ({', '.join(f'n mod {m}' for m in CRT)}) is a {'×'.join(map(str, CRT))} grid.\n"
          f"Primes > {max(CRT)} occupy only the {ncop} points with no zero coordinate.", *R4)
    cls = [int(p) for p in PRIMES if p > max(CRT)]
    cnt = Counter(p % M105 for p in cls)
    ax3 = FA([0.0, 0.365, 0.235, 0.22], projection="3d"); REF["cube"] = ax3
    ax3.set_facecolor(C("PANEL_BG"))
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color(to_rgba(C("PANEL_BG")))
        axis.pane.set_edgecolor(C("BORDER"))
        axis._axinfo["grid"]["color"] = to_rgba(C("GRID"))
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.line.set_color(C("BORDER"))
    P, G = [], []
    for r in range(M105):
        t = tuple(r % m for m in CRT)
        (P if all(t) else G).append((*t, cnt.get(r, 0)))
    P, G = np.array(P), np.array(G)
    ax3.scatter(G[:, 0], G[:, 1], G[:, 2], s=4, c=C("excluded"), alpha=0.7, lw=0, depthshade=False,
                gid="excluded")
    ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=P[:, 3] * 2.6, c=C("prime"), lw=0, depthshade=False,
                gid="prime")
    ax3.set_xlabel(f"mod {CRT[0]}", labelpad=-6); ax3.set_ylabel(f"mod {CRT[1]}", labelpad=-5)
    ax3.set_zlabel(f"mod {CRT[2]}", labelpad=-6)
    ax3.set_xticks(range(CRT[0])); ax3.set_yticks(range(CRT[1])); ax3.set_zticks(range(CRT[2]))
    ax3.tick_params(labelsize=FS["tick"] - 1, pad=-2)
    ax3.view_init(elev=18, azim=-58)
    lo_c, hi_c = int(P[:, 3].min()), int(P[:, 3].max())
    dot_label(0.018, 0.352, "prime", f"size = primes {cls[0]}–{cls[-1]} per class ({lo_c}–{hi_c})", size=8)
    ax2 = FA([0.268, 0.37, 0.12, 0.165]); style(ax2); REF["heat4"] = ax2
    Mg = np.zeros((CRT[2], CRT[1]), dtype=int)
    for p in cls:
        Mg[p % CRT[2], p % CRT[1]] += 1
    norm4 = plt.Normalize(0, Mg.max())
    ax2.imshow(Mg, origin="lower", cmap=BRAND_CMAP, norm=norm4, aspect="auto")
    for i in range(CRT[2]):
        for j in range(CRT[1]):
            v = Mg[i, j]
            cell = to_hex(BRAND_CMAP(norm4(v)))
            tc = C("MUTED") if v == 0 else max((C("BG"), C("TEXT")), key=lambda t: contrast(t, cell))
            tx = ax2.text(j, i, v, ha="center", va="center", fontsize=FS["tick"], color=tc)
            REF.setdefault("cell_texts", []).append((tx, cell))
    ax2.set_xticks(range(CRT[1])); ax2.set_yticks(range(CRT[2]))
    ax2.set_xlabel(f"p mod {CRT[1]}"); ax2.set_ylabel(f"p mod {CRT[2]}")
    ax2.set_title(f"primes {cls[0]}–{cls[-1]},\nsummed over mod {CRT[0]}", fontsize=FS["small"],
                  color=C("MUTED"), pad=5)
    CHECKS.append(("p4 coprime grid points", len(P) == ncop))

    # ================================================================ PANEL 5  (hero)
    m3, m5, m7 = CRT
    # Fixed class of p -> N - p mod each m: c ≡ N/2, using the inverse of 2.
    centre = {m: (N_EX * pow(2, -1, m)) % m for m in CRT}
    ok3 = sorted({p % m3 for p, q in pairs if p > max(CRT)})
    panel(5, "The involution p ↦ N − p",
          f"Acts coordinate-wise; mod each odd m it fixes one class, c ≡ N/2. Centred on c, it is a point reflection.\n"
          + (f"N = {N_EX}: every pair shown has p ≡ q ≡ {ok3[0]} (mod {m3}), so one (mod {m5}, mod {m7}) plane holds them all."
             if len(ok3) == 1 else f"N = {N_EX}, shown in (mod {m5}, mod {m7}) coordinates."), *R5)
    axr = FA([0.432, 0.365, 0.235, 0.185]); style(axr); REF["hero"] = axr


    def rep(x, m):
        # Representative of x in the m consecutive integers centred on c.
        lo = centre[m] - (m - 1) // 2
        return (x - lo) % m + lo


    # x = residue mod 7 (the wider axis), y = residue mod 5, each centred on c.
    # Then p and N - p are reflections through the fixed class. The axis scaling
    # is affine, so this still holds without equal aspect.
    xr = range(centre[m7] - (m7 - 1) // 2, centre[m7] + (m7 - 1) // 2 + 1)
    yr = range(centre[m5] - (m5 - 1) // 2, centre[m5] + (m5 - 1) // 2 + 1)
    for x in xr:
        for y in yr:
            bad = x % m7 in (0, N_EX % m7) or y % m5 in (0, N_EX % m5)
            axr.add_patch(Rectangle((x - .5, y - .5), 1, 1, lw=0.6, ec=C("BORDER"),
                                    fc=C("excluded") if bad else C("admissible"),
                                    alpha=0.22 if bad else 0.13,
                                    gid="excluded" if bad else "admissible"))
    refl_ok = True
    for p, q in pairs:
        if p <= max(CRT):
            continue
        pp = (rep(p % m7, m7), rep(p % m5, m5)); qq = (rep(q % m7, m7), rep(q % m5, m5))
        refl_ok &= (pp[0] + qq[0] == 2 * centre[m7]) and (pp[1] + qq[1] == 2 * centre[m5])
        axr.plot([pp[0], qq[0]], [pp[1], qq[1]], color=C("pair"), lw=2.6, solid_capstyle="round",
                 zorder=2, gid="pair")
        axr.scatter(*pp, s=190, c=C("prime"), lw=0, zorder=3, gid="prime")
        axr.scatter(*qq, s=190, c=C("partner"), lw=0, zorder=3, gid="partner")
        for (x_, y_), v_, role_ in ((pp, p, "prime"), (qq, q, "partner")):
            right = x_ >= centre[m7]
            up = y_ >= centre[m5]
            axr.text(x_ + (0.17 if right else -0.17), y_ + (0.2 if up else -0.2), str(v_),
                     ha="left" if right else "right", va="bottom" if up else "top",
                     fontsize=FS["body"], color=C(role_), zorder=4, gid=role_)
    axr.scatter(centre[m7], centre[m5], marker="D", s=70, facecolor=C("PANEL_BG"), ec=C("fixed"),
                lw=1.4, zorder=5, gid="fixed")
    axr.set_xticks(list(xr)); axr.set_xticklabels([x % m7 for x in xr])
    axr.set_yticks(list(yr)); axr.set_yticklabels([y % m5 for y in yr])
    axr.set_xlim(min(xr) - .5, max(xr) + .5); axr.set_ylim(min(yr) - .5, max(yr) + .5)
    axr.set_xlabel(f"residue mod {m7}", labelpad=2); axr.set_ylabel(f"residue mod {m5}")
    for sp in axr.spines.values():
        sp.set_visible(False)

    lx, ly, dy = 0.688, 0.540, 0.026
    dot_label(lx, ly, "prime", "p  (prime)", size=10)
    dot_label(lx, ly - dy, "partner", f"q = {N_EX} − p", size=10)
    FL([lx - 0.005, lx + 0.006], [ly - 2 * dy] * 2, color=C("pair"), lw=2.6, gid="pair")
    FT(lx + 0.011, ly - 2 * dy, "Goldbach pair", fontsize=FS["small"], va="center")
    dot_label(lx, ly - 3 * dy, "fixed", f"fixed class", marker="D", hollow=True, size=7)
    FR((lx - 0.005, ly - 4 * dy - 0.008), 0.011, 0.016,
                                 fc=C("admissible"), alpha=0.3, ec=C("BORDER"), lw=0.6, gid="admissible")
    FT(lx + 0.011, ly - 4 * dy, "admissible", fontsize=FS["small"], va="center")
    FR((lx - 0.005, ly - 5 * dy - 0.008), 0.011, 0.016,
                                 fc=C("excluded"), alpha=0.5, ec=C("BORDER"), lw=0.6, gid="excluded")
    FT(lx + 0.011, ly - 5 * dy, "blocked: p or q ≡ 0", fontsize=FS["small"], va="center")
    skipped = [f"{p} + {q}" for p, q in pairs if p <= max(CRT)]
    if skipped:
        FT(lx - 0.005, ly - 6 * dy - 0.03, f"{', '.join(skipped)} not shown:\np ≤ {max(CRT)} has a zero\ncoordinate",
                 fontsize=FS["small"], color=C("exception"), linespacing=1.3, gid="exception")
    CHECKS.append(("p5 every q reflects its p", refl_ok))

    axa = FA([0.845, 0.40, 0.125, 0.14]); style(axa); REF["comet"] = axa
    Ns = np.arange(4, COMET_MAX + 1, 2)
    # Colour: number of r mod 105 with r and N - r both coprime to 105.
    adm = np.array([np.prod([(m - 1) if n % m == 0 else (m - 2) for m in CRT]) for n in Ns])
    gb = np.array([len(goldbach_pairs(int(n))) for n in Ns])
    sc = axa.scatter(Ns, gb, c=adm, s=1.6, cmap=BRAND_CMAP, norm=LogNorm(adm.min() * 0.6, adm.max()), lw=0)
    axa.set_xticks([0, COMET_MAX // 2, COMET_MAX]); axa.set_yticks([0, 50])
    axa.set_xlabel("even N", labelpad=1)
    axa.set_title(f"Goldbach comet, N ≤ {COMET_MAX}", fontsize=FS["small"], color=C("MUTED"), pad=5)
    for s_ in ("top", "right"):
        axa.spines[s_].set_visible(False)
    FT(0.845, 0.345, f"# pairs; brighter = more of the {M105}\nclasses admissible for that N",
             fontsize=FS["small"] - 0.5, color=C("MUTED"), linespacing=1.3)

    # ================================================================ PANEL 6
    panel(6, "Goldbach pairs and sum-of-two-squares circles",
          "|z|² = p has lattice points iff p = 2 or p ≡ 1 (mod 4): 8 of them (4 for p = 2).\n"
          "Not a geometry of Goldbach pairs in general: for N ≡ 0 (mod 4), N > 4, one prime has none.", *R6)


    def pick_examples():
        # Left: N_EX, preferring a pair with one prime ≡ 1 and one ≡ 3 (mod 4).
        # Right: N ≡ 2 (mod 4) closest to N_EX with a pair both ≡ 1 (mod 4).
        mixed = [(p, q) for p, q in pairs if p > small_cut and (p % 4 == 1) != (q % 4 == 1)]
        left = (N_EX, mixed[0] if mixed else pairs[-1])
        for d in range(0, 200, 2):
            for n in (N_EX - d, N_EX + d):
                if n % 4 == 2:
                    both = [(p, q) for p, q in goldbach_pairs(n) if p % 4 == 1 and q % 4 == 1]
                    if both:
                        return left, (n, both[-1])
        return left, None


    def sq_desc(n):
        t = two_squares(n)
        return f"{n} = {t[0]}² + {t[1]}²" if t else f"{n}: no two squares"


    def lattice_circles(ax, N, p, q):
        style(ax); Rr = isqrt(max(p, q)) + 2
        REF.setdefault("circles", []).append((ax, N, p, q))
        A, B = np.meshgrid(np.arange(-Rr, Rr + 1), np.arange(-Rr, Rr + 1))
        ax.scatter(A, B, s=2, c=C("lattice"), lw=0, gid="lattice")
        t = np.linspace(0, 2 * np.pi, 400)
        for n, role in ((p, "prime"), (q, "partner")):
            pts = [(a, b) for a in range(-Rr, Rr + 1) for b in range(-Rr, Rr + 1) if a * a + b * b == n]
            ax.plot(np.sqrt(n) * np.cos(t), np.sqrt(n) * np.sin(t), color=C(role), lw=1.8,
                    ls="-" if pts else (0, (3, 3)), gid=role)
            if pts:
                pts = np.array(pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=42, c=C("pair"), lw=0, zorder=4, gid="pair")
            CHECKS.append((f"p6 point count {n}", len(pts) == (0 if two_squares(n) is None else (4 if n == 2 else 8))))
        ax.set_aspect("equal"); ax.set_xlim(-Rr - .5, Rr + .5); ax.set_ylim(-Rr - .5, Rr + .5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"N = {N}:  {p} + {q}", fontsize=FS["body"] + 1, color=C("TEXT"), pad=4)
        ax.text(0.5, -0.07, sq_desc(p), transform=ax.transAxes, ha="center", va="top",
                fontsize=FS["small"], color=C("prime"), gid="prime")
        ax.text(0.5, -0.15, sq_desc(q), transform=ax.transAxes, ha="center", va="top",
                fontsize=FS["small"], color=C("partner"), gid="partner")


    (lN, (lp, lq)), right = pick_examples()
    lattice_circles(FA([0.03, 0.065, 0.155, 0.148]), lN, lp, lq)
    if right:
        rN, (rp, rq) = right
        lattice_circles(FA([0.225, 0.065, 0.155, 0.148]), rN, rp, rq)
    dot_label(0.386, 0.20, "pair", "lattice point", size=7)
    FL([0.380, 0.391], [0.175] * 2, color=C("partner"), lw=1.8, ls=(0, (3, 3)), gid="partner")
    FT(0.397, 0.175, "no points", fontsize=FS["small"], va="center")

    # ================================================================ PANEL 7
    Ns = np.arange(4, HEAT_MAX + 1, 2)
    # Row i: count of a ≤ N/2 with neither a nor N - a divisible by any of
    # SIEVE_ZS[:i + 1]. Last row: the true Goldbach count.
    H = np.zeros((len(SIEVE_ZS) + 1, len(Ns)))
    for j, n in enumerate(Ns):
        a = np.arange(1, n // 2 + 1); b = n - a
        keep = np.ones_like(a, dtype=bool)
        for i, z in enumerate(SIEVE_ZS):
            keep &= (a % z != 0) & (b % z != 0)
            H[i, j] = keep.sum()
        H[-1, j] = len(goldbach_pairs(int(n)))
    zmax = SIEVE_ZS[-1]
    lost = [(p, q) for p, q in goldbach_pairs(HEAT_MAX) if p <= zmax]
    panel(7, "Surviving pairs after sieving",
          f"Row z counts a ≤ N/2 with a and N − a free of prime factors ≤ z. The bottom row is the true\n"
          f"Goldbach count. Vertical streaks: N divisible by {', '.join(map(str, CRT))} (panel 5).", *R7)
    norm7 = LogNorm(1, H.max())
    extent = [Ns[0] - 1, Ns[-1] + 1]
    axh = FA([0.505, 0.098, 0.41, 0.135]); style(axh); REF["heat7"] = axh
    axh.imshow(np.ma.masked_equal(H[:-1], 0), aspect="auto", cmap=BRAND_CMAP, norm=norm7,
               extent=[*extent, len(SIEVE_ZS) - 0.5, -0.5], interpolation="nearest")
    axh.set_yticks(range(len(SIEVE_ZS))); axh.set_yticklabels([f"z = {z}" for z in SIEVE_ZS])
    axh.set_xticks([])
    axt = FA([0.505, 0.065, 0.41, 0.022]); style(axt); REF["actual"] = axt
    im = axt.imshow(np.ma.masked_equal(H[-1:], 0), aspect="auto", cmap=BRAND_CMAP, norm=norm7,
                    extent=[*extent, 0.5, -0.5], interpolation="nearest")
    axt.set_yticks([0]); axt.set_yticklabels(["actual"])
    axt.get_yticklabels()[0].set_color(C("TEXT"))
    axt.set_xlabel("even N", labelpad=2)
    cax = FA([0.925, 0.065, 0.008, 0.168])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_edgecolor(C("BORDER")); cb.ax.tick_params(colors=C("MUTED"), labelsize=FS["tick"])
    cb.set_label("pairs (log)", color=C("MUTED"), fontsize=FS["small"])
    FT(0.505, 0.024,
             f"N = {HEAT_MAX}: {int(H[0, -1])} survive z = 2, {int(H[-2, -1])} survive z = {zmax}; "
             f"{int(H[-1, -1])} actual pairs, {len(lost)} with p ≤ {zmax}.",
             fontsize=FS["small"], color=C("MUTED"))
    CHECKS.append(("p7 sieve ≥ true pairs with p > z",
                   H[-2, -1] >= sum(1 for p, q in goldbach_pairs(HEAT_MAX) if p > zmax)))



# ================================================================ colour and contrast audit
def artist_colours(art):
    """Non-transparent colours an artist draws with."""
    out = []
    if isinstance(art, Line2D):
        if art.get_linestyle() not in ("None", "", " ") and art.get_linewidth() > 0:
            out.append(art.get_color())
        if art.get_marker() not in (None, "None", "", " "):
            out += [art.get_markerfacecolor(), art.get_markeredgecolor()]
    elif isinstance(art, Collection):
        if art.get_array() is not None:           # colormapped; cmap endpoints checked in audit()
            return []
        out += list(art.get_facecolor())
        if np.any(art.get_linewidths()):
            out += list(art.get_edgecolor())
    elif isinstance(art, matplotlib.patches.Patch):
        out.append(art.get_facecolor())
        if art.get_linewidth() > 0:
            out.append(art.get_edgecolor())
    elif isinstance(art, Text):
        if art.get_text().strip():
            out.append(art.get_color())
    return [to_hex(c) for c in out if c is not None and to_rgba(c)[3] > 0]


def audit():
    """Colour and contrast checks on the drawn figure. Results go to CHECKS."""
    fig.canvas.draw()
    allowed = {v.lower() for v in PALETTE.values()}
    stray, role_cols = [], defaultdict(set)
    for art in fig.findobj():
        if not art.get_visible():
            continue
        cols = artist_colours(art)
        stray += [(type(art).__name__, c) for c in cols if c.lower() not in allowed]
        if art.get_gid() in SEMANTIC:
            role_cols[art.get_gid()] |= {c.lower() for c in cols}
    role_ok = True
    for role, cols in sorted(role_cols.items()):
        extra = cols - {C(role).lower(), C("PANEL_BG").lower(), C("BORDER").lower()}
        role_ok &= C(role).lower() in cols and not extra
    CHECKS.append(("colour: no colours outside PALETTE", not stray))
    CHECKS.append(("colour: every semantic role uses exactly its colour", role_ok))
    CHECKS.append(("colour: brand cmap endpoints come from PALETTE",
                   to_hex(BRAND_CMAP(0.0)) == C("PANEL_BG") and to_hex(BRAND_CMAP(1.0)) == C("HIGHLIGHT")))
    # Text is checked against PANEL_BG, the lighter of the two backgrounds.
    # Heatmap cell labels are checked against their own cell colour.
    cell_bg = {id(t): bg for t, bg in REF.get("cell_texts", [])}
    worst, low = 99, []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        cr = contrast(t.get_color(), cell_bg.get(id(t), C("PANEL_BG")))
        worst = min(worst, cr)
        if cr < MIN_TEXT_CONTRAST:
            low.append((t.get_text()[:30], round(cr, 2)))
    CHECKS.append((f"contrast: all text ≥ {MIN_TEXT_CONTRAST}:1 (worst {worst:.2f}:1)", not low))
    CHECKS.append(("contrast: secondary (MUTED) text ≥ 7:1", contrast(C("MUTED"), C("PANEL_BG")) >= 7))
    if low:
        print("low-contrast text:", low[:10])


# ================================================================ independent verification
# verify() recomputes the plotted data and number-bearing labels without the
# numpy sieve or the drawing helpers (is_prime, goldbach_pairs, two_squares).
# It uses trial division and brute force, reads the drawn values back out of the
# artists and figure text, and compares the two.
# Several checks hardcode the defaults: WHEEL = 30, CRT = (3, 5, 7), R = 10.
def tp(n):
    """Trial-division primality. Does not use the numpy sieve."""
    n = abs(n)
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def gauss_irreducible(a, b):
    """Brute force: a + bi is irreducible iff no w with 1 < N(w) < N(z) divides it."""
    n = a * a + b * b
    if n <= 1:
        return False
    for c in range(-isqrt(n), isqrt(n) + 1):
        for d in range(-isqrt(n), isqrt(n) + 1):
            m = c * c + d * d
            if 1 < m < n and n % m == 0:
                # w | z iff z·conj(w) ≡ 0 (mod N(w)) in both parts.
                # (a+bi)(c-di) = (ac+bd) + (bc-ad)i
                if (a * c + b * d) % m == 0 and (b * c - a * d) % m == 0:
                    return False
    return True


def eis_irreducible(a, b):
    """Brute force: a + bω is irreducible iff no w with 1 < N(w) < N(z) divides it."""
    n = a * a - a * b + b * b
    if n <= 1:
        return False
    lim = 2 * isqrt(n) + 2
    for c in range(-lim, lim + 1):
        for d in range(-lim, lim + 1):
            m = c * c - c * d + d * d
            if 1 < m < n and n % m == 0:
                # Same test as above. conj(c + dω) = (c - d) - dω, and
                # (a+bω)(e+fω) = (ae - bf) + (af + be - bf)ω since ω² = -1 - ω.
                e, f = c - d, -d
                re, im = a * e - b * f, a * f + b * e - b * f
                if re % m == 0 and im % m == 0:
                    return False
    return True


def verify():
    """Independent checks against the drawn figure. Returns [(description, ok)]."""
    V = []   # (description, ok)
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
    alltext = "\n".join(texts)

    def has(s):
        return s in alltext

    # ---- panel 1
    cop = [r for r in range(WHEEL) if all(r % p for p in (2, 3, 5))]
    V.append(("p1 coprime residues = {1,7,11,13,17,19,23,29}", cop == [1, 7, 11, 13, 17, 19, 23, 29]))
    ps = [n for n in range(6, 1000) if tp(n)]
    bars = [b.get_width() for b in REF["bar"].patches]
    V.append(("p1 bar lengths = trial-division counts", bars == [sum(p % 30 == r for p in ps) for r in cop]))
    V.append(("p1 bar tick labels", [t.get_text() for t in REF["bar"].get_yticklabels()] == [str(r) for r in cop]))
    V.append(("p1 bar value labels", [t.get_text() for t in REF["bar"].texts] == [str(int(b)) for b in bars]))
    V.append(("p1 range label", has(f"Primes {ps[0]}–{ps[-1]} per class")))
    V.append(("p1 φ(30) label", has(f"φ(30) = {len(cop)}")))

    # ---- panel 2
    pr = [(p, N_EX - p) for p in range(2, N_EX // 2 + 1) if tp(p) and tp(N_EX - p)]
    main = [(p, q) for p, q in pr if p > 5]
    exc = [(p, q) for p, q in pr if p <= 5]
    groups = defaultdict(list)
    for p, q in main:
        groups[tuple(sorted((p % 30, q % 30)))].append((p, q))
    for (a, b), lst in groups.items():
        V.append((f"p2 type {a}↔{b} label + count", has(f"{a} ↔ {b}") and has(f"× {len(lst)}")))
        V.append((f"p2 type {a}↔{b} pair list", has("   ".join(f"{p}+{q}" for p, q in lst))))
    V.append(("p2 exception listed", has("   ".join(f"{p} + {q}" for p, q in exc))))
    chord_w = sorted(round(l.get_linewidth(), 3) for l in REF["wheel2"].lines if l.get_gid() == "pair")
    V.append(("p2 chord widths ∝ counts", chord_w == sorted(round(1.8 * len(v) + 0.6, 3) for v in groups.values())))
    V.append(("p2 subtitle residue", has(f"N = {N_EX} ≡ {N_EX % 30} (mod 30)")))

    # ---- panel 3: brute-force irreducibility vs drawn points
    # drawn(): points of one role. hollow=True picks the inert (unfilled) markers.
    def drawn(ax, gid, hollow=None):
        out = set()
        for c in ax.collections:
            if c.get_gid() != gid:
                continue
            fc = c.get_facecolor()
            is_hollow = len(fc) == 0 or to_rgba(fc[0])[3] == 0
            if hollow is None or hollow == is_hollow:
                out |= {(round(x, 6), round(y, 6)) for x, y in c.get_offsets()}
        return out
    axg, axe = REF["lattices"]
    exp = {"split": set(), "inert": set(), "ram": set()}
    for a in range(-10, 11):
        for b in range(-10, 11):
            if gauss_irreducible(a, b):
                n = a * a + b * b
                k = "ram" if n == 2 else ("split" if tp(n) else "inert")
                exp[k].add((float(a), float(b)))
    V.append(("p3 Gaussian split points", drawn(axg, "prime", False) == exp["split"]))
    V.append(("p3 Gaussian inert points", drawn(axg, "prime", True) == exp["inert"]))
    V.append(("p3 Gaussian ramified points", drawn(axg, "exception") == exp["ram"]))
    exp = {"split": set(), "inert": set(), "ram": set()}
    for a in range(-22, 23):
        for b in range(-22, 23):
            x, y = a - b / 2, b * np.sqrt(3) / 2
            if abs(x) <= 10 and abs(y) <= 10 and eis_irreducible(a, b):
                n = a * a - a * b + b * b
                k = "ram" if n == 3 else ("split" if tp(n) else "inert")
                exp[k].add((round(x, 6), round(y, 6)))
    V.append(("p3 Eisenstein split points", drawn(axe, "prime", False) == exp["split"]))
    V.append(("p3 Eisenstein inert points", drawn(axe, "prime", True) == exp["inert"]))
    V.append(("p3 Eisenstein ramified points", drawn(axe, "exception") == exp["ram"]))

    # ---- panel 4
    big = [n for n in range(8, LIMIT + 1) if tp(n)]
    per = Counter((p % 3, p % 5, p % 7) for p in big)
    cube = [c for c in REF["cube"].collections if c.get_gid() == "prime"][0]
    xs3, ys3, zs3 = cube._offsets3d
    got = {(int(x), int(y), int(z)): round(s / 2.6) for x, y, z, s in zip(xs3, ys3, zs3, cube._sizes3d)}
    V.append(("p4 cube: 48 classes, sizes = counts", got == dict(per) and len(got) == 48))
    grid = np.zeros((7, 5), dtype=int)
    for p in big:
        grid[p % 7, p % 5] += 1
    V.append(("p4 heatmap array", np.array_equal(REF["heat4"].images[0].get_array(), grid)))
    cells = [int(t.get_text()) for t in REF["heat4"].texts]
    V.append(("p4 heatmap cell labels", cells == [int(grid[i, j]) for i in range(7) for j in range(5)]))
    V.append(("p4 range label", has(f"({min(per.values())}–{max(per.values())})") and has(f"{big[0]}–{big[-1]}")))

    # ---- panel 5
    cen = {m: next(c for c in range(m) if (2 * c - N_EX) % m == 0) for m in (5, 7)}

    def rep(r, m):
        return next(v for v in range(cen[m] - m // 2, cen[m] + m // 2 + 1) if (v - r) % m == 0)
    ep = {(rep(p % 7, 7), rep(p % 5, 5)) for p, q in pr if p > 7}
    eq = {(rep(q % 7, 7), rep(q % 5, 5)) for p, q in pr if p > 7}
    hero = REF["hero"]
    dp = {tuple(map(int, c.get_offsets()[0])) for c in hero.collections if c.get_gid() == "prime"}
    dq = {tuple(map(int, c.get_offsets()[0])) for c in hero.collections if c.get_gid() == "partner"}
    V.append(("p5 p points", dp == ep)); V.append(("p5 q points", dq == eq))
    mids = {((l.get_xdata()[0] + l.get_xdata()[1]) / 2, (l.get_ydata()[0] + l.get_ydata()[1]) / 2)
            for l in hero.lines if l.get_gid() == "pair"}
    V.append(("p5 every pair line has midpoint at the fixed class", mids == {(cen[7], cen[5])}))
    fx = [c.get_offsets()[0] for c in hero.collections if c.get_gid() == "fixed"][0]
    V.append(("p5 fixed-class marker", tuple(fx) == (cen[7], cen[5])))
    blocked = {(int(r.get_x() + .5), int(r.get_y() + .5)) for r in hero.patches if r.get_gid() == "excluded"}
    eb = {(x, y) for x in range(cen[7] - 3, cen[7] + 4) for y in range(cen[5] - 2, cen[5] + 3)
          if x % 7 == 0 or (N_EX - x) % 7 == 0 or y % 5 == 0 or (N_EX - y) % 5 == 0}
    V.append(("p5 blocked cells", blocked == eb))
    shown = [(p, q) for p, q in pr if p > 7]
    V.append(("p5 mod-3 statement covers exactly the pairs shown", has("every pair shown has")
              and all(p % 3 == q % 3 == shown[0][0] % 3 for p, q in shown)
              and len({p for p, _ in shown}) == len(dp)))
    V.append(("p5 mod-3 statement", len({p % 3 for p, q in main if p > 7} | {q % 3 for p, q in main if p > 7}) == 1
              and has(f"p ≡ q ≡ {main[-1][0] % 3} (mod 3)")))
    comet = REF["comet"].collections[0]
    offs = comet.get_offsets()
    ok = True
    for (n, cnt), a in zip(offs, comet.get_array()):
        n = int(n)
        ok &= cnt == sum(1 for p in range(2, n // 2 + 1) if tp(p) and tp(n - p))
        ok &= a == sum(1 for r in range(105) if gcd(r, 105) == 1 and gcd(n - r, 105) == 1)
    V.append(("p5 comet counts and admissible-class colours (brute force)", ok))

    # ---- panel 6
    for ax, N, p, q in REF["circles"]:
        V.append((f"p6 {p}+{q} is a Goldbach pair of {N}", tp(p) and tp(q) and p + q == N))
        pts = {tuple(map(int, o)) for c in ax.collections if c.get_gid() == "pair" for o in c.get_offsets()}
        exp = {(a, b) for n in (p, q) for a in range(-20, 21) for b in range(-20, 21) if a * a + b * b == n}
        V.append((f"p6 lattice points for {p}, {q}", pts == exp))
        for n in (p, q):
            rep2 = [(a, b) for a in range(isqrt(n) + 1) for b in range(a, isqrt(n) + 1) if a * a + b * b == n]
            lab = f"{n} = {rep2[0][0]}² + {rep2[0][1]}²" if rep2 else f"{n}: no two squares"
            V.append((f"p6 label for {n}", has(lab)))
    claim = all(any(x % 4 == 3 for x in (p, n - p))
                for n in range(8, 2001, 4) for p in range(2, n // 2 + 1) if tp(p) and tp(n - p))
    V.append(("p6 claim: N ≡ 0 (mod 4), 4 < N ≤ 2000, every pair has a prime ≡ 3 (mod 4)",
              claim and has("N > 4")))
    (_, N1, p1, q1), (_, N2, p2, q2) = REF["circles"]
    V.append(("p6 examples: N≡0 mod 4 mixed; N≡2 mod 4 both ≡1",
              N1 % 4 == 0 and (p1 % 4) != (q1 % 4) and N2 % 4 == 2 and p2 % 4 == q2 % 4 == 1))

    # ---- panel 7
    Ns = list(range(4, HEAT_MAX + 1, 2))
    zp = [z for z in SIEVE_ZS]
    H = np.zeros((len(zp) + 1, len(Ns)))
    for j, n in enumerate(Ns):
        for i, z in enumerate(zp):
            pz = [d for d in range(2, z + 1) if tp(d)]
            H[i, j] = sum(1 for a in range(1, n // 2 + 1) if all(a % d and (n - a) % d for d in pz))
        H[-1, j] = sum(1 for p in range(2, n // 2 + 1) if tp(p) and tp(n - p))
    d7 = np.ma.filled(REF["heat7"].images[0].get_array(), 0)
    da = np.ma.filled(REF["actual"].images[0].get_array(), 0)
    V.append(("p7 sieve rows", np.array_equal(d7, H[:-1])))
    V.append(("p7 actual row", np.array_equal(da, H[-1:])))
    lost = [p for p in range(2, SIEVE_ZS[-1] + 1) if tp(p) and tp(HEAT_MAX - p) and p <= HEAT_MAX // 2]
    V.append(("p7 caption", has(f"N = {HEAT_MAX}: {int(H[0, -1])} survive z = 2, {int(H[-2, -1])} survive "
                                f"z = {SIEVE_ZS[-1]}; {int(H[-1, -1])} actual pairs, {len(lost)} with p ≤ {SIEVE_ZS[-1]}.")))
    by3 = [H[-2, j] for j, n in enumerate(Ns) if n % 3 == 0 and n > 200]
    no3 = [H[-2, j] for j, n in enumerate(Ns) if n % 3 and n > 200]
    V.append(("p7 streak claim: N divisible by 3 has more survivors on average", np.mean(by3) > np.mean(no3)))
    return V


# ================================================================ render
def render(name):
    global fig, FIG_IN, LAYOUT
    FIG_IN, full_w, prev_w = EXPORTS[name]
    LAYOUT = LAYOUTS[name]
    CHECKS.clear(); REF.clear()
    fig = plt.figure(figsize=FIG_IN)
    W, H = FIG_IN
    fig.text(0.216 / W, 1 - 0.264 / H, "Primes and Goldbach’s conjecture", fontsize=FS["title"],
             family=FONT_HEAD, va="center")
    fig.text(1 - 0.216 / W, 1 - 0.264 / H, "Seven computed views · every value below is calculated, not drawn",
             fontsize=FS["small"], color=C("MUTED"), ha="right", va="center")
    draw()
    audit()
    V = verify()
    # Any failed check exits before savefig.
    lines = [f"[{name}]"] + [("PASS " if ok else "FAIL ") + d for d, ok in CHECKS + V]
    print("\n".join(lines))
    bad = [d for d, ok in CHECKS + V if not ok]
    if bad:
        raise SystemExit(f"{len(bad)} check(s) failed for {name}; image not saved.")
    out = f"images/{name.replace('_', '-')}.png"
    fig.savefig(out, dpi=full_w / W)
    fig.savefig(out.replace(".png", "-preview.png"), dpi=prev_w / W)
    plt.close(fig)
    return lines


if __name__ == "__main__":
    # Some Windows consoles can't encode ≡, φ, ≥ etc. Escape them rather than crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    report = []
    os.makedirs("images", exist_ok=True)
    for name in EXPORTS:
        report += render(name) + [""]
    with open("verification_report.txt", "w", encoding="utf-8") as f:
        f.write("Verification report: generate_visualizations.py\n"
                "Where applicable, checks independently recompute the plotted data (trial division,\n"
                "brute-force irreducibility and sieving) and compare it with values read back from\n"
                "the drawn artists and text. Other checks cover label consistency, colours and\n"
                "text contrast.\n\n")
        f.write("\n".join(report))
    print("saved all exports and verification_report.txt")
