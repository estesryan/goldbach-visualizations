"""Draws the first-survivor figures from FirstSurvivorData. Nothing here computes
the mathematics; every number printed comes from the data.

build_figure(name, d) draws the main figure in one layout; build_blocks_figure()
draws the appendix of exact blocks. Artists carry their colour role as gid, and
ROLES maps each role to a palette key. Roles shared with the poster ("surviving",
"excluded", "pair", "exception") keep the poster's colours.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from goldbach.render import card, style_axes
from goldbach.style import BRAND_CMAP, FONT_HEAD, FS, PALETTE

from .layout import (BLOCKS_SIZE, FIGURE_SIZES, FOOTER_WRAP, HEADER_WRAP, TEXT_X, cards,
                     panel_axes)

ROLES = {
    "surviving": "ACCENT",      # surviving offsets and λ_sieve
    "excluded": "EXCLUDED",     # offsets removed by a prime below q
    "pair": "HIGHLIGHT",        # λ_prime, the nearest pair of primes
    "exception": "WARNING",     # composite survivors; λ_sieve >= W
    "boundary": "TEXT",         # the forcing boundary W
    "envelope": "ACCENT",       # dyadic-bin maxima of λ_sieve
    "fit": "ACCENT_2",          # finite-range fits to those maxima
    "block": "ACCENT",          # prime-square block maxima (appendix)
    "zoom": "MUTED",            # zoom connectors
}

LS = r"$\lambda_{\mathrm{sieve}}$"
LP = r"$\lambda_{\mathrm{prime}}$"
COMMA = FuncFormatter(lambda v, _: f"{v:,.0f}")


def col(role):
    return PALETTE[ROLES.get(role, role)]


def pow2(k):
    return f"$2^{{{k}}}$"


def inset(fig, rect, r, spines=("left", "bottom")):
    """Axes at r = (x, y, w, h) inches, measured from the card's lower-left corner."""
    cx, cy = rect[:2]
    W, H = fig.get_size_inches()
    ax = fig.add_axes([(cx + r[0]) / W, (cy + r[1]) / H, r[2] / W, r[3] / H])
    style_axes(ax)
    for s in ("left", "right", "top", "bottom"):
        ax.spines[s].set_visible(s in spines)
    return ax


def label_above(ax, s, pad_in=0.06):
    """A label pad_in inches above the axes' top-left corner."""
    h_in = ax.get_position().height * ax.figure.get_figheight()
    ax.text(0, 1 + pad_in / h_in, s, transform=ax.transAxes, fontsize=FS["small"], color=col("TEXT"),
            va="bottom")


# ---- panel A: the worked target

def offset_bars(ax, d, alive, composite=None):
    """One bar per offset: survivors full height, dead offsets short and grey,
    composite survivors in the exception colour."""
    composite = np.zeros(len(d), bool) if composite is None else composite
    good = alive & ~composite
    ax.bar(d[~alive], 0.32, width=1.45, color=col("excluded"), lw=0, gid="excluded")
    ax.bar(d[good], 1.0, width=1.45, color=col("surviving"), lw=0, gid="surviving")
    if composite.any():
        ax.bar(d[composite], 1.0, width=1.45, color=col("exception"), lw=0, gid="exception")
    ax.set_yticks([])


def zoom_fan(fig, src, x, dst):
    """Two connector lines from data x just above src's label up to the lower corners
    of dst, below its tick labels."""
    for xd in (0, 1):
        fig.add_artist(ConnectionPatch((x, 1.62), (xd, -0.3), coordsA=src.get_xaxis_transform(),
                                       coordsB=dst.transAxes, color=col("zoom"), lw=0.6, zorder=0,
                                       gid="zoom"))


def draw_example(fig, rect, e, geo, portrait):
    axes = {}
    kind = "odd" if e.parity_class else "even"

    # centre zoom
    ax = inset(fig, rect, geo["centre"], spines=("bottom",))
    axes["example_centre"] = ax
    offset_bars(ax, e.centre_d, e.centre_alive)
    ys, yp = 1.22, 1.72
    ax.scatter([e.lam_sieve], [ys], s=46, marker="v", color=col("surviving"), lw=0, zorder=5, gid="surviving")
    ax.scatter([e.lam_prime], [yp], s=95, facecolors="none", edgecolors=col("pair"), lw=1.5, zorder=5, gid="pair")
    dx = 0.035 * (e.centre_d.max() + 4)
    ax.text(e.lam_sieve + dx, ys, f"first survivor, {LS} = {e.lam_sieve}", fontsize=FS["small"],
            color=col("surviving"), va="center", gid="surviving")
    ax.text(e.lam_prime + dx, yp, f"nearest prime pair, {LP} = {e.lam_prime}", fontsize=FS["small"],
            color=col("pair"), va="center", gid="pair")
    ax.set_xlim(-3, e.centre_d.max() + 3)
    ax.set_ylim(0, 2.05)
    ax.set_xticks([0] + [x for x in range(20, int(e.centre_d.max()) + 1, 20)])
    label_above(ax, f"Near the centre: {kind} offsets d (even d gives two even numbers)", pad_in=0.04)

    # boundary zoom
    ax = inset(fig, rect, geo["edge"], spines=("bottom",))
    axes["example_edge"] = ax
    ax.axvline(e.W, color=col("boundary"), lw=1.1, ls=(0, (3, 2)), zorder=0.5, gid="boundary")
    offset_bars(ax, e.edge_d, e.edge_alive, e.edge_composite)
    lo, hi = int(e.edge_d.min()), int(e.edge_d.max())
    ax.set_xlim(lo - 3, hi + 3)
    ax.set_ylim(0, 2.05)
    k = (hi - lo) // 2
    ax.set_xticks([e.W - k, e.W, e.W + k])
    ax.set_xticklabels([f"W − {k}", f"W = {e.W:,}", f"W + {k}"])
    span = hi - lo
    for c in e.composite[(e.composite >= lo) & (e.composite <= hi)].tolist():
        ax.text(c + 0.03 * span, 1.6, f"d = W: {e.C - c:,} + {e.q}²\nsurvives, but is composite",
                fontsize=FS["small"], color=col("exception"), va="center", linespacing=1.3, gid="exception")
    ax.text(e.W - 0.03 * span, 1.6, "d < W: every survivor\nis a pair of primes", fontsize=FS["small"],
            color=col("TEXT"), ha="right", va="center", linespacing=1.3)
    label_above(ax, f"At the boundary: {kind} d within {k} of W ({e.n_outside} survivors have d ≥ W)",
                pad_in=0.04)

    # the whole offset range, with the forcing window shaded
    ax = inset(fig, rect, geo["overview"], spines=("bottom",))
    axes["example_overview"] = ax
    ax.axvspan(0, e.W, color=col("GRID"), lw=0)
    ax.axvline(e.W, color=col("boundary"), lw=1.4, gid="boundary")
    ax.set_xlim(0, e.C - 2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, e.W]); ax.set_xticklabels(["0", "W"])
    inside = "every one a pair of primes" if e.inside_all_prime else "not all pairs of primes"
    ax.text(0.015 * e.C, 0.5, f"{e.n_inside:,} survivors with d < W: {inside}", fontsize=FS["small"],
            color=col("TEXT"), va="center")
    label_above(ax, f"All offsets 0 ≤ d ≤ C − 2 = {e.C - 2:,}; the shaded window is d < W", pad_in=0.05)
    if portrait:        # the zooms sit above the ends of the range they magnify
        zoom_fan(fig, ax, 0, axes["example_centre"])
        zoom_fan(fig, ax, e.W, axes["example_edge"])

    # legend
    ax = inset(fig, rect, geo["legend"], spines=())
    ax.set_xticks([]); ax.set_yticks([])
    sq = dict(marker="s", ls="none", markersize=7, markeredgewidth=0)
    handles = [
        Line2D([], [], color=col("surviving"), label="surviving offset", **sq),
        Line2D([], [], color=col("excluded"), label="removed by a prime below q", **sq),
        Line2D([], [], color=col("exception"), label="composite survivor", **sq),
        Line2D([], [], color=col("boundary"), lw=1.4, label="forcing boundary W"),
    ]
    ax.legend(handles=handles, loc="center left", ncol=4 if portrait else 2, frameon=False,
              handlelength=1.3, columnspacing=1.4, borderaxespad=0)
    axes["example_legend"] = ax
    return axes


# ---- panel B: forcing boundary against first survivor

def draw_small(fig, rect, d, r, small_max):
    """Linear view of the complete prime-square blocks up to small_max."""
    ax = inset(fig, rect, r)
    ax.grid(True, axis="y", lw=0.5, color=col("GRID"))
    sw = d.sweep
    s = sw.N <= small_max
    N, q, W, lam = sw.N[s], sw.q[s], sw.W[s], sw.lam_sieve[s]
    ax.set_xlim(0, small_max + 4)
    ax.set_ylim(-16, 100)
    x0, x1 = ax.get_xlim()
    width_in = r[2] / (x1 - x0)
    ax.axhline(0, color=col("BORDER"), lw=0.8, zorder=1)
    for i, qq in enumerate(np.unique(q).tolist()):
        b = q == qq
        if i % 2:
            ax.axvspan(N[b][0] - 1, N[b][-1] + 1, color=col("GRID"), lw=0, zorder=0)
        ax.plot(N[b], W[b], color=col("boundary"), lw=1.6, zorder=2, gid="boundary")
        mid, wid = (N[b][0] + N[b][-1]) / 2, (N[b][-1] - N[b][0] + 2) * width_in
        if wid > 0.12:
            ax.text(mid, 97, f"q = {qq}" if wid > 0.42 else str(qq), fontsize=FS["tick"], color=col("MUTED"),
                    ha="center", va="top")
    comp = np.isin(N, d.composite_first)
    out = np.isin(N, d.prime_outside)
    ok = ~(comp | out)
    ax.scatter(N[ok], lam[ok], s=11, color=col("surviving"), lw=0, zorder=4, gid="surviving")
    ax.scatter(N[comp], lam[comp], s=30, color=col("exception"), lw=0, zorder=5, gid="exception")
    ax.scatter(N[out], lam[out], s=30, facecolors="none", edgecolors=col("exception"), lw=1.2, zorder=5,
               gid="exception")
    i = int(np.searchsorted(N, 100))
    ax.annotate(f"N = 100: λ = {lam[i]} ≥ W = {W[i]}".replace("-", "−"), xy=(100, lam[i]),
                xytext=(14, 50), fontsize=FS["small"], color=col("TEXT"), va="center",
                arrowprops=dict(arrowstyle="-", color=col("MUTED"), lw=0.8, shrinkA=2, shrinkB=4))
    k = int(np.searchsorted(N, d.max_ratio_N))
    ax.annotate(f"N = {d.max_ratio_N}: λ = {lam[k]} < W = {W[k]}", xy=(d.max_ratio_N, lam[k]),
                xytext=(small_max + 2, -9), ha="right", fontsize=FS["small"], color=col("TEXT"), va="center",
                arrowprops=dict(arrowstyle="-", color=col("MUTED"), lw=0.8, shrinkA=2, shrinkB=4))
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("even N", labelpad=2)
    ax.set_ylabel("offset d", labelpad=2)
    handles = [
        Line2D([], [], color=col("boundary"), lw=1.6, label="W(N)"),
        Line2D([], [], color=col("surviving"), marker="o", ls="none", markersize=4.5, markeredgewidth=0,
               label=f"{LS} < W"),
        Line2D([], [], color=col("exception"), marker="o", ls="none", markersize=6, markeredgewidth=0,
               label=f"{LS} ≥ W, composite ({len(d.composite_first)})"),
        Line2D([], [], color=col("exception"), marker="o", ls="none", markersize=6, markerfacecolor="none",
               markeredgewidth=1.2, label=f"{LS} ≥ W, prime pair ({len(d.prime_outside)})"),
    ]
    return ax, handles


def draw_large(fig, rect, d, r, rc):
    """λ_sieve for every N from min_N, as the share of each column of log N per row,
    against W(N)."""
    ax = inset(fig, rect, r)
    h = d.histogram
    share = np.ma.masked_equal(h.share, 0)
    mesh = ax.pcolormesh(h.x_edges, h.y_edges, share, cmap=BRAND_CMAP, rasterized=True,
                         norm=LogNorm(vmin=1e-4, vmax=1), zorder=1)
    sw = d.sweep
    s = sw.N >= d.min_N
    ax.plot(sw.N[s], sw.W[s], color=col("boundary"), lw=0.9, zorder=3, rasterized=True, gid="boundary")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10, linscale=0.7)
    ax.set_xlim(h.x_edges[0], h.x_edges[-1])
    ax.set_ylim(-0.5, 2e6)
    ax.yaxis.set_major_locator(FixedLocator([0, 10, 100, 1e3, 1e4, 1e5, 1e6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["0", "10", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
    ax.set_xlabel("even N (log scale)", labelpad=2)
    ax.set_ylabel("offset d (symlog)", labelpad=2)
    top = sw.N >= d.n_max // 2
    kt = d.n_max.bit_length() - 2
    ax.text(0.03, 0.97, f"N in [{pow2(kt)}, {pow2(kt + 1)}):  W ≥ {int(sw.W[top].min()):,},  "
            f"{LS} ≤ {int(sw.lam_sieve[top].max()):,}", transform=ax.transAxes, fontsize=FS["tick"],
            color=col("TEXT"), va="top")
    ax.text(0.03, 0.86, f"shaded: {LS}(N)", transform=ax.transAxes,
            fontsize=FS["tick"], color=col("surviving"), va="top", gid="surviving")
    ax.text(0.97, 0.66, "W(N)", transform=ax.transAxes, fontsize=FS["small"], color=col("boundary"),
            ha="right", va="center", gid="boundary")
    cax = inset(fig, rect, rc, spines=())
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_edgecolor(col("BORDER"))
    cb.set_ticks([1e-4, 1e-2, 1]); cb.set_ticklabels(["0.01%", "1%", "100%"])
    cax.tick_params(labelsize=FS["tick"] - 0.5, length=2, color=col("BORDER"))
    cb.set_label("share of the N in its column", fontsize=FS["tick"], color=col("MUTED"), labelpad=2)
    return ax


def draw_boundary(fig, rect, d, geo, small_max, portrait):
    small, handles = draw_small(fig, rect, d, geo["small"], small_max)
    small.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.0), frameon=False,
                 fontsize=FS["tick"], handlelength=1.4, borderaxespad=0.1, labelspacing=0.3, handletextpad=0.5,
                 columnspacing=1.2, ncol=4 if portrait else 2)
    label_above(small, f"Even N ≤ {small_max}: W falls linearly in each block; {LS} moves in jumps",
                pad_in=0.48 if not portrait else 0.3)
    large = draw_large(fig, rect, d, geo["large"], geo["colorbar"])
    label_above(large, f"Every even N with {d.min_N} ≤ N < {pow2(d.n_max.bit_length() - 1)}")
    return {"small": small, "large": large}


# ---- panel C: the finite-range empirical envelope

FIT_STYLES = ["-", (0, (5, 2.5)), (0, (1.2, 2))]


def draw_envelope(fig, rect, d, geo):
    ax = inset(fig, rect, geo["main"])
    ax.grid(True, lw=0.5, color=col("GRID"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(d.bin_argN, d.bin_max, s=34, color=col("envelope"), lw=0, zorder=5, gid="envelope")
    for f, ls in zip(d.fits, FIT_STYLES):
        x = np.geomspace(*f.x_range, 60)
        ax.plot(x, np.exp(f.log_A) * x ** f.alpha_N, color=col("fit"), lw=1.5, ls=ls, zorder=4, gid="fit",
                label=f"fit over N < {pow2(f.cutoff_exp)}: " + r"$\alpha_N$" + f" = {f.alpha_N:.3f}")
    ax.legend(loc="lower right", frameon=False, fontsize=FS["tick"], handlelength=2.4, labelspacing=0.35,
              borderaxespad=0.4)
    lo = 2 ** int(d.bin_exps[0])
    ax.set_xlim(lo * 0.85, d.n_max * 1.15)
    ax.set_ylim(d.bin_max.min() / 2.2, d.bin_max.max() * 2.2)
    ax.set_xlabel("even N (log scale)", labelpad=2)
    ax.set_ylabel(f"largest {LS} in the bin (log scale)", labelpad=2)
    ax.text(0.03, 0.96, f"bin maxima over [$2^k$, $2^{{k+1}}$), k = {int(d.bin_exps[0])} … {int(d.bin_exps[-1])}",
            transform=ax.transAxes, fontsize=FS["tick"], color=col("envelope"), va="top", gid="envelope")
    ax.text(0.03, 0.86, "fits by least squares on log–log axes", transform=ax.transAxes,
            fontsize=FS["tick"], color=col("fit"), va="top", gid="fit")
    label_above(ax, "Each bin maximum is placed at the N that attains it")

    # the drift: the fitted exponent against the range
    dr = inset(fig, rect, geo["drift"])
    dr.grid(True, axis="y", lw=0.5, color=col("GRID"))
    ks = [f.cutoff_exp for f in d.fits]
    al = [f.alpha_N for f in d.fits]
    dr.plot(ks, al, color=col("fit"), lw=1.2, marker="o", markersize=5, gid="fit")
    for k, a in zip(ks, al):
        dr.text(k, a + 0.022, f"{a:.3f}", fontsize=FS["tick"], color=col("TEXT"), ha="center", va="bottom")
    dr.set_xticks(ks); dr.set_xticklabels([f"N < {pow2(k)}" for k in ks])
    dr.set_xlim(min(ks) - 1.3, max(ks) + 1.3)
    pad = 0.06
    dr.set_ylim(min(al) - pad, max(al) + 1.6 * pad)
    dr.set_ylabel(r"fitted $\alpha_N$", labelpad=2)
    q_ax = dr.secondary_yaxis("right", functions=(lambda a: 2 * a, lambda a: a / 2))
    q_ax.tick_params(length=2.5, width=0.6, color=col("BORDER"))
    for sp in q_ax.spines.values():
        sp.set_color(col("BORDER"))
    q_ax.set_ylabel(r"$2\alpha_N$: q-units via $N \approx q^2$", fontsize=FS["tick"], color=col("MUTED"),
                    labelpad=3)
    label_above(dr, r"The fitted exponent $\alpha_N$ in $\lambda \approx A\,N^{\alpha_N}$ drifts with the range")
    return {"envelope": ax, "drift": dr}


# ---- the main figure

INTRO = ("Sieve the symmetric pairs (N/2 − d, N/2 + d) by every prime below q, the largest prime below √N. "
         "Survival alone does not distinguish a prime from a composite whose prime factors are all at least q. "
         "Below q², no such composite can occur, so a sieve survivor there is prime by size alone. "
         "The forcing boundary W = q² − N/2 is the offset at which N/2 + d reaches q². "
         "The experiment measures where the first survivor lands relative to that forcing boundary.")


def header(fig, title, intro, wrap, tagline=True):
    W, H = fig.get_size_inches()
    fig.text(TEXT_X / W, 1 - 0.30 / H, title, fontsize=FS["title"], family=FONT_HEAD, va="center")
    if tagline:
        fig.text(1 - TEXT_X / W, 1 - 0.30 / H, "Beyond the poster · every value below is calculated, not drawn",
                 fontsize=FS["small"], color=col("MUTED"), ha="right", va="center")
    fig.text(TEXT_X / W, 1 - 0.62 / H, textwrap.fill(intro, wrap), fontsize=FS["body"], color=col("MUTED"),
             va="top", linespacing=1.5)


def caption(d):
    """The footer sentences, every number from the data."""
    sw = d.sweep
    big = sw.N >= d.min_N
    last_q = int(sw.q[sw.N == d.exceptions.max()][0])
    f = d.fits[-1]
    growth = ("a power of log N fits the same bin maxima at least as well, so this range cannot separate a power "
              "law from slower growth." if f.rss_log <= f.rss_power else "no growth law is implied.")
    return [
        f"For every even N from {d.min_N} to {d.n_max - 2:,} the first survivor lies inside the forcing boundary, "
        "so it is a pair of primes and equals the nearest Goldbach pair.",
        f"Below {d.min_N} it lies at or beyond W for {len(d.exceptions)} values of N, "
        f"all with {keep(f'q ≤ {last_q}')}, "
        f"and {len(d.composite_first)} of those first survivors are composite.",
        f"Composite survivors occur for {d.n_with_composite:,} of the {int(big.sum()):,} even N from {d.min_N} to "
        f"{d.n_max - 2:,}, "
        "never below W.",
        f"From {d.min_N} the first-survivor offset is at most {d.max_ratio:.3f}·W (at N = {d.max_ratio_N}).",
        f"The fits describe this finite range only: {growth}",
        "This is a finite computation; it does not resolve the parity problem or prove Goldbach’s conjecture.",
    ]


NBSP = "\u00a0"     # textwrap breaks only at ASCII whitespace, so formulas joined by NBSP stay whole


def keep(s):
    return s.replace(" ", NBSP)


def subtitle(portrait, text):
    return textwrap.fill(text, 205 if portrait else 60)


def build_figure(name, d, small_max=288):
    """Draw the main figure for layout `name`. Returns (fig, axes).

    axes: "example_centre", "example_edge", "example_overview", "example_legend"
    (panel A), "small" and "large" (panel B), "envelope" and "drift" (panel C).
    """
    W, H = FIGURE_SIZES[name]
    portrait = W < H
    cs = cards(name)
    geo = panel_axes(name)
    fig = plt.figure(figsize=(W, H))
    e = d.example
    header(fig, "Where the first survivor lands", INTRO, HEADER_WRAP[name])
    axes = {}
    card(fig, cs["example"], "One target, offset by offset", subtitle(
        portrait, f"{keep(f'N = {e.N:,}')}, so {keep(f'C = {e.C:,}')} and {keep(f'q = {e.q}')}; "
                  f"{keep(f'W = q² − C = {e.W:,}')}. First survivor and nearest prime pair are defined "
                  f"separately; here both sit at {keep(f'd = {e.lam_sieve}')}: "
                  f"{keep(f'{e.C - e.lam_prime:,} + {e.C + e.lam_prime:,}')}."))
    axes.update(draw_example(fig, cs["example"], e, geo["example"], portrait))
    card(fig, cs["boundary"], "From one target to every N", subtitle(
        portrait, f"In each prime-square block {keep('q² < N < r²')}, {keep('W = q² − N/2')} falls linearly. "
                  "A first survivor that lands below W is forced to be prime."))
    axes.update(draw_boundary(fig, cs["boundary"], d, geo["boundary"], small_max, portrait))
    card(fig, cs["envelope"], "A finite-range empirical envelope", subtitle(
        portrait, "The largest first-survivor offset in each dyadic bin of N, fitted over three nested "
                  "ranges. The exponent is empirical and drifts as the range grows; none is asymptotic."))
    axes.update(draw_envelope(fig, cs["envelope"], d, geo["envelope"]))
    fig.text(TEXT_X / W, 0.16 / H, textwrap.fill(" ".join(caption(d)), FOOTER_WRAP[name]), fontsize=FS["small"],
             color=col("MUTED"), va="bottom", linespacing=1.5)
    return fig, axes


# ---- the appendix: exact blocks beyond the main range

def build_blocks_figure(blocks, d):
    """Exact block maxima of the selected prime-square blocks beyond the main range,
    beside every complete block inside it. Not joined and not fitted.
    Returns (fig, axes) with axes "maxima" and "ratio"."""
    W, H = BLOCKS_SIZE
    fig = plt.figure(figsize=(W, H))
    n_targets = sum(b.n_targets for b in blocks)
    header(fig, "Exact prime-square blocks beyond the main range",
           "Each point is one complete block q² < N < r² (r the next prime), every even N in it computed. "
           "Inside the main range every block is shown. Beyond it, "
           f"{len(blocks)} blocks are selected by a fixed rule: q is the largest prime at or below each of a "
           "list of log-spaced targets.",
           100, tagline=False)
    rect = (0.144, 1.0, W - 0.288, H - 1.0 - 1.45)
    card(fig, rect, "Block maxima of the first-survivor offset", "")
    x, y, w, h = rect
    mq = np.array([b.q for b in d.blocks]); sq = np.array([b.q for b in blocks])
    edge = float(np.sqrt(d.n_max))

    def panel(r):
        ax = fig.add_axes([(x + r[0]) / W, (y + r[1]) / H, r[2] / W, r[3] / H])
        style_axes(ax, grid=True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(mq.min() * 0.8, sq.max() * 1.3)
        ax.axvline(edge, color=col("MUTED"), lw=0.9, ls=(0, (4, 3)), zorder=1)
        return ax

    top = panel((0.95, 2.62, w - 1.25, h - 3.25))
    top.scatter(mq, [b.max_lam for b in d.blocks], s=10, color=col("block"), lw=0, zorder=3, gid="block")
    top.scatter(sq, [b.max_lam for b in blocks], s=44, facecolors="none", edgecolors=col("block"), lw=1.3,
                zorder=4, gid="block")
    top.set_ylabel(f"largest {LS} in the block", labelpad=3)
    top.tick_params(labelbottom=False)
    top.text(edge * 0.92, top.get_ylim()[1], f"N = {pow2(d.n_max.bit_length() - 1)}\nend of the main range",
             fontsize=FS["tick"], color=col("MUTED"), ha="right", va="top", linespacing=1.3)
    handles = [
        Line2D([], [], color=col("block"), marker="o", ls="none", markersize=4, markeredgewidth=0,
               label=f"every complete block inside N < {pow2(d.n_max.bit_length() - 1)} ({len(d.blocks)})"),
        Line2D([], [], color=col("block"), marker="o", ls="none", markersize=7, markerfacecolor="none",
               markeredgewidth=1.3, label=f"selected exact blocks beyond it ({len(blocks)})"),
    ]
    top.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS["small"], handletextpad=0.5)

    bot = panel((0.95, 0.62, w - 1.25, 1.62))
    bot.scatter(mq, [b.max_ratio for b in d.blocks], s=10, color=col("block"), lw=0, zorder=3, gid="block")
    bot.scatter(sq, [b.max_ratio for b in blocks], s=44, facecolors="none", edgecolors=col("block"), lw=1.3,
                zorder=4, gid="block")
    bot.axhline(1, color=col("boundary"), lw=1.0, gid="boundary")
    bot.set_ylim(min(b.max_ratio for b in blocks) / 4, 3)
    bot.text(sq.max() * 1.2, 1.25, f"{LS} = W: the forcing boundary", fontsize=FS["tick"], color=col("boundary"),
             ha="right", va="bottom", gid="boundary")
    bot.set_ylabel(f"largest {LS} / W", labelpad=3)
    bot.set_xlabel("q, the sieve depth of the block (log scale)", labelpad=3)
    bot.xaxis.set_major_formatter(COMMA)
    inside = all(b.all_inside for b in blocks) and all(b.all_inside for b in d.blocks)
    qs = f"q from {sq.min():,} to {sq.max():,}"
    fig.text(TEXT_X / W, 0.16 / H, textwrap.fill(
        f"Selected blocks: {qs}, {n_targets:,} values of N, all computed. "
        + ("In every block shown, the first survivor lies inside the forcing boundary for every N. " if inside else
           "In some block shown, the first survivor reaches the forcing boundary. ")
        + "A block maximum depends on the block's width r − q as well as on q, and the selected blocks are sparse, "
        "so these points are not an envelope; they are not joined or fitted. This is a finite computation and "
        "states no growth law.", 128), fontsize=FS["small"], color=col("MUTED"), va="bottom", linespacing=1.5)
    return fig, {"maxima": top, "ratio": bot}
