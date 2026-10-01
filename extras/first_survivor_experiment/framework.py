"""The framework figure: the first survivor and the forcing boundary, before the results.

An explanatory figure, exact rather than schematic, in three panels:

    A  one small target on its offset line, every offset 0 <= d <= C - 2 shown, with
       λ_sieve, λ_prime and W, and the composite survivor at d = W;
    B  one prime-square block: W(N) as a straight line, λ_sieve(N) as points;
    C  several consecutive blocks: the boundary resets at each q², the points stay below.

Data, layout and drawing for this figure live here; colours, roles and text helpers
are the experiment's own (render.py). Every value drawn is computed from the
definitions in goldbach/number_theory.py.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from goldbach import number_theory as nt
from goldbach.render import card, style_axes
from goldbach.style import FS

from .render import LP, LS, col, header, keep

SIZE = (16.0, 9.0)                  # inches; 3200 x 1800 px
WIDTH_PX = 3200
HEADER_IN, FOOTER_IN = 1.08, 0.14
MARGIN, GAP = 0.144, 0.09
LANDMARK_FS = 14                    # λ_sieve and W in panel A, legible at README scale


@dataclass(frozen=True)
class OffsetLine:
    """Every offset 0 <= d <= C - 2 of one target N = 2C."""
    N: int
    C: int
    q: int
    W: int
    sieve_primes: tuple         # the primes below q
    d: np.ndarray
    alive: np.ndarray           # the pair (C - d, C + d) has no prime factor below q
    prime_pair: np.ndarray      # C - d and C + d are both prime
    lam_sieve: int
    lam_prime: int


@dataclass(frozen=True)
class Blocks:
    """Every even N in the complete prime-square blocks q² < N < r² for consecutive q."""
    qs: tuple                   # the blocks' q, in order
    rs: tuple                   # the next prime after each q: block q is q² < N < r²
    N: np.ndarray
    q: np.ndarray
    W: np.ndarray
    lam_sieve: np.ndarray


@dataclass(frozen=True)
class FrameworkData:
    line: OffsetLine
    block_q: int                # the block panel B shows
    blocks: Blocks


def offset_line(N):
    C = N // 2
    spf = nt.spf_table(N)
    primes = nt.primes_from_spf(spf)
    q = int(nt.largest_prime_below_sqrt([N], primes)[0])
    d = np.arange(C - 1)
    alive = (spf[C - d] >= q) & (spf[C + d] >= q)
    prime_pair = (spf[C - d] == C - d) & (spf[C + d] == C + d)
    return OffsetLine(N, C, q, q * q - C, tuple(primes[primes < q].tolist()), d, alive, prime_pair,
                      int(nt.first_survivor_offsets([N], [q], spf)[0]), int(nt.nearest_goldbach_offsets([N], spf)[0]))


def prime_square_blocks(q_first, q_last):
    """Every even N with q_first² < N < r², r the prime after q_last."""
    primes = nt.primes_from_spf(nt.spf_table(2 * q_last + 2))     # Bertrand: the next prime is below 2·q_last
    qs = primes[(primes >= q_first) & (primes <= q_last)]
    r = int(primes[np.searchsorted(primes, q_last) + 1])
    spf = nt.spf_table(r * r)
    lo = q_first ** 2 + 1
    N = np.arange(lo + lo % 2, r * r, 2, dtype=np.int64)
    q = nt.largest_prime_below_sqrt(N, nt.primes_from_spf(spf))
    rs = tuple(qs[1:].tolist()) + (r,)
    return Blocks(tuple(qs.tolist()), rs, N, q, nt.forcing_boundary(N, q), nt.first_survivor_offsets(N, q, spf))


def framework_data(example_N=272, block_q=23, span=(11, 31)):
    blocks = prime_square_blocks(*span)
    if block_q not in blocks.qs:
        raise ValueError("the block panel's q must be one of the blocks in panel C")
    return FrameworkData(offset_line(example_N), block_q, blocks)


# ---- drawing

def axes_in(fig, rect, x, y, w, h):
    cx, cy = rect[:2]
    W, H = fig.get_size_inches()
    ax = fig.add_axes([(cx + x) / W, (cy + y) / H, w / W, h / H])
    style_axes(ax)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def draw_offset_line(fig, rect, e):
    """Panel A: the offset line of one target. The first survivor and W are the landmarks."""
    ax = axes_in(fig, rect, 0.75, 0.72, rect[2] - 1.05, rect[3] - 1.95)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.axvspan(-0.6, e.W, color=col("GRID"), lw=0, zorder=0)
    dead = ~e.alive
    good = e.alive & e.prime_pair
    comp = e.alive & ~e.prime_pair
    ax.bar(e.d[dead], 0.2, width=0.62, color=col("excluded"), alpha=0.75, lw=0, gid="excluded")
    ax.bar(e.d[good], 1.0, width=0.72, color=col("surviving"), lw=0, gid="surviving")
    ax.bar(e.d[comp], 1.0, width=0.72, color=col("exception"), lw=0, gid="exception")
    ax.axvline(e.W, color=col("boundary"), lw=2.2, zorder=0.5, gid="boundary")

    # λ_sieve and λ_prime coincide here but are separate definitions: one callout at the
    # shared offset, joined to the bar by a dotted guide. λ_sieve (triangle) is primary,
    # λ_prime (ring) secondary.
    ys, yp = 1.24, 1.84
    x = e.lam_sieve - 1.6
    ax.plot([e.lam_sieve, e.lam_sieve], [1.0, yp - 0.07], color=col("MUTED"), lw=0.8, ls=(0, (1, 2)), zorder=4)
    ax.scatter([e.lam_sieve], [ys], s=80, marker="v", color=col("surviving"), lw=0, zorder=5, gid="surviving")
    ax.scatter([e.lam_prime], [yp], s=110, facecolors="none", edgecolors=col("pair"), lw=1.6,
               zorder=5, gid="pair")
    ax.text(x, ys + 0.02, f"{LS} = {e.lam_sieve}", fontsize=LANDMARK_FS, color=col("surviving"), ha="right",
            va="bottom", gid="surviving")
    ax.text(x, ys - 0.02, "first survivor", fontsize=FS["small"], color=col("surviving"), ha="right",
            va="top", gid="surviving")
    ax.text(x, yp + 0.02, f"{LP} = {e.lam_prime}", fontsize=FS["body"], color=col("pair"), ha="right",
            va="bottom", gid="pair")
    ax.text(x, yp - 0.02, f"nearest prime pair · {e.C - e.lam_prime} + {e.C + e.lam_prime}",
            fontsize=FS["small"], color=col("pair"), ha="right", va="top", gid="pair")
    if e.W in e.d[comp].tolist():
        ax.text(e.W + 1.5, ys, f"d = W:  {e.C - e.W} + {e.q}²  survives but is composite",
                fontsize=FS["small"], color=col("exception"), va="center", gid="exception")

    # region headers, either side of the boundary
    yh = 2.26
    ax.text(e.W - 1.5, yh, "d < W:  survival forces a prime pair", fontsize=FS["small"], color=col("TEXT"),
            ha="right", va="center")
    ax.text(e.W + 1.5, yh, "d ≥ W:  composite survivors possible", fontsize=FS["small"], color=col("MUTED"),
            va="center")

    ax.set_xlim(-1, e.C - 1)
    ax.set_ylim(0, 2.4)
    ticks = sorted({0, e.lam_sieve, e.W} | set(range(60, e.C - 1, 20)))
    ax.set_xticks(ticks)
    labels = ax.set_xticklabels([f"W = {t}" if t == e.W else str(t) for t in ticks])
    for t, lab in zip(ticks, labels):
        if t == e.W:
            lab.set_color(col("boundary")); lab.set_fontsize(LANDMARK_FS); lab.set_gid("boundary")
        elif t == e.lam_sieve:
            lab.set_color(col("surviving")); lab.set_fontsize(LANDMARK_FS); lab.set_gid("surviving")
    ax.set_xlabel("offset d, standing for the pair (C − d, C + d)", labelpad=2)

    top = ax.secondary_xaxis("top", functions=(lambda d: e.C + d, lambda u: u - e.C))
    ticks = sorted({e.C, e.q * e.q} | set(range(200, e.N - 1, 50)))
    top.set_xticks(ticks)
    top.set_xticklabels([f"q² = {t}" if t == e.q * e.q else str(t) for t in ticks])
    top.tick_params(length=2.5, width=0.6, color=col("BORDER"))
    top.spines["top"].set_color(col("BORDER"))
    top.set_xlabel("upper endpoint C + d", labelpad=5, fontsize=FS["small"], color=col("MUTED"))

    sq = dict(marker="s", ls="none", markersize=6, markeredgewidth=0)
    handles = [
        Line2D([], [], color=col("excluded"), label="dead: a factor below q", **sq),
        Line2D([], [], color=col("surviving"), label="survivor, prime pair", **sq),
        Line2D([], [], color=col("exception"), label="survivor, composite", **sq),
        Line2D([], [], color=col("boundary"), lw=2.0, label="forcing boundary W"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 0.86), ncol=2, frameon=False,
              fontsize=FS["tick"], handlelength=1.2, handletextpad=0.5, columnspacing=1.2, labelspacing=0.3,
              borderaxespad=0.1)
    return ax


def draw_block(fig, rect, b, q0):
    """Panel B: one prime-square block. The boundary is a line, the first survivor points."""
    s = b.q == q0
    N, W, lam = b.N[s], b.W[s], b.lam_sieve[s]
    ax = axes_in(fig, rect, 0.82, 0.62, rect[2] - 1.1, rect[3] - 1.75)
    ax.grid(True, axis="y", lw=0.5, color=col("GRID"))
    ax.plot(N, W, color=col("boundary"), lw=2.0, zorder=3, gid="boundary")
    ax.scatter(N, lam, s=14, color=col("surviving"), lw=0, zorder=4, gid="surviving")
    ax.set_xlim(N[0] - 6, N[-1] + 6)
    ax.set_ylim(0, W.max() * 1.15)
    ax.text(N[0] + 4, W[0] + 0.04 * W.max(), f"W(N) = {q0}² − N/2:  linear, slope −1/2", fontsize=FS["small"],
            color=col("boundary"), va="bottom", gid="boundary")
    ax.text(N[len(N) // 2], lam.max() + 0.08 * W.max(), f"{LS}(N):  discrete values below W",
            fontsize=FS["small"], color=col("surviving"), ha="center", va="bottom", gid="surviving")
    ax.set_xlabel("even N", labelpad=2)
    ax.set_ylabel("offset", labelpad=2)
    return ax


def draw_blocks(fig, rect, b, q0):
    """Panel C: consecutive blocks; the boundary resets upward at every q²."""
    ax = axes_in(fig, rect, 0.82, 0.62, rect[2] - 1.1, rect[3] - 1.75)
    ax.grid(True, axis="y", lw=0.5, color=col("GRID"))
    top_tr = blended_transform_factory(ax.transData, ax.transAxes)
    prev = None
    for qq in b.qs:
        s = b.q == qq
        N, W = b.N[s], b.W[s]
        ax.axvline(qq * qq, color=col("BORDER"), lw=0.8, zorder=0.5)       # the block starts at q²
        ax.plot(N, W, color=col("boundary"), lw=1.7, zorder=3, gid="boundary")
        if prev is not None:          # the reset at q²: a jump, drawn faint and dotted, not part of W
            ax.plot([prev[0], N[0]], [prev[1], W[0]], color=col("MUTED"), lw=0.7, ls=(0, (1, 2.5)), zorder=2)
        prev = (N[-1], W[-1])
        ax.text((N[0] + N[-1]) / 2, 0.985, f"q = {qq}", transform=top_tr, fontsize=FS["tick"],
                color=col("MUTED"), ha="center", va="top")
    ax.scatter(b.N, b.lam_sieve, s=3, color=col("surviving"), lw=0, alpha=0.75, zorder=4, gid="surviving")
    s = b.q == q0
    lo, hi = b.N[s][0] - 2, b.N[s][-1] + 2
    top = b.W[s].max() * 1.06
    ax.add_patch(Rectangle((lo, 0), hi - lo, top, fill=False, ec=col("MUTED"), lw=0.9, ls=(0, (3, 2)), zorder=5))
    r = b.rs[b.qs.index(q0)]
    ax.text(hi - 4, top * 1.01, f"{q0}² < N < {r}²", fontsize=FS["tick"], color=col("MUTED"), ha="right",
            va="bottom")
    ax.set_xlim(b.N[0] - 10, b.N[-1] + 10)
    ax.set_ylim(0, b.W.max() * 1.12)
    ax.set_yticks([0, 200, 400])
    ax.set_xlabel("even N", labelpad=2)
    ax.set_ylabel("offset", labelpad=2)
    handles = [Line2D([], [], color=col("boundary"), lw=1.7, label="W(N): one straight segment per block"),
               Line2D([], [], color=col("MUTED"), lw=0.7, ls=(0, (1, 2.5)), label="reset at the next q²"),
               Line2D([], [], color=col("surviving"), marker="o", ls="none", markersize=3.5, markeredgewidth=0,
                      label=f"{LS}(N)")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0.9), frameon=True, fontsize=FS["tick"],
                    borderaxespad=0.3, labelspacing=0.3, handlelength=1.6, borderpad=0.3, framealpha=1)
    leg.get_frame().set_facecolor(col("PANEL_BG")); leg.get_frame().set_linewidth(0)   # masks the q² dividers
    leg.set_zorder(6)
    return ax


INTRO = ("For N = 2C, offset d is the pair (C − d, C + d), sieved by the primes below q, the largest prime whose "
         "square is less than N. The forcing boundary W = q² − C is where composite survivors first appear.")


def build_framework_figure(fd):
    """Draw the framework figure. Returns (fig, axes) with axes "line", "block" and "blocks"."""
    W, H = SIZE
    fig = plt.figure(figsize=SIZE)
    header(fig, "Framework: the first survivor and the forcing boundary", INTRO, 210, tagline=False)
    e, b, q0 = fd.line, fd.blocks, fd.block_q
    h_top = 3.9
    h_bot = H - HEADER_IN - FOOTER_IN - h_top - GAP
    top = (MARGIN, FOOTER_IN + h_bot + GAP, W - 2 * MARGIN, h_top)
    wb = 6.1
    left = (MARGIN, FOOTER_IN, wb, h_bot)
    right = (MARGIN + wb + GAP, FOOTER_IN, W - 2 * MARGIN - wb - GAP, h_bot)
    primes = ", ".join(map(str, e.sieve_primes))
    card(fig, top, "A · One target on its offset line",
         f"{keep(f'N = {e.N} = 2 · {e.C}')}, {keep(f'q = {e.q}')}: every pair is sieved by {primes}. "
         f"Every valid offset {keep(f'0 ≤ d ≤ C − 2 = {e.C - 2}')} is drawn; {keep(f'W = q² − C = {e.W}')}.")
    s = b.q == q0
    r = b.rs[b.qs.index(q0)]
    card(fig, left, "B · One prime-square block",
         f"Every even N with {keep(f'{q0}² < N < {r}²')} ({int(s.sum())} targets). Inside a block q is fixed,\n"
         f"so W(N) falls linearly, while the first-survivor offset moves in discrete steps.")
    card(fig, right, "C · Consecutive blocks",
         f"Every even N with {keep(f'{b.qs[0]}² < N < {b.rs[-1]}²')}. At each q² the "
         "boundary resets upward, so W(N) is a sawtooth;\nthe first-survivor offsets stay a discrete family beneath "
         "it. The full computation follows in the next figure.")
    axes = {"line": draw_offset_line(fig, top, e), "block": draw_block(fig, left, b, q0),
            "blocks": draw_blocks(fig, right, b, q0)}
    return fig, axes
