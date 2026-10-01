"""Draws the parity visualization from ParityData.

The data decides what is shown; this module decides how. Colours, fonts and the
style_axes and card helpers are shared with the poster (goldbach.style,
goldbach.render); the cards' geometry is in layout.py.
"""
import matplotlib.pyplot as plt

from goldbach.render import card, style_axes
from goldbach.style import C, FONT_HEAD, FS

from .layout import FIGURE_SIZES, LAYOUTS


def card_axes(fig, rect):
    """Axes inside a card, below its heading, with room for tick and axis labels."""
    x, y, w, h = rect
    W, H = fig.get_size_inches()
    ax = fig.add_axes([(x + 0.85) / W, (y + 0.62) / H, (w - 1.2) / W, (h - 1.62) / H])
    style_axes(ax, grid=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlim(2, 1900)
    ax.set_xticks([2, 10, 100, 1000]); ax.set_xticklabels(["2", "10", "100", "1000"])
    ax.set_xlabel("sieve level z (largest prime sieved out)", labelpad=4)
    return ax


def build_figure(name, d):
    """Draw the parity figure for layout `name` from ParityData. Returns (fig, axes).

    axes: "average" (S/M for each N), "sums" (M and S for the featured N) and
    "classes" (sign splits of groups of pairs). Lines carry their role as gid
    ("surviving" for M, "pair" for S and S/M) and their N as label; split bars and
    their counts carry "sign_plus" or "sign_minus".
    """
    W, H = FIGURE_SIZES[name]
    cards = LAYOUTS[name]
    fig = plt.figure(figsize=(W, H))
    fig.text(0.216 / W, 1 - 0.30 / H, "What the sieve can’t see", fontsize=FS["title"],
             family=FONT_HEAD, va="center")
    fig.text(1 - 0.216 / W, 1 - 0.30 / H, "Beyond the poster · every value below is calculated, not drawn",
             fontsize=FS["small"], color=C("MUTED"), ha="right", va="center")
    fig.text(0.216 / W, 1 - 0.62 / H,
             "A pair (a, N − a) with 2 ≤ a ≤ N/2 survives the sieve at level z if neither a nor N − a has "
             "a prime factor ≤ z. Liouville’s $\\lambda(n) = (-1)^{\\Omega(n)}$, where $\\Omega(n)$ counts prime factors\n"
             "with multiplicity, is −1 on primes and +1 on products of two primes. Classical sieve information "
             "has difficulty distinguishing integers by the parity of Ω(n), which is exactly what λ records.",
             fontsize=FS["body"], color=C("MUTED"), va="top", linespacing=1.5)
    axes = {}

    # 1. The average sign S/M for each N. Same colour, different dashes: the curves nearly coincide.
    card(fig, cards["average"], "Parity stays hidden at shallow sieve depth",
         "Average of λ(a)·λ(N − a) over surviving pairs, S(z)/M(z),\n"
         "for three nearby N. The curves nearly coincide.")
    axl = card_axes(fig, cards["average"]); axes["average"] = axl
    for (N, sw), ls in zip(d.sweeps.items(), ["-", (0, (5, 3)), (0, (1, 2.5))]):
        axl.plot(sw.zs, sw.average, color=C("pair"), lw=1.6, ls=ls, label=f"N = {N:,}", gid="pair")
    axl.set_ylim(-0.05, 1.08)
    axl.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axl.legend(loc="upper left", frameon=False, handlelength=2.6)

    # 2. M and S for the featured N.
    N0 = d.featured
    sw = d.sweeps[N0]
    zs, M, S = sw.zs, sw.M, sw.S
    card(fig, cards["sums"], "Parity resolves only when forced",
         f"Survivor count M(z) and parity-sensitive sum S(z), N = {N0:,}.\n"
         f"They meet only near √N: from z = {d.meet_level}, every survivor has sign +1.")
    axr = card_axes(fig, cards["sums"]); axes["sums"] = axr
    axr.plot(zs, M, color=C("surviving"), lw=2.2, label="M", gid="surviving")
    axr.plot(zs, S, color=C("pair"), lw=2.2, label="S", gid="pair")
    axr.set_yscale("symlog", linthresh=100)
    axr.set_ylim(-1000, M.max() * 1.6)
    axr.set_ylabel("pairs (symlog scale)")
    axr.text(3, M[1] * 1.35, "survivor count  M(z)", fontsize=FS["small"], color=C("surviving"), gid="surviving")
    axr.text(40, 8, "parity-sensitive sum  S(z)", fontsize=FS["small"], color=C("pair"), gid="pair")
    axr.scatter([zs[-1]], [S[-1]], s=70, facecolor=C("PANEL_BG"), ec=C("TEXT"), lw=1.5, zorder=5,
                gid="endpoint")
    axr.annotate(f"z = {zs[-1]} (largest prime ≤ √N):\nS = M = {S[-1]:,}",
                 xy=(zs[-1], S[-1]), xytext=(1700, S[-1] * 7), ha="right", va="bottom",
                 fontsize=FS["small"], color=C("TEXT"), linespacing=1.4,
                 arrowprops=dict(arrowstyle="-", color=C("MUTED"), lw=0.8, shrinkB=6))

    # 3. Sign splits: what the sieve leaves balanced, and where the Goldbach pairs lie.
    card(fig, cards["classes"], "Goldbach lies entirely in one parity class",
         f"Pairs split by the sign of λ(a)·λ(N − a), N = {N0:,}.")
    x, y, w, h = cards["classes"]
    axc = fig.add_axes([(x + 0.3) / W, (y + 0.95) / H, (w - 0.6) / W, (h - 2.05) / H])
    axes["classes"] = axc
    style_axes(axc)
    for s_ in axc.spines.values():
        s_.set_visible(False)
    labels = {"all": "all pairs, 2 ≤ a ≤ N/2", "goldbach": "Goldbach pairs"}
    rows = d.class_splits
    for yy, c in zip(range(len(rows) - 1, -1, -1), rows):
        t = c.plus + c.minus
        axc.barh(yy, c.plus / t, height=0.4, color=C("sign_plus"), lw=0, gid="sign_plus")
        axc.barh(yy, c.minus / t, left=c.plus / t, height=0.4, color=C("sign_minus"), lw=0, gid="sign_minus")
        lab = labels.get(c.kind, f"survivors at z = {c.z}")
        axc.text(0, yy + 0.25, lab, fontsize=FS["small"], color=C("TEXT"), va="bottom")
        axc.text(0, yy - 0.25, f"+1: {c.plus:,}", fontsize=FS["small"], color=C("sign_plus"),
                 va="top", gid="sign_plus")
        axc.text(1, yy - 0.25, f"−1: {c.minus:,}", fontsize=FS["small"], color=C("sign_minus"),
                 va="top", ha="right", gid="sign_minus")
    axc.axvline(0.5, color=C("TEXT"), lw=1.0, zorder=3)
    axc.set_xlim(0, 1); axc.set_ylim(-0.75, len(rows) - 0.35)
    axc.set_yticks([]); axc.set_xticks([0, 0.5, 1]); axc.set_xticklabels(["0%", "50%", "100%"])
    fig.text((x + 0.3) / W, (y + 0.3) / H, "surviving a shallow sieve ⇏ being a Goldbach pair",
             fontsize=FS["body"], color=C("TEXT"), va="bottom")

    fig.text(0.216 / W, 0.18 / H,
             "At shallow sieve levels the surviving pairs split almost evenly between λ(a)·λ(N − a) = +1 and −1, "
             "while every Goldbach pair lies in the +1 class: divisibility by small primes does not\n"
             "determine the sign, so surviving a shallow sieve does not make a pair a Goldbach pair. The rise of "
             "S(z)/M(z) toward 1 happens only as z nears √N, where survival forces both numbers to be\n"
             "prime; it is not the sieve overcoming the parity barrier. This illustrates the classical parity "
             "problem at three values of N and proves nothing about Goldbach’s conjecture.",
             fontsize=FS["small"], color=C("MUTED"), va="bottom", linespacing=1.5)
    return fig, axes
