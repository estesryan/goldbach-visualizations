"""Draws the seven panels from the prepared visualization data.

The data decides what is shown; this module decides how. Nothing here
computes the mathematics a panel represents.

Each draw_*(cv, d, axes) function draws one visualization onto a PanelCanvas
and records its main axes by name in `axes`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, to_hex, to_rgba
from matplotlib.patches import Circle, Rectangle

from .layout import FIGURE_SIZES, LAYOUTS, PanelCanvas
from .style import BRAND_CMAP, C, FONT_HEAD, FS, contrast


# ---- helpers

def panel(cv, num, title, sub):
    cv.set_panel(num)
    x, y, w, h = cv.new
    W, H = cv.fig_in
    cv.fig.patches.append(Rectangle((x / W, y / H), w / W, h / H, transform=cv.fig.transFigure,
                                    fc=C("PANEL_BG"), ec=C("BORDER"), lw=0.8, zorder=-10))
    top = (y + h) / H
    cv.fig.text((x + 0.162) / W, top - 0.24 / H, f"{num:02d}", fontsize=FS["num"], color=C("ACCENT"),
                va="center")
    cv.fig.text((x + 0.54) / W, top - 0.24 / H, title, fontsize=FS["head"], family=FONT_HEAD, va="center")
    if sub:
        cv.fig.text((x + 0.162) / W, top - 0.516 / H, sub, fontsize=FS["sub"], va="top",
                    color=C("MUTED"), linespacing=1.4)


def style_axes(ax, grid=False):
    ax.set_facecolor(C("PANEL_BG"))
    for s in ax.spines.values():
        s.set_color(C("BORDER")); s.set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.6, color=C("BORDER"))
    if grid:
        ax.grid(True, lw=0.5, color=C("GRID"))


def wheel(ax, modulus, coprime, label_extra=()):
    """Residue wheel mod modulus. Coprime classes large, the rest small and grey."""
    ang = np.pi / 2 - 2 * np.pi * np.arange(modulus) / modulus
    xs, ys = np.cos(ang), np.sin(ang)
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=C("EXCLUDED"), lw=0.8, zorder=1))
    for k in range(modulus):
        good = k in coprime
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


def dot_label(cv, x, y, role, text, marker="o", hollow=False, size=9):
    """Legend marker and label at landscape figure coords."""
    kw = dict(mfc="none", mec=C(role), mew=1.6) if hollow else dict(mfc=C(role), mec="none")
    cv.line([x], [y], marker=marker, ms=size, ls="", gid=role, **kw)
    cv.text(x + 0.011, y, text, fontsize=FS["small"], va="center")


# ---- panels

def draw_prime_wheel(cv, d, axes):
    """The prime wheel."""
    wheel_primes, coprime = d.wheel_primes, d.coprime
    wp = "·".join(map(str, wheel_primes))
    panel(cv, 1, f"The prime wheel (mod {d.wheel})",
          f"Every prime > {max(wheel_primes)} is coprime to {d.wheel} = {wp}, so it lies in\n"
          f"one of φ({d.wheel}) = {len(coprime)} classes.")
    ax = cv.axes([0.012, 0.668, 0.175, 0.21])
    xs, ys = wheel(ax, d.wheel, coprime, label_extra=wheel_primes)
    for r in wheel_primes:
        ax.scatter(xs[r], ys[r], s=130, facecolor="none", ec=C("exception"), lw=1.3, zorder=4,
                   gid="exception")
    big = d.primes
    axb = cv.axes([0.222, 0.70, 0.095, 0.155]); style_axes(axb); axes["bar"] = axb
    vals = d.counts
    axb.barh(range(len(coprime)), vals, color=C("prime"), height=0.62, gid="prime")
    axb.set_yticks(range(len(coprime))); axb.set_yticklabels(coprime)
    axb.invert_yaxis(); axb.set_xticks([])
    for s in ("top", "right", "bottom"):
        axb.spines[s].set_visible(False)
    for i, v in enumerate(vals):
        axb.text(v + 0.6, i, str(v), va="center", fontsize=FS["tick"], color=C("MUTED"))
    axb.set_xlim(0, max(vals) * 1.3)
    cv.text(0.205, 0.875, f"Primes {big[0]}–{big[-1]} per class", fontsize=FS["small"],
            color=C("MUTED"))
    dot_label(cv, 0.205, 0.684, "exception", f"{', '.join(map(str, wheel_primes))}: the sieving primes",
              hollow=True)


def draw_goldbach_wheel(cv, d, axes):
    """Goldbach pairs on the wheel."""
    small_cut, exc = d.small_cut, d.exceptions
    s, types, by_type = d.residue, d.types, d.by_type
    panel(cv, 2, "Goldbach pairs on the wheel",
          f"N = {d.N} ≡ {s} (mod {d.wheel}), so p ≡ r forces q ≡ {s} − r.\n"
          f"With p, q > {small_cut}, only {len(types)} residue types are possible.")
    ax = cv.axes([0.343, 0.668, 0.175, 0.21]); axes["wheel2"] = ax
    xs, ys = wheel(ax, d.wheel, d.coprime,
                   label_extra=sorted({p % d.wheel for p, _ in exc} | {q % d.wheel for _, q in exc}
                                      - set(d.coprime)))
    for (a, b), lst in by_type.items():
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=C("pair"), lw=1.8 * len(lst) + 0.6,
                solid_capstyle="round", zorder=2, gid="pair")
    for p, q in exc:
        a, b = p % d.wheel, q % d.wheel
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=C("exception"), lw=1.3, ls=(0, (2, 2)),
                zorder=2, gid="exception")
    y0 = 0.862
    for t in types:
        lst = by_type.get(t, [])
        cv.text(0.528, y0, f"{t[0]} ↔ {t[1]}", fontsize=FS["head"] - 1, family=FONT_HEAD,
                color=C("pair"), va="top", gid="pair")
        cv.text(0.610, y0 - 0.002, f"× {len(lst)}", fontsize=FS["body"], color=C("MUTED"), va="top")
        cv.text(0.528, y0 - 0.029, "   ".join(f"{p}+{q}" for p, q in lst) or "none",
                fontsize=FS["small"], color=C("TEXT"), va="top")
        y0 -= 0.07
    if exc:
        cv.text(0.528, y0, "   ".join(f"{p} + {q}" for p, q in exc), fontsize=FS["body"],
                color=C("exception"), va="top", gid="exception")
        cv.text(0.528, y0 - 0.025, f"small-prime exception:\n{', '.join(str(p) for p, _ in exc)} "
                f"is removed by the wheel", fontsize=FS["small"], color=C("MUTED"), va="top",
                linespacing=1.3)


def draw_lattice_primes(cv, d, axes):
    """Gaussian and Eisenstein primes."""
    panel(cv, 3, "Gaussian and Eisenstein primes",
          "Dots are prime elements z. Style shows how the rational\n"
          "prime p below z factors: split, inert or ramified.")
    R = d.radius

    def lattice_plot(rect, lp, title):
        ax = cv.axes(rect); style_axes(ax); axes.setdefault("lattices", []).append(ax)
        allp = lp.points
        ax.scatter(allp[:, 0], allp[:, 1], s=1.5, c=C("lattice"), lw=0, gid="lattice")
        g = lp.split
        ax.scatter(g[:, 0], g[:, 1], s=9, c=C("prime"), lw=0, alpha=0.75, gid="prime")
        g = lp.inert
        ax.scatter(g[:, 0], g[:, 1], s=26, facecolor="none", ec=C("prime"), lw=1.2, gid="prime")
        g = lp.ram
        ax.scatter(g[:, 0], g[:, 1], s=46, c=C("exception"), lw=0, zorder=3, gid="exception")
        ax.set_aspect("equal"); ax.set_xlim(-R - .8, R + .8); ax.set_ylim(-R - .8, R + .8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=FS["body"], color=C("TEXT"), pad=4)

    lattice_plot([0.690, 0.712, 0.125, 0.14], d.gaussian, "Gaussian  ℤ[i]")
    lattice_plot([0.845, 0.712, 0.125, 0.14], d.eisenstein, "Eisenstein  ℤ[ω]")
    dot_label(cv, 0.680, 0.694, "prime", "split    p ≡ 1 (mod 4) · p ≡ 1 (mod 3)", size=6)
    dot_label(cv, 0.680, 0.679, "prime", "inert    p ≡ 3 (mod 4) · p ≡ 2 (mod 3)", hollow=True, size=8)
    dot_label(cv, 0.680, 0.664, "exception", "ramified   above 2 · above 3", size=8)


def draw_residue_space(cv, d, axes):
    """Residue space mod the CRT moduli."""
    crt = d.moduli
    mstr = ", ".join(map(str, crt))
    ncop = d.admissible
    panel(cv, 4, f"Residue space (mod {mstr})",
          f"By CRT, n ↦ ({', '.join(f'n mod {m}' for m in crt)}) is a {'×'.join(map(str, crt))} grid.\n"
          f"Primes > {max(crt)} occupy only the {ncop} points with no zero coordinate.")
    cls = d.primes
    ax3 = cv.axes([0.0, 0.365, 0.235, 0.22], projection="3d"); axes["cube"] = ax3
    ax3.set_facecolor(C("PANEL_BG"))
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color(to_rgba(C("PANEL_BG")))
        axis.pane.set_edgecolor(C("BORDER"))
        axis._axinfo["grid"]["color"] = to_rgba(C("GRID"))
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis.line.set_color(C("BORDER"))
    P, G = d.grid, d.excluded
    ax3.scatter(G[:, 0], G[:, 1], G[:, 2], s=4, c=C("excluded"), alpha=0.7, lw=0, depthshade=False,
                gid="excluded")
    ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=P[:, 3] * 2.6, c=C("prime"), lw=0, depthshade=False,
                gid="prime")
    ax3.set_xlabel(f"mod {crt[0]}", labelpad=-6); ax3.set_ylabel(f"mod {crt[1]}", labelpad=-5)
    ax3.set_zlabel(f"mod {crt[2]}", labelpad=-6)
    ax3.set_xticks(range(crt[0])); ax3.set_yticks(range(crt[1])); ax3.set_zticks(range(crt[2]))
    ax3.tick_params(labelsize=FS["tick"] - 1, pad=-2)
    ax3.view_init(elev=18, azim=-58)
    lo_c, hi_c = int(P[:, 3].min()), int(P[:, 3].max())
    dot_label(cv, 0.018, 0.352, "prime", f"size = primes {cls[0]}–{cls[-1]} per class ({lo_c}–{hi_c})", size=8)
    ax2 = cv.axes([0.268, 0.37, 0.12, 0.165]); style_axes(ax2); axes["heat4"] = ax2
    Mg = d.heat
    norm4 = plt.Normalize(0, Mg.max())
    ax2.imshow(Mg, origin="lower", cmap=BRAND_CMAP, norm=norm4, aspect="auto")
    for i in range(crt[2]):
        for j in range(crt[1]):
            v = Mg[i, j]
            cell = to_hex(BRAND_CMAP(norm4(v)))
            tc = C("MUTED") if v == 0 else max((C("BG"), C("TEXT")), key=lambda t: contrast(t, cell))
            ax2.text(j, i, v, ha="center", va="center", fontsize=FS["tick"], color=tc)
    ax2.set_xticks(range(crt[1])); ax2.set_yticks(range(crt[2]))
    ax2.set_xlabel(f"p mod {crt[1]}"); ax2.set_ylabel(f"p mod {crt[2]}")
    ax2.set_title(f"primes {cls[0]}–{cls[-1]},\nsummed over mod {crt[0]}", fontsize=FS["small"],
                  color=C("MUTED"), pad=5)


def draw_involution(cv, d, axes):
    """The involution p -> N - p, and the Goldbach comet."""
    m3, m5, m7 = d.moduli
    # Fixed class of p -> N - p mod each m: c ≡ N/2.
    centre = d.centre
    ok3 = d.first_mod_residues
    panel(cv, 5, "The involution p ↦ N − p",
          f"Acts coordinate-wise; mod each odd m it fixes one class, c ≡ N/2. Centred on c, it is a point reflection.\n"
          + (f"N = {d.N}: every pair shown has p ≡ q ≡ {ok3[0]} (mod {m3}), so one (mod {m5}, mod {m7}) plane holds them all."
             if len(ok3) == 1 else f"N = {d.N}, shown in (mod {m5}, mod {m7}) coordinates."))
    axr = cv.axes([0.432, 0.365, 0.235, 0.185]); style_axes(axr); axes["hero"] = axr

    # x = residue mod 7 (the wider axis), y = residue mod 5, each centred on c.
    # Then p and N - p are reflections through the fixed class. The axis scaling
    # is affine, so this still holds without equal aspect.
    xr, yr = d.xs, d.ys
    for x, y, bad in d.cells:
        axr.add_patch(Rectangle((x - .5, y - .5), 1, 1, lw=0.6, ec=C("BORDER"),
                                fc=C("excluded") if bad else C("admissible"),
                                alpha=0.22 if bad else 0.13,
                                gid="excluded" if bad else "admissible"))
    for p, q, pp, qq in d.shown:
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
    dot_label(cv, lx, ly, "prime", "p  (prime)", size=10)
    dot_label(cv, lx, ly - dy, "partner", f"q = {d.N} − p", size=10)
    cv.line([lx - 0.005, lx + 0.006], [ly - 2 * dy] * 2, color=C("pair"), lw=2.6, gid="pair")
    cv.text(lx + 0.011, ly - 2 * dy, "Goldbach pair", fontsize=FS["small"], va="center")
    dot_label(cv, lx, ly - 3 * dy, "fixed", f"fixed class", marker="D", hollow=True, size=7)
    cv.rect((lx - 0.005, ly - 4 * dy - 0.008), 0.011, 0.016,
            fc=C("admissible"), alpha=0.3, ec=C("BORDER"), lw=0.6, gid="admissible")
    cv.text(lx + 0.011, ly - 4 * dy, "admissible", fontsize=FS["small"], va="center")
    cv.rect((lx - 0.005, ly - 5 * dy - 0.008), 0.011, 0.016,
            fc=C("excluded"), alpha=0.5, ec=C("BORDER"), lw=0.6, gid="excluded")
    cv.text(lx + 0.011, ly - 5 * dy, "blocked: p or q ≡ 0", fontsize=FS["small"], va="center")
    skipped = [f"{p} + {q}" for p, q in d.skipped]
    if skipped:
        cv.text(lx - 0.005, ly - 6 * dy - 0.03, f"{', '.join(skipped)} not shown:\np ≤ {max(d.moduli)} has a zero\ncoordinate",
                fontsize=FS["small"], color=C("exception"), linespacing=1.3, gid="exception")

    axa = cv.axes([0.845, 0.40, 0.125, 0.14]); style_axes(axa); axes["comet"] = axa
    # Colour: number of classes r mod the product of the moduli with r and N - r
    # both coprime to it.
    Ns, adm, gb = d.comet_N, d.comet_admissible, d.comet_pairs
    axa.scatter(Ns, gb, c=adm, s=1.6, cmap=BRAND_CMAP, norm=LogNorm(adm.min() * 0.6, adm.max()), lw=0)
    axa.set_xticks([0, d.comet_max // 2, d.comet_max]); axa.set_yticks([0, 50])
    axa.set_xlabel("even N", labelpad=1)
    axa.set_title(f"Goldbach comet, N ≤ {d.comet_max}", fontsize=FS["small"], color=C("MUTED"), pad=5)
    for s_ in ("top", "right"):
        axa.spines[s_].set_visible(False)
    cv.text(0.845, 0.345, f"# pairs; brighter = more of the {d.modulus}\nclasses admissible for that N",
            fontsize=FS["small"] - 0.5, color=C("MUTED"), linespacing=1.3)


def draw_sum_of_squares(cv, d, axes):
    """Goldbach pairs and sum-of-two-squares circles."""
    panel(cv, 6, "Goldbach pairs and sum-of-two-squares circles",
          "|z|² = p has lattice points iff p = 2 or p ≡ 1 (mod 4): 8 of them (4 for p = 2).\n"
          "Not a geometry of Goldbach pairs in general: for N ≡ 0 (mod 4), N > 4, one prime has none.")

    def sq_desc(n, t):
        return f"{n} = {t[0]}² + {t[1]}²" if t else f"{n}: no two squares"

    def lattice_circles(ax, ex):
        N, p, q = ex.N, ex.p, ex.q
        style_axes(ax); Rr = ex.radius
        axes.setdefault("circles", []).append(ax)
        A, B = np.meshgrid(np.arange(-Rr, Rr + 1), np.arange(-Rr, Rr + 1))
        ax.scatter(A, B, s=2, c=C("lattice"), lw=0, gid="lattice")
        t = np.linspace(0, 2 * np.pi, 400)
        for n, pts, role in ((p, ex.p_points, "prime"), (q, ex.q_points, "partner")):
            ax.plot(np.sqrt(n) * np.cos(t), np.sqrt(n) * np.sin(t), color=C(role), lw=1.8,
                    ls="-" if pts else (0, (3, 3)), gid=role)
            if pts:
                pts = np.array(pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=42, c=C("pair"), lw=0, zorder=4, gid="pair")
        ax.set_aspect("equal"); ax.set_xlim(-Rr - .5, Rr + .5); ax.set_ylim(-Rr - .5, Rr + .5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"N = {N}:  {p} + {q}", fontsize=FS["body"] + 1, color=C("TEXT"), pad=4)
        ax.text(0.5, -0.07, sq_desc(p, ex.p_squares), transform=ax.transAxes, ha="center", va="top",
                fontsize=FS["small"], color=C("prime"), gid="prime")
        ax.text(0.5, -0.15, sq_desc(q, ex.q_squares), transform=ax.transAxes, ha="center", va="top",
                fontsize=FS["small"], color=C("partner"), gid="partner")

    lattice_circles(cv.axes([0.03, 0.065, 0.155, 0.148]), d.left)
    if d.right:
        lattice_circles(cv.axes([0.225, 0.065, 0.155, 0.148]), d.right)
    dot_label(cv, 0.386, 0.20, "pair", "lattice point", size=7)
    cv.line([0.380, 0.391], [0.175] * 2, color=C("partner"), lw=1.8, ls=(0, (3, 3)), gid="partner")
    cv.text(0.397, 0.175, "no points", fontsize=FS["small"], va="center")


def draw_sieve(cv, d, axes):
    """Surviving pairs after sieving."""
    # Row i: count of a ≤ N/2 with neither a nor N - a divisible by any of
    # zs[:i + 1]. Last row: the true Goldbach count.
    Ns, H = d.Ns, d.counts
    zmax = d.zs[-1]
    lost = d.lost
    panel(cv, 7, "Surviving pairs after sieving",
          f"Row z counts a ≤ N/2 with a and N − a free of prime factors ≤ z. The bottom row is the true\n"
          f"Goldbach count. Vertical streaks: N divisible by {', '.join(map(str, d.moduli))} (panel 5).")
    norm7 = LogNorm(1, H.max())
    extent = [Ns[0] - 1, Ns[-1] + 1]
    axh = cv.axes([0.505, 0.098, 0.41, 0.135]); style_axes(axh); axes["heat7"] = axh
    axh.imshow(np.ma.masked_equal(H[:-1], 0), aspect="auto", cmap=BRAND_CMAP, norm=norm7,
               extent=[*extent, len(d.zs) - 0.5, -0.5], interpolation="nearest")
    axh.set_yticks(range(len(d.zs))); axh.set_yticklabels([f"z = {z}" for z in d.zs])
    axh.set_xticks([])
    axt = cv.axes([0.505, 0.065, 0.41, 0.022]); style_axes(axt); axes["actual"] = axt
    im = axt.imshow(np.ma.masked_equal(H[-1:], 0), aspect="auto", cmap=BRAND_CMAP, norm=norm7,
                    extent=[*extent, 0.5, -0.5], interpolation="nearest")
    axt.set_yticks([0]); axt.set_yticklabels(["actual"])
    axt.get_yticklabels()[0].set_color(C("TEXT"))
    axt.set_xlabel("even N", labelpad=2)
    cax = cv.axes([0.925, 0.065, 0.008, 0.168])
    cb = cv.fig.colorbar(im, cax=cax)
    cb.outline.set_edgecolor(C("BORDER")); cb.ax.tick_params(colors=C("MUTED"), labelsize=FS["tick"])
    cb.set_label("pairs (log)", color=C("MUTED"), fontsize=FS["small"])
    cv.text(0.505, 0.024,
            f"N = {d.heat_max}: {int(H[0, -1])} survive z = 2, {int(H[-2, -1])} survive z = {zmax}; "
            f"{int(H[-1, -1])} actual pairs, {len(lost)} with p ≤ {zmax}.",
            fontsize=FS["small"], color=C("MUTED"))


# ---- figure

def build_figure(name, data):
    """Draw all seven panels for layout `name` from VisualizationData.

    Returns (fig, axes). axes names the panels' main axes: bar (panel 1),
    wheel2 (2), lattices (3; Gaussian, Eisenstein), cube and heat4 (4), hero and
    comet (5), circles (6; left, right), heat7 and actual (7).
    """
    fig_in = FIGURE_SIZES[name]
    fig = plt.figure(figsize=fig_in)
    W, H = fig_in
    fig.text(0.216 / W, 1 - 0.264 / H, "Primes and Goldbach’s conjecture", fontsize=FS["title"],
             family=FONT_HEAD, va="center")
    fig.text(1 - 0.216 / W, 1 - 0.264 / H, "Seven computed views · every value below is calculated, not drawn",
             fontsize=FS["small"], color=C("MUTED"), ha="right", va="center")
    cv = PanelCanvas(fig, fig_in, LAYOUTS[name])
    axes = {}
    draw_prime_wheel(cv, data.prime_wheel, axes)
    draw_goldbach_wheel(cv, data.goldbach_wheel, axes)
    draw_lattice_primes(cv, data.lattice_primes, axes)
    draw_residue_space(cv, data.residue_space, axes)
    draw_involution(cv, data.involution, axes)
    draw_sum_of_squares(cv, data.sum_of_squares, axes)
    draw_sieve(cv, data.sieve, axes)
    return fig, axes
