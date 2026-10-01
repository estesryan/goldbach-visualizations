"""Figure tests for the first-survivor experiment.

Both main layouts and the blocks figure are built once (first_survivor_figures,
conftest.py). These tests read the drawn bars, points, lines, mesh and labels back
and compare them with the data, and check palette colours, colour roles, contrast
and glyphs with the poster's helpers. That the data itself is right is covered by
test_first_survivor.py.
"""
import logging
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from extras.first_survivor_experiment.framework import build_framework_figure, framework_data
from extras.first_survivor_experiment.layout import FIGURE_SIZES
from extras.first_survivor_experiment.render import ROLES, build_blocks_figure, build_figure, col
from goldbach.style import PALETTE, apply_style
from figure_test_helpers import all_text, artist_colours, low_contrast_text, points, rounded, stray_colours

pytestmark = pytest.mark.first_survivor


@pytest.fixture(params=list(FIGURE_SIZES))
def figure(request, first_survivor_figures):
    """(fig, axes) for each main layout in turn."""
    return first_survivor_figures[request.param]


def lines_with(ax, gid):
    return [ln for ln in ax.lines if ln.get_gid() == gid]


def bar_xs(ax, gid):
    """Centres of one role's bars, as a sorted list of ints."""
    return sorted(int(round(p.get_x() + p.get_width() / 2)) for p in ax.patches if p.get_gid() == gid)


def text_of(fig):
    return all_text(fig).replace(" ", " ")


# ---- panel A

def test_example_strips_show_the_offsets(figure, first_survivor):
    _, axes = figure
    e = first_survivor.example
    ax = axes["example_centre"]
    assert bar_xs(ax, "surviving") == e.centre_d[e.centre_alive].tolist()
    assert bar_xs(ax, "excluded") == e.centre_d[~e.centre_alive].tolist()
    # Separate markers for the two definitions, at the same offset but different heights.
    (ps,), (pp,) = points(ax, "surviving"), points(ax, "pair", hollow=True)
    assert ps[0] == e.lam_sieve and pp[0] == e.lam_prime and ps[1] != pp[1]
    ax = axes["example_edge"]
    assert bar_xs(ax, "exception") == e.composite[(e.composite >= e.edge_d.min())
                                                  & (e.composite <= e.edge_d.max())].tolist() == [e.W]
    assert bar_xs(ax, "surviving") == e.edge_d[e.edge_alive & ~e.edge_composite].tolist()
    (bd,) = lines_with(ax, "boundary")
    assert bd.get_xdata()[0] == e.W and bd.get_zorder() < 1      # behind the bars
    (bd,) = lines_with(axes["example_overview"], "boundary")
    assert bd.get_xdata()[0] == e.W
    assert axes["example_overview"].get_xlim() == (0, e.C - 2)


def test_portrait_is_its_own_layout(first_survivor_figures):
    # Portrait: the two zooms side by side above the overview. Landscape: stacked.
    def pos(name, key):
        return first_survivor_figures[name][1][key].get_position()
    assert pos("preview_4x5", "example_edge").x0 > pos("preview_4x5", "example_centre").x1
    assert pos("preview_4x5", "example_edge").y0 > pos("preview_4x5", "example_overview").y1
    assert pos("landscape", "example_edge").y1 < pos("landscape", "example_overview").y0
    assert pos("preview_4x5", "large").x0 > pos("preview_4x5", "small").x1
    assert pos("landscape", "large").y1 < pos("landscape", "small").y0


# ---- panel B

def test_small_view_shows_blocks_and_points(figure, first_survivor):
    from extras.first_survivor_experiment import generate
    _, axes = figure
    ax = axes["small"]
    d = first_survivor
    sw = d.sweep
    s = sw.N <= generate.SMALL_MAX
    N, q, W, lam = sw.N[s], sw.q[s], sw.W[s], sw.lam_sieve[s]
    segs = lines_with(ax, "boundary")
    assert len(segs) == len(np.unique(q))
    for seg, qq in zip(segs, np.unique(q)):
        assert np.array_equal(seg.get_xdata(), N[q == qq]) and np.array_equal(seg.get_ydata(), W[q == qq])
        assert np.all(np.diff(seg.get_ydata()) == -1)                # one straight segment per block
    comp, out = np.isin(N, d.composite_first), np.isin(N, d.prime_outside)
    assert points(ax, "surviving") == rounded(zip(N[~(comp | out)], lam[~(comp | out)]))
    assert points(ax, "exception", hollow=False) == rounded(zip(N[comp], lam[comp]))
    assert points(ax, "exception", hollow=True) == rounded(zip(N[out], lam[out]))
    ann = {t.get_text() for t in ax.texts if hasattr(t, "xy")}
    assert ann == {"N = 100: λ = 3 ≥ W = −1", "N = 272: λ = 27 < W = 33"}


def test_large_view_shows_the_distribution_and_w(figure, first_survivor):
    _, axes = figure
    ax = axes["large"]
    d = first_survivor
    (mesh,) = ax.collections
    assert np.allclose(mesh.get_array().filled(0).reshape(d.histogram.share.shape), d.histogram.share)
    sw = d.sweep
    s = sw.N >= d.min_N
    (w,) = lines_with(ax, "boundary")
    assert np.array_equal(w.get_xdata(), sw.N[s]) and np.array_equal(w.get_ydata(), sw.W[s])
    assert ax.get_yscale() == "symlog" and ax.get_xscale() == "log"
    assert ax.get_ylim()[0] < 0                                       # the λ = 0 row is on the axes


# ---- panel C

def test_envelope_panel(figure, first_survivor):
    _, axes = figure
    ax = axes["envelope"]
    d = first_survivor
    assert points(ax, "envelope") == rounded(zip(d.bin_argN, d.bin_max))
    fits = lines_with(ax, "fit")
    assert len(fits) == len(d.fits)
    for ln, f in zip(fits, d.fits):
        x, y = ln.get_xdata(), ln.get_ydata()
        assert (x[0], x[-1]) == pytest.approx(f.x_range)
        assert np.allclose(y, np.exp(f.log_A) * x ** f.alpha_N)
        assert f"{f.alpha_N:.3f}" in ln.get_label()
    assert not lines_with(ax, "boundary")                            # W is not drawn over the envelope
    (drift,) = lines_with(axes["drift"], "fit")
    assert drift.get_xdata().tolist() == [14, 17, 20]
    assert drift.get_ydata().tolist() == [f.alpha_N for f in d.fits]
    assert [t.get_text() for t in axes["drift"].texts][:-1] == [f"{f.alpha_N:.3f}" for f in d.fits]


# ---- text

def test_labels(figure):
    fig, _ = figure
    text = text_of(fig).replace("\n", " ")
    for s in ("Where the first survivor lands",
              "Survival alone does not distinguish a prime from a composite whose prime factors are all at least q.",
              "Below q², no such composite can occur, so a sieve survivor there is prime by size alone.",
              "The experiment measures where the first survivor lands relative to that forcing boundary.",
              "N = 999,008, so C = 499,504 and q = 997; W = q² − C = 494,505.",
              "here both sit at d = 45: 499,459 + 499,549.",
              "d = W: 4,999 + 997²",
              "3,956 survivors with d < W: every one a pair of primes",
              "51 survivors have d ≥ W",
              "For every even N from 122 to 1,048,574 the first survivor lies inside the forcing boundary",
              "Below 122 it lies at or beyond W for 22 values of N, all with q ≤ 7, and 13 of those",
              "Composite survivors occur for 175,468 of the 524,227 even N from 122 to 1,048,574, never below W.",
              "Every even N with 122 ≤ N < $2^{20}$",
              "at most 0.818·W (at N = 272)",
              "this range cannot separate a power law from slower growth",
              "none is asymptotic",
              r"$2\alpha_N$: q-units via $N \approx q^2$"):
        assert s in text, s
    # The q-unit value is only ever a conversion, never a fitted q exponent, and no
    # exponent is called asymptotic.
    assert "fitted q exponent" not in text.lower()
    assert "asymptotic" not in text.replace("none is asymptotic", "")
    # The range is N < 2^20: no wording that could include 2^20 itself, and no claim
    # about all sieve information.
    assert "to $2^{20}$" not in text and "through $2^{20}$" not in text
    assert "Sieve information alone" not in text
    # The figure stays descriptive: no statements about proof status.
    assert "prove" not in text.lower() and "parity problem" not in text and "conjecture" not in text


def test_blocks_figure(first_survivor_figures, first_survivor_blocks, first_survivor):
    fig, axes = first_survivor_figures["blocks"]
    d, blocks = first_survivor, first_survivor_blocks
    for key, attr in (("maxima", "max_lam"), ("ratio", "max_ratio")):
        ax = axes[key]
        assert points(ax, "block", hollow=True) == rounded((b.q, getattr(b, attr)) for b in blocks)
        assert points(ax, "block", hollow=False) == rounded((b.q, getattr(b, attr)) for b in d.blocks)
        assert not lines_with(ax, "fit") and not [ln for ln in ax.lines if len(ln.get_xdata()) > 2]
    (one,) = lines_with(axes["ratio"], "boundary")                   # the boundary only as the ratio 1
    assert set(one.get_ydata()) == {1}
    text = text_of(fig).replace("\n", " ")
    for s in ("16 blocks are selected by a fixed rule", "largest prime at or below each","q from 1,097 to 49,999, 3,395,436 values of N, all computed",
              "the first survivor lies inside the forcing boundary for every N", "not an envelope",
              "they are not joined or fitted", "states no growth law"):
        assert s in text, s


# ---- colours, contrast and glyphs

@pytest.mark.parametrize("name", list(FIGURE_SIZES) + ["blocks"])
def test_colours_roles_and_contrast(first_survivor_figures, name):
    fig, _ = first_survivor_figures[name]
    assert stray_colours(fig) == []
    role_cols = defaultdict(set)
    for a in fig.findobj():
        if a.get_visible() and a.get_gid() in ROLES:
            role_cols[a.get_gid()] |= {c.lower() for c in artist_colours(a)}
    assert role_cols
    for role, cols in role_cols.items():
        assert col(role).lower() in cols, role
        assert cols <= {col(role).lower(), PALETTE["PANEL_BG"].lower(), PALETTE["BORDER"].lower()}, role
    assert low_contrast_text(fig) == []


# ---- the framework figure

@pytest.fixture(scope="module")
def framework():
    """The framework figure at the settings in generate.py, as (data, fig, axes)."""
    from extras.first_survivor_experiment import generate
    with matplotlib.rc_context():
        apply_style()
        fd = framework_data(generate.FRAMEWORK_N, generate.FRAMEWORK_BLOCK_Q, generate.FRAMEWORK_SPAN)
        fig, axes = build_framework_figure(fd)
        fig.canvas.draw()
        yield fd, fig, axes
    plt.close(fig)


def test_framework_example_is_exact(framework, first_survivor):
    fd, _, axes = framework
    e = fd.line
    assert (e.N, e.C, e.q, e.W, e.lam_sieve, e.lam_prime) == (272, 136, 13, 33, 27, 27)
    assert e.d.tolist() == list(range(e.C - 1))                       # every valid offset is drawn
    assert e.d[e.alive & ~e.prime_pair].tolist() == [e.W]             # the only composite survivor is at d = W
    sw = first_survivor.sweep                                         # agrees with the main computation
    i = int(np.searchsorted(sw.N, e.N))
    assert (sw.q[i], sw.W[i], sw.lam_sieve[i], sw.lam_prime[i]) == (e.q, e.W, e.lam_sieve, e.lam_prime)
    ax = axes["line"]
    assert bar_xs(ax, "surviving") == e.d[e.alive & e.prime_pair].tolist()
    assert bar_xs(ax, "exception") == [e.W]
    (ps,), (pp,) = points(ax, "surviving"), points(ax, "pair", hollow=True)
    assert ps[0] == e.lam_sieve and pp[0] == e.lam_prime and ps[1] != pp[1]


def test_framework_blocks_match_the_sweep(framework, first_survivor):
    fd, _, axes = framework
    b, sw = fd.blocks, first_survivor.sweep
    s = (sw.N >= b.N[0]) & (sw.N <= b.N[-1])
    assert np.array_equal(b.N, sw.N[s]) and np.array_equal(b.W, sw.W[s])
    assert np.array_equal(b.lam_sieve, sw.lam_sieve[s])
    assert b.qs == (11, 13, 17, 19, 23, 29, 31) and (b.lam_sieve < b.W).all()
    (line,) = lines_with(axes["block"], "boundary")
    inb = b.q == fd.block_q
    assert np.array_equal(line.get_xdata(), b.N[inb]) and np.all(np.diff(line.get_ydata()) == -1)
    assert points(axes["block"], "surviving") == rounded(zip(b.N[inb], b.lam_sieve[inb]))
    assert len(lines_with(axes["blocks"], "boundary")) == len(b.qs)  # one straight segment per block


def test_framework_labels_size_and_output(framework, tmp_path):
    from matplotlib.image import imread

    from extras.first_survivor_experiment import generate
    _, fig, axes = framework
    assert tuple(fig.get_size_inches()) == (16.0, 9.0) and set(axes) == {"line", "block", "blocks"}
    text = text_of(fig)
    for s in ("Framework: the first survivor and the forcing boundary", r"\lambda_{\mathrm{sieve}}",
              r"\lambda_{\mathrm{prime}}", "W = q² − C", "W(N) = 23² − N/2", "prime-square block", "q² = 169",
              "W = 33", "d = W:  103 + 13²  survives but is composite", "d < W:  survival forces a prime pair",
              "d ≥ W:  composite survivors possible", "nearest prime pair · 109 + 163", "first survivor",
              "23² < N < 29²"):
        assert s in text, s
    # every block in panel C is labelled by its q, and the highlight names the block, not a panel
    labels = [t.get_text() for t in axes["blocks"].texts]
    assert all(f"q = {q}" in labels for q in (11, 13, 17, 19, 23, 29, 31)) and "panel B" not in text
    assert stray_colours(fig) == [] and low_contrast_text(fig) == []
    with matplotlib.rc_context():
        apply_style()
        out = generate.save_framework(str(tmp_path))
    assert imread(out).shape[:2] == (1800, 3200)


def test_text_has_every_glyph(first_survivor, first_survivor_blocks, caplog):
    with matplotlib.rc_context(), caplog.at_level(logging.WARNING, logger="matplotlib"):
        apply_style()
        figs = [build_figure(n, first_survivor)[0] for n in FIGURE_SIZES]
        figs.append(build_blocks_figure(first_survivor_blocks, first_survivor)[0])
        figs.append(build_framework_figure(framework_data())[0])
        for fig in figs:
            fig.canvas.draw()
            plt.close(fig)
    assert [r.getMessage() for r in caplog.records if "does not have a glyph" in r.getMessage()] == []
