"""Figure integration tests: the rendered figures show the computed data.

Both layouts are built once from the same visualization data (see conftest.py).
These tests read values back from the drawn artists and text and compare them
with that data, and check colours and text contrast. That the data itself is
right is covered by test_visualization_data.py and, against independent
brute-force implementations, by test_cross_checks.py.

Label and caption checks use literal strings at the default settings.
"""
from collections import defaultdict

import matplotlib
import numpy as np
import pytest
from matplotlib.collections import Collection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.text import Text

from goldbach.style import C, PALETTE, SEMANTIC, contrast

MIN_TEXT_CONTRAST = 4.5     # WCAG AA


@pytest.fixture(params=["landscape", "linkedin_4x5"])
def figure(request, figures):
    """(fig, axes) for each layout in turn."""
    return figures[request.param]


def all_text(fig):
    return "\n".join(t.get_text() for t in fig.findobj(Text) if t.get_visible())


def points(ax, gid, hollow=None):
    """Offsets of one role's scatter points, rounded. hollow=True: unfilled markers only."""
    out = set()
    for c in ax.collections:
        if c.get_gid() != gid:
            continue
        fc = c.get_facecolor()
        is_hollow = len(fc) == 0 or to_rgba(fc[0])[3] == 0
        if hollow is None or hollow == is_hollow:
            out |= {(round(float(x), 6), round(float(y), 6)) for x, y in c.get_offsets()}
    return out


def rounded(xy):
    return {(round(float(x), 6), round(float(y), 6)) for x, y in xy}


def wheel_xy(k, m):
    ang = np.pi / 2 - 2 * np.pi * k / m
    return np.cos(ang), np.sin(ang)


# ---- panel 1: the prime wheel

def test_prime_wheel_bars_show_the_class_counts(figure, default_data):
    _, axes = figure
    d, ax = default_data.prime_wheel, axes["bar"]
    assert [b.get_width() for b in ax.patches] == d.counts
    assert [t.get_text() for t in ax.get_yticklabels()] == [str(r) for r in d.coprime]
    assert [t.get_text() for t in ax.texts] == [str(c) for c in d.counts]


def test_prime_wheel_labels(figure):
    text = all_text(figure[0])
    assert "The prime wheel (mod 30)" in text
    assert "Every prime > 5 is coprime to 30 = 2·3·5" in text
    assert "one of φ(30) = 8 classes." in text
    assert "Primes 7–997 per class" in text
    assert "2, 3, 5: the sieving primes" in text


# ---- panel 2: Goldbach pairs on the wheel

def test_goldbach_wheel_chords_join_the_residues_of_each_type(figure, default_data):
    _, axes = figure
    d, ax = default_data.goldbach_wheel, axes["wheel2"]
    chords = [ln for ln in ax.lines if ln.get_gid() == "pair"]
    assert len(chords) == len(d.by_type)
    for ln, ((a, b), lst) in zip(chords, d.by_type.items()):
        assert ln.get_linewidth() == pytest.approx(1.8 * len(lst) + 0.6)
        (xa, ya), (xb, yb) = wheel_xy(a, 30), wheel_xy(b, 30)
        assert np.allclose(ln.get_xdata(), [xa, xb]) and np.allclose(ln.get_ydata(), [ya, yb])
    exc = [ln for ln in ax.lines if ln.get_gid() == "exception"]
    assert len(exc) == len(d.exceptions)
    for ln, (p, q) in zip(exc, d.exceptions):
        (xa, ya), (xb, yb) = wheel_xy(p % 30, 30), wheel_xy(q % 30, 30)
        assert np.allclose(ln.get_xdata(), [xa, xb]) and np.allclose(ln.get_ydata(), [ya, yb])


def test_goldbach_wheel_labels(figure):
    text = all_text(figure[0])
    assert "N = 100 ≡ 10 (mod 30), so p ≡ r forces q ≡ 10 − r." in text
    assert "With p, q > 5, only 2 residue types are possible." in text
    for label in ("11 ↔ 29", "× 3", "11+89   29+71   41+59", "17 ↔ 23", "× 2", "17+83   47+53",
                  "3 + 97", "small-prime exception:\n3 is removed by the wheel"):
        assert label in text, label


# ---- panel 3: Gaussian and Eisenstein primes

def test_lattice_primes_points_show_the_classification(figure, default_data):
    _, axes = figure
    d = default_data.lattice_primes
    for ax, lp in zip(axes["lattices"], (d.gaussian, d.eisenstein)):
        assert points(ax, "lattice") == rounded(lp.points)
        assert points(ax, "prime", hollow=False) == rounded(lp.split)
        assert points(ax, "prime", hollow=True) == rounded(lp.inert)
        assert points(ax, "exception") == rounded(lp.ram)


def test_lattice_primes_labels(figure):
    _, axes = figure
    assert [ax.get_title() for ax in axes["lattices"]] == ["Gaussian  ℤ[i]", "Eisenstein  ℤ[ω]"]
    text = all_text(figure[0])
    assert "split    p ≡ 1 (mod 4) · p ≡ 1 (mod 3)" in text
    assert "inert    p ≡ 3 (mod 4) · p ≡ 2 (mod 3)" in text


# ---- panel 4: residue space

def test_residue_space_cube_sizes_are_the_class_counts(figure, default_data):
    _, axes = figure
    d, ax = default_data.residue_space, axes["cube"]
    cube = [c for c in ax.collections if c.get_gid() == "prime"][0]
    # 3D scatter positions and sizes have no public getter; these are what mplot3d draws from.
    xs, ys, zs = cube._offsets3d
    drawn = {(int(x), int(y), int(z)): round(s / 2.6) for x, y, z, s in zip(xs, ys, zs, cube._sizes3d)}
    assert drawn == {(int(a), int(b), int(c)): int(n) for a, b, c, n in d.grid}
    assert len(drawn) == 48
    excluded = [c for c in ax.collections if c.get_gid() == "excluded"][0]
    assert len(excluded._offsets3d[0]) == len(d.excluded)


def test_residue_space_heatmap_shows_the_counts(figure, default_data):
    _, axes = figure
    d, ax = default_data.residue_space, axes["heat4"]
    assert np.array_equal(ax.images[0].get_array(), d.heat)
    cells = [(int(t.get_text()), t.get_position()) for t in ax.texts]
    assert cells == [(int(d.heat[i, j]), (j, i)) for i in range(7) for j in range(5)]


def test_residue_space_labels(figure):
    text = all_text(figure[0])
    assert "By CRT, n ↦ (n mod 3, n mod 5, n mod 7) is a 3×5×7 grid." in text
    assert "Primes > 7 occupy only the 48 points with no zero coordinate." in text
    assert "size = primes 11–9973 per class (22–29)" in text
    assert "primes 11–9973,\nsummed over mod 3" in text


# ---- panel 5: the involution and the comet

def test_involution_pairs_reflect_through_the_fixed_class(figure, default_data):
    _, axes = figure
    d, ax = default_data.involution, axes["hero"]
    centre = (d.centre[7], d.centre[5])
    drawn_p = {tuple(map(int, c.get_offsets()[0])) for c in ax.collections if c.get_gid() == "prime"}
    drawn_q = {tuple(map(int, c.get_offsets()[0])) for c in ax.collections if c.get_gid() == "partner"}
    assert drawn_p == {pp for _, _, pp, _ in d.shown}
    assert drawn_q == {qq for _, _, _, qq in d.shown}
    lines = [ln for ln in ax.lines if ln.get_gid() == "pair"]
    assert len(lines) == len(d.shown)
    for ln in lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        assert ((x[0] + x[1]) / 2, (y[0] + y[1]) / 2) == centre
    fixed = [c.get_offsets()[0] for c in ax.collections if c.get_gid() == "fixed"]
    assert len(fixed) == 1 and tuple(fixed[0]) == centre
    labels = {t.get_text() for t in ax.texts}
    assert labels == {str(v) for p, q, _, _ in d.shown for v in (p, q)}


def test_involution_blocked_cells(figure, default_data):
    _, axes = figure
    d, ax = default_data.involution, axes["hero"]
    cell = lambda r: (int(r.get_x() + .5), int(r.get_y() + .5))
    blocked = {cell(r) for r in ax.patches if r.get_gid() == "excluded"}
    open_ = {cell(r) for r in ax.patches if r.get_gid() == "admissible"}
    assert blocked == {(x, y) for x, y, b in d.cells if b}
    assert open_ == {(x, y) for x, y, b in d.cells if not b}
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(x % 7) for x in d.xs]
    assert [t.get_text() for t in ax.get_yticklabels()] == [str(y % 5) for y in d.ys]


def test_involution_comet_shows_pair_counts_and_admissible_classes(figure, default_data):
    _, axes = figure
    d = default_data.involution
    comet = axes["comet"].collections[0]
    assert np.array_equal(comet.get_offsets(), np.column_stack([d.comet_N, d.comet_pairs]))
    assert np.array_equal(comet.get_array(), d.comet_admissible)


def test_involution_labels(figure):
    text = all_text(figure[0])
    assert ("N = 100: every pair shown has p ≡ q ≡ 2 (mod 3), so one (mod 5, mod 7) plane "
            "holds them all.") in text
    assert "3 + 97 not shown:\np ≤ 7 has a zero\ncoordinate" in text
    assert "q = 100 − p" in text
    assert "Goldbach comet, N ≤ 2000" in text
    assert "# pairs; brighter = more of the 105\nclasses admissible for that N" in text


# ---- panel 6: sum-of-two-squares circles

def test_sum_of_squares_lattice_points_match_the_examples(figure, default_data):
    _, axes = figure
    d = default_data.sum_of_squares
    for ax, ex in zip(axes["circles"], (d.left, d.right)):
        drawn = {tuple(map(int, o)) for c in ax.collections if c.get_gid() == "pair" for o in c.get_offsets()}
        assert drawn == set(ex.p_points) | set(ex.q_points)
        for role, pts in (("prime", ex.p_points), ("partner", ex.q_points)):
            circle = [ln for ln in ax.lines if ln.get_gid() == role][0]
            assert (circle.get_linestyle() == "-") == bool(pts), role   # dashed: no lattice points


def test_sum_of_squares_labels(figure):
    _, axes = figure
    assert [ax.get_title() for ax in axes["circles"]] == ["N = 100:  11 + 89", "N = 98:  37 + 61"]
    text = all_text(figure[0])
    for label in ("11: no two squares", "89 = 5² + 8²", "37 = 1² + 6²", "61 = 5² + 6²",
                  "for N ≡ 0 (mod 4), N > 4, one prime has none."):
        assert label in text, label


# ---- panel 7: sieve survivors

def test_sieve_heatmaps_show_survivors_and_actual_counts(figure, default_data):
    _, axes = figure
    d = default_data.sieve
    assert np.array_equal(np.ma.filled(axes["heat7"].images[0].get_array(), 0), d.counts[:-1])
    assert np.array_equal(np.ma.filled(axes["actual"].images[0].get_array(), 0), d.counts[-1:])
    assert [t.get_text() for t in axes["heat7"].get_yticklabels()] == [f"z = {z}" for z in d.zs]


def test_sieve_labels(figure):
    text = all_text(figure[0])
    assert "N = 1000: 250 survive z = 2, 26 survive z = 23; 28 actual pairs, 3 with p ≤ 23." in text
    assert "Vertical streaks: N divisible by 3, 5, 7 (panel 5)." in text


# ---- colours and contrast

def artist_colours(art):
    """Non-transparent colours an artist draws with. Colormapped collections are skipped."""
    out = []
    if isinstance(art, Line2D):
        if art.get_linestyle() not in ("None", "", " ") and art.get_linewidth() > 0:
            out.append(art.get_color())
        if art.get_marker() not in (None, "None", "", " "):
            out += [art.get_markerfacecolor(), art.get_markeredgecolor()]
    elif isinstance(art, Collection):
        if art.get_array() is not None:
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


def test_only_palette_colours_are_used(figure):
    fig, _ = figure
    allowed = {v.lower() for v in PALETTE.values()}
    stray = [(type(a).__name__, c) for a in fig.findobj() if a.get_visible()
             for c in artist_colours(a) if c.lower() not in allowed]
    assert stray == []


def test_each_semantic_role_uses_its_colour(figure):
    fig, _ = figure
    role_cols = defaultdict(set)
    for a in fig.findobj():
        if a.get_visible() and a.get_gid() in SEMANTIC:
            role_cols[a.get_gid()] |= {c.lower() for c in artist_colours(a)}
    assert role_cols
    for role, cols in role_cols.items():
        assert C(role).lower() in cols, role
        # Only the role's own colour, plus the panel fill and border used as edges.
        assert cols <= {C(role).lower(), C("PANEL_BG").lower(), C("BORDER").lower()}, role


def test_text_contrast(figure):
    # Text is checked against PANEL_BG, the lighter background; heatmap cell labels
    # against their own cell colour.
    fig, axes = figure
    heat = axes["heat4"]
    im = heat.images[0]
    cell_texts = set(heat.texts)
    low = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bg = to_hex(im.cmap(im.norm(int(t.get_text())))) if t in cell_texts else C("PANEL_BG")
        if contrast(t.get_color(), bg) < MIN_TEXT_CONTRAST:
            low.append((t.get_text()[:30], round(contrast(t.get_color(), bg), 2)))
    assert low == []
    assert contrast(C("MUTED"), C("PANEL_BG")) >= 7
