"""Figure tests for the parity visualization.

Both layouts are built once from the same data (parity_figures, conftest.py). These
tests read the drawn lines, bars, markers and labels back and compare them with the
data, and check palette colours, colour roles, contrast and glyphs with the shared
helpers. That the data itself is right is covered by test_parity_visualization.py.
"""
import matplotlib
import numpy as np
import pytest

from figure_test_helpers import all_text, assert_semantic_roles, low_contrast_text, stray_colours


@pytest.fixture(params=["landscape", "preview_4x5"])
def parity_figure(request, parity_figures):
    """(fig, axes) for each parity layout in turn."""
    return parity_figures[request.param]


def parity_lines(ax, gid):
    return [ln for ln in ax.lines if ln.get_gid() == gid]


@pytest.mark.parity
def test_parity_average_lines_show_s_over_m(parity_figure, parity):
    _, axes = parity_figure
    lines = parity_lines(axes["average"], "pair")
    assert [ln.get_label() for ln in lines] == [f"N = {N:,}" for N in parity.sweeps]
    for ln, sw in zip(lines, parity.sweeps.values()):
        assert np.array_equal(ln.get_xdata(), sw.zs)
        assert np.array_equal(ln.get_ydata(), sw.S / sw.M)


@pytest.mark.parity
def test_parity_sums_show_m_and_s(parity_figure, parity):
    _, axes = parity_figure
    sw = parity.sweeps[parity.featured]
    (m,), (s,) = parity_lines(axes["sums"], "surviving"), parity_lines(axes["sums"], "pair")
    assert np.array_equal(m.get_xdata(), sw.zs) and np.array_equal(m.get_ydata(), sw.M)
    assert np.array_equal(s.get_xdata(), sw.zs) and np.array_equal(s.get_ydata(), sw.S)


@pytest.mark.parity
def test_parity_endpoint_marker_and_annotation(parity_figure, parity):
    _, axes = parity_figure
    ax = axes["sums"]
    sw = parity.sweeps[parity.featured]
    (marker,) = [c for c in ax.collections if c.get_gid() == "endpoint"]
    assert marker.get_offsets().tolist() == [[sw.zs[-1], sw.S[-1]]]
    (ann,) = [t for t in ax.texts if hasattr(t, "xy")]
    assert ann.xy == (sw.zs[-1], sw.S[-1])
    assert ann.get_text() == f"z = {sw.zs[-1]} (largest prime ≤ √N):\nS = M = {sw.S[-1]:,}"


@pytest.mark.parity
def test_parity_labels(parity_figure):
    # Number-bearing statements at the default settings; the numbers are cross-checked
    # against trial division in test_cross_checks.py.
    fig, axes = parity_figure
    text = all_text(fig)
    for N in ("999,000", "999,998", "1,000,000"):
        assert f"N = {N}" in text, N
    assert "2 ≤ a ≤ N/2" in text          # the range parity_sieve uses
    assert "S = M = 5,382" in text
    assert "from z = 983, every survivor has sign +1" in text
    for label in ("all pairs, 2 ≤ a ≤ N/2", "survivors at z = 7", "survivors at z = 31", "Goldbach pairs"):
        assert label in text, label
    assert "+1: 5,402" in text and "−1: 0" in text
    assert "surviving a shallow sieve ⇏ being a Goldbach pair" in text
    assert "proves nothing about Goldbach’s conjecture" in text
    assert [t.get_text() for t in axes["average"].get_legend().get_texts()] == [
        "N = 999,000", "N = 999,998", "N = 1,000,000"]


@pytest.mark.parity
def test_parity_split_bars_show_the_class_splits(parity_figure, parity):
    # Each row: a +1 bar from 0 to plus/total and a -1 bar from there to 1, top row first.
    _, axes = parity_figure
    ax = axes["classes"]
    plus = [p for p in ax.patches if p.get_gid() == "sign_plus"]
    minus = [p for p in ax.patches if p.get_gid() == "sign_minus"]
    assert len(plus) == len(minus) == len(parity.class_splits)
    n = len(parity.class_splits)
    for k, (c, bp, bm) in enumerate(zip(parity.class_splits, plus, minus)):
        t = c.plus + c.minus
        assert bp.get_y() + bp.get_height() / 2 == pytest.approx(n - 1 - k)
        assert bp.get_x() == 0 and bp.get_width() == pytest.approx(c.plus / t)
        assert bm.get_x() == pytest.approx(c.plus / t) and bm.get_width() == pytest.approx(c.minus / t)
    assert [c.kind for c in parity.class_splits] == ["all", "survivors", "survivors", "goldbach"]
    assert any(ln.get_xdata()[0] == 0.5 for ln in ax.lines)    # the 50% divider


@pytest.mark.parity
def test_parity_curves_nearly_coincide(parity):
    # Figure regression, not a mathematical claim: the subtitle says the three curves
    # nearly coincide, which holds for these N (largest gap 0.024, at z = 479).
    averages = np.vstack([sw.average for sw in parity.sweeps.values()])
    assert np.ptp(averages, axis=0).max() < 0.03


@pytest.mark.parity
def test_parity_text_has_every_glyph(parity, caplog):
    # A line with mathtext renders its plain text without per-glyph font fallback, so
    # Greek letters outside $...$ would be drawn as dummy symbols in Poppins. Mathtext
    # reports that through logging. (With DejaVu, which has every glyph, this always passes.)
    import logging
    import matplotlib.pyplot as plt
    from extras.parity.layout import FIGURE_SIZES
    from extras.parity.render import build_figure
    from goldbach.style import apply_style
    with matplotlib.rc_context(), caplog.at_level(logging.WARNING, logger="matplotlib"):
        apply_style()
        for name in FIGURE_SIZES:
            fig, _ = build_figure(name, parity)
            fig.canvas.draw()
            plt.close(fig)
    assert [r.getMessage() for r in caplog.records if "does not have a glyph" in r.getMessage()] == []


@pytest.mark.parity
def test_parity_colours_and_contrast(parity_figure):
    fig, _ = parity_figure
    assert stray_colours(fig) == []
    assert_semantic_roles(fig)
    assert low_contrast_text(fig) == []
