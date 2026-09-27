"""Style, layout and figure construction.

These check the rendering API and its boundaries. Whether the drawn figures
show the right values is in test_figures.py.
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from goldbach import layout, style
from goldbach.visualization_data import build_visualization_data

REPO = Path(__file__).resolve().parents[1]
AXES_KEYS = {"bar", "wheel2", "lattices", "cube", "heat4", "hero", "comet", "circles", "heat7", "actual"}


# ---- style

def test_importing_modules_does_not_change_rcparams():
    # Fresh interpreter, so nothing earlier in the test run has touched rcParams.
    code = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "before = matplotlib.rcParams.copy()\n"
        "import goldbach.number_theory, goldbach.visualization_data\n"
        "import goldbach.style, goldbach.layout\n"
        "import goldbach.render, generate_visualizations\n"
        "from extras import generate_parity_visualization\n"
        "assert matplotlib.rcParams == before\n"
        "goldbach.style.apply_style()\n"
        "assert matplotlib.rcParams != before\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_apply_style_sets_every_rc_param():
    with matplotlib.rc_context():
        style.apply_style()
        for key, value in style.RC_PARAMS.items():
            assert matplotlib.rcParams[key] == matplotlib.RcParams({key: value})[key], key
        assert matplotlib.rcParams["font.family"] == ["Poppins", "DejaVu Sans"]
        assert matplotlib.rcParams["figure.facecolor"] == "#0f0e0d"
        assert matplotlib.rcParams["legend.labelcolor"] == "#f1ebe2"


def test_semantic_roles_resolve_to_palette_colours():
    assert set(style.SEMANTIC.values()) <= set(style.PALETTE)
    assert style.C("prime") == style.PALETTE["ACCENT"]
    assert style.C("partner") == style.PALETTE["ACCENT_2"]
    assert style.C("MUTED") == "#aaa39c"


def test_brand_cmap_endpoints():
    from matplotlib.colors import to_hex
    assert to_hex(style.BRAND_CMAP(0.0)) == style.PALETTE["PANEL_BG"]
    assert to_hex(style.BRAND_CMAP(1.0)) == style.PALETTE["HIGHLIGHT"]


def test_contrast():
    assert style.contrast("#ffffff", "#000000") == pytest.approx(21.0)
    assert style.contrast("#aaa39c", "#171513") == pytest.approx(style.contrast("#171513", "#aaa39c"))
    assert style.contrast(style.C("MUTED"), style.C("PANEL_BG")) == pytest.approx(7.309, abs=1e-3)


# ---- layout

EXPECTED_RECTS = {
    "landscape": {1: (0.144, 7.86, 5.868, 3.66), 2: (6.102, 7.86, 5.868, 3.66),
                  3: (12.06, 7.86, 5.796, 3.66), 4: (0.144, 4.02, 7.2, 3.744),
                  5: (7.434, 4.02, 10.422, 3.744), 6: (0.144, 0.096, 7.92, 3.828),
                  7: (8.154, 0.096, 9.702, 3.828)},
    "linkedin_4x5": {1: (0.144, 15.151333333333, 7.911, 4.618666666667),
                     2: (8.145, 15.151333333333, 7.911, 4.618666666667),
                     3: (0.144, 10.442666666667, 7.911, 4.618666666667),
                     4: (8.145, 10.442666666667, 7.911, 4.618666666667),
                     5: (0.144, 4.852666666667, 15.912, 5.5),
                     6: (0.144, 0.144, 7.6, 4.618666666667),
                     7: (7.834, 0.144, 8.222, 4.618666666667)},
}


def test_figure_sizes():
    assert layout.FIGURE_SIZES == {"landscape": (18.0, 12.0), "linkedin_4x5": (16.2, 20.25)}


@pytest.mark.parametrize("name", ["landscape", "linkedin_4x5"])
def test_panel_rects(name):
    rects = layout.LAYOUTS[name]
    assert set(rects) == set(range(1, 8))
    for k, r in rects.items():
        assert r == pytest.approx(EXPECTED_RECTS[name][k], abs=1e-9), k
    W, H = layout.FIGURE_SIZES[name]
    for x, y, w, h in rects.values():
        assert 0 <= x and x + w <= W + 1e-9 and 0 <= y and y + h <= H + 1e-9


def _canvas(name, num):
    cv = layout.PanelCanvas(None, layout.FIGURE_SIZES[name], layout.LAYOUTS[name])
    cv.set_panel(num)
    return cv


@pytest.mark.parametrize("num", range(1, 8))
def test_landscape_mapping_is_the_identity(num):
    cv = _canvas("landscape", num)
    for x, y in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.05)]:
        assert cv.point(x, y) == pytest.approx((x, y), abs=1e-12)
    assert cv.size(0.1, 0.2) == pytest.approx((0.1, 0.2), abs=1e-12)


@pytest.mark.parametrize("num", range(1, 8))
def test_portrait_mapping_maps_the_content_area(num):
    # The content area (below the HDR_IN header) of the landscape panel maps onto
    # the content area of the portrait panel. The header itself is placed by
    # render.panel() in inches, not through this mapping.
    cv = _canvas("linkedin_4x5", num)
    W, H = layout.FIGURE_SIZES["linkedin_4x5"]
    lx, ly, lw, lh = layout._in(layout.LAND_RECTS[num], layout.LAND)
    nx, ny, nw, nh = layout.LAYOUTS["linkedin_4x5"][num]
    header_bottom = (ly + lh - layout.HDR_IN) / layout.LAND[1]
    assert cv.point(lx / layout.LAND[0], header_bottom) == pytest.approx((nx / W, (ny + nh - layout.HDR_IN) / H))
    assert cv.point((lx + lw) / layout.LAND[0], ly / layout.LAND[1]) == pytest.approx(((nx + nw) / W, ny / H))


@pytest.mark.parametrize("name, num, pt, sz, expected_pt, expected_sz", [
    ("linkedin_4x5", 5, (0.5, 0.45), (0.1, 0.05),
     (0.15647668393782393, 0.34986341615877486), (0.169641143734408, 0.04792415481585664)),
    ("linkedin_4x5", 2, (0.528, 0.862), (0.175, 0.21),
     (0.7858912747102932, 0.9134880658436213), (0.262142126789366, 0.16766935050993023)),
    ("landscape", 5, (0.5, 0.45), (0.1, 0.05), (0.5, 0.45), (0.1, 0.05000000000000001)),
])
def test_mapping_matches_the_original_implementation(name, num, pt, sz, expected_pt, expected_sz):
    # Values from the pre-refactor M() and MS(). Exact equality: float operation
    # order affects rendered pixels.
    cv = _canvas(name, num)
    assert cv.point(*pt) == expected_pt
    assert cv.size(*sz) == expected_sz


@pytest.mark.parametrize("name", ["landscape", "linkedin_4x5"])
def test_size_agrees_with_point(name):
    cv = _canvas(name, 5)
    x0, y0 = cv.point(0.5, 0.4)
    x1, y1 = cv.point(0.6, 0.45)
    assert cv.size(0.1, 0.05) == pytest.approx((x1 - x0, y1 - y0))


# ---- figure construction (figures: both layouts from one VisualizationData, conftest.py)

def test_both_layouts_build_from_the_same_data(figures):
    figs = figures
    for name, (fig, _) in figs.items():
        assert tuple(fig.get_size_inches()) == layout.FIGURE_SIZES[name]
    # Same data, same drawn content: both layouts have the same artists per axes.
    (fa, ha), (fb, hb) = figs["landscape"], figs["linkedin_4x5"]
    # 2 + 1 + 2 + 2 + 2 + 2 + 3 (panel 7 heatmap, actual row, colorbar) axes.
    assert len(fa.axes) == len(fb.axes) == 14
    assert [len(t.get_text()) for t in fa.texts] == [len(t.get_text()) for t in fb.texts]
    assert np.array_equal(ha["heat7"].images[0].get_array(), hb["heat7"].images[0].get_array())


def test_build_figure_does_not_change_the_data(figures, default_data):
    import generate_visualizations as gv
    data = default_data
    fresh = build_visualization_data(gv.EXAMPLE_N, gv.WHEEL_MODULUS, gv.CRT_MODULI,
                                     gv.HEATMAP_MAX, gv.SIEVE_PRIMES, gv.COMET_MAX)
    assert data.prime_wheel == fresh.prime_wheel and data.goldbach_wheel == fresh.goldbach_wheel
    assert data.sum_of_squares == fresh.sum_of_squares
    assert np.array_equal(data.residue_space.grid, fresh.residue_space.grid)
    assert np.array_equal(data.sieve.counts, fresh.sieve.counts)


def test_build_figure_returns_named_axes(figures):
    for fig, axes in figures.values():
        assert set(axes) == AXES_KEYS
        assert len(axes["lattices"]) == 2 and len(axes["circles"]) == 2
        for key, value in axes.items():
            for ax in value if isinstance(value, list) else [value]:
                assert ax.figure is fig, key


# ---- the parity figure (optional extra) and its generator

PARITY_OUTPUTS = {"images/extras/parity-landscape.png": (3200, 1800),
                  "images/extras/parity-linkedin-4x5.png": (2160, 2700)}
POSTER_OUTPUTS = {"images/landscape.png", "images/landscape-preview.png",
                  "images/linkedin-4x5.png", "images/linkedin-4x5-preview.png"}


def written_files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_poster_generator_is_independent_of_the_parity_extra(tmp_path, monkeypatch):
    # generate_visualizations.py knows nothing of the extra and writes only the poster.
    import inspect
    import generate_visualizations as gv
    source = inspect.getsource(gv).lower()
    assert "parity" not in source and "extras" not in source
    assert gv.EXPORTS == {"landscape": (3600, 1200), "linkedin_4x5": (2160, 1080)}
    assert set(layout.FIGURE_SIZES) == set(layout.LAYOUTS) == {"landscape", "linkedin_4x5"}
    monkeypatch.chdir(tmp_path)
    with matplotlib.rc_context():
        gv.main()
    assert written_files(tmp_path) == POSTER_OUTPUTS


@pytest.mark.parity
def test_parity_figure_sizes_and_cards():
    assert layout.PARITY_FIGURE_SIZES == {"landscape": (16.0, 9.0), "linkedin_4x5": (16.2, 20.25)}
    for name, (W, H) in layout.PARITY_FIGURE_SIZES.items():
        cards = layout.PARITY_LAYOUTS[name]
        for x, y, w, h in cards.values():
            assert 0 <= x and x + w <= W + 1e-9 and 0 <= y and y + h <= H + 1e-9, name
        (ax_, ay, aw, ah), (sx, sy, sw, sh) = cards["average"], cards["sums"]
        if name == "landscape":
            assert ay == sy and ax_ + aw < sx                   # side by side
        else:
            assert ax_ == sx and sy + sh < ay                   # stacked, average on top
        assert (aw, ah) == pytest.approx((sw, sh))              # same size, not stretched


@pytest.mark.parity
def test_parity_figure_axes(parity_figures):
    for name, (fig, axes) in parity_figures.items():
        assert tuple(fig.get_size_inches()) == layout.PARITY_FIGURE_SIZES[name]
        assert set(axes) == {"average", "sums"} and len(fig.axes) == 2
        for ax in axes.values():
            assert ax.figure is fig and ax.get_xscale() == "log"
        assert axes["sums"].get_yscale() == "symlog"
    # Same data, same drawn text in both layouts.
    (fa, _), (fb, _) = parity_figures.values()
    assert [t.get_text() for t in fa.texts] == [t.get_text() for t in fb.texts]


@pytest.mark.parity
def test_parity_generator_data_checks_pass(parity):
    from extras import generate_parity_visualization as gp
    checks = gp.sanity_checks(parity)
    assert checks and all(ok for _, ok in checks), [d for d, ok in checks if not ok]


@pytest.mark.parity
def test_parity_generator_outputs(tmp_path):
    # Run as documented, as a script by path, with outputs relative to the working
    # directory (here a temporary one): both layouts at the export sizes, nothing else.
    from PIL import Image
    r = subprocess.run([sys.executable, str(REPO / "extras" / "generate_parity_visualization.py")],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert written_files(tmp_path) == set(PARITY_OUTPUTS)
    for path, size in PARITY_OUTPUTS.items():
        with Image.open(tmp_path / path) as im:
            assert im.size == size, path
