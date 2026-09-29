import matplotlib

# Tests never open a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def default_data():
    """Visualization data at the settings in generate_visualizations.py."""
    import generate_visualizations as gv
    from goldbach.visualization_data import build_visualization_data
    return build_visualization_data(gv.EXAMPLE_N, gv.WHEEL_MODULUS, gv.CRT_MODULI,
                                    gv.HEATMAP_MAX, gv.SIEVE_PRIMES, gv.COMET_MAX)


@pytest.fixture(scope="session")
def figures(default_data):
    """Both layouts, built once from the same visualization data, as {name: (fig, axes)}.

    Built under the figure style and drawn once, as saving would, so tick labels
    and text exist before tests read them.
    """
    from goldbach.layout import FIGURE_SIZES
    from goldbach.render import build_figure
    from goldbach.style import apply_style
    with matplotlib.rc_context():
        apply_style()
        figs = {name: build_figure(name, default_data) for name in FIGURE_SIZES}
        for fig, _ in figs.values():
            fig.canvas.draw()
        yield figs
    for fig, _ in figs.values():
        plt.close(fig)


@pytest.fixture(scope="session")
def parity():
    """Parity figure data at the settings in extras/generate_parity_visualization.py."""
    from extras import generate_parity_visualization as gp
    from goldbach.visualization_data import parity_data
    return parity_data(gp.PARITY_NS, gp.SPLIT_LEVELS)


@pytest.fixture(scope="session")
def parity_figures(parity):
    """Both parity layouts, built once from the same data, as {name: (fig, axes)}.
    Built under the figure style and drawn once, as saving would."""
    from goldbach.layout import PARITY_FIGURE_SIZES
    from goldbach.render import build_parity_figure
    from goldbach.style import apply_style
    with matplotlib.rc_context():
        apply_style()
        figs = {name: build_parity_figure(name, parity) for name in PARITY_FIGURE_SIZES}
        for fig, _ in figs.values():
            fig.canvas.draw()
        yield figs
    for fig, _ in figs.values():
        plt.close(fig)
