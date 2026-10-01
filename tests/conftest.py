import matplotlib

# Tests never open a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def default_data():
    """Visualization data at the settings in generate_poster.py."""
    import generate_poster as gv
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
    """Parity visualization data at the settings in extras/parity/generate.py."""
    from extras.parity import generate as gp
    from extras.parity.data import parity_data
    return parity_data(gp.PARITY_NS, gp.SPLIT_LEVELS)


@pytest.fixture(scope="session")
def parity_figures(parity):
    """Both parity layouts, built once from the same data, as {name: (fig, axes)}.
    Built under the figure style and drawn once, as saving would."""
    from extras.parity.layout import FIGURE_SIZES
    from extras.parity.render import build_figure
    from goldbach.style import apply_style
    with matplotlib.rc_context():
        apply_style()
        figs = {name: build_figure(name, parity) for name in FIGURE_SIZES}
        for fig, _ in figs.values():
            fig.canvas.draw()
        yield figs
    for fig, _ in figs.values():
        plt.close(fig)


@pytest.fixture(scope="session")
def first_survivor():
    """First-survivor data at the settings in extras/first_survivor_experiment/generate.py."""
    from extras.first_survivor_experiment import generate
    return generate.load()


@pytest.fixture(scope="session")
def first_survivor_blocks():
    """The exact selected blocks at the settings in extras/first_survivor_experiment/generate.py."""
    from extras.first_survivor_experiment import generate
    from extras.first_survivor_experiment.data import exact_blocks
    return exact_blocks(generate.BLOCK_TARGETS)


@pytest.fixture(scope="session")
def first_survivor_figures(first_survivor, first_survivor_blocks):
    """Both main layouts and the blocks figure, built once, as {name: (fig, axes)}.
    Built under the figure style and drawn once, as saving would."""
    from extras.first_survivor_experiment import generate
    from extras.first_survivor_experiment.layout import FIGURE_SIZES as FS_SIZES
    from extras.first_survivor_experiment.render import build_blocks_figure, build_figure
    from goldbach.style import apply_style
    with matplotlib.rc_context():
        apply_style()
        figs = {name: build_figure(name, first_survivor, generate.SMALL_MAX) for name in FS_SIZES}
        figs["blocks"] = build_blocks_figure(first_survivor_blocks, first_survivor)
        for fig, _ in figs.values():
            fig.canvas.draw()
        yield figs
    for fig, _ in figs.values():
        plt.close(fig)
