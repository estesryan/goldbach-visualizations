"""Parity visualization: data, brute-force cross-checks, layout and generator.

Data at the default settings (the `parity` fixture, conftest.py) is checked against
known values and invariants, then recomputed independently with the pure-Python
helpers in tests/brute_force.py. Layout, figure construction and the generator
follow. The drawn figure is checked in test_parity_figures.py.
"""
import subprocess
import sys
from math import isqrt
from pathlib import Path

import numpy as np
import pytest

import brute_force as bf
from extras.parity import data as pd
from extras.parity import layout as pl
from figure_test_helpers import written_files
from goldbach import number_theory as nt
from goldbach import visualization_data as vd

REPO = Path(__file__).resolve().parents[1]
PARITY_OUTPUTS = {"images/extras/parity-landscape.png": (3200, 1800),
                  "images/extras/parity-preview-4x5.png": (2160, 2700)}


# ---- data at the default settings

@pytest.mark.parity
def test_parity_default_settings():
    from extras.parity import generate as gp
    assert gp.PARITY_NS == (999_000, 999_998, 1_000_000)
    assert gp.SPLIT_LEVELS == (7, 31)


@pytest.mark.parity
def test_parity_known_values(parity):
    assert list(parity.sweeps) == [999_000, 999_998, 1_000_000] and parity.featured == 1_000_000
    endpoint = {N: (len(sw.zs), int(sw.zs[-1]), int(sw.M[-1]), int(sw.S[-1])) for N, sw in parity.sweeps.items()}
    assert endpoint == {999_000: (168, 997, 11_083, 11_083), 999_998: (168, 997, 4_191, 4_191),
                        1_000_000: (168, 997, 5_382, 5_382)}
    sw = parity.sweeps[1_000_000]
    assert (sw.M[0], sw.S[0]) == (249_999, -347)
    assert (parity.goldbach_total, parity.goldbach_small) == (5_402, 20)
    assert parity.meet_level == 983
    assert parity.class_splits == (pd.ClassSplit("all", None, 249_778, 250_221),
                                   pd.ClassSplit("survivors", 7, 23_924, 23_695),
                                   pd.ClassSplit("survivors", 31, 10_458, 10_222),
                                   pd.ClassSplit("goldbach", None, 5_402, 0))


@pytest.mark.parity
def test_parity_split_level_must_be_a_sieve_level():
    with pytest.raises(ValueError):
        pd.parity_data((1_000,), (8,))


@pytest.mark.parity
def test_parity_invariants(parity):
    for N, sw in parity.sweeps.items():
        assert np.all(np.diff(sw.M) <= 0), N                    # sieving more never adds survivors
        assert np.all(np.abs(sw.S) <= sw.M), N                  # each term is ±1
        assert np.all((sw.M - sw.S) % 2 == 0), N                # so S ≡ M (mod 2)
        assert sw.S[-1] == sw.M[-1], N                          # at the last level every term is +1
        assert np.array_equal(sw.average, sw.S / sw.M), N
    assert parity.goldbach_total - parity.goldbach_small == parity.sweeps[parity.featured].M[-1]


@pytest.mark.parity
def test_parity_dtypes(parity):
    for sw in parity.sweeps.values():
        assert sw.zs.dtype == np.int64 and sw.M.dtype == np.int64 and sw.S.dtype == np.int64
        assert sw.average.dtype == np.float64


@pytest.mark.parity
def test_parity_data_needs_even_n():
    with pytest.raises(ValueError):
        pd.parity_data((1001,))


@pytest.mark.parity
def test_parity_data_is_not_part_of_the_poster():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(vd.VisualizationData)}
    assert len(fields) == 7 and not any("parity" in f for f in fields)


# ---- brute-force helpers against known values

@pytest.mark.parity
def test_bf_liouville_known_values():
    assert [bf.omega(n) for n in (1, 2, 12, 64, 97, 1_000_000)] == [0, 1, 3, 6, 1, 12]
    assert [bf.liouville(n) for n in range(1, 13)] == [1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1]


@pytest.mark.parity
def test_bf_liouville_summatory_values(bf_parity_tables):
    # L(10^k) = sum of λ(n) for n <= 10^k, k = 1..6 (OEIS A090410).
    _, lam = bf_parity_tables
    assert [sum(lam[1:10 ** k + 1]) for k in range(1, 7)] == [0, -2, -14, -94, -288, -530]


@pytest.mark.parity
def test_bf_parity_sweep_by_hand():
    # Same hand-worked N = 30 example as test_number_theory.test_parity_sieve_by_hand.
    spf = bf.smallest_prime_factors(30)
    lam = bf.liouville_from_spf(spf)
    assert bf.parity_sweep(30, [2, 3, 5], spf, lam) == ([7, 4, 3], [5, 2, 3])
    assert spf[2:16] == [2, 3, 2, 5, 2, 7, 2, 3, 2, 11, 2, 13, 2, 3]


# ---- the data against brute force, at the settings in extras/parity/generate.py

@pytest.fixture(scope="module")
def bf_parity_tables():
    """Smallest prime factors and λ up to 10⁶, in pure Python (spf recursion)."""
    spf = bf.smallest_prime_factors(1_000_000)
    return spf, bf.liouville_from_spf(spf)


@pytest.mark.parity
def test_liouville_table_matches_spf_recursion(bf_parity_tables):
    _, lam = bf_parity_tables
    assert nt.liouville_table(1_000_000).tolist() == lam


@pytest.mark.parity
def test_liouville_table_matches_trial_division_sample():
    import random
    lam = nt.liouville_table(1_000_000)
    for n in random.Random(1).sample(range(1, 1_000_001), 3000):
        assert lam[n] == bf.liouville(n), n


@pytest.mark.parity
def test_parity_levels_are_the_primes_up_to_sqrt_n(parity):
    for N, sw in parity.sweeps.items():
        assert list(sw.zs) == bf.primes_between(2, isqrt(N) + 1), N
        # The last level is the largest prime <= √N: no prime lies between it and √N.
        assert not any(bf.is_prime(k) for k in range(sw.zs[-1] + 1, isqrt(N) + 1)), N


@pytest.mark.parity
def test_parity_counts_and_sums_match_least_prime_factor_count(parity, bf_parity_tables):
    # Every plotted level for every N: M and S recounted without sieving.
    spf, lam = bf_parity_tables
    for N, sw in parity.sweeps.items():
        Ms, Ss = bf.parity_sweep(N, [int(z) for z in sw.zs], spf, lam)
        assert sw.M.tolist() == Ms, N
        assert sw.S.tolist() == Ss, N


@pytest.mark.parity
def test_parity_endpoint_survivors_are_prime_pairs(parity, bf_parity_tables):
    spf, _ = bf_parity_tables
    for N, sw in parity.sweeps.items():
        surv = bf.survivors(N, int(sw.zs[-1]), spf)
        assert len(surv) == sw.M[-1], N
        assert all(bf.is_prime(a) and bf.is_prime(N - a) for a in surv), N
        assert all(bf.liouville(a) * bf.liouville(N - a) == 1 for a in surv), N


@pytest.mark.parity
def test_parity_endpoint_counts_goldbach_pairs_above_sqrt_n(parity):
    # At the last level S = M = the trial-division count of pairs whose smaller prime exceeds √N.
    for N, sw in parity.sweeps.items():
        pairs = bf.goldbach_pairs(N)
        assert sw.S[-1] == sw.M[-1] == sum(1 for p, _ in pairs if p > isqrt(N)), N


@pytest.mark.parity
def test_parity_caption_goldbach_counts(parity):
    pairs = bf.goldbach_pairs(parity.featured)
    assert parity.goldbach_total == len(pairs)
    assert parity.goldbach_small == sum(1 for p, _ in pairs if p <= isqrt(parity.featured))


@pytest.mark.parity
def test_parity_class_splits_match_brute_force(parity, bf_parity_tables):
    # Every split recounted by λ from the pure-Python table, without sieving.
    spf, lam = bf_parity_tables
    N = parity.featured
    sw = parity.sweeps[N]
    for c in parity.class_splits:
        if c.kind == "all":
            signs = [lam[a] * lam[N - a] for a in range(2, N // 2 + 1)]
        elif c.kind == "survivors":
            signs = [lam[a] * lam[N - a] for a in bf.survivors(N, c.z, spf)]
            i = list(sw.zs).index(c.z)
            assert c.plus + c.minus == sw.M[i] and c.plus - c.minus == sw.S[i]
        else:
            pairs = bf.goldbach_pairs(N)
            signs = [bf.liouville(p) * bf.liouville(q) for p, q in pairs]
            assert c.minus == 0 and c.plus == len(pairs)        # every Goldbach pair has sign +1
        assert (c.plus, c.minus) == (signs.count(1), signs.count(-1)), c


@pytest.mark.parity
def test_parity_meet_level(parity, bf_parity_tables):
    # From the pure-Python sweep: S = M at every level from meet_level on, and not just before.
    spf, lam = bf_parity_tables
    zs = [int(z) for z in parity.sweeps[parity.featured].zs]
    Ms, Ss = bf.parity_sweep(parity.featured, zs, spf, lam)
    i = zs.index(parity.meet_level)
    assert Ss[i:] == Ms[i:]
    assert i == 0 or Ss[i - 1] != Ms[i - 1]


# ---- layout, figure construction and generator

@pytest.mark.parity
def test_parity_figure_sizes_and_cards():
    assert pl.FIGURE_SIZES == {"landscape": (16.0, 9.0), "preview_4x5": (16.2, 20.25)}
    for name, (W, H) in pl.FIGURE_SIZES.items():
        cards = pl.LAYOUTS[name]
        for x, y, w, h in cards.values():
            assert 0 <= x and x + w <= W + 1e-9 and 0 <= y and y + h <= H + 1e-9, name
        assert list(cards) == ["average", "sums", "classes"]
        (ax_, ay, aw, ah), (sx, sy, sw, sh), (cx, cy, cw, ch) = cards.values()
        if name == "landscape":
            assert ay == sy == cy and ax_ + aw < sx and sx + sw < cx     # side by side, in order
        else:
            assert ax_ == sx == cx and cy + ch < sy and sy + sh < ay     # stacked, average on top
        assert (aw, ah) == pytest.approx((sw, sh)) and (cw, ch) == pytest.approx((sw, sh))   # same size


@pytest.mark.parity
def test_parity_figure_axes(parity_figures):
    for name, (fig, axes) in parity_figures.items():
        assert tuple(fig.get_size_inches()) == pl.FIGURE_SIZES[name]
        assert set(axes) == {"average", "sums", "classes"} and len(fig.axes) == 3
        for ax in axes.values():
            assert ax.figure is fig
        assert axes["average"].get_xscale() == axes["sums"].get_xscale() == "log"
        assert axes["sums"].get_yscale() == "symlog"
        assert axes["classes"].get_xlim() == (0, 1)
    # Same data, same drawn text in both layouts.
    (fa, _), (fb, _) = parity_figures.values()
    assert [t.get_text() for t in fa.texts] == [t.get_text() for t in fb.texts]


@pytest.mark.parity
def test_parity_generator_data_checks_pass(parity):
    from extras.parity import generate as gp
    checks = gp.sanity_checks(parity)
    assert checks and all(ok for _, ok in checks), [d for d, ok in checks if not ok]


@pytest.mark.parity
def test_parity_generator_outputs(tmp_path):
    # Run as documented, as a module, with outputs relative to the working directory
    # (here a temporary one, with the repository on the path): both layouts at the
    # export sizes, nothing else.
    import os
    from PIL import Image
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-m", "extras.parity.generate"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert written_files(tmp_path) == set(PARITY_OUTPUTS)
    for path, size in PARITY_OUTPUTS.items():
        with Image.open(tmp_path / path) as im:
            assert im.size == size, path


@pytest.mark.parity
def test_shared_modules_do_not_know_the_parity_visualization():
    # Shared poster modules own nothing of the parity figure; it lives in extras.parity.
    import inspect
    from goldbach import layout, render
    for module in (layout, render, vd):
        source = inspect.getsource(module)
        assert "parity" not in source.lower() and "Parity" not in source, module.__name__
