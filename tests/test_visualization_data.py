"""Visualization data at the default settings: known values, invariants and dtypes.

Dtypes are pinned where the drawing code depends on them, so a change that
would alter the rendered figure shows up here first.
"""
import numpy as np
import pytest

import generate_visualizations as gv
from goldbach import visualization_data as vd


def test_default_settings():
    # The expected values below assume these settings.
    assert (gv.EXAMPLE_N, gv.WHEEL_MODULUS, gv.CRT_MODULI) == (100, 30, (3, 5, 7))
    assert (gv.HEATMAP_MAX, gv.COMET_MAX) == (1000, 2000)
    assert gv.SIEVE_PRIMES == [2, 3, 5, 7, 11, 13, 17, 19, 23]


@pytest.fixture(scope="module")
def prime_wheel_data():
    return vd.prime_wheel_data(30)


@pytest.fixture(scope="module")
def goldbach_wheel_data():
    return vd.goldbach_wheel_data(100, 30)


@pytest.fixture(scope="module")
def lattice_prime_data():
    return vd.lattice_prime_data()


@pytest.fixture(scope="module")
def residue_space_data():
    return vd.residue_space_data((3, 5, 7))


@pytest.fixture(scope="module")
def involution_data():
    return vd.involution_data(100, (3, 5, 7), 2000)


@pytest.fixture(scope="module")
def sum_of_squares_data():
    return vd.sum_of_squares_data(100, 30)


@pytest.fixture(scope="module")
def sieve_data():
    return vd.sieve_data(1000, [2, 3, 5, 7, 11, 13, 17, 19, 23], (3, 5, 7))


# ---- panel 1

def test_prime_wheel(prime_wheel_data):
    assert prime_wheel_data.wheel_primes == [2, 3, 5]
    assert prime_wheel_data.coprime == [1, 7, 11, 13, 17, 19, 23, 29]
    assert (prime_wheel_data.primes[0], prime_wheel_data.primes[-1], len(prime_wheel_data.primes)) == (7, 997, 165)
    assert prime_wheel_data.counts == [18, 24, 22, 20, 22, 18, 21, 20]
    assert sum(prime_wheel_data.counts) == len(prime_wheel_data.primes)


# ---- panel 2

def test_goldbach_wheel(goldbach_wheel_data):
    assert goldbach_wheel_data.residue == 10 and goldbach_wheel_data.small_cut == 5
    assert goldbach_wheel_data.pairs == [(3, 97), (11, 89), (17, 83), (29, 71), (41, 59), (47, 53)]
    assert goldbach_wheel_data.exceptions == [(3, 97)]
    assert goldbach_wheel_data.main == goldbach_wheel_data.pairs[1:]
    assert goldbach_wheel_data.types == [(11, 29), (17, 23)]


def test_goldbach_wheel_grouping_keeps_first_appearance_order(goldbach_wheel_data):
    # Chords are drawn in this order.
    assert list(goldbach_wheel_data.by_type.items()) == [
        ((11, 29), [(11, 89), (29, 71), (41, 59)]),
        ((17, 23), [(17, 83), (47, 53)]),
    ]
    assert all(t in goldbach_wheel_data.types for t in goldbach_wheel_data.by_type)


# ---- panel 3

def test_lattice_primes_counts(lattice_prime_data):
    g, e = lattice_prime_data.gaussian, lattice_prime_data.eisenstein
    assert (len(g.points), len(g.split), len(g.inert), len(g.ram)) == (441, 128, 8, 4)
    assert (len(e.points), len(e.split), len(e.inert), len(e.ram)) == (471, 176, 16, 6)


def test_lattice_primes_ramified_points(lattice_prime_data):
    assert {tuple(p) for p in lattice_prime_data.gaussian.ram} == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    # The six associates of 1 - ω all have norm 3, so they lie on the circle |z| = √3.
    for x, y in lattice_prime_data.eisenstein.ram:
        assert np.isclose(x * x + y * y, 3)


def test_lattice_primes_dtypes(lattice_prime_data):
    assert lattice_prime_data.gaussian.points.dtype == np.int64 and lattice_prime_data.gaussian.points.shape == (441, 2)
    assert lattice_prime_data.eisenstein.points.dtype == np.float64 and lattice_prime_data.eisenstein.points.shape == (471, 2)


# ---- panel 4

def test_residue_space(residue_space_data):
    assert residue_space_data.modulus == 105 and residue_space_data.admissible == 48
    assert (residue_space_data.primes[0], residue_space_data.primes[-1], len(residue_space_data.primes)) == (11, 9973, 1225)
    assert residue_space_data.grid.shape == (48, 4) and residue_space_data.excluded.shape == (57, 4)
    assert all(residue_space_data.grid[:, :3].all(axis=1))
    assert not residue_space_data.excluded[:, :3].all(axis=1).any()
    assert (residue_space_data.grid[:, 3].min(), residue_space_data.grid[:, 3].max()) == (22, 29)
    assert residue_space_data.grid[:, 3].sum() == len(residue_space_data.primes)
    assert (residue_space_data.excluded[:, 3] == 0).all()


def test_residue_space_heat(residue_space_data):
    assert residue_space_data.heat.shape == (7, 5)
    assert residue_space_data.heat.sum() == len(residue_space_data.primes)
    assert (residue_space_data.heat[0, :] == 0).all() and (residue_space_data.heat[:, 0] == 0).all()


def test_residue_space_dtypes(residue_space_data):
    assert residue_space_data.grid.dtype == np.int64 and residue_space_data.heat.dtype == np.int64


# ---- panel 5

def test_involution_fixed_class_and_shown_pairs(involution_data):
    assert involution_data.centre == {3: 2, 5: 0, 7: 1}
    assert involution_data.first_mod_residues == [2]
    assert list(involution_data.xs) == [-2, -1, 0, 1, 2, 3, 4] and list(involution_data.ys) == [-2, -1, 0, 1, 2]
    assert involution_data.skipped == [(3, 97)]
    assert [(p, q) for p, q, _, _ in involution_data.shown] == [(11, 89), (17, 83), (29, 71), (41, 59), (47, 53)]
    assert involution_data.reflects
    for p, q, pp, qq in involution_data.shown:
        assert ((pp[0] + qq[0]) / 2, (pp[1] + qq[1]) / 2) == (involution_data.centre[7], involution_data.centre[5])
        # The subtitle's claim: every pair shown has p ≡ q ≡ 2 (mod 3).
        assert p % 3 == q % 3 == involution_data.first_mod_residues[0]


def test_involution_cells(involution_data):
    assert len(involution_data.cells) == 35
    # Blocked: x ≡ 0 or 2 (mod 7) gives 2 columns of 5, y ≡ 0 (mod 5) one row of 7, 2 overlap.
    assert sum(1 for *_, blocked in involution_data.cells if blocked) == 15
    # Every shown p and q sits in an admissible cell.
    open_cells = {(x, y) for x, y, blocked in involution_data.cells if not blocked}
    for _, _, pp, qq in involution_data.shown:
        assert pp in open_cells and qq in open_cells


def test_involution_comet(involution_data):
    assert len(involution_data.comet_N) == 999 and involution_data.comet_N[0] == 4 and involution_data.comet_N[-1] == 2000
    i = list(involution_data.comet_N).index(100)
    assert involution_data.comet_pairs[i] == 6 and involution_data.comet_admissible[i] == 20
    assert (involution_data.comet_admissible.min(), involution_data.comet_admissible.max()) == (15, 48)
    assert involution_data.modulus == 105 and involution_data.comet_max == 2000


def test_involution_dtypes(involution_data):
    assert involution_data.comet_N.dtype == np.int64
    assert involution_data.comet_pairs.dtype == np.int64 and involution_data.comet_admissible.dtype == np.int64
    assert all(type(v) is int for _, _, pp, qq in involution_data.shown for v in pp + qq)


# ---- panel 6

def test_sum_of_squares_examples(sum_of_squares_data):
    left, right = sum_of_squares_data.left, sum_of_squares_data.right
    assert (left.N, left.p, left.q, left.radius) == (100, 11, 89, 11)
    assert (right.N, right.p, right.q, right.radius) == (98, 37, 61, 9)
    assert left.N % 4 == 0 and (left.p % 4) != (left.q % 4)
    assert right.N % 4 == 2 and right.p % 4 == right.q % 4 == 1


def test_sum_of_squares_lattice_points(sum_of_squares_data):
    left, right = sum_of_squares_data.left, sum_of_squares_data.right
    assert (len(left.p_points), len(left.q_points)) == (0, 8)
    assert (len(right.p_points), len(right.q_points)) == (8, 8)
    assert (left.p_squares, left.q_squares) == (None, (5, 8))
    assert (right.p_squares, right.q_squares) == ((1, 6), (5, 6))


# ---- panel 7

def test_sieve(sieve_data):
    assert sieve_data.counts.shape == (10, 499) and sieve_data.counts.dtype == np.float64
    assert list(sieve_data.Ns[[0, -1]]) == [4, 1000] and sieve_data.Ns.dtype == np.int64
    assert list(sieve_data.counts[:, -1]) == [250, 83, 66, 48, 40, 34, 31, 28, 26, 28]
    assert len(sieve_data.pairs) == 28
    assert sieve_data.lost == [(3, 997), (17, 983), (23, 977)]
    assert sieve_data.moduli == (3, 5, 7)


def test_sieve_survivors_are_not_only_primes(sieve_data):
    # 26 pairs survive z = 23, but only 25 actual pairs have p > 23.
    assert sieve_data.counts[-2, -1] == 26
    assert sum(1 for p, _ in sieve_data.pairs if p > 23) == 25


def test_sieve_streak_claim_mod_3(sieve_data):
    # The subtitle's vertical streaks, checked for mod 3 on average (N > 200).
    last_z = sieve_data.counts[-2]
    by3 = [c for n, c in zip(sieve_data.Ns, last_z) if n % 3 == 0 and n > 200]
    no3 = [c for n, c in zip(sieve_data.Ns, last_z) if n % 3 and n > 200]
    assert np.mean(by3) > np.mean(no3)


# ---- all visualizations

def test_visualization_data_is_frozen(prime_wheel_data):
    with pytest.raises(AttributeError):
        prime_wheel_data.wheel = 12


def test_build_visualization_data_matches_the_individual_functions(prime_wheel_data, goldbach_wheel_data, residue_space_data, involution_data, sum_of_squares_data, sieve_data):
    data = vd.build_visualization_data(100, 30, (3, 5, 7), 1000, [2, 3, 5, 7, 11, 13, 17, 19, 23], 2000)
    assert data.prime_wheel == prime_wheel_data and data.goldbach_wheel == goldbach_wheel_data and data.sum_of_squares == sum_of_squares_data
    assert data.residue_space.primes == residue_space_data.primes and np.array_equal(data.residue_space.grid, residue_space_data.grid)
    assert data.involution.shown == involution_data.shown and np.array_equal(data.involution.comet_pairs, involution_data.comet_pairs)
    assert np.array_equal(data.sieve.counts, sieve_data.counts) and data.sieve.moduli == sieve_data.moduli
    assert len(data.lattice_primes.gaussian.ram) == 4
