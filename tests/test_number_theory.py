"""Known values and properties of goldbach.number_theory."""
import numpy as np
import pytest

from goldbach import number_theory as nt


# ---- primes

def test_prime_table_size():
    assert nt.PRIME_LIMIT == 10_000
    assert len(nt.primes_between(0, nt.PRIME_LIMIT + 1)) == 1229


@pytest.mark.parametrize("n", [2, 3, 5, 97, 9973])
def test_is_prime_true(n):
    assert nt.is_prime(n)


@pytest.mark.parametrize("n", [0, 1, 4, 91, 9999])
def test_is_prime_false(n):
    assert not nt.is_prime(n)


def test_is_prime_uses_absolute_value():
    assert nt.is_prime(-7)
    assert not nt.is_prime(-9)


def test_is_prime_raises_above_limit():
    with pytest.raises(ValueError):
        nt.is_prime(nt.PRIME_LIMIT + 1)


def test_primes_between():
    assert nt.primes_between(6, 1000)[:3] == [7, 11, 13]
    assert nt.primes_between(6, 1000)[-1] == 997
    assert nt.primes_between(8, nt.PRIME_LIMIT + 1)[-1] == 9973
    assert all(type(p) is int for p in nt.primes_between(0, 100))
    with pytest.raises(ValueError):
        nt.primes_between(0, nt.PRIME_LIMIT + 2)


def test_prime_table_is_read_only():
    with pytest.raises(ValueError):
        nt._SIEVE[4] = True


# ---- Goldbach pairs

def test_goldbach_pairs_of_100():
    assert nt.goldbach_pairs(100) == [(3, 97), (11, 89), (17, 83), (29, 71), (41, 59), (47, 53)]


@pytest.mark.parametrize("N, count", [(4, 1), (98, 3), (100, 6), (1000, 28)])
def test_goldbach_pair_counts(N, count):
    assert len(nt.goldbach_pairs(N)) == count


def test_goldbach_pairs_are_ordered_prime_pairs():
    for N in range(4, 2001, 2):
        pairs = nt.goldbach_pairs(N)
        assert pairs, N
        assert [p for p, _ in pairs] == sorted(p for p, _ in pairs)
        for p, q in pairs:
            assert p <= q and p + q == N
            assert nt.is_prime(p) and nt.is_prime(q)


# ---- sums of two squares

@pytest.mark.parametrize("n, expected", [
    (2, (1, 1)), (5, (1, 2)), (37, (1, 6)), (61, (5, 6)), (89, (5, 8)), (11, None),
])
def test_two_squares_known_values(n, expected):
    assert nt.two_squares(n) == expected


def test_two_squares_for_primes():
    # Fermat: an odd prime is a sum of two squares iff p ≡ 1 (mod 4).
    for p in nt.primes_between(0, nt.PRIME_LIMIT + 1):
        t = nt.two_squares(p)
        assert (t is not None) == (p == 2 or p % 4 == 1), p
        if t:
            a, b = t
            assert a <= b and a * a + b * b == p


@pytest.mark.parametrize("n, count", [(2, 4), (5, 8), (11, 0), (89, 8), (25, 12)])
def test_lattice_points_on_circle_counts(n, count):
    assert len(nt.lattice_points_on_circle(n, 11)) == count


def test_lattice_points_on_circle_row_order():
    assert nt.lattice_points_on_circle(5, 3) == [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                                 (1, -2), (1, 2), (2, -1), (2, 1)]


# ---- residues

def test_wheel_30():
    assert nt.wheel_primes(30) == [2, 3, 5]
    assert nt.coprime_residues(30) == [1, 7, 11, 13, 17, 19, 23, 29]


def test_residue_types_for_100_mod_30():
    assert nt.residue_types(100, 30) == [(11, 29), (17, 23)]
    assert nt.residue_type(89, 11, 30) == (11, 29)


def test_goldbach_pairs_fall_in_residue_types():
    for N in range(8, 2001, 2):
        types = nt.residue_types(N, 30)
        for p, q in nt.goldbach_pairs(N):
            if p > 5:
                assert nt.residue_type(p, q, 30) in types, (N, p, q)


def test_fixed_class_for_100():
    assert {m: nt.fixed_class(100, m) for m in (3, 5, 7)} == {3: 2, 5: 0, 7: 1}


def test_fixed_class_is_fixed_by_the_involution():
    for m in (3, 5, 7, 9, 11, 13):
        for N in range(4, 400, 2):
            c = nt.fixed_class(N, m)
            assert 0 <= c < m and (N - c) % m == c


def test_centred_range_and_rep():
    assert list(nt.centred_range(7, 1)) == [-2, -1, 0, 1, 2, 3, 4]
    assert list(nt.centred_range(5, 0)) == [-2, -1, 0, 1, 2]
    for m in (3, 5, 7):
        for c in range(m):
            window = nt.centred_range(m, c)
            for x in range(-30, 30):
                v = nt.centred_rep(x, m, c)
                assert v in window and (v - x) % m == 0


def test_involution_is_a_point_reflection_in_centred_coordinates():
    # p -> N - p maps rep(p) to 2c - rep(p) for every residue p.
    for N in (98, 100, 1000):
        for m in (3, 5, 7):
            c = nt.fixed_class(N, m)
            for p in range(m):
                assert nt.centred_rep(p, m, c) + nt.centred_rep(N - p, m, c) == 2 * c


def test_involution_has_exactly_one_fixed_class_mod_odd_m():
    for m in (3, 5, 7, 9, 11):
        for N in range(4, 200, 2):
            fixed = [x for x in range(m) if (N - x) % m == x]
            assert fixed == [nt.fixed_class(N, m)], (N, m)


def test_admissible_count():
    assert nt.admissible_count(100, (3, 5, 7)) == 20
    counts = [nt.admissible_count(N, (3, 5, 7)) for N in range(4, 2001, 2)]
    assert (min(counts), max(counts)) == (15, 48)
    assert isinstance(nt.admissible_count(100, (3, 5, 7)), np.integer)


# ---- Gaussian and Eisenstein integers

@pytest.mark.parametrize("ab, cls", [((1, 1), "ram"), ((3, 0), "inert"), ((2, 1), "split"), ((0, 5), None),
                                     ((0, 0), None), ((1, 0), None)])
def test_classify_gaussian(ab, cls):
    assert nt.classify_gaussian(*ab) == cls


@pytest.mark.parametrize("ab, cls", [((2, 1), "ram"), ((2, 0), "inert"), ((3, 1), "split"),
                                     ((0, 0), None), ((1, 0), None)])
def test_classify_eisenstein(ab, cls):
    assert nt.classify_eisenstein(*ab) == cls


def test_lattice_windows():
    g = list(nt.gaussian_points(10))
    e = list(nt.eisenstein_points(10))
    assert len(g) == 441 and len(e) == 471
    assert all((a, b) == (x, y) for a, b, x, y in g)
    for a, b, x, y in e:
        assert x == a - b / 2 and y == b * np.sqrt(3) / 2
        assert abs(x) <= 10 and abs(y) <= 10


# ---- sieving

def test_sieve_survivors_at_1000():
    zs = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    assert nt.sieve_survivors(1000, zs) == [250, 83, 66, 48, 40, 34, 31, 28, 26]


def test_sieve_survivors_do_not_increase_with_z():
    zs = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for N in range(4, 1001, 2):
        counts = nt.sieve_survivors(N, zs)
        assert all(a >= b for a, b in zip(counts, counts[1:])), N


def test_sieve_survivors_cover_goldbach_pairs_above_z():
    # Every pair with p > z survives z; survivors can also include non-primes.
    zs = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for N in range(4, 1001, 2):
        counts = nt.sieve_survivors(N, zs)
        pairs = nt.goldbach_pairs(N)
        for z, c in zip(zs, counts):
            assert c >= sum(1 for p, _ in pairs if p > z), (N, z)
