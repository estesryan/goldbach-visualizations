"""Independent mathematical cross-checks.

The first group pins the brute-force helpers to known values, so the reference
itself is checked. The second group compares number_theory with the brute-force
helpers, and the third compares the visualization data with them, over the ranges the
figures use. The parity figure's checks follow the same pattern at the end.
"""
from collections import Counter
from math import isqrt

import numpy as np
import pytest

import brute_force as bf
from goldbach import number_theory as nt
from goldbach import visualization_data as vd

ZS = [2, 3, 5, 7, 11, 13, 17, 19, 23]


# ---- brute-force helpers against known values

def test_bf_prime_count():
    assert len(bf.primes_between(0, 10_001)) == 1229


def test_bf_wheel_30():
    assert bf.prime_factors(30) == [2, 3, 5]
    assert bf.coprime_residues(30) == [1, 7, 11, 13, 17, 19, 23, 29]


def test_bf_prime_wheel_class_counts():
    # Primes greater than 5 and below 1000, by class mod 30.
    ps = bf.primes_between(6, 1000)
    assert (ps[0], ps[-1], len(ps)) == (7, 997, 165)
    counts = Counter(p % 30 for p in ps)
    assert [counts[r] for r in bf.coprime_residues(30)] == [18, 24, 22, 20, 22, 18, 21, 20]


def test_bf_goldbach_wheel_residue_types_for_100():
    types = {tuple(sorted((p % 30, q % 30))) for p, q in bf.goldbach_pairs(100) if p > 5}
    assert sorted(types) == [(11, 29), (17, 23)]


def test_bf_residue_space_crt_classes():
    # Primes greater than 7 up to 10 000 fill exactly the 48 classes with no zero coordinate.
    ps = bf.primes_between(8, 10_001)
    assert (ps[0], ps[-1]) == (11, 9973)
    counts = Counter((p % 3, p % 5, p % 7) for p in ps)
    assert len(counts) == 48 and all(all(k) for k in counts)
    assert (min(counts.values()), max(counts.values())) == (22, 29)


@pytest.mark.parametrize("ab, cls", [((1, 1), "ram"), ((3, 0), "inert"), ((2, 1), "split"), ((0, 5), None)])
def test_bf_gaussian_class_samples(ab, cls):
    assert bf.gaussian_class(*ab) == cls


@pytest.mark.parametrize("ab, cls", [((2, 1), "ram"), ((2, 0), "inert"), ((3, 1), "split")])
def test_bf_eisenstein_class_samples(ab, cls):
    assert bf.eisenstein_class(*ab) == cls


def test_bf_gaussian_window_counts():
    window = bf.gaussian_window(10)
    counts = Counter(bf.gaussian_class(a, b) for a, b in window)
    assert len(window) == 441
    assert (counts["split"], counts["inert"], counts["ram"]) == (128, 8, 4)


def test_bf_eisenstein_window_counts():
    window = bf.eisenstein_window(10)
    counts = Counter(bf.eisenstein_class(a, b) for a, b in window)
    assert len(window) == 471
    assert (counts["split"], counts["inert"], counts["ram"]) == (176, 16, 6)


def test_bf_fixed_class_for_100():
    assert {m: bf.fixed_class(100, m) for m in (3, 5, 7)} == {3: 2, 5: 0, 7: 1}


def test_bf_pairs_reflect_through_fixed_class():
    # p -> N - p is a point reflection through the fixed class in centred coordinates.
    N = 100
    for m in (3, 5, 7):
        c = bf.fixed_class(N, m)
        for p, q in bf.goldbach_pairs(N):
            if p > 7:
                assert bf.centred_rep(p, m, c) + bf.centred_rep(q, m, c) == 2 * c


def test_bf_admissible_counts():
    counts = [bf.admissible_count(N, (3, 5, 7)) for N in range(4, 2001, 2)]
    assert bf.admissible_count(100, (3, 5, 7)) == 20
    assert (min(counts), max(counts)) == (15, 48)


def test_bf_sieve_survivors_at_1000():
    zs = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    column = [bf.sieve_survivors(1000, z) for z in zs]
    assert column == [250, 83, 66, 48, 40, 34, 31, 28, 26]
    # Survivors are not all prime pairs: 26 survive z = 23, but only 25
    # actual pairs have p > 23 (3 of the 28 pairs have p <= 23).
    pairs = bf.goldbach_pairs(1000)
    assert len(pairs) == 28
    assert sum(1 for p, _ in pairs if p > 23) == 25


def test_bf_streak_claim_mod_3():
    # Panel 7: for N > 200, N divisible by 3 has more survivors at z = 23 on average.
    by3, no3 = [], []
    for N in range(202, 1001, 2):
        (by3 if N % 3 == 0 else no3).append(bf.sieve_survivors(N, 23))
    assert sum(by3) / len(by3) > sum(no3) / len(no3)


def test_bf_pairs_for_n_divisible_by_4_include_a_prime_3_mod_4():
    # The panel 6 claim, checked for 4 < N <= 2000.
    for N in range(8, 2001, 4):
        for p, q in bf.goldbach_pairs(N):
            assert p % 4 == 3 or q % 4 == 3, (N, p, q)


# ---- number_theory against brute force

def test_is_prime_matches_trial_division():
    for n in range(-50, nt.PRIME_LIMIT + 1):
        assert nt.is_prime(n) == bf.is_prime(n), n


def test_primes_between_matches_trial_division():
    for lo, hi in [(0, 100), (6, 1000), (8, nt.PRIME_LIMIT + 1), (97, 98), (98, 100)]:
        assert nt.primes_between(lo, hi) == bf.primes_between(lo, hi), (lo, hi)


def test_goldbach_pairs_match_trial_division():
    for N in range(4, 2001, 2):
        assert nt.goldbach_pairs(N) == bf.goldbach_pairs(N), N


def test_two_squares_matches_search():
    for n in range(0, 2001):
        assert nt.two_squares(n) == bf.two_squares(n), n


def test_lattice_points_on_circle_match_search():
    for n in range(0, 500):
        R = isqrt(n) + 2
        assert set(nt.lattice_points_on_circle(n, R)) == bf.lattice_points(n), n


def test_lattice_point_counts_follow_two_squares():
    # Panel 6: |z|² = p has 4 lattice points for p = 2, 8 if p ≡ 1 (mod 4), else 0.
    for p in bf.primes_between(2, 2000):
        expected = 0 if nt.two_squares(p) is None else (4 if p == 2 else 8)
        assert len(bf.lattice_points(p)) == expected, p


@pytest.mark.parametrize("m", [2, 6, 7, 12, 30, 210])
def test_wheel_primes_and_coprime_residues_match_brute_force(m):
    assert nt.wheel_primes(m) == bf.prime_factors(m)
    assert nt.coprime_residues(m) == bf.coprime_residues(m)


def test_residue_types_match_brute_force():
    # Types {r, N - r} with neither residue sharing a prime factor with 30.
    ps = bf.prime_factors(30)
    for N in range(4, 2001, 2):
        expected = sorted({tuple(sorted((r, (N - r) % 30))) for r in range(30)
                           if all(r % p and (N - r) % p for p in ps)})
        assert nt.residue_types(N, 30) == expected, N


def test_fixed_class_and_centred_rep_match_search():
    for m in (3, 5, 7, 9, 11, 13):
        for N in range(4, 2001, 2):
            assert nt.fixed_class(N, m) == bf.fixed_class(N, m), (N, m)
        for c in range(m):
            for r in range(-20, 40):
                assert nt.centred_rep(r, m, c) == bf.centred_rep(r, m, c), (r, m, c)


def test_admissible_count_matches_gcd_count():
    for N in range(4, 2001, 2):
        assert nt.admissible_count(N, (3, 5, 7)) == bf.admissible_count(N, (3, 5, 7)), N


def test_gaussian_classification_matches_brute_force_irreducibility():
    points = {(a, b) for a, b, _, _ in nt.gaussian_points(10)}
    assert points == set(bf.gaussian_window(10))
    for a, b in points:
        assert nt.classify_gaussian(a, b) == bf.gaussian_class(a, b), (a, b)


def test_eisenstein_classification_matches_brute_force_irreducibility():
    points = {(a, b) for a, b, _, _ in nt.eisenstein_points(10)}
    assert points == set(bf.eisenstein_window(10))
    for a, b in points:
        assert nt.classify_eisenstein(a, b) == bf.eisenstein_class(a, b), (a, b)


def test_sieve_survivors_match_brute_force():
    for N in range(4, 1001, 2):
        assert nt.sieve_survivors(N, ZS) == [bf.sieve_survivors(N, z) for z in ZS], N


# ---- visualization data against brute force

def test_prime_wheel_counts_match_brute_force():
    d = vd.prime_wheel_data(30)
    ps = bf.primes_between(6, 1000)
    counts = Counter(p % 30 for p in ps)
    assert d.primes == ps
    assert d.counts == [counts[r] for r in bf.coprime_residues(30)]


def test_goldbach_wheel_grouping_matches_brute_force():
    d = vd.goldbach_wheel_data(100, 30)
    main = [(p, q) for p, q in bf.goldbach_pairs(100) if p > 5]
    assert d.main == main
    for t, lst in d.by_type.items():
        assert lst == [(p, q) for p, q in main if tuple(sorted((p % 30, q % 30))) == t]
    assert sum(len(v) for v in d.by_type.values()) == len(main)


def test_goldbach_wheel_partition_matches_brute_force():
    # Includes N such as 10 = 5 + 5, where p equals the largest wheel prime.
    for N in range(4, 401, 2):
        d = vd.goldbach_wheel_data(N, 30)
        pairs = bf.goldbach_pairs(N)
        assert d.main == [(p, q) for p, q in pairs if p > 5], N
        assert d.exceptions == [(p, q) for p, q in pairs if p <= 5], N


def test_sum_of_squares_example_choice_matches_its_definition():
    # Left: first pair with p > 5 and exactly one of p, q ≡ 1 (mod 4), else the last pair.
    # Right: the nearest n ≡ 2 (mod 4), trying N - d before N + d, and its pair with
    # both primes ≡ 1 (mod 4) and the largest p.
    for N in range(6, 401, 2):
        d = vd.sum_of_squares_data(N, 30)
        pairs = bf.goldbach_pairs(N)
        mixed = [(p, q) for p, q in pairs if p > 5 and (p % 4 == 1) != (q % 4 == 1)]
        assert (d.left.p, d.left.q) == (mixed[0] if mixed else pairs[-1]), N
        right = None
        for n in sorted({n for n in range(max(4, N - 198), N + 199, 2) if n % 4 == 2},
                        key=lambda n: (abs(n - N), n > N)):
            both = [(p, q) for p, q in bf.goldbach_pairs(n) if p % 4 == 1 and q % 4 == 1]
            if both:
                right = (n, *both[-1])
                break
        assert (d.right.N, d.right.p, d.right.q) == right, N


def test_lattice_primes_points_match_brute_force():
    d = vd.lattice_prime_data()
    for lp, window, classify, xy in (
            (d.gaussian, bf.gaussian_window(10), bf.gaussian_class, lambda a, b: (a, b)),
            (d.eisenstein, bf.eisenstein_window(10), bf.eisenstein_class,
             lambda a, b: (round(a - b / 2, 6), round(b * np.sqrt(3) / 2, 6)))):
        drawn = {k: {(round(float(x), 6), round(float(y), 6)) for x, y in getattr(lp, k)}
                 for k in ("split", "inert", "ram")}
        for k in drawn:
            expected = {tuple(round(float(v), 6) for v in xy(a, b))
                        for a, b in window if classify(a, b) == k}
            assert drawn[k] == expected, k


def test_residue_space_matches_brute_force():
    d = vd.residue_space_data((3, 5, 7))
    ps = bf.primes_between(8, nt.PRIME_LIMIT + 1)
    per = Counter((p % 3, p % 5, p % 7) for p in ps)
    assert d.primes == ps
    assert {(int(x), int(y), int(z)): int(c) for x, y, z, c in d.grid} == dict(per)
    heat = np.zeros((7, 5), dtype=int)
    for p in ps:
        heat[p % 7, p % 5] += 1
    assert np.array_equal(d.heat, heat)


def test_involution_matches_brute_force():
    N = 100
    d = vd.involution_data(N, (3, 5, 7), 2000)
    cen = {m: bf.fixed_class(N, m) for m in (3, 5, 7)}
    assert d.centre == cen
    shown = [(p, q) for p, q in bf.goldbach_pairs(N) if p > 7]
    assert [(p, q) for p, q, _, _ in d.shown] == shown
    for p, q, pp, qq in d.shown:
        assert pp == (bf.centred_rep(p, 7, cen[7]), bf.centred_rep(p, 5, cen[5]))
        assert qq == (bf.centred_rep(q, 7, cen[7]), bf.centred_rep(q, 5, cen[5]))
    blocked = {(x, y) for x, y, b in d.cells if b}
    expected = {(x, y) for x in range(cen[7] - 3, cen[7] + 4) for y in range(cen[5] - 2, cen[5] + 3)
                if x % 7 == 0 or (N - x) % 7 == 0 or y % 5 == 0 or (N - y) % 5 == 0}
    assert blocked == expected


def test_involution_comet_matches_brute_force():
    d = vd.involution_data(100, (3, 5, 7), 2000)
    for n, count, adm in zip(d.comet_N, d.comet_pairs, d.comet_admissible):
        n = int(n)
        assert count == len(bf.goldbach_pairs(n)), n
        assert adm == bf.admissible_count(n, (3, 5, 7)), n


def test_sum_of_squares_matches_brute_force():
    d = vd.sum_of_squares_data(100, 30)
    for ex in (d.left, d.right):
        assert bf.is_prime(ex.p) and bf.is_prime(ex.q) and ex.p + ex.q == ex.N
        assert set(ex.p_points) == bf.lattice_points(ex.p)
        assert set(ex.q_points) == bf.lattice_points(ex.q)
        assert ex.p_squares == bf.two_squares(ex.p) and ex.q_squares == bf.two_squares(ex.q)


def test_sieve_matches_brute_force():
    d = vd.sieve_data(1000, ZS, (3, 5, 7))
    for j, n in enumerate(d.Ns):
        n = int(n)
        assert list(d.counts[:-1, j]) == [bf.sieve_survivors(n, z) for z in ZS], n
        assert d.counts[-1, j] == len(bf.goldbach_pairs(n)), n
    assert d.lost == [(p, q) for p, q in bf.goldbach_pairs(1000) if p <= 23]


# ---- parity figure: brute-force helpers against known values

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


# ---- parity figure against brute force, at the extras/generate_parity_visualization.py settings

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
