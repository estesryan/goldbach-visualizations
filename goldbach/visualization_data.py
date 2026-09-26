"""Data behind the seven visualizations. No plotting, colours or layout.

Each *_data() function returns the values the drawing code needs, computed from
the run settings it is given. Arrays keep the dtypes the drawing code expects.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isqrt

import numpy as np

from . import number_theory as nt


# ---- prime wheel

@dataclass(frozen=True)
class PrimeWheelData:
    wheel: int
    wheel_primes: list      # prime factors of the wheel
    coprime: list           # the φ(wheel) coprime residue classes
    primes: list            # primes counted in the bar chart
    counts: list            # number of those primes in each coprime class, in class order


def prime_wheel_data(wheel, bar_limit=1000):
    """Primes above the wheel primes and below bar_limit, counted by class mod wheel."""
    wp = nt.wheel_primes(wheel)
    coprime = nt.coprime_residues(wheel)
    big = nt.primes_between(max(wp) + 1, bar_limit)
    counts = Counter(p % wheel for p in big)
    return PrimeWheelData(wheel, wp, coprime, big, [counts[r] for r in coprime])


# ---- Goldbach pairs on the wheel

@dataclass(frozen=True)
class GoldbachWheelData:
    N: int
    wheel: int
    coprime: list
    residue: int            # N mod wheel
    small_cut: int          # largest wheel prime; pairs with p <= small_cut are exceptions
    pairs: list             # all Goldbach pairs of N
    main: list              # pairs with p > small_cut
    exceptions: list        # pairs with p <= small_cut
    types: list             # possible residue types {r, N - r}
    by_type: dict           # residue type -> main pairs of that type, in order of first appearance


def goldbach_wheel_data(N, wheel):
    pairs = nt.goldbach_pairs(N)
    small_cut = max(nt.wheel_primes(wheel))
    main = [(p, q) for p, q in pairs if p > small_cut]
    exc = [(p, q) for p, q in pairs if p <= small_cut]
    by_type = defaultdict(list)
    for p, q in main:
        by_type[nt.residue_type(p, q, wheel)].append((p, q))
    return GoldbachWheelData(N, wheel, nt.coprime_residues(wheel), N % wheel, small_cut, pairs, main, exc,
                      nt.residue_types(N, wheel), dict(by_type))


# ---- Gaussian and Eisenstein primes

@dataclass(frozen=True)
class LatticePrimes:
    points: np.ndarray      # (x, y) of every lattice point in the window
    split: np.ndarray       # (x, y) of prime elements, by how the rational prime below factors
    inert: np.ndarray
    ram: np.ndarray


@dataclass(frozen=True)
class LatticePrimeData:
    radius: int
    gaussian: LatticePrimes
    eisenstein: LatticePrimes


def _lattice_primes(points, classify):
    groups = {"split": [], "inert": [], "ram": []}
    allp = []
    for a, b, x, y in points:
        allp.append((x, y))
        c = classify(a, b)
        if c:
            groups[c].append((x, y))
    return LatticePrimes(np.array(allp), np.array(groups["split"]), np.array(groups["inert"]),
                         np.array(groups["ram"]))


def lattice_prime_data(radius=10):
    return LatticePrimeData(radius,
                      _lattice_primes(nt.gaussian_points(radius), nt.classify_gaussian),
                      _lattice_primes(nt.eisenstein_points(radius), nt.classify_eisenstein))


# ---- residue space mod the CRT moduli

@dataclass(frozen=True)
class ResidueSpaceData:
    moduli: tuple
    modulus: int            # product of the moduli
    admissible: int         # classes with no zero coordinate
    primes: list            # primes above max(moduli) up to PRIME_LIMIT
    grid: np.ndarray        # (r mod m1, r mod m2, r mod m3, count) for classes with no zero coordinate
    excluded: np.ndarray    # the same for classes with a zero coordinate (count is 0)
    heat: np.ndarray        # counts by (p mod m3, p mod m2), summed over mod m1


def residue_space_data(moduli):
    modulus = int(np.prod(moduli))
    ncop = int(np.prod([m - 1 for m in moduli]))
    cls = nt.primes_between(max(moduli) + 1, nt.PRIME_LIMIT + 1)
    cnt = Counter(p % modulus for p in cls)
    P, G = [], []
    for r in range(modulus):
        t = tuple(r % m for m in moduli)
        (P if all(t) else G).append((*t, cnt.get(r, 0)))
    P, G = np.array(P), np.array(G)
    Mg = np.zeros((moduli[2], moduli[1]), dtype=int)
    for p in cls:
        Mg[p % moduli[2], p % moduli[1]] += 1
    return ResidueSpaceData(tuple(moduli), modulus, ncop, cls, P, G, Mg)


# ---- involution p -> N - p, and the Goldbach comet

@dataclass(frozen=True)
class InvolutionData:
    N: int
    moduli: tuple
    centre: dict            # m -> fixed class c ≡ N/2 (mod m)
    first_mod_residues: list  # distinct p mod moduli[0] over the pairs shown (p > max(moduli))
    xs: range               # centred residues mod moduli[2] (x axis)
    ys: range               # centred residues mod moduli[1] (y axis)
    cells: list             # (x, y, blocked) in drawing order; blocked if p or N - p ≡ 0
    shown: list             # (p, q, (x, y) of p, (x, y) of q) for pairs with p > max(moduli)
    skipped: list           # pairs with p <= max(moduli); they have a zero coordinate
    reflects: bool          # every shown q is the reflection of its p through the fixed class
    comet_max: int          # largest N in the comet
    comet_N: np.ndarray     # even N from 4 to comet_max
    comet_pairs: np.ndarray # Goldbach pair count for each N
    comet_admissible: np.ndarray  # admissible classes mod prod(moduli) for each N
    modulus: int


def involution_data(N, moduli, comet_max):
    m3, m5, m7 = moduli
    pairs = nt.goldbach_pairs(N)
    centre = {m: nt.fixed_class(N, m) for m in moduli}
    ok3 = sorted({p % m3 for p, q in pairs if p > max(moduli)})
    xr = nt.centred_range(m7, centre[m7])
    yr = nt.centred_range(m5, centre[m5])
    cells = []
    for x in xr:
        for y in yr:
            cells.append((x, y, x % m7 in (0, N % m7) or y % m5 in (0, N % m5)))
    shown, refl_ok = [], True
    for p, q in pairs:
        if p <= max(moduli):
            continue
        pp = (nt.centred_rep(p % m7, m7, centre[m7]), nt.centred_rep(p % m5, m5, centre[m5]))
        qq = (nt.centred_rep(q % m7, m7, centre[m7]), nt.centred_rep(q % m5, m5, centre[m5]))
        refl_ok &= (pp[0] + qq[0] == 2 * centre[m7]) and (pp[1] + qq[1] == 2 * centre[m5])
        shown.append((p, q, pp, qq))
    skipped = [(p, q) for p, q in pairs if p <= max(moduli)]
    Ns = np.arange(4, comet_max + 1, 2)
    adm = np.array([nt.admissible_count(n, moduli) for n in Ns])
    gb = np.array([len(nt.goldbach_pairs(int(n))) for n in Ns])
    return InvolutionData(N, tuple(moduli), centre, ok3, xr, yr, cells, shown, skipped, refl_ok,
                      comet_max, Ns, gb, adm, int(np.prod(moduli)))


# ---- sum-of-two-squares circles

@dataclass(frozen=True)
class CircleExample:
    N: int
    p: int
    q: int
    radius: int             # lattice window |a|, |b| <= radius
    p_points: list          # lattice points on |z|² = p
    q_points: list          # lattice points on |z|² = q
    p_squares: tuple        # two_squares(p), or None
    q_squares: tuple        # two_squares(q), or None


@dataclass(frozen=True)
class SumOfSquaresData:
    left: CircleExample     # N, preferring a pair with one prime ≡ 1 and one ≡ 3 (mod 4)
    right: CircleExample    # N ≡ 2 (mod 4) closest to N with a pair both ≡ 1 (mod 4), or None


def _circle_example(N, p, q):
    Rr = isqrt(max(p, q)) + 2
    return CircleExample(N, p, q, Rr, nt.lattice_points_on_circle(p, Rr),
                         nt.lattice_points_on_circle(q, Rr), nt.two_squares(p), nt.two_squares(q))


def pick_examples(N, wheel):
    """(N, pair) for the left example and (n, pair) for the right one, or None."""
    pairs = nt.goldbach_pairs(N)
    small_cut = max(nt.wheel_primes(wheel))
    mixed = [(p, q) for p, q in pairs if p > small_cut and (p % 4 == 1) != (q % 4 == 1)]
    left = (N, mixed[0] if mixed else pairs[-1])
    for d in range(0, 200, 2):
        for n in (N - d, N + d):
            if n % 4 == 2:
                both = [(p, q) for p, q in nt.goldbach_pairs(n) if p % 4 == 1 and q % 4 == 1]
                if both:
                    return left, (n, both[-1])
    return left, None


def sum_of_squares_data(N, wheel):
    (lN, (lp, lq)), right = pick_examples(N, wheel)
    if right:
        rN, (rp, rq) = right
        right = _circle_example(rN, rp, rq)
    return SumOfSquaresData(_circle_example(lN, lp, lq), right)


# ---- surviving pairs after sieving

@dataclass(frozen=True)
class SieveData:
    heat_max: int
    zs: list                # sieve levels
    Ns: np.ndarray          # even N from 4 to heat_max
    counts: np.ndarray      # rows: survivors after each z; last row: actual Goldbach count
    pairs: list             # Goldbach pairs of heat_max
    lost: list              # those with p <= max(zs), which the sieve removes
    moduli: tuple           # the panel 5 moduli; N divisible by them shows as vertical streaks


def sieve_data(heat_max, zs, moduli):
    Ns = np.arange(4, heat_max + 1, 2)
    H = np.zeros((len(zs) + 1, len(Ns)))
    for j, n in enumerate(Ns):
        for i, c in enumerate(nt.sieve_survivors(n, zs)):
            H[i, j] = c
        H[-1, j] = len(nt.goldbach_pairs(int(n)))
    pairs = nt.goldbach_pairs(heat_max)
    lost = [(p, q) for p, q in pairs if p <= zs[-1]]
    return SieveData(heat_max, list(zs), Ns, H, pairs, lost, tuple(moduli))


# ---- all visualizations

@dataclass(frozen=True)
class VisualizationData:
    prime_wheel: PrimeWheelData
    goldbach_wheel: GoldbachWheelData
    lattice_primes: LatticePrimeData
    residue_space: ResidueSpaceData
    involution: InvolutionData
    sum_of_squares: SumOfSquaresData
    sieve: SieveData


def build_visualization_data(n_ex, wheel, crt, heat_max, sieve_zs, comet_max):
    """Data for all seven visualizations from the run settings. Built once per run."""
    return VisualizationData(
        prime_wheel_data(wheel),
        goldbach_wheel_data(n_ex, wheel),
        lattice_prime_data(),
        residue_space_data(crt),
        involution_data(n_ex, crt, comet_max),
        sum_of_squares_data(n_ex, wheel),
        sieve_data(heat_max, sieve_zs, crt),
    )
