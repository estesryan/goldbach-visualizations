"""Number theory used by the figures. No plotting.

Primality is precomputed through PRIME_LIMIT when the module is imported.
is_prime() raises ValueError above that limit.
"""
from math import gcd, isqrt

import numpy as np

PRIME_LIMIT = 10_000


def _build_sieve(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return sieve


_SIEVE = _build_sieve(PRIME_LIMIT)
_PRIMES = np.nonzero(_SIEVE)[0]
_SIEVE.flags.writeable = False
_PRIMES.flags.writeable = False


# ---- primes

def is_prime(n):
    n = abs(int(n))
    if n > PRIME_LIMIT:
        raise ValueError(f"{n} exceeds PRIME_LIMIT")
    return bool(_SIEVE[n])


def primes_between(lo, hi):
    """Primes p with lo <= p < hi, as ints."""
    if hi - 1 > PRIME_LIMIT:
        raise ValueError(f"{hi - 1} exceeds PRIME_LIMIT")
    return [int(p) for p in _PRIMES if lo <= p < hi]


def goldbach_pairs(N):
    """Pairs (p, N - p) with p <= N - p and both prime."""
    return [(int(p), int(N - p)) for p in _PRIMES if p <= N // 2 and is_prime(N - p)]


def two_squares(n):
    """(a, b) with a <= b and a² + b² = n, or None."""
    for a in range(isqrt(n // 2) + 1):
        b = isqrt(n - a * a)
        if a * a + b * b == n:
            return a, b
    return None


def lattice_points_on_circle(n, R):
    """Integer (a, b) with |a|, |b| <= R and a² + b² = n, in row order."""
    return [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1) if a * a + b * b == n]


# ---- residues

def wheel_primes(m):
    """Prime factors of m."""
    return [p for p in range(2, m + 1) if m % p == 0 and is_prime(p)]


def coprime_residues(m):
    return [r for r in range(m) if gcd(r, m) == 1]


def residue_type(p, q, m):
    """Unordered pair of residues {p mod m, q mod m}."""
    return tuple(sorted((p % m, q % m)))


def residue_types(N, m):
    """Unordered residue types {r, N - r} with both classes coprime to m.

    A Goldbach pair with p, q coprime to m must fall in one of these.
    """
    cop = coprime_residues(m)
    s = N % m
    return sorted({tuple(sorted((r, (s - r) % m))) for r in cop if (s - r) % m in cop})


def fixed_class(N, m):
    """The class c ≡ N/2 (mod m) fixed by p -> N - p. Needs m odd."""
    return (N * pow(2, -1, m)) % m


def centred_range(m, c):
    """The m consecutive integers centred on c (m odd)."""
    return range(c - (m - 1) // 2, c + (m - 1) // 2 + 1)


def centred_rep(x, m, c):
    """Representative of x mod m in centred_range(m, c)."""
    lo = c - (m - 1) // 2
    return (x - lo) % m + lo


def admissible_count(N, moduli):
    """Classes r mod prod(moduli) with r and N - r both coprime to it.

    For prime moduli this is the product of (m - 1) if m | N else (m - 2).
    """
    return np.prod([(m - 1) if N % m == 0 else (m - 2) for m in moduli])


# ---- Gaussian and Eisenstein integers

def classify_gaussian(a, b):
    """How the rational prime below a + bi factors, if a + bi is prime.

    Returns 'split', 'inert', 'ram' or None.
    """
    # Off-axis with prime norm: split, or ramified if the norm is 2.
    # On an axis: inert iff |a + b| is a prime ≡ 3 (mod 4).
    n = a * a + b * b
    if a and b and is_prime(n):
        return "ram" if n == 2 else "split"
    if (a == 0) != (b == 0) and is_prime(a + b) and abs(a + b) % 4 == 3:
        return "inert"
    return None


def classify_eisenstein(a, b):
    """Same for z = a + b·ω, N(z) = a² − ab + b²."""
    n = a * a - a * b + b * b
    if is_prime(n):
        return "ram" if n == 3 else "split"
    # Inert: z = unit · r for a rational prime r ≡ 2 (mod 3), so N(z) = r².
    r = isqrt(n)
    if n and r * r == n and is_prime(r) and r % 3 == 2 and a % r == 0 and b % r == 0:
        return "inert"
    return None


def gaussian_points(R):
    """(a, b, x, y) for a + bi in the square |x|, |y| <= R."""
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            yield a, b, a, b


def eisenstein_points(R):
    """(a, b, x, y) for a + bω in the square |x|, |y| <= R, with ω = (-1 + i√3)/2."""
    for a in range(-2 * R - 2, 2 * R + 3):
        for b in range(-2 * R - 2, 2 * R + 3):
            x, y = a - b / 2, b * np.sqrt(3) / 2
            if abs(x) <= R and abs(y) <= R:
                yield a, b, x, y


# ---- sieving

def sieve_survivors(N, zs):
    """For each z in zs, the count of a <= N/2 with neither a nor N - a divisible by
    any of zs up to and including z. Survivors need not be prime."""
    a = np.arange(1, N // 2 + 1); b = N - a
    keep = np.ones_like(a, dtype=bool)
    counts = []
    for z in zs:
        keep &= (a % z != 0) & (b % z != 0)
        counts.append(keep.sum())
    return counts
