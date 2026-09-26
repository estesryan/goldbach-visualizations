"""Independent brute-force implementations used by the tests.

Slow and simple on purpose. Nothing here imports the code under test, so a bug
in a fast implementation cannot also hide in the value it is checked against.
Several come from the runtime checks the script used to run before saving.
"""
from functools import cache
from math import gcd, isqrt, prod


@cache
def is_prime(n):
    """Trial division."""
    n = abs(n)
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def primes_between(lo, hi):
    """Primes p with lo <= p < hi."""
    return [n for n in range(lo, hi) if is_prime(n)]


def goldbach_pairs(N):
    """Pairs (p, N - p) with p <= N - p, both prime."""
    return [(p, N - p) for p in range(2, N // 2 + 1) if is_prime(p) and is_prime(N - p)]


def prime_factors(m):
    return [p for p in range(2, m + 1) if m % p == 0 and is_prime(p)]


def coprime_residues(m):
    """Residues mod m divisible by none of m's prime factors."""
    ps = prime_factors(m)
    return [r for r in range(m) if all(r % p for p in ps)]


def two_squares(n):
    """First (a, b) with a <= b and a² + b² = n, by search; None if there is none."""
    reps = [(a, b) for a in range(isqrt(n) + 1) for b in range(a, isqrt(n) + 1) if a * a + b * b == n]
    return reps[0] if reps else None


def lattice_points(n):
    """All integer (a, b) with a² + b² = n."""
    r = isqrt(n) + 1
    return {(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if a * a + b * b == n}


def gauss_irreducible(a, b):
    """a + bi is irreducible iff no w with 1 < N(w) < N(z) divides it."""
    n = a * a + b * b
    if n <= 1:
        return False
    for c in range(-isqrt(n), isqrt(n) + 1):
        for d in range(-isqrt(n), isqrt(n) + 1):
            m = c * c + d * d
            if 1 < m < n and n % m == 0:
                # w | z iff z·conj(w) ≡ 0 (mod N(w)) in both parts.
                if (a * c + b * d) % m == 0 and (b * c - a * d) % m == 0:
                    return False
    return True


def eis_irreducible(a, b):
    """a + bω is irreducible iff no w with 1 < N(w) < N(z) divides it."""
    n = a * a - a * b + b * b
    if n <= 1:
        return False
    lim = 2 * isqrt(n) + 2
    for c in range(-lim, lim + 1):
        for d in range(-lim, lim + 1):
            m = c * c - c * d + d * d
            if 1 < m < n and n % m == 0:
                # conj(c + dω) = (c - d) - dω, and ω² = -1 - ω.
                e, f = c - d, -d
                re, im = a * e - b * f, a * f + b * e - b * f
                if re % m == 0 and im % m == 0:
                    return False
    return True


def gaussian_class(a, b):
    """'split', 'inert' or 'ram' for a Gaussian prime, from its norm; None otherwise."""
    if not gauss_irreducible(a, b):
        return None
    n = a * a + b * b
    return "ram" if n == 2 else ("split" if is_prime(n) else "inert")


def eisenstein_class(a, b):
    """'split', 'inert' or 'ram' for an Eisenstein prime, from its norm; None otherwise."""
    if not eis_irreducible(a, b):
        return None
    n = a * a - a * b + b * b
    return "ram" if n == 3 else ("split" if is_prime(n) else "inert")


def gaussian_window(R):
    """(a, b) with |a|, |b| <= R."""
    return [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1)]


def eisenstein_window(R):
    """(a, b) whose point a + bω lies in the square |x|, |y| <= R.

    x = a - b/2 and y = b·√3/2, so the test is done in integers:
    |2a - b| <= 2R and 3b² <= 4R².
    """
    span = range(-2 * R - 2, 2 * R + 3)
    return [(a, b) for a in span for b in span if abs(2 * a - b) <= 2 * R and 3 * b * b <= 4 * R * R]


def fixed_class(N, m):
    """The c in 0..m-1 with 2c ≡ N (mod m), by search."""
    return next(c for c in range(m) if (2 * c - N) % m == 0)


def centred_rep(r, m, c):
    """The v ≡ r (mod m) in the window of m integers centred on c, by search."""
    return next(v for v in range(c - m // 2, c + m // 2 + 1) if (v - r) % m == 0)


def admissible_count(N, moduli):
    """Classes r mod prod(moduli) with r and N - r both coprime to it."""
    M = prod(moduli)
    return sum(1 for r in range(M) if gcd(r, M) == 1 and gcd(N - r, M) == 1)


def sieve_survivors(N, z):
    """a <= N/2 with neither a nor N - a divisible by any prime <= z."""
    pz = [d for d in range(2, z + 1) if is_prime(d)]
    return sum(1 for a in range(1, N // 2 + 1) if all(a % d and (N - a) % d for d in pz))
