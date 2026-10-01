"""Brute-force references for the first-survivor experiment.

Slow and simple on purpose, and built only on tests/brute_force.py: nothing here
imports the code under test. The sieve condition is checked from its definition,
gcd(C² − d², P_<q) = 1, with big-integer primorials, or from a pure-Python
least-prime-factor list.
"""
from functools import cache
from math import gcd, isqrt, prod

from brute_force import is_prime, primes_between


def sieve_depth(N):
    """q: the largest prime p with p² < N, by trial division."""
    p = isqrt(N)
    while p * p >= N or not is_prime(p):
        p -= 1
    return p


def sieve_depth_spf(N, spf):
    """The same, from a least-prime-factor list."""
    p = isqrt(N)
    while p * p >= N or spf[p] != p:
        p -= 1
    return p


@cache
def primorial_below(q):
    """Product of the primes below q (1 if there are none)."""
    return prod(primes_between(2, q))


def survives_gcd(C, d, q):
    """The definition itself: gcd(C² − d², P_<q) = 1."""
    return gcd(C * C - d * d, primorial_below(q)) == 1


def first_survivor_gcd(N):
    """λ_sieve(N) from the gcd definition, over 0 <= d <= C - 2."""
    C, q = N // 2, sieve_depth(N)
    return next((d for d in range(C - 1) if survives_gcd(C, d, q)), None)


def first_survivor_any_d(N):
    """λ_sieve(N) from the gcd definition over every integer d >= 0, with no range
    limit: admits d = C - 1, the pair (1, N - 1), and d >= C."""
    C, q = N // 2, sieve_depth(N)
    d = 0
    while not survives_gcd(C, d, q):
        d += 1
    return d


def central_gap_trial(N):
    """λ_prime(N): least d with C - d and C + d both prime, by trial division."""
    C = N // 2
    return next((d for d in range(C - 1) if is_prime(C - d) and is_prime(C + d)), None)


def both_lambdas_spf(N, spf):
    """(λ_sieve, λ_prime) from a least-prime-factor list, in one scan."""
    C, q = N // 2, sieve_depth_spf(N, spf)
    ls = lp = None
    for d in range(C - 1):
        lo, hi = C - d, C + d
        if ls is None and spf[lo] >= q and spf[hi] >= q:
            ls = d
        if lp is None and spf[lo] == lo and spf[hi] == hi:
            lp = d
        if ls is not None and lp is not None:
            break
    return ls, lp


def surviving_offsets_spf(N, spf):
    """Every d, 0 <= d <= C - 2, whose pair survives the primes below q, as (d, both_prime)."""
    C, q = N // 2, sieve_depth_spf(N, spf)
    out = []
    for d in range(C - 1):
        lo, hi = C - d, C + d
        if spf[lo] >= q and spf[hi] >= q:
            out.append((d, spf[lo] == lo and spf[hi] == hi))
    return out


def ols(xs, ys):
    """(slope, intercept, residual sum of squares) of the least-squares line, from the
    normal equations."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    return b, a, sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
