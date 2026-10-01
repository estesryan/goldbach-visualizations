"""Data for the parity visualization. No plotting, colours or layout.

parity_data() sieves the pairs (a, N - a) of each N by the primes up to √N and records,
at every level, the survivor count M(z) and the parity-sensitive sum S(z) of
λ(a)·λ(N - a), plus the sign splits the third card draws. The arithmetic itself
(parity_sieve, liouville_table, goldbach_smaller_primes) is shared, in
goldbach/number_theory.py.
"""
from dataclasses import dataclass
from math import isqrt
from typing import Optional

import numpy as np

from goldbach import number_theory as nt


@dataclass(frozen=True)
class ParitySweep:
    N: int
    zs: np.ndarray          # sieve levels: the primes <= √N; the last is the largest such prime
    M: np.ndarray           # surviving pairs (a, N - a), 2 <= a <= N/2, at each level
    S: np.ndarray           # sum of λ(a)·λ(N - a) over those survivors
    average: np.ndarray     # S / M


@dataclass(frozen=True)
class ClassSplit:
    """One group of pairs (a, N - a) of the featured N, split by the sign of λ(a)·λ(N - a)."""
    kind: str               # "all" (2 <= a <= N/2), "survivors" (at level z) or "goldbach"
    z: Optional[int]        # the sieve level for "survivors", else None
    plus: int               # pairs with λ(a)·λ(N - a) = +1
    minus: int              # pairs with λ(a)·λ(N - a) = -1


@dataclass(frozen=True)
class ParityData:
    sweeps: dict            # N -> ParitySweep, in the order given
    featured: int           # N shown in the right panel: the last one given
    goldbach_total: int     # Goldbach pairs of the featured N
    goldbach_small: int     # of those, pairs with p <= √N
    meet_level: int         # featured N: first level from which S(z) = M(z) at every later level
    class_splits: tuple     # ClassSplit for all pairs, survivors at each split level, Goldbach pairs


def parity_data(Ns, split_levels=(7, 31)):
    """Data for the parity visualization. Separate from the poster's
    build_visualization_data: sieving near N = 10⁶ takes a few seconds.

    split_levels: sieve levels (primes <= √N) at which the featured N's survivors
    are split by sign. Each split comes from M and S: plus = (M + S)/2, minus = (M - S)/2."""
    if any(N % 2 for N in Ns):
        raise ValueError("N must be even")
    lam = nt.liouville_table(max(Ns))
    sweeps = {}
    for N in Ns:
        zs = np.array(nt.primes_between(2, isqrt(N) + 1))
        M, S = nt.parity_sieve(N, zs, lam)
        sweeps[N] = ParitySweep(N, zs, M, S, S / M)
    featured = Ns[-1]
    sw = sweeps[featured]
    ps = nt.goldbach_smaller_primes(featured)

    unequal = np.nonzero(sw.S != sw.M)[0]
    meet_level = int(sw.zs[unequal[-1] + 1 if len(unequal) else 0])

    a = np.arange(2, featured // 2 + 1)
    sign = lam[a] * lam[featured - a]
    splits = [ClassSplit("all", None, int((sign == 1).sum()), int((sign == -1).sum()))]
    for z in split_levels:
        i = np.nonzero(sw.zs == z)[0]
        if len(i) != 1:
            raise ValueError(f"split level {z} is not a sieve level of N = {featured}")
        M, S = int(sw.M[i[0]]), int(sw.S[i[0]])
        splits.append(ClassSplit("survivors", int(z), (M + S) // 2, (M - S) // 2))
    g = lam[ps] * lam[featured - ps]
    splits.append(ClassSplit("goldbach", None, int((g == 1).sum()), int((g == -1).sum())))
    return ParityData(sweeps, featured, len(ps), int((ps <= isqrt(featured)).sum()),
                      meet_level, tuple(splits))
