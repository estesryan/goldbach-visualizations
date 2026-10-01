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


# ---- Liouville's function and a parity-sensitive sieve sum

def liouville_table(limit):
    """λ(n) = (-1)^Ω(n) for 0 <= n <= limit, with λ(0) = 0 as a placeholder.

    Ω(n) counts prime factors with multiplicity: 1 is added for every prime
    power dividing n.
    """
    primes = np.nonzero(_build_sieve(limit))[0]
    omega = np.zeros(limit + 1, dtype=np.int16)
    for p in primes:
        pk = int(p)
        while pk <= limit:
            omega[pk::pk] += 1
            pk *= int(p)
    lam = np.where(omega % 2 == 0, 1, -1).astype(np.int64)
    lam[0] = 0
    return lam


def parity_sieve(N, zs, lam):
    """(M, S) for each sieve level z in zs, sieving the pairs (a, N - a), 2 <= a <= N/2.

    A pair survives level z if neither a nor N - a is divisible by any of zs up to
    and including z. M counts the survivors; S sums λ(a)·λ(N - a) over them.
    lam is a table of λ covering N. Unlike sieve_survivors, a starts at 2: 1 has
    no prime factors, so (1, N - 1) would survive every level without being a
    pair of primes.
    """
    a = np.arange(2, N // 2 + 1); b = N - a
    keep = np.ones_like(a, dtype=bool)
    Ms, Ss = [], []
    for z in zs:
        keep &= (a % z != 0) & (b % z != 0)
        Ms.append(int(keep.sum()))
        Ss.append(int((lam[a[keep]] * lam[b[keep]]).sum()))
    return np.array(Ms), np.array(Ss)


def goldbach_smaller_primes(N):
    """The primes p <= N/2 with N - p prime, as an array. Not limited by PRIME_LIMIT."""
    sieve = _build_sieve(N)
    p = np.nonzero(sieve[:N // 2 + 1])[0]
    return p[sieve[N - p]]


# ---- symmetric offsets around N/2: first sieve survivor, nearest Goldbach pair
#
# For even N = 2C, q is the largest prime below √N and the pair at offset d is
# (C - d, C + d). Offsets run over 0 <= d <= C - 2, so both numbers are at least 2;
# (1, N - 1) is left out, as in parity_sieve. A pair survives if neither number has
# a prime factor below q. A composite n < q² has a prime factor below q, so a
# survivor with C + d < q², i.e. d < W = q² - C, is a pair of primes. The functions
# below compute each quantity separately and do not assume that.

def spf_table(limit):
    """Least prime factor of n for 0 <= n <= limit, as an int32 array (0 for n = 0, 1)."""
    spf = np.zeros(limit + 1, dtype=np.int32)
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == 0:
            s = spf[i * i::i]
            s[s == 0] = i
    n = np.arange(limit + 1, dtype=np.int32)
    unset = (spf == 0) & (n >= 2)
    spf[unset] = n[unset]
    return spf


def primes_from_spf(spf):
    """The primes up to len(spf) - 1, as an int64 array."""
    n = np.arange(len(spf))
    return n[(n >= 2) & (spf == n)].astype(np.int64)


def largest_prime_below_sqrt(Ns, primes):
    """q(N), the largest prime below √N, for each even N >= 6.

    An even N >= 6 is not the square of a prime, so p < √N iff p <= isqrt(N).
    primes must reach isqrt(max(Ns)).
    """
    Ns = np.asarray(Ns, dtype=np.int64)
    if (Ns < 6).any() or (Ns % 2).any():
        raise ValueError("N must be even and at least 6")
    s = np.floor(np.sqrt(Ns.astype(np.float64))).astype(np.int64)
    s -= s * s > Ns                     # exact isqrt after float rounding
    s += (s + 1) * (s + 1) <= Ns
    return primes[np.searchsorted(primes, s, side="right") - 1]


def forcing_boundary(Ns, qs):
    """W(N) = q² - N/2. A survivor at offset d < W has C + d < q²."""
    Ns, qs = np.asarray(Ns, dtype=np.int64), np.asarray(qs, dtype=np.int64)
    return qs * qs - Ns // 2


def _first_offset(Cs, ok):
    """For each centre C, the least d with 0 <= d <= C - 2 and ok(C - d, C + d, idx), or -1.

    ok gets the lows, highs and indices (into Cs) of the centres still searching.
    """
    out = np.full(len(Cs), -1, dtype=np.int64)
    todo = np.arange(len(Cs))
    d = 0
    while len(todo):
        todo = todo[Cs[todo] - d >= 2]
        if not len(todo):
            break
        c = Cs[todo]
        hit = ok(c - d, c + d, todo)
        out[todo[hit]] = d
        todo = todo[~hit]
        d += 1
    return out


def first_survivor_offsets(Ns, qs, spf):
    """λ_sieve(N): the least d, 0 <= d <= C - 2, with neither C - d nor C + d divisible
    by a prime below q. Uses least prime factors only; no primality test. -1 if none."""
    Ns, qs = np.asarray(Ns, dtype=np.int64), np.asarray(qs, dtype=np.int64)
    return _first_offset(Ns // 2, lambda lo, hi, i: (spf[lo] >= qs[i]) & (spf[hi] >= qs[i]))


def nearest_goldbach_offsets(Ns, spf):
    """λ_prime(N), the nearest Goldbach-pair offset: the least d, 0 <= d <= C - 2, with
    C - d and C + d both prime (OEIS A047160 at C). The two primes are 2d apart.
    -1 if none."""
    Ns = np.asarray(Ns, dtype=np.int64)
    return _first_offset(Ns // 2, lambda lo, hi, i: (spf[lo] == lo) & (spf[hi] == hi))


def composite_survivors(Ns, qs, spf, primes):
    """(first, count): for each N, the least offset of a surviving pair that is not a
    pair of primes, and how many such offsets there are (-1 and 0 if none).

    Precondition: each q must be the sieve depth of its N, the largest prime below √N,
    as largest_prime_below_sqrt returns; primes must contain every prime p' with
    q_i·p' < N_i, i.e. reach at least max_i floor((N_i - 1) / q_i). The reasoning
    below depends on it and the function does not check it.

    A composite m < N with no prime factor below q is then q·p' with p' a prime >= q:
    two factors >= r, the next prime after q, would give m >= r² > N, and three
    factors >= q would give m >= q³ >= N. The pair is (m, N - m), at offset |C - m|,
    and survives if N - m >= 2 has no prime factor below q. Each surviving pair is
    counted once.
    """
    Ns, qs = np.asarray(Ns, dtype=np.int64), np.asarray(qs, dtype=np.int64)
    Cs = Ns // 2
    first = np.full(len(Ns), -1, dtype=np.int64)
    count = np.zeros(len(Ns), dtype=np.int64)
    j0 = np.searchsorted(primes, qs)
    j = 0
    while True:
        k = j0 + j
        live = k < len(primes)
        m = qs * primes[np.minimum(k, len(primes) - 1)]
        live &= m < Ns
        if not live.any():
            break
        partner = np.where(live, Ns - m, 2)
        live &= (partner >= 2) & (spf[partner] >= qs)
        # A pair with both numbers composite is met twice, once from each side; keep the
        # meeting where m is the larger number (m >= C).
        live &= (m >= Cs) | (spf[partner] == partner)
        d = np.abs(Cs - m)
        first = np.where(live & ((first < 0) | (d < first)), d, first)
        count += live
        j += 1
    return first, count


def rough_segment(a, b, q, primes):
    """Boolean array over a..b: True where n has no prime factor below q. Needs a > q."""
    if a <= q:
        raise ValueError("segment must start above q")
    ok = np.ones(b - a + 1, dtype=bool)
    for p in primes[primes < q].tolist():
        ok[(-a) % p::p] = False
    return ok


def block_first_survivor_offsets(q, r, primes, margin=4096):
    """(Ns, λ_sieve) for every even N with q² < N < r², q and r consecutive primes.

    Every N in the block has sieve depth q. One segmented sieve by the primes below
    q covers every C ± d needed; the margin doubles until every centre has a survivor.
    """
    lo = q * q + 1
    lo += lo % 2
    hi = r * r - 1
    hi -= hi % 2
    Ns = np.arange(lo, hi + 1, 2, dtype=np.int64)
    Cs = Ns // 2
    while True:
        a = int(Cs[0]) - margin
        ok = rough_segment(a, int(Cs[-1]) + margin, q, primes)
        lam = np.full(len(Cs), -1, dtype=np.int64)
        todo = np.arange(len(Cs))
        k = Cs - a
        for d in range(min(margin, int(Cs[0]) - 2) + 1):
            if not len(todo):
                break
            hit = ok[k[todo] - d] & ok[k[todo] + d]
            lam[todo[hit]] = d
            todo = todo[~hit]
        if not len(todo):
            return Ns, lam
        margin *= 2
