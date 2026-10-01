"""Optional extra, beyond the poster: what the sieve can't see.

For even N and sieve level z, a pair (a, N - a) with 2 <= a <= N/2 survives if
neither a nor N - a has a prime factor <= z. The visualization plots, against z,

    M(z) = number of surviving pairs
    S(z) = sum over the survivors of λ(a)·λ(N - a),   λ(n) = (-1)^Ω(n),

and splits groups of pairs (all pairs, survivors at two sieve levels, Goldbach pairs)
by the sign of λ(a)·λ(N - a).

    data.py      ParityData and parity_data() (no plotting)
    layout.py    figure sizes and the three cards, landscape and 4:5
    render.py    the figure, its text and captions
    generate.py  settings and image generation:
                 python -m extras.parity.generate

The shared arithmetic (parity_sieve, liouville_table, goldbach_smaller_primes) is in
goldbach/number_theory.py; colours and the style_axes and card helpers are shared
with the poster. The poster does not use this package. The independent checks are the
tests: python -m pytest -m parity. Nothing here proves Goldbach's conjecture.
"""
