"""Optional extra, beyond the poster: where the first survivor lands.

For even N = 2C, let q be the largest prime below √N and sieve the symmetric pairs
(C - d, C + d), 0 <= d <= C - 2, by every prime below q. Three quantities are kept
separate:

    λ_sieve(N)   the first-survivor offset: the least d whose pair survives
    λ_prime(N)   the nearest Goldbach-pair offset: the least d with C - d and C + d
                 both prime (the two primes are 2d apart)
    W(N)         the forcing boundary q² - N/2

A composite n < q² has a prime factor below q, so a survivor with d < W is a pair of
primes, and is then also the nearest one: any closer pair of primes would survive
too. The experiment measures where the first survivor lands relative to W.

    data.py      computation, fitting and the exact blocks (no plotting)
    layout.py    figure sizes and panel geometry, landscape and portrait separately
    render.py    the figures, their colours, annotations and captions
    generate.py  settings and image generation:
                 python -m extras.first_survivor_experiment.generate [--blocks]

Shared mathematical primitives are in goldbach/number_theory.py. The poster and the
parity extra do not use this package. The independent checks are the tests:
python -m pytest -m first_survivor. Nothing here proves Goldbach's conjecture,
states an asymptotic exponent or resolves the parity problem.
"""
