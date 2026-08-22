"""Interval arithmetic the evidence layer shares: slices now, derived gates next.

A rate on its own says nothing about how much evidence stands behind it; the
Wilson lower bound is what a floor is derived from, so it lives here once.
"""

from __future__ import annotations

import math

# Two-sided 95 % normal quantile.
Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n`` Bernoulli trials.

    Raises on ``n <= 0`` rather than returning ``(0.0, 0.0)``: a bound on no
    evidence is not a bound, and a caller that wants "no rate" must say so.
    """
    if n <= 0:
        raise ValueError(f"wilson_interval needs n > 0; got n={n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must lie in [0, n]; got {successes}/{n}")
    # At the boundaries the algebra gives exactly 0 or 1; float rounding does not.
    if successes == 0:
        return 0.0, _wilson_upper(0, n, z)
    if successes == n:
        return _wilson_lower_raw(n, n, z), 1.0
    return _wilson_lower_raw(successes, n, z), _wilson_upper(successes, n, z)


def _wilson_centre_half(successes: int, n: int, z: float) -> tuple[float, float]:
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return centre, half


def _wilson_lower_raw(successes: int, n: int, z: float) -> float:
    centre, half = _wilson_centre_half(successes, n, z)
    return max(0.0, centre - half)


def _wilson_upper(successes: int, n: int, z: float) -> float:
    centre, half = _wilson_centre_half(successes, n, z)
    return min(1.0, centre + half)


def wilson_lower(successes: int, n: int, *, z: float = Z95) -> float:
    return wilson_interval(successes, n, z=z)[0]
