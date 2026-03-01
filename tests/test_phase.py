"""Tests for HQIVPhaseLift and δθ′, ˙δθ′, lapse."""

import numpy as np

from pyhqiv.constants import GAMMA, LAPSE_COMPRESSION_PAPER
from pyhqiv.phase import (
    HQIVPhaseLift,
    adm_lapse_compression_factor,
    apparent_age_from_wall_clock,
    delta_theta_prime,
    delta_theta_prime_dot_homogeneous,
)


def test_delta_theta_prime_zero():
    assert abs(delta_theta_prime(0.0)) < 1e-10


def test_delta_theta_prime_monotonic():
    x = np.linspace(0, 1, 11)
    d = delta_theta_prime(x)
    assert np.all(np.diff(d) >= 0)


def test_delta_theta_prime_bounded():
    assert 0 <= delta_theta_prime(1.0) <= np.pi / 2 + 0.01


def test_homogeneous_dot():
    H = 1e-18
    assert delta_theta_prime_dot_homogeneous(H) == H


def test_lapse_compression():
    # When phi_over_c2 and dtdc are small, factor ≈ 1
    f = adm_lapse_compression_factor(0.0, 0.0, gamma=GAMMA)
    assert abs(f - 1.0) < 1e-10


def test_apparent_age():
    t_app = apparent_age_from_wall_clock(51.2, lapse_compression=LAPSE_COMPRESSION_PAPER)
    assert abs(t_app - 51.2 / LAPSE_COMPRESSION_PAPER) < 0.1


def test_phase_lift_delta_theta_prime():
    pl = HQIVPhaseLift(gamma=0.40)
    assert abs(pl.delta_theta_prime(0.5) - np.arctan(0.5) * (np.pi / 2)) < 1e-10


def test_phase_lift_dot_homogeneous():
    pl = HQIVPhaseLift()
    assert pl.delta_theta_prime_dot(H_homogeneous=1e-18) == 1e-18
