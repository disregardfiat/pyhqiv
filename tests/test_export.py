"""Tests for export: VESTA/OVITO charge density, pyscf_hqiv_shift."""

import os
import tempfile
import numpy as np
import pytest

from pyhqiv.export import (
    export_charge_density_vesta,
    export_charge_density_ovito,
    pyscf_hqiv_shift,
)
from pyhqiv.crystal import hqiv_potential_shift
from pyhqiv.constants import GAMMA


def test_export_charge_density_vesta_shape():
    """VESTA export with grid and cell; optional phi_grid same shape."""
    grid = np.ones((4, 4, 4))
    cell = np.eye(3) * 10.0
    with tempfile.NamedTemporaryFile(suffix=".xsf", delete=False) as f:
        path = f.name
    try:
        export_charge_density_vesta(grid, cell, path)
        assert os.path.getsize(path) > 0
        with open(path) as f:
            text = f.read()
        assert "CRYSTAL" in text or "PRIMVEC" in text
    finally:
        os.unlink(path)


def test_export_charge_density_vesta_with_phi():
    """VESTA with phi_grid applies HQIV modulation."""
    grid = np.ones((3, 3, 3))
    phi_grid = np.ones((3, 3, 3)) * 0.1
    cell = np.eye(3) * 5.0
    with tempfile.NamedTemporaryFile(suffix=".xsf", delete=False) as f:
        path = f.name
    try:
        export_charge_density_vesta(grid, cell, path, phi_grid=phi_grid, gamma=0.4)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_export_charge_density_vesta_phi_wrong_shape_raises():
    """phi_grid must match grid shape."""
    grid = np.ones((2, 2, 2))
    phi_grid = np.ones((3, 3, 3))
    cell = np.eye(3)
    with tempfile.NamedTemporaryFile(suffix=".xsf", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError):
            export_charge_density_vesta(grid, cell, path, phi_grid=phi_grid)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_export_charge_density_ovito():
    """OVITO export writes a file."""
    grid = np.ones((2, 2, 2))
    cell = np.eye(3) * 5.0
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = f.name
    try:
        export_charge_density_ovito(grid, cell, path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_pyscf_hqiv_shift():
    """pyscf_hqiv_shift equals hqiv_potential_shift (same formula)."""
    phi = 1e-10
    dot = 1e-18
    v1 = pyscf_hqiv_shift(phi, dot, gamma=0.4)
    v2 = hqiv_potential_shift(phi, dot, gamma=0.4)
    assert abs(v1 - v2) < 1e-30
    assert np.isfinite(v1)
