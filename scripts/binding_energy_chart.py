#!/usr/bin/env python3
"""
Binding energy: HQIV first-principles vs experimental (test-case nuclides).

Run from repo root: uv run python scripts/binding_energy_chart.py
Saves docs/binding_energy_chart.png and prints a small table to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import numpy as np

from pyhqiv.nuclear import binding_energy_mev

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# Test-case nuclides: (P, N), label, experimental B (MeV) from AME/NNDC
TEST_NUCLIDES = [
    ((1, 1), "²H", 2.2246),
    ((1, 2), "³H", 8.4818),
    ((2, 1), "³He", 7.7180),
    ((2, 2), "⁴He", 28.2957),
    ((6, 6), "¹²C", 92.1618),
    ((6, 8), "¹⁴C", 105.284),
    ((7, 7), "¹⁴N", 104.659),
    ((8, 8), "¹⁶O", 127.6193),
    ((26, 30), "⁵⁶Fe", 492.257),
]


def main() -> None:
    labels = []
    b_exp = []
    b_hqiv = []
    for (P, N), label, be_exp in TEST_NUCLIDES:
        be_hq = binding_energy_mev(P, N)
        labels.append(label)
        b_exp.append(be_exp)
        b_hqiv.append(be_hq)

    labels = np.asarray(labels)
    b_exp = np.asarray(b_exp)
    b_hqiv = np.asarray(b_hqiv)

    # Table to stdout
    print("Nuclide   B_exp (MeV)   B_HQIV (MeV)   B_HQIV/B_exp")
    print("-" * 55)
    for i, lab in enumerate(labels):
        ratio = b_hqiv[i] / b_exp[i] if b_exp[i] != 0 else float("nan")
        print(f"  {lab:4s}   {b_exp[i]:10.3f}   {b_hqiv[i]:10.3f}   {ratio:.4f}")
    print()

    # Chart: grouped bars (total B) if matplotlib available
    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        w = 0.36
        ax.bar(x - w / 2, b_exp, w, label="Experiment", color="steelblue", alpha=0.9)
        ax.bar(x + w / 2, b_hqiv, w, label="HQIV", color="darkorange", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Binding energy (MeV)")
        ax.set_title("Nuclear binding energy: HQIV first-principles vs experiment (test cases)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = root / "docs" / "binding_energy_chart.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved {out}")
    else:
        print("Install matplotlib to generate docs/binding_energy_chart.png")


if __name__ == "__main__":
    main()
