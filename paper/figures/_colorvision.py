"""Colour-vision arithmetic: perceptual distance, with and without CVD.

Used by the palette definition in ``_common`` and enforced by
``paper/tests/test_palette.py``. Kept separate from the palette itself so the
numbers that justify a colour choice are reproducible rather than asserted in a
comment: any claim in this repo that palette A separates better than palette B
should come from ``min_separation`` here, not from someone's eye.

Two standard pieces, both implemented from the published coefficients:

* sRGB -> CIE L*a*b* (D65), and CIE76 Delta-E between pairs. Delta-E 2000 is more
  faithful for near-neighbours but far more code; CIE76 is monotone enough to RANK
  candidate palettes, which is all it is used for.
* Machado, Oliveira & Fernandes (2009) severity-1.0 matrices for protanopia,
  deuteranopia and tritanopia -- the same model matplotlib's own accessibility
  tooling and most CVD simulators use.
"""

from __future__ import annotations

import itertools

import numpy as np

# Machado et al. (2009), Table 1, severity 1.0. Applied to LINEAR sRGB.
CVD_MATRICES = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}

_XYZ_FROM_LINEAR_RGB = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65 = np.array([0.95047, 1.00000, 1.08883])


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _lab(rgb: np.ndarray) -> np.ndarray:
    xyz = _XYZ_FROM_LINEAR_RGB @ _to_linear(rgb) / _D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def simulate(hex_color: str, kind: str) -> np.ndarray:
    """Return the sRGB (0-1) a viewer with `kind` perceives for `hex_color`."""
    if kind == "normal":
        return hex_to_rgb(hex_color)
    lin = _to_linear(hex_to_rgb(hex_color))
    return np.clip(CVD_MATRICES[kind] @ lin, 0, 1) ** (1 / 2.4) * 1.055 - 0.055


def delta_e(a: str, b: str, kind: str = "normal") -> float:
    """CIE76 Delta-E between two hex colours, optionally under simulated CVD."""
    return float(np.linalg.norm(_lab(simulate(a, kind)) - _lab(simulate(b, kind))))


def min_separation(colors: list[str]) -> dict[str, float]:
    """Worst-case Delta-E inside a categorical palette, per vision type.

    The minimum is the number that matters: a palette is only as readable as its
    closest pair, and averages hide exactly the collision that breaks a figure.
    """
    kinds = ["normal", *CVD_MATRICES]
    return {
        k: min(delta_e(a, b, k) for a, b in itertools.combinations(colors, 2))
        for k in kinds
    }


def closest_pair(colors: list[str], kind: str) -> tuple[str, str, float]:
    pairs = ((a, b, delta_e(a, b, kind)) for a, b in itertools.combinations(colors, 2))
    return min(pairs, key=lambda t: t[2])
