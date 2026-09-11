"""The palette's rules, enforced.

These are not style preferences. Each one failed at least once in this repo's history
and cost a figure round-trip, so they are checks rather than comments.
"""

from __future__ import annotations

import re
import tokenize
from pathlib import Path

import pytest

from paper.figures import _palette as P
from paper.figures._colorvision import closest_pair, delta_e, min_separation

# CIE76 Delta-E. Below roughly 12 two colours read as "the same colour, maybe a shade
# off" at figure mark sizes; the floor is set just under FAMILY's measured 15.6 so the
# nine-way identity palette passes while anything genuinely colliding does not.
MIN_DELTA_E = 14.0

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def test_no_hex_means_two_things():
    """A colour may not appear in two semantic namespaces."""
    seen: dict[str, str] = {}
    for ns, mapping in P.SEMANTIC.items():
        for role, hex_ in mapping.items():
            key = hex_.upper()
            if key in seen:
                pytest.fail(f"{hex_} is both {seen[key]} and {ns}.{role}")
            seen[key] = f"{ns}.{role}"


def test_furniture_is_not_a_data_colour():
    """Neutral greys must not read as a data series.

    Judged under NORMAL vision only, deliberately. A mid-grey is where every
    mid-lightness hue lands under deuteranopia -- that is what colour blindness IS --
    so requiring separation there would fail every palette ever drawn. What keeps
    furniture legible for a colour-blind reader is that it is thin rules and small
    labels next to filled, larger marks, not its hue.

    The one grey that is data is METHOD["hbi"], and that one has to clear the
    furniture under every vision type, because both are grey to everyone.
    """
    for fname, fhex in P.NEUTRAL.items():
        for ns, mapping in P.SEMANTIC.items():
            for role, hex_ in mapping.items():
                d = delta_e(fhex, hex_, "normal")
                assert d >= 10.0, (
                    f"NEUTRAL.{fname} {fhex} is only {d:.1f} from {ns}.{role} {hex_} "
                    "under normal vision"
                )

    hbi = P.METHOD["hbi"]
    for fname, fhex in P.NEUTRAL.items():
        d = min(min_separation([fhex, hbi]).values())
        assert d >= 10.0, (
            f"NEUTRAL.{fname} {fhex} is only {d:.1f} from the HBI grey {hbi}; "
            "grey-the-baseline and grey-the-furniture must not be the same grey"
        )


@pytest.mark.parametrize("group", sorted(P.CATEGORICAL))
def test_categorical_groups_survive_colour_blindness(group):
    """Worst-case separation inside a group, under normal vision and all three CVD types."""
    colors = P.CATEGORICAL[group]
    assert len(set(c.upper() for c in colors)) == len(colors), (
        f"{group} repeats a colour"
    )
    sep = min_separation(colors)
    worst_kind = min(sep, key=sep.get)
    a, b, d = closest_pair(colors, worst_kind)
    assert d >= MIN_DELTA_E, (
        f"{group}: {a} and {b} are {d:.1f} apart under {worst_kind} "
        f"(floor {MIN_DELTA_E})"
    )


def test_figure_scripts_hold_no_bare_hex():
    """Plotting scripts import palette names; a literal hex cannot be audited.

    _palette.py is where the literals live; _colorvision.py holds none. Covers
    supplementary.py too -- it plots, so it is bound by the same rule.

    Tokenised rather than regexed over raw lines: a colour literal starts with "#", so
    naive comment-stripping deletes exactly what this test is looking for. (It did, and
    the test passed against scripts full of bare hex until the tokeniser went in.)
    """
    offenders = []
    scripts = sorted(FIGURES.glob("figure_*.py")) + [FIGURES / "supplementary.py"]
    for path in scripts:
        with path.open("rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type != tokenize.STRING:
                    continue
                if re.fullmatch(r"[\"\']#[0-9A-Fa-f]{6}[\"\']", tok.string):
                    offenders.append(f"{path.name}:{tok.start[0]}: {tok.string}")
    assert not offenders, (
        f"{len(offenders)} bare hex literal(s) in figure scripts -- import a name from "
        "paper.figures._palette instead:\n  " + "\n  ".join(offenders)
    )
