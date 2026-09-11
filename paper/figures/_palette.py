"""Every colour the manuscript's figures use, in one place, one meaning each.

Rules this module exists to enforce (``paper/tests/test_palette.py`` checks them):

1. **A hex means exactly one thing.** No colour appears in two semantic namespaces.
   Before this file, ``#E69F00`` was ToxFam~(emb+tax) in Fig. 2 and "toxin" in Fig. 1,
   ``#BBBBBB`` was HBI in one figure and neutral furniture in another, and the venom
   metalloproteinase inset reused the toxin green and the ToxFam amber for two of its
   three classes. A reader who learns a colour in one figure should not be re-taught it
   in the next.
2. **Figures import names, never hex.** A literal ``"#0072B2"`` in a plotting script is
   a colour that cannot be audited, because nothing knows what it means.
3. **Colour-blind readability is measured, not asserted.** Every categorical group has a
   minimum pairwise CIE76 Delta-E under normal vision and under simulated protanopia,
   deuteranopia and tritanopia (``_colorvision.py``), and the test holds those above a
   floor. Comments quoting a separation number must come from that code.

Measured worst-case separations (min pairwise Delta-E over normal + 3 CVD types), from
``uv run python -m paper.figures._palette``:  METHOD 40.4, CLASS 19.8, VERDICT 23.5,
RANK 21.9, CALIBRATION 53.8, FAMILY 15.6, SUBSTRUCTURE 28.4. FAMILY is the floor because
nine categorical colours is past what colour alone separates; it is also the only group
whose members carry a legend entry each.

Known near-pairs across namespaces, all of which are fine because the two colours never
appear in the same figure, and all of which are measured rather than hoped:
  * ``CLASS["toxin"]`` green vs ``VERDICT["incorrect"]`` red -- the deuteranopia failure
    pair, but Fig. 1 has no verdicts and Fig. 3 has no classes.
  * ``SUBSTRUCTURE`` violet vs ``FAMILY`` indigo (Delta-E 9.1, protanopia). Both are in
    Fig. S3, but the substructure ramp lives in its own inset with its own legend, and it
    is read by LIGHTNESS (L* 88 / 60 / 32), which every CVD type preserves.
"""

from __future__ import annotations

# ── Methods: the series compared in every performance figure ───────────────────────
# Grey for the baseline is deliberate and long-standing: it pushes homology visually
# behind the learned models without ever claiming it is worse. Okabe-Ito blue/orange is
# the most CVD-robust contrast pair there is, and both survive greyscale printing.
METHOD = {
    "hbi": "#BBBBBB",
    "toxfam_emb": "#0072B2",
    "toxfam_embtax": "#E69F00",
}

# Darker companions, for text and marker edges where the canonical fill is too pale to
# read on white. Same role, same meaning -- a shade, not a new colour -- so they live
# beside their base rather than as literals in a plotting script.
METHOD_DARK = {"hbi": "#6F6F6F", "toxfam_emb": "#00517F", "toxfam_embtax": "#B06A00"}

# ── Data classes: what a sequence IS, in the dataset-side figures ──────────────────
# Toxin is the focus and takes the saturated hue; non-toxins are the 95% majority and
# recede into slate. "splits" holds BOTH classes so it must look like neither. "removed"
# is a filter step, not a judgement -- it is a different red from VERDICT["incorrect"]
# on purpose (they were 4.8 apart before, i.e. the same colour meaning two things).
# "splits" is a saturated violet rather than the muted plum it started as: the plum sat
# 6.4 from NEUTRAL["muted"], and both appear in Fig. 1, so the split block and the tool
# labels beside it were nearly the same colour. The test caught that, not a reviewer.
CLASS = {
    "toxin": "#009E73",
    "nontoxin": "#7F98AA",
    "splits": "#7A5195",
    "removed": "#CC3311",
}

CLASS_DARK = {"toxin": "#00654A", "nontoxin": "#4E677A"}

# ── Verdicts: the blind curation's good -> bad categories ──────────────────────────
# Blue = upheld, teal = partly upheld, red = overturned. The middle used to be Tol's
# sand #DDAA33, which sat 5.2 from ToxFam's amber -- close enough to read as the same
# colour one page apart. Teal is 75.4 away from it. Figures using these also vary marker
# SHAPE, so the encoding survives total colour loss.
VERDICT = {
    "correct": "#004488",
    "partial": "#66AAAA",
    "incorrect": "#BB5566",
}

# ── Rank bands: an ordered 4-step "how close was the right answer" scale ───────────
# Deliberately built from VERDICT's endpoints so that blue still means right and red
# still means wrong; the two middles are a lightness ramp between them.
RANK = ["#004488", "#3C7DBF", "#8FB8DC", "#BB5566"]

# ── Calibration: before vs after temperature scaling ───────────────────────────────
CALIBRATION = {"calibrated": "#882255", "uncalibrated": "#999933"}

# ── Family identity: nominal labels in the embedding-space projection ──────────────
# Paul Tol's "muted" qualitative scheme, which replaced Kelly's maximum-contrast nine on
# measurement, not taste. Kelly wins on normal vision (min Delta-E 35.0 vs 22.8) but
# collapses under protanopia (8.6, the #0067A5/#875692 pair) where Tol holds 15.6 -- and
# colour-blind readability is the constraint that matters. Tol also avoids every hex in
# the namespaces above, which Kelly did not (#F38400 vs ToxFam amber, #0067A5 vs ToxFam
# blue, #008856 vs toxin green were all near-collisions).
FAMILY = [
    "#CC6677",
    "#332288",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
]

# ── Ordered sub-structure: the P-I/P-II/P-III metalloproteinase inset ──────────────
# An ORDERED variable (domain count), so it gets a sequential lightness ramp rather than
# three categorical hues -- which is also the honest encoding, since the note beside the
# figure says the separation tracks domain count and therefore length.
SUBSTRUCTURE = ["#E4D4EF", "#A87FC7", "#5E3191"]

# ── Neutral furniture: never a data category ──────────────────────────────────────
# Axis glyphs, tick labels, backdrops, reference rules. Kept clear of METHOD["hbi"] so
# that grey-the-baseline and grey-the-furniture are not the same grey.
NEUTRAL = {
    "ink": "#1F1F1F",
    "muted": "#5F5F5F",
    "faint": "#9A9A9A",
    "backdrop": "#DCDCDC",
    "rule": "#E5E5E5",
    "fallback": "#767676",
    "card": "#F5F5F3",
    "card_edge": "#D9D9D6",
}

# Every group whose members are compared WITHIN one figure, so every group the
# separation floor applies to. NEUTRAL is excluded: furniture is not a category.
CATEGORICAL = {
    "METHOD": list(METHOD.values()),
    "CLASS": list(CLASS.values()),
    "VERDICT": list(VERDICT.values()),
    "RANK": RANK,
    "CALIBRATION": list(CALIBRATION.values()),
    "FAMILY": FAMILY,
    "SUBSTRUCTURE": SUBSTRUCTURE,
}

# Namespaces that must not share a hex. RANK is exempt as a deliberate quotation of
# VERDICT's endpoints, and NEUTRAL is furniture.
SEMANTIC = {
    "METHOD": METHOD,
    "CLASS": CLASS,
    "VERDICT": VERDICT,
    "CALIBRATION": CALIBRATION,
}

# Darker shades of an existing role. Exempt from the cross-namespace uniqueness rule --
# they are the same meaning, not a new one -- but still must not collide with a DIFFERENT
# role's base colour, which the test checks.
DARK = {"METHOD_DARK": METHOD_DARK, "CLASS_DARK": CLASS_DARK}


def _report() -> None:  # pragma: no cover - a developer convenience
    from paper.figures._colorvision import closest_pair, min_separation

    for name, colors in CATEGORICAL.items():
        sep = min_separation(colors)
        worst = min(sep, key=sep.get)
        a, b, d = closest_pair(colors, worst)
        print(f"{name:14s} min dE {min(sep.values()):5.1f}  ({worst}: {a} / {b})")


if __name__ == "__main__":  # pragma: no cover
    _report()
