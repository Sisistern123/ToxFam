# Confident-error curation — combined_run (2026-07-19 round, completed 2026-08-06)

Blind expert curation of the **238** confident errors (calibrated confidence >= 0.8) of
the `combined_run` checkpoint stamped `2026-07-17T17:54:20`, pinned to split manifest
`959e4d5b`. This is the **live** curation: the three files here are copied verbatim to
`paper/data/curation/` where `paper._paths.curated_verdicts_tsv()` reads them.

Supersedes the 255-verdict 2026-07-10 round, preserved in
`../archive/2026-07-10_combined_run_e236807/`.

## How the 238 came about

The set is a property of the **checkpoint**, not a running total — retraining re-derives
it. Against the previous round:

| | count |
|---|---|
| carried over (in both rounds) | 182 |
| new in this round | 56 |
| dropped out of the 2026-07-10 round | 73 |

182 + 56 = 238. The drop is a *better* result: the Jul-17 checkpoint makes 73 fewer
confident mistakes. Of the 15 dropped test-split rows, 3 are now predicted correctly
(`P34966` 0.810->0.912, `P86286` 0.803->0.918, `A0A7S8RGC8` 0.934->0.515) and 12 are
still wrong but fell below the 0.8 threshold (e.g. `P0DSM5` 0.999->0.660). So the
improvement is mostly **calibration**, not accuracy — worth stating precisely rather
than as "fewer errors".

## What the curator was asked

Only **58** rows needed an answer: 56 genuinely new proteins, plus 2 carried-over rows
(`Q86MA1`, `P83416`) that prefill refused because the predicted family changed
(`Cationic peptide family` -> `Non-disulfide-bridged peptide (NDBP) superfamily`). Both
came back with an identical verdict, assessment and fp_category — prefill matches on
identifier + Swiss-Prot label + prediction, and both predictions were wrong the same
way, so re-asking gained nothing.

The remaining 180 transferred automatically from `../archive/.../prior_verdicts.csv`.

## Files

- `confident_errors_curated.tsv`    — the completed 238-row sheet. Assembled by
  `scripts/assemble_curated_round.py` from the 180 prefilled rows plus the 58 returned
  ones; `fp_category` comes from the returned file for the new rows and from
  `prior_verdicts.csv` for the transferred ones (the prefilled sheet has no such column).
- `confident_errors_key.tsv`        — un-blinding key (split, actual/predicted, confidence).
- `confident_errors_to_curate.tsv`  — the sheet as generated: 180 prefilled, 58 blank.
- `confident_errors_new_to_curate.tsv` — just the 58 blanks, as handed to the curator.
- `confident_errors_new_curated.tsv`   — those 58, returned answered.
- `curation_provenance.json`        — manifest hash, checkpoint stamp, commit, threshold.

## Result

n=238 · verdict tox 193 / nontox 45 · assessment correct 133 / partial 33 / incorrect 72
· by split train 78 / val 77 / test 83 · 146 annotation gaps (Swiss-Prot files it
non-toxin, curator confirmed toxin — the model was right and the database is incomplete).

## Known limitation

`verdict` and `fp_category` are properties of the *protein* and never need re-asking;
`assessment` judges one specific predicted family and so is checkpoint-dependent. The
sheet conflates the two, which is why a retrain costs curator time at all. 350 distinct
proteins have now been judged across all rounds (63 + 255 + 58, deduplicated), including
4 of this round's "new" asks that the retired n=63 file had already answered — 2 agreeing,
2 flipping (`Q58T40`, `Q58T51`, nontox -> tox). A cumulative identifier-keyed verdict
store plus a `true_family` column would make `assessment` derivable and end re-curation
for known proteins; it would also need to record round provenance, since those 2 flips
show a verdict can be revised.
