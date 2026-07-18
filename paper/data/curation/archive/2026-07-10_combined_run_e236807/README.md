# Archived confident-error curation — combined_run (2026-07-10, selin, commit e236807)

Blind expert curation of the **255** confident errors (>=0.8 calibrated confidence)
of the combined (emb+tax) model checkpoint trained on 2026-07-10 by a collaborator
(`/Users/selin/...`), pinned to split manifest `959e4d5b`. Splits: train 88 (1/3
subsample), val 82, test 85.

This is the curation behind the confidence-curation figure/paragraph of the
manuscript as of the 2026-07-18 data-v2 reconciliation. It is preserved here (not
deleted) so the analysis can be restored without re-adjudicating if we revert or
re-run against this checkpoint.

- `confident_errors_curated.tsv` — the filled blind sheet (expert verdicts).
- `confident_errors_key.tsv`     — un-blinding key (split, actual/predicted, confidence).
- `confident_errors_to_curate.tsv` — the blind sheet as handed to the curator.
- `curation_provenance.json`     — manifest hash, checkpoint stamp, commit.
- `prior_verdicts.csv`           — derived: identifier, verdict, assessment,
  assessment_note, fp_category, actual_label, predicted_label. Used as
  `--prefill-from` when re-curating against a new checkpoint so unchanged
  (identifier + Swiss-Prot label + prediction) verdicts transfer automatically.
