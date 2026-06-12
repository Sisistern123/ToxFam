# ToxFam — Results Section Design

**Date:** 2026-06-12
**Status:** Draft for author review (Selin/Tobias/Ivan)
**Scope:** Three main display items + core messages for the (currently empty) `3_Results.tex`.
All numbers below were verified directly from `benchmark/test_set/*/predictions.csv` and `metrics.json`.

---

## 1. Strategic frame (locked: *capability-first*)

Reframe the paper from **"a learned method that matches MMseqs2"** to **"the first family-resolved,
metazoan-wide toxin atlas, validated against homology."** Three facts force this frame:

1. **94.73 % of the test set is the easy non-toxin class** (9 264 / 9 779). All-class accuracy
   (HBI 98.15 % → NN-combined 98.60 %) is near-ceiling for *both* methods and is not the real task.
   The honest headline is **toxin-only accuracy: 0.924 (NN-combined) vs 0.854 (HBI)** over 515 toxins.
2. **The gap is significant under the correct *paired* test** — McNemar χ²=8.80, p≈0.003 (discordant
   127 NN-right/HBI-wrong vs 83 the other way); paired-bootstrap Δacc = +0.0045, 95 % CI
   [+0.0016, +0.0074] (excludes zero). *Do not* argue from marginal-CI overlap — it falsely suggests parity.
3. **HBI's macro-F1 "lead" (0.851 vs 0.792) is a 58-sequence artifact.** It lives entirely in 20
   ultra-rare families (support ≤ 5). On adequately-supported families (support > 5),
   **NN wins macro-F1, 0.882 vs 0.846.**

**Honesty guardrail:** there is **no twilight-zone win**. HBI stays 0.985–0.994 across all
sequence-identity bins and NN never overtakes it *within* a bin. ToxFam's advantage is **coverage**
(no usable homolog) and **very short toxins (<30 aa)** — the Introduction must promise *coverage*,
not *low identity*. Drop the "~50 aa pLM threshold" framing: NN actually *wins* below 30 aa.

(The architecture schematic already lives in Methods as Figure `fig:arch`; the three items below are the
Results display items and must not re-spend a slot on an approach overview.)

---

## 2. Core messages (4)

| # | Message | Evidence (verified) | Guardrail |
|---|---------|---------------------|-----------|
| **M1 — Capability** | First metazoan-wide, multi-class toxin-**family** classifier: one taxonomy-fused ProtT5 MLP resolves 38 classes (36 ToxProt families + catch-all toxin + non-toxin) across Metazoa — a capability homology cannot provide where no homolog exists. | 38-class stratified test; HBI returns "no hit" on 74 proteins. Taxonomic breadth = dataset sunbursts (Porifera→Soricidae). | Carry on the 38-class structure + toxin-only performance, **never** on inflated all-class accuracy. |
| **M2 — Superiority on toxins** | On the scientifically relevant toxin population the taxonomy-fused NN is **significantly more accurate than homology** (0.924 vs 0.854). | Toxin-only n=515; McNemar p≈0.003; paired-bootstrap Δacc CI excludes 0. NN-combined > NN-standard on every aggregate (acc 0.984→0.986, MCC 0.862→0.874, macro-F1 0.770→0.792). | Use the **paired** test. Frame taxonomy fusion as a *small consistent gain* unless the per-family/per-taxon breakdown shows where it helps. |
| **M3 — Beyond homology = coverage** | The advantage is concentrated **where alignment breaks**: no-hit coverage and toxins <30 aa — *not* the twilight zone. | No-hit subset n=74, HBI 0 % (by construction) vs NN 94.6 %; toxins <30 aa n=62, HBI 0.565 vs NN 0.903; NN ≈flat across length. Identity-null reported in Supp. | The 74 no-hits are **64 non-toxin + 10 toxin** — report the toxin coverage separately; don't let easy negatives carry the panel. |
| **M4 — Calibrated readout + database curation** | A temperature-calibrated binary toxicity readout falls out of the family head; its confident "errors" are **enriched for candidate annotation gaps** in ToxProt — positioning ToxFam as a curation tool. | Expert adjudication of the 63 confident (≥0.8) errors (`analysis/model_test_wrong_conf_annotated.csv`, by I. Koludarov): **43/63 (68 %) model-vindicated** (33 correct + 10 partial) vs 20 genuine model false-positives; **38 are nontox-labelled but verdict = toxin** (candidate ToxProt additions), 39 venom-secreted yet only 8 carry the toxin keyword. Worked examples P00601, F8J2F6 also in notes.md. | Honest **mixed** result (not "all errors are label noise") — 20/63 are real FPs. Binary ROC-AUC/PR-AUC/ECE **still do not exist** and must be recomputed (Methods promises them). |

---

## 3. Three main display items (locked recommended compositions)

### Figure 1 — *Capability + validated superiority* (multi-panel figure)
- **Panel A — Capability map (new):** per-(super)family test support × NN-combined F1 across taxonomic
  breadth; the metazoan-wide family resolution that has *no* homology baseline. (Derivative of the sunbursts
  + `classification_report`.)
- **Panel B — Headline bars (new):** **toxin-only accuracy** (NN 0.924 vs HBI 0.854) as the hero number,
  with all-class accuracy / MCC / micro-MCC as a labelled reference set. Annotate McNemar p≈0.003 and the
  paired-bootstrap Δacc CI; caption states the 94.73 % non-toxin prior.
- **Panel C — macro & weighted P/R/F1** for HBI / NN-standard / NN-combined with bootstrap CIs (exposes the
  NN macro-precision deficit *and* the standard→combined fusion gain honestly).
- *Existing assets:* `test_set_headline_metrics.png`, `test_set_avg_precision_recall_f1.png`.
- *New work:* toxin-only bars + McNemar/bootstrap annotation; Panel A capability plot.

### Figure 2 — *Where homology breaks* (multi-panel figure)
- **Panel A — Toxin-only accuracy vs length (new overlay):** rolling-window accuracy, log-x, **HBI + NN
  overlaid** with CI ribbon + length histogram underlay; annotate the <30 aa collapse (HBI 0.565, NN 0.903)
  and that NN is ≈flat while HBI degrades only at the short end.
- **Panel B — No-hit coverage:** HBI 0 % vs NN 94.6 % on the 74 no-hit proteins — **split toxin (10) vs
  non-toxin (64)** so the toxin-coverage claim is clean.
- **Panel C — Out-of-distribution recognition (regenerate OOD):** non-metazoan binary toxin recognition
  (812 toxins from families outside the 38) — surfaces the calibrated readout in the main text. Requires
  `toxfam eval model non_metazoan --model-dir model/model_output/combined_run`.
- *New work:* regenerate `test_set_accuracy_vs_seq_length_rolling.png` with an HBI line + toxin-only filter;
  regenerate `benchmark/non_metazoan/` predictions and score the binary axis only.

### Figure 3 — *Per-family resolution + label quality* (1 load-bearing + 1 adjudicated panel)
- **Panel A — Per-family F1 difference (exists):** sorted NN−HBI per-family F1, marker size = support;
  annotate that negative bars are support ≤ 5 and that NN wins macro-F1 on supported families (0.882 vs
  0.846). Inset = support-stratified decomposition (≤5: HBI 0.856 vs NN 0.710 over 58 seqs; >5: NN 0.882 vs
  HBI 0.846). Existing asset: `test_set_class_f1_difference_top40.png`.
- **Panel B — Confident-error adjudication (data EXISTS — `analysis/model_test_wrong_conf_annotated.csv`):**
  expert review (I. Koludarov) of the **63** confident (≥0.8) errors, each annotated with `verdict`,
  `assessment` (correct / partial / incorrect) and `assessment_category`. Stacked bar:
  **33 correct + 10 partial (43/63 = 68 % model-vindicated) vs 20 genuine model FPs**; categories
  `family_correct` 31, false-positive variants 17 (nonspecific 8 / homolog 5 / analog 3 / spurious 1),
  `family_adjacent`/`family_related` 9, `false_negative` 3. **38** cases are nontox-labelled but verdict =
  toxin (candidate ToxProt additions; 39 venom-secreted, only 8 keyworded). Worked examples: P00601 (nontox →
  Phospholipase, 0.99), F8J2F6 (nontox → Venom Kunitz, 0.98); honest counter-examples P31398 (moth immune
  protein — a genuine false positive) and P54107 (non-toxic CRISP the model *correctly* calls nontox, 0.99 —
  taxonomy-fusion evidence). **Sources:** Ivan's CSV (quantitative) + the notes.md "Misclassified
  Venom/Toxin Proteins" curated list (qualitative, incl. PLA2 cluster). Reconcile the adjudicated 63 against
  the ~81 confident errors the analysis flagged (threshold / calibrated-vs-raw difference).

---

## 4. Supplementary items
- Performance-vs-identity table with the explicit **null** (HBI 0.985–0.994 across bins; NN never overtakes
  within a bin) — so the main text is transparent, not accused of hiding it.
- 38×38 row-normalised confusion matrices (HBI, NN-standard, NN-combined) — systematic family collisions.
- Full per-class P/R/F1 + per-family ROC for NN-combined and HBI (completeness for the capability claim).
- Reliability diagram with **ECE before/after temperature scaling** (must be computed) + confidence histograms.
- Binary toxic/non-toxic ROC + PR curves with AUROC/AUPRC (report against the **0.053** positive prior) and
  default-vs-Youden thresholds (must be recomputed from `predictions.csv`).
- OOD confidence histograms for unreviewed TrEMBL (no labels → confidence only) + full non-metazoan
  binary-recognition detail.
- Validation-set headline metrics for NN (no HBI val eval exists → internal check, not a comparison).
- Length-binned (bar) accuracy + per-family F1-vs-length scatter (toxin-only, per-bin n).
- Paired McNemar for the standard→combined taxonomy ablation + per-taxon/family map of where taxonomy helps.
- Training curves + dataset taxonomic sunbursts.

---

## 5. Analysis to-do (all four tracks in scope)
1. **Existing data:** toxin-only accuracy bars + paired-test annotations (Fig 1B); per-family F1 diff with
   support stratification (Fig 3A); macro/weighted P/R/F1 with CIs (Fig 1C).
2. **Recompute binary + ECE** *(closes a blocker):* AUROC, AUPRC (vs 0.053 prior), default + Youden
   thresholds, ECE before/after temperature scaling, from `predictions.csv` → Fig 2C / Supp.
3. **Regenerate OOD:** `toxfam eval model non_metazoan` (binary axis) and `unreviewed` (confidence only) →
   `benchmark/{non_metazoan,unreviewed}/` are currently absent.
4. **Expert adjudication (H2) — DONE:** `analysis/model_test_wrong_conf_annotated.csv` (63 confident errors,
   adjudicated by I. Koludarov). Remaining: render it as the Fig 3 Panel B stacked bar + worked examples and
   reconcile the 63 adjudicated vs ~81 flagged.
5. **No-hit convention:** lock the *fair* macro-F1 (HBI no-hit scored wrong, ≈0.849) as primary; report both
   (vs 0.872 restricted); state the convention in every caption and the abstract.

---

## 6. Open items the author still owns
- **Abstract hook (locked = capability claim):** "first metazoan-wide, multi-class toxin-family classifier";
  decide which macro-F1 value (0.849 fair vs 0.872 restricted) the abstract quotes.
- **Adjudication count:** reconcile the 63 expert-annotated confident errors
  (`analysis/model_test_wrong_conf_annotated.csv`) with the ~81 the analysis flagged (likely a
  confidence-threshold or calibrated-vs-raw difference); lock the denominator before quoting "68 % vindicated".
- **Taxonomy-fusion contribution:** headline as architectural (needs the per-taxon/family "where it helps"
  breakdown — PLA2 / Peptidase S1 / Insulin / CRISP confusable families) or downgrade to "small consistent
  improvement."
- **Length/rarity confound:** quick cross-tab confirming short toxins aren't merely the rare/no-hit ones
  (pre-empts a reviewer objection); report toxin-only with per-bin n + CIs throughout.
- **Intro↔Results audit:** ensure the Introduction promises a *coverage* advantage, not twilight-zone.
