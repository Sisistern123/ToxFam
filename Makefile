# ToxFam figure regeneration.
#
# Manuscript figures live in paper/figures/ and are rendered from the repository
# root. They read the trained models (model/model_output/) and evaluation results
# (benchmark/), which are gitignored, so a CLEAN CHECKOUT CANNOT rebuild figures
# without first producing those inputs. Full regeneration chain:
#
#   1. uv run toxfam train configs/standard.yaml   # -> model/model_output/standard_run
#   2. uv run toxfam train configs/combined.yaml   # -> model/model_output/combined_run
#   3. uv run toxfam eval {hbi,eat,model} test_set # -> benchmark/test_set/...
#   4. make figures                                # render everything
#
# Individual figures are `make fig-<name>`; `make figures` builds all of them.

PY := uv run python -m paper.figures

.PHONY: figures numbers fig-pipeline fig-capability fig-confidence-curation \
        fig-supp-accuracy fig-supp-perfamily fig-supplementary unreviewed-predictions

## Build every manuscript figure + the results-numbers manifest.
figures: numbers fig-pipeline fig-capability fig-confidence-curation \
         fig-supp-accuracy fig-supp-perfamily fig-supplementary

## Emit paper/figures/output/results_numbers.{json,tex} (every cited number).
numbers:
	$(PY).numbers_manifest

fig-pipeline:
	$(PY).figure_pipeline

fig-capability:
	$(PY).figure_capability

fig-confidence-curation:
	$(PY).figure_confidence_curation

fig-supp-accuracy:
	$(PY).figure_supp_accuracy

fig-supp-perfamily:
	$(PY).figure_supp_perfamily

fig-supplementary:
	$(PY).supplementary

## Manual data-prep step (NOT part of `figures`): run the combined model on the
## unreviewed TrEMBL set -> paper/figures/output/unreviewed_predictions.csv.
## Requires model/model_output/combined_run + the unreviewed evaluation H5.
unreviewed-predictions:
	$(PY).run_unreviewed_inference
