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
#   4. uv run toxfam predict {non_metazoan,unreviewed} \
#        --model-dir model/model_output/combined_run \
#        -o benchmark/<set>/predict/predictions.tsv  # -> the two supp generalisation figs
#   5. make figures                                # render everything
#
# Individual figures are `make fig-<name>`; `make figures` builds all of them.

PY := uv run python -m paper.figures

.PHONY: figures numbers fig-pipeline fig-capability fig-confidence-curation \
        fig-supp-accuracy fig-supp-perfamily fig-supp-nonmetazoan fig-supp-unreviewed \
        fig-supplementary coverage

## Ad-hoc test-coverage report (no CI gate; run occasionally).
coverage:
	uv run pytest --cov=toxfam --cov-report=term-missing

## Build every manuscript figure + the results-numbers manifest.
figures: numbers fig-pipeline fig-capability fig-confidence-curation \
         fig-supp-accuracy fig-supp-perfamily fig-supp-nonmetazoan \
         fig-supp-unreviewed fig-supplementary

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

## The two generalisation figures read `toxfam predict` output, not `toxfam eval`:
## neither set is a labelled benchmark, and predict builds the taxonomy vectors from
## each set's own organism IDs so the combined model's taxonomy branch is live.
## Produce their inputs first (see the chain in the header).
fig-supp-nonmetazoan:
	$(PY).figure_supp_nonmetazoan

fig-supp-unreviewed:
	$(PY).figure_supp_unreviewed

fig-supplementary:
	$(PY).supplementary
