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
#   5. make protspace                              # -> protspace/ (UMAP bundles)
#   6. make figures                                # render everything
#
# Individual figures are `make fig-<name>`; `make figures` builds all of them.
# `make protspace` is deliberately NOT a prerequisite of `figures`: it is minutes of
# UMAP over 65k embeddings rather than a render, and its output is stable across
# figure iterations. Run it once (step 5); fig-supp-embedding-space fails with a
# clear message if you skip it.

PY := uv run python -m paper.figures

.PHONY: figures numbers verify fig-pipeline fig-capability fig-confidence-curation \
        fig-supp-accuracy fig-supp-perfamily fig-supp-nonmetazoan fig-supp-unreviewed \
        fig-supp-embedding-space fig-supplementary protspace coverage

## Ad-hoc test-coverage report (no CI gate; run occasionally).
coverage:
	uv run pytest --cov=toxfam --cov-report=term-missing

## Fail loudly if any split-derived artifact is stale relative to the manifest.
## Every figure derives from benchmark predictions, so this gates figure builds.
verify:
	uv run toxfam verify --dataset test_set

## Build every manuscript figure + the results-numbers manifest.
## `numbers` self-verifies (see numbers_manifest._gate_on_pipeline_verification);
## the explicit `verify` prerequisite gives an early, clear failure.
figures: verify numbers fig-pipeline fig-capability fig-confidence-curation \
         fig-supp-accuracy fig-supp-perfamily fig-supp-nonmetazoan \
         fig-supp-unreviewed fig-supp-embedding-space fig-supplementary

## Build the ProtSpace UMAP bundles (protspace/out_{all,toxin}).
## Skips if already built; `make protspace FORCE=--force` recomputes from scratch.
## Also emits the shareable .parquetbundle files for protspace.app/explore.
protspace:
	uv run python -m paper.protspace_bundles $(FORCE)

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

## ProtT5 embedding space: toxin/non-toxin globally + families in a toxin-only refit.
## Requires `make protspace` (see the header chain).
fig-supp-embedding-space:
	$(PY).figure_embedding_space

fig-supplementary:
	$(PY).supplementary
