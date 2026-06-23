# ToxFam Notes
## 17th June Meeting Notes
0. npj review?
1. explain whats been done, plots etc.
  - Ivans overview in new plots?
2. see if 3 key hypotheses are covered
  1. we have a classifier thats better than hbi (top 3 most confident results in jupyter notebook)
    - notebook link in paper
  2. classifier is still right even if its confident and wrong/the toxin naming conventions are bananas
  3. ~50 aas as the PLM struggle threshold
  despite plms being known to perform badly for short seqs, we still perform well
3. should we still focus on the other datasets, currently it's only the test/val set
discussion:
  - unreviewed
  - non-metazoa
  - proteome


#### hbi run
Running HBI evaluation on 'non_metazoan'
f   Loaded 702 sequences from non_metazoan.tsv
   Mapping 45 train-only labels to 'other'
a   Creating MMseqs2 databases (702 queries)...

-------------------- Running a mmseqs2 command --------------------
✓ Detailed execution log has been saved
✓ Database creation completed successfully
  Results saved to: /Users/selin/PycharmProjects/ToxFam/benchmark/non_metazoan/hbi/tmp/queryDB

-------------------- Running a mmseqs2 command --------------------
m✓ Detailed execution log has been saved
✓ Database creation completed successfully
  Results saved to: /Users/selin/PycharmProjects/ToxFam/benchmark/non_metazoan/hbi/tmp/targetDB

-------------------- Running a mmseqs2 command --------------------
✓ Detailed execution log has been saved
✓ Search completed successfully
  Results saved to: /Users/selin/PycharmProjects/ToxFam/benchmark/non_metazoan/hbi/tmp/resultDB
Output is not readable. Executing convertalis command to convert the alignment database to a readable format.

-------------------- Running a mmseqs2 command --------------------
✓ Detailed execution log has been saved
✓ ConvertAlis completed successfully
  Results saved to: /Users/selin/PycharmProjects/ToxFam/benchmark/non_metazoan/hbi/tmp/resultDB.tsv
   Extracted best hits for 699/702 queries
   Coverage: 99.6% (699/702)
/Users/selin/PycharmProjects/ToxFam/.venv/lib/python3.11/site-packages/sklearn/metrics/_classification.py:534: UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
  warnings.warn(
/Users/selin/PycharmProjects/ToxFam/.venv/lib/python3.11/site-packages/sklearn/metrics/_classification.py:534: UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
  warnings.warn(
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Method ┃ Accuracy ┃    MCC ┃ Micro-MCC ┃ Std Error ┃ Samples ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ HBI    │   1.0000 │ 0.0000 │    1.0000 │    0.0000 │     702 │
└────────┴──────────┴────────┴───────────┴───────────┴─────────┘
   Results saved to: /Users/selin/PycharmProjects/ToxFam/benchmark/non_metazoan/hbi

## previous notes:
### Test Set:
- most confident model errors are actually correct -- keyword annotation errors in SwissProt
  - TODO: find confidence threshold
- TODO: find out why less confident errors are happening
  - maybe shorter sequence --> less confidence? limitation of PLMs?
- TODO: specifically evaluate similar protein families but different toxicity (specifically PLA2, Peptidase S1, Insulin Family)
  - improvement in one sentence
  - compare correctness to overall score (significant difference in scores?)
- TODO: specifically evaluate "other" category
  - write down which protein families that were too small were bundled up together in "other" and why wrongful predictions make sense because of its diversity

### Non Metazoan:
- outlook: potential to expand to non-metazoan
- TODO: find out how to use this, cant find a proper use case of the predictions yet
- paragraph on how generalizable is toxicity in principle? (Ivan thinks not generalizable)
  - protein can be toxic or nontoxic in different contexts

### Unreviewed
- TODO: manually look through the predictions

### Proteome
- use predicted proteome without KW-0800 to make suggestions for KW-0800 candidates
  - snake/scorpio/newly added species

### Outlook
- outlook: expand this by making more fine-grained predictions (3FTX Ancestral, Long, Short or different Conotoxins)
  - limitations: incorrect annotations in SwissProt + limited data availability for specific subfamilies






------------










## TODO
- side by side bars of correct vs incorrect (2 bars each) bins for 0.0-0.1, 0.1-0.2, 0.2-0.3
  - before and after temp scaling

- +-2 std errors (+- next to the number)

- plot performance vs seq length by protein
- bin by length and avg performance for those

get the list of proteins for ivan: model_test.iloc[nn_wrong_conf[nn_wrong_conf >= 0.8].index] # everything above 0.8 confidence and wrongly predicted on calibrated confidences

- PLMs struggle with less than 50 aas (Michael)

key hypotheses:
1. we have a classifier thats better than hbi (top 3 most confident results in jupyter notebook)
2. classifier is still right even if its confident and wrong/the toxin naming conventions are bananas
3. ~50 aas as the PLM struggle threshold

tobi: jupyter notebook
ivan: look through confident and wrong
selin: write results

### notes HBI "no hit" handling in test-set metrics
HBI (homology-based inference via MMseqs2) does not always return a prediction: when no homolog passes the search threshold, the query gets the label "no hit" instead of one of the 38 canonical family classes.

Decision: a "no hit" is counted as a **wrong prediction in every metric**. All metrics are computed over the union of labels actually present, so the extra "no hit" label is simply an always-incorrect prediction:
- Accuracy, MCC, micro-MCC — "no hit" never equals a true family label, so it counts as wrong (same as metrics.json).
- Macro / weighted precision-recall-F1 — "no hit" is **included** as its own class. It has no true instances, so it gets F1 = 0 and drags the unweighted macro mean down (HBI macro-F1 0.849 with it included vs. 0.872 when excluded). Weighted averages are unaffected, because "no hit" has zero true support and therefore zero weight; the true family of each "no hit" query is still penalised on recall.

Note: this differs from metrics.json, which **restricts** the report-based (macro/weighted) metrics to the 38 canonical classes and therefore reports HBI macro-F1 = 0.872. The notebook deliberately counts "no hit" as wrong, so its HBI macro numbers are slightly lower. NN Combined never emits "no hit", so its numbers are identical either way.


----
preprocessing output:
source /Users/selin/PycharmProjects/ToxFam/.venv/bin/activate
selin@Selins-MacBook-Pro ToxFam % source /Users/selin/PycharmProjects/ToxFam/.venv/bin/activat
e
(toxfam) selin@Selins-MacBook-Pro ToxFam % toxfam
                                                                                              
 Usage: toxfam [OPTIONS] COMMAND [ARGS]...                                                    
                                                                                              
 Animal toxin protein family classification using MLP on ProtT5 embeddings.                   
                                                                                              
╭─ Options ──────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion            Install completion for the current shell.                  │
│ --show-completion               Show completion for the current shell, to copy it or       │
│                                 customize the installation.                                │
│ --help                -h        Show this message and exit.                                │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────╮
│ download-data  Download raw and processed data from GitHub Releases.                       │
│ preprocess     Run the full data preprocessing pipeline.                                   │
│ embed          Generate per-protein ProtT5 embeddings from a FASTA file.                   │
│ taxonomy       Generate multi-hot taxonomy vectors for the combined training strategy.     │
│ train          Train a toxin family classifier from a YAML config file.                    │
│ eval           Run evaluations and compare methods.                                        │
│ plot           Generate plots and visualizations.                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────╯
(toxfam) selin@Selins-MacBook-Pro ToxFam % toxfam download-data  
  0800.tsv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 16.5 MB/s
  nontox.tsv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.1/52.1 MB 13.2 MB/s
  training_data.csv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 30.4/30.4 MB 19.2 MB/s
  embeddings.h5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 315.8/315.8 MB 10.4 MB/s
  hbi_train_all.csv ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 33.1/33.1 MB 12.3 MB/s
  hbi_train_all.fasta ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 31.5/31.5 MB 12.4 MB/s
  sp6_cache.zip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.7/12.7 MB 9.2 MB/s
  evaluation_data.zip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.6/66.6 MB 14.0 MB/s
Done.
(toxfam) selin@Selins-MacBook-Pro ToxFam % toxfam preprocess 

1. Loading raw data
   5567 toxin sequences (45 families), 98850 non-toxin sequences

2. SignalP6 signal peptide removal
   All 104417 sequences cached

3. MMseqs2 clustering (46 families, min_seq_id=0.9)
⠸ Clustering Bradykinin-related_peptide_family ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  2/46x
⠧ Clustering FARP__FMRFamide_related_peptide__family ━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━ 11/46
   65179 representative sequences (3416 toxin, 61763 non-toxin)

4. Stratified train/val/test splits

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Split               ┃ Sequences ┃ Families ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ train (reps)        │     45621 │       38 │
│ val                 │      9779 │       38 │
│ test                │      9779 │       38 │
│ train (all members) │     72413 │       46 │
└─────────────────────┴───────────┴──────────┘

Done.
(toxfam) selin@Selins-MacBook-Pro ToxFam % toxfam taxonomy 
Resolving lineage for 2583 unique taxon IDs ...
Taxonomy cache is stale. Refreshing ...
toxfam embed Loading taxopy database from /Users/selin/.cache/taxopy_db
  Resolving taxonomy lineages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2583/2583
Built multi-hot taxonomy vectors for 65179 identifiers (vector length: 50)
  Writing taxonomy vectors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65179/65179

Multi-hot taxonomy pipeline complete!
  Total proteins: 65179
  Matched with taxonomy: 65179
  Unmatched (zero vector): 0
Output: data/processed/taxonomy_vectors.h5
(toxfam) selin@Selins-MacBook-Pro ToxFam % toxfam embed 

1. Device: mps

2. Reading data/intermediate/mmseqs/representatives/all.fasta
   65179 sequences (longest: 2243 residues)
   65179 already embedded, 0 remaining

All sequences already embedded. Nothing to do.
(toxfam) selin@Selins-MacBook-Pro ToxFam % 




Manuscript TODO:
- write methods now that everything has been established
- make inference easier for biologists (only input sequence (worse model) or also add Organism ID if known (better model))
  - web interface?
- Results:
  - find out how to plot previous assumptions the best
  - Non-Metazoa Confusion Matrix
  - Unreviewed Supplementary Bar Plot on which are actual toxins and which are not based on our model
    - based on confidence



Hypotheses:
1. Current curated protein databases (such as SwissProt and ToxProt) systematically under-annotate animal toxin proteins, and highly confident "false positives" generated by our predictive model are frequently true positives that expose these database gaps.
  - Evidence: most confident model errors on the test set are actually correct (e.g., keyword annotation errors in SwissProt). manual checks revealed that proteins like A5X2H7, A6MGY1, and the PLA2 cluster (P00601, P08872) are expressed in venom glands but are labeled as nontox and excluded from ToxProt.
  - Why it matters: This directly supports the idea that ToxFam can improve toxin protein annotations and positions your model not just as a classifier, but as a database-correction tool

2. Relying solely on Protein Language Model (PLM) embeddings is insufficient for distinguishing between toxic and non-toxic variants of structurally similar protein families (e.g., PLA2, Insulin, CRISP); integrating taxonomic embeddings (Organism ID) significantly improves classification accuracy
  - TODO: show difference
  - Evidence: observed edge cases where nontox proteins share families with toxins (CRISP, Insulin). You also spent significant effort building a model that concatenates/integrates taxonomic data with sequence embeddings.
  - Why it matters: This justifies your architectural choices and the dual-input interface you plan to offer biologists (sequence only vs. sequence + Organism ID). It shows that toxicity is an evolutionary trait, not just a structural one.

3. A model trained to navigate the complex functional overlaps of toxin families can successfully generalize to unreviewed datasets (TrEMBL), allowing for the high-throughput discovery and reclassification of putative toxins from predicted protein/translated transcriptomic data
  - TODO: manually evaluate more thoroughly
  - Evidence: You are benchmarking against TrEMBL (unreviewed data) and noted the intention to evaluate fresh transcriptomes. You specifically mention stating: "we can correctly assess x proteins that are in trembl and reassign their KW/pfam."
  - Why it matters: This provides the "outlook" and practical application of your tool. It proves ToxFam is useful for biologists discovering novel sequences in non-metazoan or unreviewed datasets, moving beyond just benchmark datasets into real-world utility.












- rerun everything

- do unreviewed benchmark (no HBI needed)
  - (keyword:KW-0800) AND (reviewed:false) AND (taxonomy_id:33208) AND (fragment:false)
  - we say we think we can improve the labels of these proteins






- took longer than expected since i dont have LRZ access anymore for some reason, had to compute embeds myself
- cant run inference with new taxa model, needs taxa vector as input
  - should we use it or keep the standard model? just 0.01 better MCC
  - HBI wrecks us

- HBI with normal seqs (no sp cutoff)
- get taxa with taxopy



TODO: rewrite evaluation, focus on manuscript
final model: COMBINED!!!!,standard als vergleich!
get a timeline until next week (3 main figures)
- specify tasks on what needs to be done (eval, preprocess, cleanup)
- protspace plot of data
- evaluation (bar plot mcc)
- FPs + confidences etc
- plot that shows the # of distinct protein families compared to the sequence variance
- put plots into github

signalp6 --fastafile ../../Desktop/unreviewed.fasta --output_dir unreviewed/ --organism eukarya --mode slow --model_dir /Users/selin/Desktop/signalp6_slow_sequential/signalp-6-package/models

evaluation:
- hbi baseline
- wichtig ist: einfaches inference + eval script schreiben
- preprocessing, embeds, inference, eval
- pipeline immer gleich behalten (remove SP, dann redred, dann inference, etc)
- try TMBed (statt sp6)
- inference:
  - kann interessant sein, muss nicht: pla2 (0.9 redred on test and val)
  - muss: unreviewed (mit 0.9 redundancy reduction), test on KW-0800 excl. metazoa


- update readme for models
- inference on PL snake data
- check FPs with highest confidences

- focal loss + 8 taxon embeds
- HBI comparison
- try to train on taxa first and then the MLP instead of joining

- Single Mutable Instance: Replaced inefficient dual-model weight copying with one ModularMLP that dynamically swaps input layers in-place to save memory.
- Modular Design: Decoupled the Projector (input mapping) from the Backbone (hidden logic) to enable surgical architectural changes.
- Stage 1 (Pre-training): The Backbone optimizes internal representations using clean 56-dim taxonomy data.
- The Swap: swap_input_layer hot-swaps the input dimension (56→1024) while preserving the trained Backbone's "intelligence."
- Stage 2 (Fine-tuning): A fresh input projector learns to map complex 1024-dim embeddings into the pre-structured Backbone space for faster convergence.

- concatenate ivans one hot encoding
- different input channels for tax and embeds
move on after 1 hour of getting nowhere
- concatenate taxon to embeds as taxonomy IDs
- if doesnt work, normalize between -2 and 2 (z-normalization)
- if doesnt work, use different input channels for taxonomy and embeds
  - concatenate taxon to embeds as one-hot encoding
- try poincarré embeds https://github.com/facebookresearch/poincare-embeddings



- use validation set instead of test set for comparison to see if nontox PLA2 ended up in validation set instead of training or test

- add taxonomy to per-protein embeds (concat to the end phylum, class, order, family, genus, species) -> 0 if not exist
- https://github.com/tsenoner/protspace/blob/main/src/protspace/data/features/retrievers/taxonomy_retriever.py
- maybe make an embedding script?

add 2nd layer or dropout or both (bisschen rumprobieren, 128 oder 64 nodes oder höher noch)
extract probability/confidence from softmax from cross entropy loss (logits) on validation data
based on that, 10% der highest confident FPs extracten und genauer angucken die distribution der confidence etc

benchmark on:
- 256_256_05_200ep model
- (keyword:KW-0800) AND (reviewed:false) AND (taxonomy_id:33208) AND (fragment:false)
- HBI on our test set (HBI on train data oder so)

WRITE UP

newest queries:
- Tox
  - (keyword:KW-0800) AND (reviewed:true) AND (taxonomy_id:33208) AND (fragment:false)
- Nontox
  - (reviewed:true) AND (taxonomy_id:33208) AND (fragment:false) AND ((existence:1) OR (existence:2)) NOT (keyword:KW-0800)

## To-Do
- cleanup
  - benchmark scripts
    - hbi
    - trembl
  - preprocessing scripts
  - inference script
  - data
- manuscript - for Oxford Bioinformatics/BMC
- double check roc curve
- look at 10% of FP, manually check them
- rerun trembl? based on manual assessment
  - if most nontox FP are actually correct in test set, we look at trembl? --> "we can correctly assess x proteins that are in trembl and reassign their KW/pfam"



- mmseqs with inf evalue - done
- predict data of KW-0800 (non venom tissue)
  - if less than 90% correct -> train on it as well
- plot evalues log scaled x and y axis and color by true + false predictions?
- sunburst plot of nontox non venom tissue (484) von test predictions
  - phylum und class von den proteinen
  - plotly für sunburst plot
  - taxopy
- look at those where we are wrong + hbi is right
  - make a csv with wrong preds of hbi + of ours
  - include identifier, actual label, hbi pred, our pred, function, tissue expression
- update plots on slides

- Misclassifications on test set
  - misclassified by ToxProt?
    - look at below
    - validate further (look at other overlapping nontox/tox clusters)
  - misclassified by ToxFam?
    - look at prediction confidence of misclassifications
    - look below
## Manuscript
- we show the predictor
- describe it
- how we create families, why we merge them etc
- show performance, compare hbi
- show performance on trembl
- show use cases (3 fresh transcriptomes as proteins, ants etc)
- show issue of families and annotation in discussion

## Misclassified Venom/Toxin Proteins
The following proteins are present in the **nontox** dataset but are **not listed in ToxProt**, despite being **confirmed venom/toxin proteins by us**.

![tiktok proud](https://em-content.zobj.net/content/2020/07/27/proud.png)

### PLA2 Cluster (Phospholipase A2)
Each entry below matches the following criteria:
- `taxonomy_id: 33208`
- `cc_tissue_specificity: venom`
- `reviewed: true`
- `fragment: false`

**Query:**  
```text
(taxonomy_id:33208) AND (cc_tissue_specificity:venom) AND (reviewed:true) AND (fragment:false) 
```

#### Identified Proteins:
- [**P00601**](https://www.uniprot.org/uniprotkb/P00601)
- [**P08872**](https://www.uniprot.org/uniprotkb/P08872)
- [**A0A1L4BJ46**](https://www.uniprot.org/uniprotkb/A0A1L4BJ46)
- [**P14615**](https://www.uniprot.org/uniprotkb/P14615)


### 🧬 Other Notable Cases
These proteins appear in the **nontox** dataset but are either **toxins** or represent **edge cases** due to their family association.

| Accession | Predicted Class | Dataset Label               | Notes                                                                |
|-----------|-----------------|-----------------------------|----------------------------------------------------------------------|
| [A5X2H7](https://www.uniprot.org/uniprotkb/A5X2H7) | Scoloptoxin     | **nontox** | is actually expressed in venom glands                                |
| [A6MGY1](https://www.uniprot.org/uniprotkb/A6MGY1) | Venom Kunitz    | **nontox** | is actually expressed in venom glands |
| [P54107](https://www.uniprot.org/uniprotkb/P54107) | CRISP           | **nontox** | actually a nontox but is also a CRISP  |
| [Q7KUD5](https://www.uniprot.org/uniprotkb/Q7KUD5) | Insulin         | **nontox** | actually a nontox but is also an Insulin  |
| [F8J2F6](https://www.uniprot.org/uniprotkb/F8J2F6) | Venom Kunitz    | **nontox** | is actually expressed in venom glands            |


