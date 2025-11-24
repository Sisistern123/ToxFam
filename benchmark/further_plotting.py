#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-converted from Jupyter notebook:
  - Imports hoisted
  - Executable cells wrapped in main()
  - Notebook magics stripped
  - Shell escapes (!cmd) mapped to subprocess.run
"""

import subprocess

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_shell(cmd: str) -> None:
    """
    Run a shell command safely.
    Example: run_shell("echo hello")
    """
    print(f"[shell] $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main() -> int:
    test_preds = pd.read_csv("../model/model_output/test_predictions.csv")
    test_preds

    nontox_tissue = pd.read_csv("nontox_tissue.tsv", sep="\t").rename(columns={"Entry": "identifier"})
    #nontox_tissue
    nontox_tissue[nontox_tissue['Tissue specificity'].str.contains('venom', case=False, na=False)]

    mask = (
        nontox_tissue['Tissue specificity']
          .str.contains('venom', case=False, na=False)
        & ~nontox_tissue['Tissue specificity']
          .str.contains('venom gland', case=False, na=False)
    )

    venom_only = nontox_tissue[mask]
    venom_only

    # create mask for “venom” (case-insensitive, NaNs→False)
    mask = nontox_tissue['Tissue specificity'].str.contains('venom', case=False, na=False)

    # label those rows “venom”
    nontox_tissue.loc[mask, 'Tissue specificity'] = 'venom tissue'

    # label all others “non venom”
    nontox_tissue.loc[~mask, 'Tissue specificity'] = 'no venom tissue'

    nontox_tissue

    # 1) Get all identifiers in test_preds whose actual_label is "nontox"
    nontox_ids = test_preds.loc[
        test_preds['actual_label'] == 'nontox',
        'identifier'
    ]

    # 2) Build a mask of which of those appear in nontox_tissue
    mask = nontox_ids.isin(nontox_tissue['identifier'])

    # 3) Are they all there?
    all_present = mask.all()
    print("All test_preds nontox IDs present in nontox_tissue?", all_present)

    # 4) If not, list the missing ones
    if not all_present:
        missing = nontox_ids[~mask].unique()
        print(f"{len(missing)} missing IDs:", missing)

    # 1) Merge the Tissue specificity from nontox_tissue into test_preds
    test_preds = test_preds.merge(
        nontox_tissue[['identifier','Tissue specificity']],
        on='identifier',
        how='left'
    )

    # 2) For any row whose actual_label isn’t “nontox”, overwrite with “venom”
    test_preds.loc[
        test_preds['actual_label'] != 'nontox',
        'Tissue specificity'
    ] = 'venom'
    test_preds

    # 1) Build an ordered Categorical for the row index
    row_cat = pd.Categorical(
        np.where(
            test_preds['prediction'] == 'nontox',
            'Predicted Nontoxin',
            'Predicted Toxin'
        ),
        categories=['Predicted Toxin', 'Predicted Nontoxin'],  # desired order
        ordered=True
    )

    # 2) Build the column MultiIndex as before
    col = [
        np.where(
            test_preds['actual_label'] == 'nontox',
            'Annotated Nontoxin (no KW)',
            'Annotated Toxin (KW-0800)'
        ),
        test_preds['Tissue specificity']
    ]

    # 3) Pass the categorical index into crosstab
    matrix = pd.crosstab(
        index   = row_cat,
        columns = col
    )

    # 4) Name the column levels (optional)a
    matrix.columns.names = ['Annotated Class', 'Tissue specificity']

    matrix

    false_positives_nonvenom = test_preds[
        (test_preds['actual_label'] == 'nontox') &
        (test_preds['prediction'] != 'nontox') &
        (test_preds['Tissue specificity'] == 'venom tissue')
    ]
    false_positives_nonvenom

    test_preds[test_preds["actual_label"].str.contains("Flavin")]

    test_metrics = json.load(open('../model/model_output/test_metrics.json', 'r'))

    # classification_report
    filtered = {
        cls: stats
        for cls, stats in test_metrics['classification_report'].items()
        if isinstance(stats, dict)
    }
    test_classes = (
        pd.DataFrame.from_dict(filtered, orient='index')
          .reset_index()
          .rename(columns={'index': 'class'})
    )

    # numeric_metrics/overall metrics
    test_overall = (
        pd.DataFrame.from_dict(
            test_metrics['numeric_metrics'],
            orient='index',
            columns=['value']
        )
        .reset_index()
        .rename(columns={'index': 'metric'})
    )

    test_classes = test_classes[~test_classes["class"].isin(["weighted avg", "macro avg"])]
    test_classes

    test_overall

    # 1) Per-class F1-score bar chart
    df = test_classes.set_index('class').sort_values('f1-score')
    plt.figure(figsize=(10,10), dpi=180)
    plt.barh(df.index, df['f1-score'])
    plt.xlabel('F1 Score')
    plt.title('Per-class F1 Scores')
    plt.tight_layout()
    plt.show()

    # 2) Per-class precision & recall grouped bars
    plt.figure(figsize=(10,10), dpi=180)
    x = range(len(df))
    width = 0.4
    plt.barh([i - width/2 for i in x], df['precision'], height=width, label='Precision')
    plt.barh([i + width/2 for i in x], df['recall'],    height=width, label='Recall')
    plt.yticks(x, df.index)
    plt.xlabel('Score')
    plt.legend(loc='lower right')
    plt.title('Per-class Precision & Recall')
    plt.tight_layout()
    plt.show()

    # 3) Overall metrics as a simple bar chart
    test_overall = test_overall[~test_overall["metric"].isin(["Test_Avg_Loss"])]

    ov = test_overall.set_index('metric')
    plt.figure(figsize=(8,6), dpi=180)
    plt.bar(ov.index, ov['value'])
    plt.ylim(0,1)
    plt.ylabel('Value')
    plt.title('Overall Test Metrics')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main()
