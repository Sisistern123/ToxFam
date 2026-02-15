"""Evaluation script for unreviewed metazoan proteins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
from pymmseqs.commands import createdb, search
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.preprocessing import label_binarize

from toxfam._paths import get_project_root


# ---------- Protein Family Normalization ----------


def normalize_protein_families(
    df: pd.DataFrame, column: str = "Protein families"
) -> pd.DataFrame:
    df = df.copy()

    df[column] = df[column].str.split(";").str[0]
    df[column] = df[column].str.split(",").str[0]

    conotoxin_repl = {
        "I1 superfamily": "Conotoxin I1 superfamily",
        "O1 superfamily": "Conotoxin O1 superfamily",
        "O2 superfamily": "Conotoxin O2 superfamily",
        "E superfamily": "Conotoxin E superfamily",
        "F superfamily": "Conotoxin F superfamily",
    }
    df[column] = df[column].replace(conotoxin_repl)

    mapping = {
        r"Conotoxin.*": "Conotoxin family",
        r"Neurotoxin.*": "Neurotoxin family",
        r"Scoloptoxin.*|Scolopendra.*": "Scoloptoxin family",
        r"Caterpillar.*": "Caterpillar family",
        r"Teretoxin.*": "Teretoxin family",
        r"Limacoditoxin.*": "Limacoditoxin family",
        r"Scutigerotoxin.*": "Scutigerotoxin family",
        r"Cationic peptide.*": "Cationic peptide family",
        r"Formicidae venom.*": "Formicidae venom family",
        r"Bradykinin-potentiating peptide family|Natriuretic peptide family|Natriuretic": "Natriuretic, Bradykinin potentiating peptide family",
        r".*phospholipase.*|.*Phospholipase.*": "Phospholipase family",
    }

    for pattern, replacement in mapping.items():
        df[column] = df[column].str.replace(pattern, replacement, regex=True)

    known_families = set(mapping.values())
    family_counts = df[column].value_counts()

    def should_keep_family(family):
        if family in known_families:
            return True
        if family_counts.get(family, 0) >= 10:
            return True
        return False

    df[column] = df[column].apply(lambda x: x if should_keep_family(x) else "other")

    return df


# ---------- Data Loading ----------


def load_preprocessed_data(
    input_tsv: Path,
) -> pd.DataFrame:
    print("Loading preprocessed data...")

    if not input_tsv.exists():
        raise FileNotFoundError(f"TSV file not found: {input_tsv}")

    df = pd.read_csv(input_tsv, sep="\t")
    print(f"   Loaded {len(df)} sequences from {input_tsv}")

    column_mapping = {
        "Entry": "identifier",
        "entry": "identifier",
        "accession": "identifier",
        "Accession": "identifier",
        "Protein_families": "Protein families",
        "protein_families": "Protein families",
    }
    df = df.rename(columns=column_mapping)

    required_cols = ["identifier", "Sequence", "Protein families"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"   Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    initial_len = len(df)
    df = df.dropna(subset=["Protein families"]).copy()
    if len(df) < initial_len:
        print(
            f"   Dropped {initial_len - len(df)} entries without "
            f"protein family annotation"
        )

    print("Normalizing protein families...")
    df = normalize_protein_families(df)

    print(f"Loaded {len(df)} sequences")
    print(f"\nProtein family distribution (top 10):")
    print(df["Protein families"].value_counts().head(10))

    return df


def load_embeddings(
    identifiers: list, input_h5: Path
) -> Dict[str, np.ndarray]:
    print("\nLoading embeddings from H5 file...")

    if not input_h5.exists():
        raise FileNotFoundError(f"H5 file not found: {input_h5}")

    embeddings = {}

    with h5py.File(input_h5, "r") as f:
        if len(f.keys()) == 0:
            raise ValueError(f"H5 file is empty: {input_h5}")

        for identifier in identifiers:
            possible_keys = [
                identifier,
                identifier.split("|")[-1],
                f"protein_{identifier}",
            ]

            found = False
            for key in possible_keys:
                if key in f:
                    embeddings[identifier] = f[key][:]
                    found = True
                    break

            if not found:
                print(f"   Warning: Embedding not found for {identifier}")

    print(f"Loaded {len(embeddings)} embeddings")

    if len(embeddings) == 0:
        raise ValueError("No embeddings loaded. Check identifier format in H5 file.")

    return embeddings


# ---------- HBI Evaluation ----------


def _write_fasta(df: pd.DataFrame, filename: Path) -> None:
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['identifier']}\n{row['Sequence']}\n")


def calculate_metrics(df: pd.DataFrame, truth_col: str, pred_col: str) -> Dict:
    class_list = sorted(list(df[truth_col].unique()))
    cls2idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    y_true = df[truth_col].map(cls2idx).to_numpy()
    y_pred = df[pred_col].map(cls2idx).to_numpy()

    n_samples = len(y_true)
    n_classes = len(class_list)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(n_classes))

    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        y_pred_bin = np.hstack((1 - y_pred_bin, y_pred_bin))

    micro_mcc = matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel())

    std_error = np.sqrt((acc * (1 - acc)) / n_samples)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(n_classes),
        target_names=class_list,
        output_dict=True,
        zero_division=0,
    )

    return {
        "acc": acc,
        "mcc": mcc,
        "micro_mcc": micro_mcc,
        "std_error": std_error,
        "n_samples": n_samples,
        "report": report,
        "class_list": class_list,
        "y_true_encoded": y_true,
        "y_pred_encoded": y_pred,
    }


def run_hbi_evaluation(
    query_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_fasta: Path,
    results_dir: Path,
) -> Dict:
    print("\nRunning HBI Evaluation...")

    q_labels = set(query_df["Protein families"].unique())
    t_labels = set(train_df["Protein families"].unique())
    only_in_train = t_labels - q_labels

    if only_in_train:
        print(
            f"Found {len(only_in_train)} train labels not in query set. "
            f"Mapping to 'other'..."
        )
        train_df = train_df.copy()
        train_df["Protein families"] = train_df["Protein families"].replace(
            {lbl: "other" for lbl in only_in_train}
        )

    tmp_dir = results_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    query_fasta = tmp_dir / "query.fasta"
    _write_fasta(query_df, query_fasta)

    print("   Creating databases...")
    query_db = createdb(str(query_fasta), str(tmp_dir / "queryDB"))
    target_db = createdb(str(train_fasta), str(tmp_dir / "targetDB"))

    print("   Running sequence search...")
    search_res = search(
        query_db.to_path(),
        target_db.to_path(),
        str(tmp_dir / "resultDB"),
        str(tmp_dir / "search_tmp"),
        s=9,
        e="inf",
        min_seq_id=0.0,
        max_seqs=100_000,
    )

    df_search = search_res.to_pandas()

    if df_search.empty:
        print("No search hits found.")
        predictions = query_df.copy()
        predictions["hbi_prediction"] = "no hit"
        predictions["hbi_confidence"] = 0.0
        predictions["evalue"] = np.nan
    else:
        best_hits = df_search.loc[
            df_search.groupby("query")["evalue"].idxmin()
        ].copy()

        train_label_map = dict(
            zip(train_df["identifier"], train_df["Protein families"])
        )
        best_hits["hbi_prediction"] = best_hits["target"].map(train_label_map)
        best_hits["hbi_confidence"] = best_hits["fident"]

        predictions = query_df.merge(
            best_hits[["query", "hbi_prediction", "hbi_confidence", "evalue"]],
            left_on="identifier",
            right_on="query",
            how="left",
        )
        predictions.drop(columns="query", inplace=True, errors="ignore")

        predictions["hbi_prediction"] = predictions["hbi_prediction"].fillna("no hit")
        predictions["hbi_confidence"] = predictions["hbi_confidence"].fillna(0.0)

    valid_labels = set(query_df["Protein families"].unique())
    hbi_labels = set(predictions["hbi_prediction"].unique())
    labels_not_in_ground_truth = hbi_labels - valid_labels

    if labels_not_in_ground_truth:
        print(
            f"Found {len(labels_not_in_ground_truth)} HBI labels not in ground truth. "
            f"Mapping to 'other'..."
        )
        repl_map_hbi = {lbl: "other" for lbl in labels_not_in_ground_truth}
        predictions["hbi_prediction"] = predictions["hbi_prediction"].replace(
            repl_map_hbi
        )

    metrics = calculate_metrics(
        predictions, truth_col="Protein families", pred_col="hbi_prediction"
    )

    return {"predictions": predictions, "metrics": metrics}


# ---------- Main Pipeline ----------


def run_eval_unreviewed(
    input_tsv: Path,
    input_fasta: Path,
    input_h5: Path,
    train_data: Path | None = None,
    train_fasta: Path | None = None,
) -> None:
    """Run unreviewed metazoan evaluation pipeline."""
    root = get_project_root()

    if train_data is None:
        train_data = root / "benchmark" / "HBI" / "train_all_df.csv"
    if train_fasta is None:
        train_fasta = root / "benchmark" / "HBI" / "train_all_members.fasta"

    results_dir = root / "benchmark" / "new" / "evaluation" / "unreviewed_metazoan" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load preprocessed data
    df = load_preprocessed_data(input_tsv)

    # Step 2: Load embeddings
    embeddings = load_embeddings(df["identifier"].tolist(), input_h5)

    # Step 3: Load training data
    print("\nLoading training data...")
    if not train_data.exists():
        raise FileNotFoundError(f"Training data not found: {train_data}")

    train_df = pd.read_csv(train_data)
    train_df = normalize_protein_families(train_df)
    print(f"Training data: {len(train_df)} sequences")

    # Step 4: Run HBI evaluation
    hbi_results = run_hbi_evaluation(df, train_df, train_fasta, results_dir)

    # Step 5: Combine results
    combined = hbi_results["predictions"].copy()

    # Step 6: Save results
    combined.to_csv(results_dir / "all_predictions.csv", index=False)
    print(f"\nSaved predictions to: {results_dir / 'all_predictions.csv'}")

    with open(results_dir / "hbi_metrics.json", "w") as f:
        json.dump(
            {
                "numeric_metrics": {
                    "Accuracy": hbi_results["metrics"]["acc"],
                    "MCC": hbi_results["metrics"]["mcc"],
                    "Micro_MCC": hbi_results["metrics"]["micro_mcc"],
                    "Std_Error": hbi_results["metrics"]["std_error"],
                    "Sample_Size": hbi_results["metrics"]["n_samples"],
                },
                "classification_report": hbi_results["metrics"]["report"],
            },
            f,
            indent=4,
        )

    summary_data = [
        {
            "Method": "HBI (Sequence Similarity)",
            "Accuracy": hbi_results["metrics"]["acc"],
            "MCC": hbi_results["metrics"]["mcc"],
            "Micro_MCC": hbi_results["metrics"]["micro_mcc"],
            "Std_Error": hbi_results["metrics"]["std_error"],
            "Sample_Size": hbi_results["metrics"]["n_samples"],
        }
    ]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "metric_comparison.csv", index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    hbi_m = hbi_results["metrics"]
    print(f"\nHBI Performance:")
    print(f"   Accuracy:  {hbi_m['acc']:.4f} (+/-{hbi_m['std_error']:.4f})")
    print(f"   MCC:       {hbi_m['mcc']:.4f}")
    print(f"   Micro-MCC: {hbi_m['micro_mcc']:.4f}")
    print(f"   Samples:   {hbi_m['n_samples']}")

    print(f"\nEvaluation complete! Results saved to {results_dir}")
