from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

import pandas as pd
import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from toxfam.config import TrainConfig
from toxfam.data.dataset import ToxDataset, analyze_data_splits
from toxfam.data.split_manifest import apply_manifest, write_split_provenance
from toxfam.device import get_device
from toxfam.model.calibration import ModelWithTemperature
from toxfam.training.strategies import (
    DataSelector,
    evaluate_label_on_dataset,
    run_binary_strategy,
    run_combined_strategy,
    run_standard_strategy,
)
from toxfam.training.trainer import get_class_weights, set_seed
from toxfam.visualization.analysis import analyze_label_distribution_for_split

console = Console()

# Fixed seed order for multi-seed runs, so `--seeds 3` always means {42, 7, 13}
# and the selection is reproducible.
SEED_SEQUENCE = (42, 7, 13, 23, 100, 7777, 2024, 1, 88, 314)


def run_training(config: TrainConfig) -> float:
    """Train one model per the config. Returns its validation MCC (for seed selection)."""
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Set seed early for full reproducibility
    set_seed(config.seed)

    # Save resolved config to output dir
    config_copy = out_root / "config.yaml"
    config_copy.write_text(yaml.dump(config.model_dump(mode="json"), sort_keys=False))

    # wandb setup (optional — gracefully degrade if not logged in)
    _use_wandb = False
    if wandb is not None:
        try:
            wandb.login()
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=config.model_dump(mode="json"),
            )
            _use_wandb = True
        except Exception:
            console.print(
                "[yellow]wandb login failed — continuing without wandb[/yellow]"
            )

    device = get_device()
    console.print(f"Using device: [bold]{device}[/bold]")

    # Create organized subdirectories
    models_dir = out_root / "models"
    plots_dir = out_root / "plots"
    metrics_dir = out_root / "metrics"
    predictions_dir = out_root / "predictions"
    for d in (models_dir, plots_dir, metrics_dir, predictions_dir):
        d.mkdir(exist_ok=True)

    # Save run environment for reproducibility
    import platform
    import sys
    from datetime import datetime

    from toxfam.evaluation.runner import git_commit_short

    env_info = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit_short(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }
    (out_root / "run_environment.json").write_text(json.dumps(env_info, indent=2))

    # 1. Load Data
    console.print("Loading data...")
    strategy = config.training_strategy
    # Split comes from the git-tracked manifest, never the CSV's own Split column,
    # so training and evaluation cannot disagree about what "test" means.
    df = apply_manifest(pd.read_csv(config.input_csv))
    train_df, val_df, test_df = analyze_data_splits(df)

    label_col = "Protein families"
    is_binary = strategy == "binary"

    if is_binary:
        from toxfam.evaluation.metrics import to_binary_class

        for split_df in (train_df, val_df, test_df):
            split_df["binary_label"] = split_df[label_col].apply(to_binary_class)
        effective_label_col = "binary_label"
    else:
        effective_label_col = label_col

    analyze_label_distribution_for_split(
        train_df, val_df, test_df, effective_label_col, out_root
    )

    # 2. Init Datasets
    h5_paths = [str(p) for p in config.h5_paths]
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None

    train_ds = ToxDataset(
        train_df,
        h5_paths,
        is_train=True,
        label_col=effective_label_col,
        tax_h5_path=tax_h5,
    )

    # Validate taxonomy vector dimension matches config
    if tax_h5 is not None:
        import h5py as _h5py

        with _h5py.File(tax_h5, "r") as _f:
            _first = next(iter(_f))
            actual_dim = _f[_first][:].shape[0]
            if actual_dim != config.tax_dim:
                raise ValueError(
                    f"Taxonomy H5 has vectors of dim {actual_dim} but config.tax_dim={config.tax_dim}. "
                    f"Regenerate with `toxfam taxonomy` or update tax_dim in your config."
                )

    class_indices = {int(i): label for i, label in enumerate(train_ds.le.classes_)}
    class_json_path = out_root / "class_indices.json"
    with open(class_json_path, "w") as f:
        json.dump(class_indices, f, indent=4)
    console.print(f"Saved class mapping to {class_json_path}")

    # Save model config for deterministic architecture reconstruction at inference
    from toxfam.model.model_config import ModelConfig

    num_classes = len(class_indices)
    model_config = ModelConfig(
        architecture="MultiInputMLP"
        if config.training_strategy == "combined"
        else "ModularMLP",
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        num_classes=num_classes,
        dropout=config.dropout,
        tax_dim=config.tax_dim if config.training_strategy == "combined" else None,
    )
    model_config.save(out_root / "model_config.json")

    val_ds = ToxDataset(
        val_df,
        h5_paths,
        label_encoder=train_ds.le,
        is_train=False,
        label_col=effective_label_col,
        tax_h5_path=tax_h5,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    _, w_tensor, _ = get_class_weights(train_ds, power=config.class_weight_power)

    # 3. Dispatch Strategy
    final_model = None

    if strategy == "standard":
        final_model = run_standard_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "binary":
        final_model = run_binary_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "combined":
        final_model = run_combined_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Evaluation: Uncalibrated
    console.print("\n[bold]Running Final Evaluation (Uncalibrated)...[/bold]")
    val_metrics = evaluate_label_on_dataset(
        final_model,
        val_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "val",
        out_root,
        config,
    )
    test_metrics = evaluate_label_on_dataset(
        final_model,
        test_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "test",
        out_root,
        config,
    )

    # 5. Calibration (Temperature Scaling)
    console.print("\n[bold]Running Calibration (Temperature Scaling)...[/bold]")

    val_selector = DataSelector(
        val_loader, "both" if strategy == "combined" else "emb_only"
    )

    final_model = final_model.to(device)

    scaled_model = ModelWithTemperature(final_model, device)
    scaled_model.set_temperature(val_selector)
    calibrated_path = models_dir / "best_model_calibrated.pt"
    torch.save(scaled_model.state_dict(), calibrated_path)
    # Bind the checkpoint to the split it trained on, here rather than at run start:
    # a run that dies before calibration leaves the previous checkpoint in place, and
    # must not leave a stamp claiming that checkpoint matches the current split.
    digest = write_split_provenance(out_root)
    console.print(f"Saved calibrated model to {calibrated_path}")
    console.print(f"Pinned to split manifest [dim]{digest[:12]}[/]")

    # Log calibrated model as a wandb artifact
    if _use_wandb:
        calibrated_artifact = wandb.Artifact(
            name="toxfam-best-model-calibrated",
            type="model",
            metadata={"strategy": strategy},
        )
        calibrated_artifact.add_file(str(calibrated_path))
        wandb.log_artifact(calibrated_artifact)

    # 6. Evaluation: Calibrated
    console.print("\n[bold]Running Final Evaluation (Calibrated)...[/bold]")
    val_cal_metrics = evaluate_label_on_dataset(
        scaled_model,
        val_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "val_calibrated",
        out_root,
        config,
    )
    test_cal_metrics = evaluate_label_on_dataset(
        scaled_model,
        test_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "test_calibrated",
        out_root,
        config,
    )

    # wandb summary
    if _use_wandb:
        for tag, m in [
            ("val", val_metrics),
            ("test", test_metrics),
            ("val_calibrated", val_cal_metrics),
            ("test_calibrated", test_cal_metrics),
        ]:
            for k, v in m.items():
                wandb.run.summary[k] = v

    # 7. Binary Metrics Pipeline
    # For binary strategy, the model directly outputs binary classes,
    # so we pass the effective label col. For other strategies, we derive
    # binary from the original family labels.
    from toxfam.evaluation.binary import run_binary_evaluation

    run_binary_evaluation(
        scaled_model, train_ds.le, val_df, test_df, config, out_root, effective_label_col
    )

    train_ds.close()
    val_ds.close()

    if _use_wandb:
        wandb.finish()

    return float(val_metrics.get("val_mcc", float("nan")))


def run_multiseed_training(config: TrainConfig, n_seeds: int) -> None:
    """Train ``n_seeds`` models on different seeds; promote the best-on-val.

    Each seed trains into ``<output_dir>/seeds/seed_<s>/``. The seed with the
    highest validation MCC is copied up to ``<output_dir>/`` as the canonical run
    (so ``eval``/``predict`` find ``models/best_model_calibrated.pt`` +
    ``split_provenance.json`` unchanged). A ``seeds_summary.json`` records every
    seed's val/test MCC plus mean±sd — a robustness record, not the published
    error bar (those come from bootstrap in the figures pipeline).

    ``n_seeds == 1`` short-circuits to a plain single-seed run for back-compat.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    if n_seeds == 1:
        run_training(config)
        return
    if n_seeds > len(SEED_SEQUENCE):
        raise ValueError(
            f"n_seeds={n_seeds} exceeds the {len(SEED_SEQUENCE)} predefined seeds."
        )

    out_root = Path(config.output_dir)
    seeds = SEED_SEQUENCE[:n_seeds]
    console.print(f"[bold]Multi-seed training[/] over seeds {list(seeds)}")

    records: list[dict] = []
    for i, seed in enumerate(seeds, 1):
        seed_dir = out_root / "seeds" / f"seed_{seed}"
        console.print(f"\n[bold cyan]═══ Seed {seed} ({i}/{n_seeds}) ═══[/]")
        # model_copy does not re-validate, so pass output_dir as the Path the
        # config field expects (a str would break `config.output_dir / ...`).
        seed_cfg = config.model_copy(update={"seed": seed, "output_dir": seed_dir})
        val_mcc = run_training(seed_cfg)
        test_mcc = _read_metric(seed_dir / "metrics" / "test_metrics.json", "test_mcc")
        records.append({"seed": seed, "dir": str(seed_dir),
                        "val_mcc": val_mcc, "test_mcc": test_mcc})

    best = max(records, key=lambda r: r["val_mcc"])
    console.print(
        f"\n[bold green]Best seed: {best['seed']}[/] "
        f"(val MCC {best['val_mcc']:.4f}, test MCC {best['test_mcc']:.4f})"
    )
    _promote_seed(Path(best["dir"]), out_root)

    val_mccs = [r["val_mcc"] for r in records]
    test_mccs = [r["test_mcc"] for r in records if r["test_mcc"] == r["test_mcc"]]
    summary = {
        "n_seeds": n_seeds,
        "seeds": list(seeds),
        "best_seed": best["seed"],
        "selection_metric": "val_mcc",
        "per_seed": records,
        "val_mcc_mean": statistics.mean(val_mccs),
        "val_mcc_sd": statistics.stdev(val_mccs) if len(val_mccs) > 1 else 0.0,
        "test_mcc_mean": statistics.mean(test_mccs) if test_mccs else float("nan"),
        "test_mcc_sd": statistics.stdev(test_mccs) if len(test_mccs) > 1 else 0.0,
        "note": "Error bars for publication come from bootstrap, not seed sd.",
    }
    (out_root / "seeds_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    console.print(
        f"test MCC across seeds: {summary['test_mcc_mean']:.4f} "
        f"± {summary['test_mcc_sd']:.4f}  → seeds_summary.json"
    )


def _read_metric(path: Path, key: str) -> float:
    try:
        return float(json.loads(path.read_text())["numeric_metrics"][key])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return float("nan")


def _promote_seed(seed_dir: Path, out_root: Path) -> None:
    """Copy the winning seed's artifacts up to the canonical run directory."""
    for item in ("models", "metrics", "plots", "predictions", "config.yaml",
                 "class_indices.json", "model_config.json"):
        src = seed_dir / item
        if not src.exists():
            continue
        dst = out_root / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
