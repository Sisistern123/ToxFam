from __future__ import annotations

import os
import time
from pathlib import Path

import h5py
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import T5EncoderModel, T5Tokenizer

from toxfam.device import get_device  # re-export for backward compat

console = Console()


def get_T5_model(model_dir, transformer_link, device):
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)

    if device.type == "cuda":
        model.half()
    elif device.type == "cpu":
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()

    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, vocab


def _load_model_quietly(model_dir, model_name, device):
    """Load ProtT5 model with all stdout/stderr noise suppressed at the OS fd level."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        return get_T5_model(model_dir, model_name, device)
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def read_fasta(fasta_path: str | Path) -> dict[str, str]:
    sequences = dict()
    current_id = None
    with open(fasta_path, "r") as fasta_f:
        for line in fasta_f:
            if line.startswith(">"):
                current_id = line.replace(">", "").strip()
                current_id = current_id.replace("/", "_").replace(".", "_")
                sequences[current_id] = ""
            elif current_id is not None:
                sequences[current_id] += "".join(line.split()).upper().replace("-", "")
    return sequences


def _process_batch(batch, hf_file, model, tokenizer, device, duplicates, use_amp):
    pdb_ids, seqs, seq_lens = zip(*batch)

    token_encoding = tokenizer(
        list(seqs),
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )

    input_ids = token_encoding["input_ids"].to(device)
    attention_mask = token_encoding["attention_mask"].to(device)

    with torch.no_grad(), torch.amp.autocast(device.type, enabled=use_amp):
        embedding_repr = model(input_ids, attention_mask=attention_mask)

    for batch_idx, identifier in enumerate(pdb_ids):
        s_len = seq_lens[batch_idx]
        emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
        emb = emb.mean(dim=0)
        emb_np = emb.float().cpu().numpy()

        if identifier in hf_file:
            duplicates.append(identifier)
        else:
            hf_file.create_dataset(identifier, data=emb_np)


def generate_embeddings(
    input_fasta: str | Path,
    output_h5: str | Path,
    *,
    model_dir: str | None = None,
    model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
    max_residues: int = 4000,
    max_batch: int = 100,
    force: bool = False,
) -> None:
    """Generate per-protein ProtT5 embeddings from a FASTA file and write to HDF5."""
    # -- Step 1: Device --
    device = get_device()
    use_amp = device.type == "cuda"
    console.print(f"\n[bold]1.[/] Device: [cyan]{device}[/]")

    # -- Step 2: Read FASTA --
    console.print(f"\n[bold]2.[/] Reading [cyan]{input_fasta}[/]")
    seq_dict = read_fasta(input_fasta)
    sorted_seqs = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    max_len = len(sorted_seqs[0][1]) if sorted_seqs else 0
    console.print(f"   {len(seq_dict)} sequences (longest: {max_len} residues)")

    # -- Step 2b: Skip already-embedded sequences --
    output_path = Path(output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h5_mode = "w" if force else "a"

    existing_keys: set[str] = set()
    if not force and output_path.exists():
        with h5py.File(output_path, "r") as hf_read:
            existing_keys = set(hf_read.keys())

    if existing_keys:
        sorted_seqs = [(k, v) for k, v in sorted_seqs if k not in existing_keys]
        console.print(
            f"   [dim]{len(existing_keys)} already embedded, "
            f"{len(sorted_seqs)} remaining[/]"
        )

    if not sorted_seqs:
        console.print("\n[bold green]All sequences already embedded. Nothing to do.[/]")
        return

    # -- Step 3: Load model --
    console.print(f"\n[bold]3.[/] Loading model [cyan]{model_name}[/]")
    with console.status("Loading model & tokenizer..."):
        model, tokenizer = _load_model_quietly(model_dir, model_name, device)

    # -- Step 4: Embed --
    console.print(f"\n[bold]4.[/] Generating embeddings → [cyan]{output_h5}[/]")
    duplicates: list[str] = []
    embedded = 0
    start_time = time.time()

    with h5py.File(output_h5, h5_mode) as hf:
        batch: list[tuple[str, str, int]] = []
        batch_res_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Embedding", total=len(sorted_seqs))

            for pdb_id, seq in sorted_seqs:
                seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
                seq_len = len(seq)
                seq_spaced = " ".join(list(seq))

                if batch and (
                    len(batch) >= max_batch or batch_res_count + seq_len > max_residues
                ):
                    _process_batch(
                        batch, hf, model, tokenizer, device, duplicates, use_amp
                    )
                    hf.flush()
                    embedded += len(batch)
                    progress.update(task, completed=embedded)
                    batch = []
                    batch_res_count = 0

                batch.append((pdb_id, seq_spaced, seq_len))
                batch_res_count += seq_len

            # Final batch
            if batch:
                _process_batch(batch, hf, model, tokenizer, device, duplicates, use_amp)
                hf.flush()
                embedded += len(batch)
                progress.update(task, completed=embedded)

    elapsed = time.time() - start_time

    # -- Summary --
    if duplicates:
        console.print(
            f"   [yellow]Skipped {len(duplicates)} duplicate identifier(s)[/]"
        )

    console.print(
        f"\n[bold green]Done.[/] {embedded - len(duplicates)} embeddings "
        f"in {elapsed:.1f}s"
    )

    # -- Step 5: Prune stale embeddings not in input FASTA --
    fasta_ids = set(seq_dict.keys())
    with h5py.File(output_h5, "a") as hf:
        extra = set(hf.keys()) - fasta_ids
        if extra:
            for key in extra:
                del hf[key]
            console.print(f"\n[bold]5.[/] Pruned [red]{len(extra)}[/] stale embeddings")
