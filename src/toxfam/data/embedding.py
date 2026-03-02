from __future__ import annotations

import logging
import time
from pathlib import Path

import h5py
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_T5_model(model_dir, transformer_link, device):
    logging.info(f"Loading: {transformer_link}")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)

    if device.type == "cpu":
        logging.info("Casting model to full precision for CPU execution...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()

    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, vocab


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


def _process_batch(batch, hf_file, model, tokenizer, device):
    pdb_ids, seqs, seq_lens = zip(*batch)

    token_encoding = tokenizer(
        list(seqs),
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )

    input_ids = token_encoding["input_ids"].to(device)
    attention_mask = token_encoding["attention_mask"].to(device)

    try:
        with torch.no_grad():
            embedding_repr = model(input_ids, attention_mask=attention_mask)
    except RuntimeError as e:
        logging.error(
            f"RuntimeError during embedding. Try lowering batch size. Error: {e}"
        )
        return

    for batch_idx, identifier in enumerate(pdb_ids):
        s_len = seq_lens[batch_idx]
        emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
        emb = emb.mean(dim=0)
        emb_np = emb.cpu().numpy()

        if identifier in hf_file:
            logging.warning(f"Duplicate identifier found: {identifier}. Skipping.")
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
) -> None:
    """Generate per-protein ProtT5 embeddings from a FASTA file and write to HDF5."""
    device = get_device()
    logging.info(f"Using device: {device}")

    seq_dict = read_fasta(input_fasta)
    sorted_seqs = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    model, tokenizer = get_T5_model(model_dir, model_name, device)

    logging.info(f"Total sequences: {len(seq_dict)}")

    with h5py.File(output_h5, "w") as hf:
        batch = []
        batch_res_count = 0

        start_time = time.time()

        for seq_idx, (pdb_id, seq) in enumerate(
            tqdm(sorted_seqs, desc="Processing"), 1
        ):
            seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
            seq_len = len(seq)
            seq_spaced = " ".join(list(seq))

            if batch:
                if (len(batch) >= max_batch) or (
                    batch_res_count + seq_len > max_residues
                ):
                    _process_batch(batch, hf, model, tokenizer, device)
                    batch = []
                    batch_res_count = 0

            batch.append((pdb_id, seq_spaced, seq_len))
            batch_res_count += seq_len

            if seq_idx == len(seq_dict):
                _process_batch(batch, hf, model, tokenizer, device)

    logging.info(f"Finished in {time.time() - start_time:.2f} seconds")
