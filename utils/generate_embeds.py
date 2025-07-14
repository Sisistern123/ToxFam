#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

import os

# Detect device with MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def get_T5_model(model_dir: Path, transformer_link: str):
    """
    Load only the T5 encoder model and slow tokenizer from HuggingFace,
    compile it for speed if on CUDA, and move to the target device.
    """
    logging.info(f"Loading T5EncoderModel {transformer_link}")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)

    # Only compile on CUDA—skip on MPS to avoid compiler errors
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            logging.info("Compiled model with torch.compile() for additional speed.")
        except Exception as e:
            logging.warning(f"torch.compile() failed ({e}); continuing without compilation.")
    else:
        logging.info("Skipping torch.compile(): not running on CUDA device.")

    model.to(device).eval()

    # Use slow tokenizer to avoid conversion issues
    tokenizer = T5Tokenizer.from_pretrained(
        transformer_link,
        do_lower_case=False,
        use_fast=False
    )

    return model, tokenizer

def read_fasta(fasta_path: Path) -> dict:
    """
    Read a FASTA file and return dict {seq_id: sequence}.
    """
    sequences = {}
    with fasta_path.open('r') as f:
        seq_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
                sequences[seq_id] = ''
            else:
                sequences[seq_id] += line.upper().replace('-', '')
    return sequences

def process_batch(
    batch: list,
    hf: h5py.File,
    model: torch.nn.Module,
    tokenizer,
    per_protein: bool
):
    """
    Tokenize, embed, and write a batch of sequences to the HDF5 file.
    """
    pdb_ids, seqs, seq_lens = zip(*batch)
    encoding = tokenizer.batch_encode_plus(
        seqs,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    hidden_states = outputs.last_hidden_state  # (B, L, D)
    for idx, seq_id in enumerate(pdb_ids):
        L = seq_lens[idx]
        emb = hidden_states[idx, :L]
        if per_protein:
            emb = emb.mean(dim=0)
        data = emb.cpu().numpy()
        hf.create_dataset(seq_id, data=data)
        logging.debug(f"Wrote embedding for {seq_id}, shape {data.shape}")

def get_embeddings(
    seq_path: Path,
    emb_path: Path,
    model_dir: Path,
    transformer_link: str,
    per_protein: bool,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch: int = 100
) -> None:
    """
    Main embedding loop: reads FASTA, batches sequences by residue count,
    embeds with ProtT5 encoder, and writes to an HDF5 file.
    """
    sequences = read_fasta(seq_path)
    model, tokenizer = get_T5_model(model_dir, transformer_link)

    total = len(sequences)
    lengths = [len(s) for s in sequences.values()]
    avg_len = sum(lengths) / total if total > 0 else 0
    n_long = sum(1 for L in lengths if L > max_seq_len)
    sorted_seqs = sorted(
        sequences.items(), key=lambda kv: len(kv[1]), reverse=True)

    logging.info(f"Read {total} sequences; avg len {avg_len:.2f}; >{max_seq_len}: {n_long}")
    start_time = time.time()

    with h5py.File(str(emb_path), 'w') as hf:
        batch = []
        cum_res = 0
        for idx, (seq_id, raw_seq) in enumerate(
            tqdm(sorted_seqs, total=total, desc="Embedding sequences", unit="seq"),
            start=1
        ):
            seq = raw_seq.translate(str.maketrans('UZO', 'XXX'))
            L = len(seq)
            spaced = ' '.join(seq)

            # Flush batch if limits are exceeded or single sequence is too long
            if batch and (
                len(batch) + 1 > max_batch or
                cum_res + L > max_residues or
                L > max_seq_len
            ):
                process_batch(batch, hf, model, tokenizer, per_protein)
                batch, cum_res = [], 0

            batch.append((seq_id, spaced, L))
            cum_res += L

            # Flush the final batch at the end
            if idx == total and batch:
                process_batch(batch, hf, model, tokenizer, per_protein)

    elapsed = time.time() - start_time
    logging.info(
        f"Embeddings saved to {emb_path}. "
        f"Processed {total} sequences in {elapsed:.2f}s (avg {elapsed/total:.4f}s/seq)."
    )

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Create ProtT5 encoder embeddings for protein sequences and save them in HDF5.'
    )
    parser.add_argument(
        '-i', '--input', required=True, type=Path,
        help='FASTA file with protein sequences.'
    )
    parser.add_argument(
        '-o', '--output', required=True, type=Path,
        help='HDF5 file path to save embeddings.'
    )
    parser.add_argument(
        '--model-dir', type=Path, default=None,
        help='Cache dir for the pre-trained encoder checkpoint.'
    )
    parser.add_argument(
        '--model-name', type=str,
        default='Rostlab/prot_t5_xl_half_uniref50-enc',
        help='Identifier of the ProtT5 encoder model.'
    )
    parser.add_argument(
        '--per-protein', action='store_true',
        help='Output mean-pooled embedding per protein instead of per-residue.'
    )
    parser.add_argument(
        '--max-residues', type=int, default=4000,
        help='Max total residues per batch.'
    )
    parser.add_argument(
        '--max-seq-len', type=int, default=1000,
        help='Threshold for single-sequence processing.'
    )
    parser.add_argument(
        '--max-batch', type=int, default=100,
        help='Max sequences per batch.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable debug logging.'
    )
    return parser

def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Using device: {device}")

    get_embeddings(
        seq_path=args.input,
        emb_path=args.output,
        model_dir=args.model_dir,
        transformer_link=args.model_name,
        per_protein=args.per_protein,
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch
    )


def show_entries(h5_path):
    # sanity check
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"No such file: {h5_path!r} (check spelling, relative path, and extension)")

    with h5py.File(h5_path, 'r') as f:
        def recurse(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    first = obj[...].flat[0]
                    print(f"Dataset: '{name}'")
                    print("  Shape:", obj.shape)
                    print("  Dtype:", obj.dtype)
                    print("  Entry:", first, "\n")
                except Exception as e:
                    print(f"  [!] Couldn’t read entry of '{name}': {e}\n")

        f.visititems(recurse)

def show_first_dataset_contents(h5_path):
    """
    Open the HDF5 file at h5_path, find the first dataset (by visit order),
    and print its entire contents.
    """
    with h5py.File(h5_path, 'r') as f:
        # Collect dataset names in visit order
        dataset_names = []
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_names.append(name)
        f.visititems(collect)

        if not dataset_names:
            print("No datasets found in the file.")
            return

        first_name = dataset_names[0]
        ds = f[first_name]
        data = ds[...]  # load entire dataset into memory

        print(f"First dataset: '{first_name}'")
        print(f"  Shape: {ds.shape}")
        print(f"  Dtype: {ds.dtype}")
        print("  Contents:")
        print(data)

if __name__ == '__main__':
    # usage:
    # python utility/generate_embeds.py
    # --input dataset/sequences/sequences_batch_1.fasta
    # --output dataset/embeddings/embeddings_batch_1.h5
    # --model ProstT5
    main()

    #show_entries("../dataset/embeddings/embeddings_batch_1.h5")
    #show_first_dataset_contents("../dataset/embeddings/embeddings_batch_1.h5")
