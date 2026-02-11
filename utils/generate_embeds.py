#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def get_device():
    # Helper to manage device selection safely
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # MPS is often unstable for T5, fall back to CPU if needed, but here we try MPS
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_T5_model(model_dir, transformer_link, device):
    logging.info(f"Loading: {transformer_link}")

    # Load model
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)

    # CRITICAL FIX: Cast to float32 if on CPU, otherwise half-precision models crash
    if device.type == "cpu":
        logging.info("Casting model to full precision for CPU execution...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()

    # Load tokenizer
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    return model, vocab


def read_fasta(fasta_path):
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                # CRITICAL FIX: Replicate original ID sanitization
                # HDF5 fails if keys contain "/" or "."
                uniprot_id = line.replace('>', '').strip()
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                sequences[uniprot_id] += ''.join(line.split()).upper().replace("-", "")
    return sequences


def process_batch(batch, hf_file, model, tokenizer, device, per_protein):
    # Unzip the batch
    pdb_ids, seqs, seq_lens = zip(*batch)

    # FIX: Use tokenizer directly (equivalent to batch_encode_plus but safer across versions)
    token_encoding = tokenizer(
        list(seqs),
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    )

    input_ids = token_encoding['input_ids'].to(device)
    attention_mask = token_encoding['attention_mask'].to(device)

    # Embed
    try:
        with torch.no_grad():
            embedding_repr = model(input_ids, attention_mask=attention_mask)
    except RuntimeError as e:
        logging.error(f"RuntimeError during embedding. Try lowering batch size. Error: {e}")
        return

    # Extract and Write
    for batch_idx, identifier in enumerate(pdb_ids):
        s_len = seq_lens[batch_idx]
        # slice-off padded tokens: seq_len x embed_dim
        emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

        if per_protein:
            emb = emb.mean(dim=0)

        # Move to numpy
        emb_np = emb.cpu().numpy()

        if identifier in hf_file:
            logging.warning(f"Duplicate identifier found: {identifier}. Skipping.")
        else:
            hf_file.create_dataset(identifier, data=emb_np)


def get_embeddings(args):
    device = get_device()
    logging.info(f"Using device: {device}")

    seq_dict = read_fasta(args.input)

    # Sort for efficiency (batch similar lengths together)
    # Original script logic: process longest first
    sorted_seqs = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    model, tokenizer = get_T5_model(args.model_dir, args.model_name, device)

    # Stats
    logging.info(f"Total sequences: {len(seq_dict)}")

    # Open HDF5 file once and keep open
    with h5py.File(args.output, "w") as hf:
        batch = []
        batch_res_count = 0

        start_time = time.time()

        for seq_idx, (pdb_id, seq) in enumerate(tqdm(sorted_seqs, desc="Processing"), 1):
            # ProtT5 expects spaces between residues
            # Original script replaces U,Z,O with X
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            seq_len = len(seq)
            seq_spaced = ' '.join(list(seq))

            # Logic to decide if we run the batch
            # 1. Current batch size limit reached
            # 2. Total residues limit reached
            # 3. Current sequence is too long (process individually)

            # Check if adding this sequence would break limits
            if batch:
                if (len(batch) >= args.max_batch) or \
                        (batch_res_count + seq_len > args.max_residues):
                    process_batch(batch, hf, model, tokenizer, device, args.per_protein)
                    batch = []
                    batch_res_count = 0

            batch.append((pdb_id, seq_spaced, seq_len))
            batch_res_count += seq_len

            # Flush final batch
            if seq_idx == len(seq_dict):
                process_batch(batch, hf, model, tokenizer, device, args.per_protein)

    logging.info(f"Finished in {time.time() - start_time:.2f} seconds")


def create_arg_parser():
    parser = argparse.ArgumentParser(description='ProtT5 Embedder')
    parser.add_argument('-i', '--input', required=True, type=str, help='Input FASTA file')
    parser.add_argument('-o', '--output', required=True, type=str, help='Output H5 file')
    parser.add_argument('--model-dir', type=str, default=None, help='Cache directory for model')
    parser.add_argument('--model-name', type=str, default="Rostlab/prot_t5_xl_half_uniref50-enc", help='Model name')

    parser.add_argument('--per-protein', action='store_true', help="Output mean-pooled per-protein embeddings")

    parser.add_argument('--max_residues', type=int, default=4000, help='Max residues per batch')
    parser.add_argument('--max_batch', type=int, default=100, help='Max sequences per batch')
    return parser


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    get_embeddings(args)