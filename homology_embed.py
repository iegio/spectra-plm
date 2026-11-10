import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from plm_wrapper import PLM, PLM_dim
from utils import read_fasta

ASTRAL_FASTA = "datasets/SCOPe/astral40.fa"
OUT_ROOT = Path("outputs/astral_benchmark")

def atomic_pickle_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def checked_array(value, expected_shape, label):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{label}: expected shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label}: non-finite values")
    return array

def embed_astral40(fasta_path=ASTRAL_FASTA, out_root=OUT_ROOT, model_key="esm2-650M", layer_ix=33, overwrite=False, checkpoint_every=500):
    if model_key not in PLM:
        raise ValueError(f"Unknown model {model_key!r}; choose from {sorted(PLM)}")
    if layer_ix < 0:
        raise ValueError("layer_ix must be nonnegative")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")

    headers, sequences = read_fasta(str(fasta_path), return_headers=True)
    if len(sequences) == 0 or len(headers) != len(sequences):
        raise ValueError("FASTA must contain matching, nonempty headers and sequences")

    records = []
    seen_ids = set()
    for header, sequence in zip(headers, sequences):
        protein_id = str(header).lstrip(">").split()[0]
        sequence = str(sequence).upper()
        if not protein_id or protein_id in (".", "..") or "/" in protein_id or "\\" in protein_id:
            raise ValueError(f"Invalid domain ID: {protein_id!r}")
        if protein_id in seen_ids:
            raise ValueError(f"Duplicate domain ID in FASTA: {protein_id}")
        if not sequence:
            raise ValueError(f"Empty sequence: {protein_id}")
        seen_ids.add(protein_id)
        records.append((protein_id, sequence))

    root = Path(out_root)
    residue_dir = root / "embeddings_astral40" / model_key / f"layer_{layer_ix}"
    residue_dir.mkdir(parents=True, exist_ok=True)

    is_esm = model_key.startswith("esm2-")
    cls_path = (root / "compress_astral40" / model_key / f"layer_{layer_ix}"
                / "cls_pool" / "cls_pool.pkl") if is_esm else None
    dim = PLM_dim[model_key]

    cls_vectors = {}
    if cls_path is not None and cls_path.exists() and not overwrite:
        with open(cls_path, "rb") as handle:
            saved = pickle.load(handle)
        if not isinstance(saved, dict):
            raise ValueError(f"Expected domain-to-vector dictionary in {cls_path}")
        unknown = set(saved) - seen_ids
        if unknown:
            raise ValueError(f"CLS cache contains IDs absent from FASTA: {sorted(unknown)[:5]}")
        cls_vectors = {
            protein_id: checked_array(vector, (dim,), f"CLS {protein_id}")
            for protein_id, vector in saved.items()
        }

    # Check existing residue files before loading a potentially large GPU model.
    missing_residues = set()
    for protein_id, sequence in tqdm(records, desc="Check cached residue embeddings"):
        path = residue_dir / f"{protein_id}.pkl"
        if overwrite or not path.exists():
            missing_residues.add(protein_id)
        else:
            with open(path, "rb") as handle:
                checked_array(pickle.load(handle), (len(sequence), dim), f"Residues {protein_id}")

    missing_cls = set(seen_ids) - set(cls_vectors) if is_esm else set()
    print(f"ASTRAL proteins: {len(records):,}")
    print(f"Model: {model_key}; layer: {layer_ix}; residue dimension: {dim}")
    print(f"Residue output: {residue_dir}")
    if cls_path is not None:
        print(f"CLS output: {cls_path}")
    print(f"Residue embeddings to compute: {len(missing_residues):,}")
    if is_esm:
        print(f"CLS embeddings to compute: {len(missing_cls):,}")

    if not missing_residues and not missing_cls:
        print("All requested embeddings are already cached.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("The provided PLM wrappers load models on CUDA; no CUDA device is available.")
    model = PLM[model_key]()
    model.eval()

    new_cls = 0
    try:
        with torch.inference_mode():
            for protein_id, sequence in tqdm(records, desc="Extract ASTRAL embeddings"):
                needs_residue = protein_id in missing_residues
                needs_cls = protein_id in missing_cls
                if not needs_residue and not needs_cls:
                    continue
                try:
                    if needs_residue:
                        residues = model([sequence], hidden_layer=layer_ix)[0].detach().cpu().numpy()
                        residues = checked_array(residues, (len(sequence), dim),
                                                 f"Residues {protein_id}")
                        atomic_pickle_dump(residues, residue_dir / f"{protein_id}.pkl")
                    if needs_cls:
                        # ESMWrapper returns CLS only when explicitly requested.
                        cls = model([sequence], hidden_layer=layer_ix, return_cls=True)[0, 0]
                        cls_vectors[protein_id] = checked_array(cls.detach().cpu().numpy(),
                                                                (dim,), f"CLS {protein_id}")
                        new_cls += 1
                        if new_cls % checkpoint_every == 0:
                            atomic_pickle_dump(cls_vectors, cls_path)
                except Exception as exc:
                    raise RuntimeError(f"Failed on domain {protein_id} (length {len(sequence)})") from exc
    finally:
        # Preserve completed CLS work if processing is interrupted by an exception.
        if cls_path is not None and new_cls:
            atomic_pickle_dump(cls_vectors, cls_path)

    print(f"Saved residue embeddings under {residue_dir}")
    if cls_path is not None:
        print(f"Saved {len(cls_vectors):,} CLS vectors to {cls_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_key", choices=sorted(PLM))
    parser.add_argument("layer_ix", type=int)
    parser.add_argument("--fasta", default=ASTRAL_FASTA)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Atomically save the CLS dictionary every N newly extracted proteins (default: 500)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute residue and CLS embeddings even if cached")
    args = parser.parse_args()
    embed_astral40(fasta_path=args.fasta, out_root=args.out_root,
                   model_key=args.model_key, layer_ix=args.layer_ix,
                   overwrite=args.overwrite, checkpoint_every=args.checkpoint_every)

if __name__ == "__main__":
    main()
