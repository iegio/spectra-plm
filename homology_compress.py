import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
from Bio import SeqIO
from tqdm.auto import tqdm

from utils import PLM_dim, physical_dct, physical_dft, prost_idct, relative_dct, relative_dft, TRANSFORMS
from configs import method_tag, POOL_METHODS

ASTRAL_FASTA = Path("datasets/SCOPe/astral40.fa")
ASTRAL_ROOT = Path("/astral_benchmark")
FITS_ROOT = Path("/output/pca_tools/fits")

def load_fit(folder):
    with open(folder / "scaler.pkl", "rb") as handle:
        scaler = pickle.load(handle)
    with open(folder / "pca.pkl", "rb") as handle:
        pca = pickle.load(handle)
    return scaler, pca

def atomic_pickle_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def compress_astral(fasta, root, fits_root, model, layer, method, N, maxF,
                    overwrite=False):
    if model not in PLM_dim:
        raise ValueError(f"Unsupported model: {model}")
    if method not in POOL_METHODS and method not in TRANSFORMS.keys():
        raise ValueError(f"Unknown method: {method}")
    if method in TRANSFORMS.keys():
        minimum_n = 2
        if N < minimum_n or not (0 < maxF <= 0.5):
            raise ValueError(f"{method}: require N >= {minimum_n} and 0 < maxF <= 0.5")

    dim = PLM_dim[model]
    tag = method_tag(method, N, maxF)
    residue_dir = root / "embeddings_astral40" / model / f"layer_{layer}"
    output = root / "compress_astral40" / model / f"layer_{layer}" / method / f"{tag}.pkl"
    if output.exists() and not overwrite:
        print(f"Already exists; skipping: {output}")
        return output

    mean_scaler = mean_pca = spec_scaler = spec_pca = None
    fit_base = fits_root / model / f"layer_{layer}"
    if method == "mean_pool_standardized":
        mean_scaler, _ = load_fit(fit_base / "mean_pool")
    elif method != "mean_pool" and method != "2d_idct":
        mean_scaler, mean_pca = load_fit(fit_base / "mean_pool")
        spec_scaler, spec_pca = load_fit(fit_base / method / tag)

    data = {}
    expected_width = None
    for record in tqdm(SeqIO.parse(str(fasta), "fasta"), desc=f"Compress {method}"):
        domain_id = record.id
        if domain_id in data:
            raise ValueError(f"Duplicate ASTRAL domain ID: {domain_id}")
        path = residue_dir / f"{domain_id}.pkl"
        with open(path, "rb") as handle:
            X = np.asarray(pickle.load(handle), dtype=np.float32)
        if X.shape != (len(record.seq), dim) or not np.isfinite(X).all():
            raise ValueError(f"Invalid residue embedding for {domain_id}: {X.shape} at {path}")

        if method in POOL_METHODS:
            pooled = X.astype(np.float64).mean(axis=0).astype(np.float32)[None, :]
            if method == "mean_pool_standardized":
                pooled = mean_scaler.transform(pooled)
            vector = pooled.reshape(-1)
        elif method == "2d_idct":
            vector = np.asarray(TRANSFORMS[method](X, N, maxF)).reshape(-1)
        else:
            mean_features = X.mean(axis=0).astype(np.float32)[None, :]
            spectral_features = np.asarray(TRANSFORMS[method](X, N, maxF),
                                           dtype=np.float32).reshape(1, -1)
            mean_emb = mean_pca.transform(mean_scaler.transform(mean_features))
            spectral_emb = spec_pca.transform(spec_scaler.transform(spectral_features))
            vector = np.concatenate([mean_emb, spectral_emb], axis=1).reshape(-1)

        vector = np.asarray(vector, dtype=np.float32)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid compressed vector for {domain_id}")
        if method != "2d_idct" and vector.shape != (dim,):
            raise ValueError(f"Expected compressed dimension {dim}, got {vector.shape} for {domain_id}")
        if expected_width is None:
            expected_width = vector.size
        elif vector.size != expected_width:
            raise ValueError(f"Inconsistent compressed dimensions: {domain_id} has {vector.size}, expected {expected_width}")
        data[domain_id] = vector

    if not data:
        raise ValueError(f"No sequences in {fasta}")
    atomic_pickle_dump(data, output)
    print(f"Saved {len(data):,} vectors of dimension {expected_width:,} to {output}")
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=sorted([*POOL_METHODS, *TRANSFORMS]))
    parser.add_argument("layer", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("maxF", type=float)
    parser.add_argument("model", choices=sorted(PLM_dim))
    parser.add_argument("--fasta", type=Path, default=ASTRAL_FASTA)
    parser.add_argument("--root", type=Path, default=ASTRAL_ROOT)
    parser.add_argument("--fits-root", type=Path, default=FITS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("layer must be nonnegative")
    compress_astral(args.fasta, args.root, args.fits_root, args.model, args.layer,
                    args.method, args.N, args.maxF, args.overwrite)

if __name__ == "__main__":
    main()
