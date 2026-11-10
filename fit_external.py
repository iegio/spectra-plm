import random
random.seed(0)

import argparse
import gc
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import ASTRAL_FASTA, UR50_FASTA_PATH, UR50_SAMPLE_FASTA
from plm_wrapper import PLM, PLM_dim
from utils import (
    parse_astral,
    physical_dct,
    physical_dft,
    read_fasta,
    relative_dct,
    relative_dft,
    TRANSFORMS
)

DIMS = {"esm2-650M": 1280, "prottrans": 1024}
OUT_ROOT = Path("/output/pca_tools/fits")
EMBEDDINGS_EXTRACTIONS = "output/pca_tools/embeddings_extractions"

def embed_fasta(fasta_path, out_path, model_key, layer_ix):
    header_list, seq_list = read_fasta(fasta_path, return_headers=True)

    if model_key not in PLM:
        raise ValueError(f"Unknown model: {model_key}")

    if len(header_list) != len(seq_list):
        raise ValueError("FASTA headers and sequences have different lengths")

    out_dir = os.path.join(out_path, f"{model_key}_layer{layer_ix}")
    os.makedirs(out_dir, exist_ok=True)

    model = PLM[model_key]()
    model.eval()

    for header, seq in tqdm(zip(header_list, seq_list), total=len(seq_list), desc=f"{model_key} layer {layer_ix}"):
        protein_id = header.lstrip(">").split()[0]
        save_path = os.path.join(out_dir, f"{protein_id}.pkl")

        if os.path.exists(save_path):
            continue

        with torch.inference_mode():
            emb = model([seq], hidden_layer=layer_ix)

        # (1, L, D) -> (L, D)
        emb = emb[0].detach().cpu().numpy().astype(np.float32)

        if emb.shape != (len(seq), PLM_dim[model_key]):
            raise ValueError(f"Unexpected shape for {protein_id}: {emb.shape}")

        with open(save_path, "wb") as f:
            pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Embeddings saved to: {out_dir}")

def load_residue(path, dim):
    with open(path, "rb") as f:
        x = pickle.load(f)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != dim or x.shape[0] < 1:
        raise ValueError(f"Invalid residue embedding shape {x.shape} in {path}")
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite values in {path}")
    return x

def get_embedding_files(folder):
    files = sorted(Path(folder).glob("*.pkl"))
    if not files:
        files = sorted(Path(folder).glob("*.pickle"))
    if not files:
        raise FileNotFoundError(f"No .pkl/.pickle files directly in {folder}")
    return files

def output_dir(model, layer, method, n, maxf, out_root):
    base = out_root / model / f"layer_{layer}" / method
    if method == "mean_pool":
        return base
    tag = f"N{n}_fmax{maxf:g}".replace(".", "p")
    return base / tag

def write_objects(outdir, scaler, pca, metadata, overwrite):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / "scaler.pkl", outdir / "pca.pkl", outdir / "metadata.json"]
    if not overwrite and any(p.exists() for p in paths):
        raise FileExistsError(f"Output already exists in {outdir}; use --overwrite")
    for name, obj in [("scaler.pkl", scaler), ("pca.pkl", pca)]:
        tmp = outdir / (name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, outdir / name)
    tmp = outdir / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp, outdir / "metadata.json")
    print(f"Saved fits: {outdir}", flush=True)

def fit_pca(matrix, components, seed):
    if components > min(matrix.shape):
        raise ValueError(f"Cannot fit {components} components from shape {matrix.shape}")
    return PCA(n_components=components, svd_solver="randomized", random_state=seed).fit(matrix)

def fit_mean_pool(args, embedding_dir, dim, outdir):
    files = get_embedding_files(embedding_dir)
    x = np.stack([
        load_residue(path, dim).mean(axis=0)
        for path in tqdm(files, desc="Mean-pool embeddings")
    ])
    print(f"Mean-pool input: {x.shape}", flush=True)
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    pca = fit_pca(standardized, dim // 2, args.seed)
    metadata = dict(model=args.model, layer=args.layer, method="mean_pool",
                    n_proteins=len(files), input_dim=dim, output_dim=dim // 2,
                    embedding_dir=embedding_dir, seed=args.seed,
                    sklearn_pca_solver="randomized", feature_order="PLM dimension")
    write_objects(outdir, scaler, pca, metadata, args.overwrite)

def fit_spectral(args, embedding_dir, dim, outdir):
    files = get_embedding_files(embedding_dir)
    if len(files) != 25000:
        print(f"WARNING: found {len(files)} pickle files (expected approximately 25000)", flush=True)
    # Assumes each transform returns (N-1)*D real-valued features.
    n_features = (args.N - 1) * dim
    n_samples = len(files)
    if n_samples < dim // 2:
        raise ValueError("Insufficient proteins for requested PCA components")
    print(f"Spectral matrix: ({n_samples}, {n_features}); "
          f"float32 disk size ~{n_samples*n_features*4/1e9:.2f} GB", flush=True)
    fn = TRANSFORMS[args.method]
    # Use local scratch where possible. The scratch files are removed on completion.
    scratch = args.scratch or os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or str(outdir.parent)
    Path(scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="external_pca_", dir=scratch) as tmp:
        raw = np.memmap(Path(tmp) / "features.dat", dtype="float32", mode="w+",
                        shape=(n_samples, n_features))
        for i, path in enumerate(tqdm(files, desc="Extract spectral features")):
            emb = load_residue(path, dim)
            raw[i] = fn(emb, args.N, args.maxF)
        raw.flush()
        scaler = StandardScaler()
        for start in tqdm(range(0, n_samples, args.batch_size), desc="Fit scaler"):
            scaler.partial_fit(raw[start:start + args.batch_size])

        # Standardize into a second memmap, leaving the original untouched.
        standardized = np.memmap(Path(tmp) / "standardized.dat", dtype="float32",
                                 mode="w+", shape=(n_samples, n_features))
        for start in tqdm(range(0, n_samples, args.batch_size), desc="Apply scaler"):
            end = min(start + args.batch_size, n_samples)
            standardized[start:end] = scaler.transform(raw[start:end])
        standardized.flush()
        del raw
        gc.collect()
        print("Fitting randomized PCA (may require substantial RAM)...", flush=True)
        pca = fit_pca(standardized, dim // 2, args.seed)
        metadata = dict(model=args.model, layer=args.layer, method=args.method,
                        N=args.N,
                        maxF=args.maxF, n_proteins=n_samples, input_dim=n_features,
                        output_dim=dim // 2, embedding_dir=embedding_dir,
                        seed=args.seed, sklearn_pca_solver="randomized",
                        dc_removed=True, feature_order="flattened C-order (N-1,D)",
                        normalization="per-protein spectral norm then fitted StandardScaler")
        write_objects(outdir, scaler, pca, metadata, args.overwrite)
        del standardized, pca

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("layer", type=int)
    ap.add_argument("method")
    ap.add_argument("N", type=int, nargs="?", default=0)
    ap.add_argument("maxF", type=float, nargs="?", default=0.5)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--scratch", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.method != "mean_pool":
        if args.N < 2 or not 0 < args.maxF <= 0.5:
            ap.error("Spectral methods require N >= 2 and 0 < maxF <= 0.5")

    # Randomly sample without replacement
    if not os.path.exists(UR50_SAMPLE_FASTA):
        astral_seqs = [x["seq"] for x in parse_astral(ASTRAL_FASTA)]
        header_list, seq_list = read_fasta(UR50_FASTA_PATH, return_headers=True)

        num_to_sample = 25000
        total_positions = list(range(0, len(header_list)))
        positions_to_sample = random.sample(total_positions, min(num_to_sample * 2, len(total_positions)))

        valid = 0
        with open(UR50_SAMPLE_FASTA, "w") as f:
            for p in positions_to_sample:
                header = header_list[p]
                seq = seq_list[p]
                if len(seq) > 1022 or seq in astral_seqs:
                    continue
                f.write(">" + header + "\n")
                f.write(seq + "\n")
                valid += 1
                if valid == num_to_sample:
                    break

    # Embed sampled UR50 proteins
    embed_fasta(UR50_SAMPLE_FASTA, EMBEDDINGS_EXTRACTIONS, args.model, args.layer)

    # Fit parameters
    embedding_dir = f"{EMBEDDINGS_EXTRACTIONS}/{args.model}_layer{args.layer}"
    dim = DIMS[args.model]
    outdir = output_dir(args.model, args.layer, args.method,
                        args.N, args.maxF, args.out_root)
    if outdir.exists() and not args.overwrite and any(outdir.glob("*.pkl")):
        ap.error(f"Fit already exists at {outdir}; use --overwrite")
    if args.method == "mean_pool":
        fit_mean_pool(args, embedding_dir, dim, outdir)
    else:
        fit_spectral(args, embedding_dir, dim, outdir)

if __name__ == "__main__":
    main()
