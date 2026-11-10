import os
import pickle
import tempfile
import numpy as np

from Bio import SeqIO
from tqdm.auto import tqdm
from config import *
from utils import PLM_dim, N_LAYERS

def _distance_block(A, X, metric, X_sq=None):
    if metric == "l2":
        A_sq = np.sum(A * A, axis=1)
        D2 = A_sq[:, None] + X_sq[None, :] - 2.0 * (A @ X.T)
        np.maximum(D2, 0, out=D2)
        np.sqrt(D2, out=D2)
        return D2

    if metric == "cosine":
        return 1.0 - A @ X.T

    raise ValueError(f"Unknown distance metric: {metric}")

def _distance_auc_parts(pos_parts, neg_parts, search_chunk_size=1_000_000):
    pos_parts = [np.asarray(x, dtype=np.float32).ravel() for x in pos_parts if x.size > 0]
    neg_parts = [x for x in neg_parts if x.size > 0]

    n_pos = sum(x.size for x in pos_parts)
    n_neg = sum(x.size for x in neg_parts)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    positives = np.concatenate(pos_parts)
    positives.sort()

    wins = 0.0

    for part in neg_parts:
        part = np.asarray(part, dtype=np.float32).ravel()

        for start in range(0, part.size, search_chunk_size):
            negatives = part[start:start + search_chunk_size]
            left = np.searchsorted(positives, negatives, side="left")
            right = np.searchsorted(positives, negatives, side="right")
            wins += np.sum(left + 0.5 * (right - left), dtype=np.float64)

    return wins / (n_pos * n_neg)

def _make_memmap(path, n):
    if n == 0:
        return np.empty(0, dtype=np.float32)

    return np.memmap(path, dtype=np.float32, mode="w+", shape=(n,))

def evaluate_astral40(records, do_embedding, distance_metric="cosine", batch_size=128, tmp_dir=None, show_progress=True):
    if distance_metric not in ["l2", "cosine"]:
        raise ValueError(f"distance_metric must be 'l2' or 'cosine', got {distance_metric!r}")

    n = len(records)
    print("Loading/computing embeddings...")
    iterator = tqdm(records, desc="Embeddings") if show_progress else records

    embeddings = [
        np.asarray(do_embedding(record), dtype=np.float32).ravel()
        for record in iterator
    ]

    dims = {x.shape[0] for x in embeddings}

    if len(dims) != 1:
        raise ValueError(f"do_embedding must return fixed-length vectors; "f"observed dimensions: {sorted(dims)}")

    X = np.stack(embeddings).astype(np.float32, copy=False)
    del embeddings

    # For cosine, normalize once here so cosine similarity = A @ X.T.
    if distance_metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Cannot compute cosine distance for zero-norm embeddings.")

        X = X / norms
        X_sq = None

    else:
        X_sq = np.sum(X * X, axis=1)

    families = np.asarray([record["family"] for record in records])
    superfamilies = np.asarray([record["superfamily"] for record in records])

    family_names = sorted(np.unique(families))
    superfamily_names = sorted(np.unique(superfamilies))

    for family in family_names:
        sf = np.unique(superfamilies[families == family])

        if len(sf) != 1:
            raise ValueError(f"Family {family} belongs to multiple superfamilies: {sf}")

    _, family_counts = np.unique(families, return_counts=True)
    _, superfamily_counts = np.unique(superfamilies, return_counts=True)

    n_pairs = n * (n - 1) // 2
    n_same_family = int(np.sum(family_counts * (family_counts - 1) // 2))
    n_same_superfamily = int(
        np.sum(superfamily_counts * (superfamily_counts - 1) // 2)
    )
    n_cross_family_positive = n_same_superfamily - n_same_family
    n_negative = n_pairs - n_same_superfamily

    print(f"Proteins: {n:,}")
    print(f"Embedding dimension: {X.shape[1]:,}")
    print(f"Distance metric: {distance_metric}")
    print(f"Total unordered pairs: {n_pairs:,}")
    print(f"Same-family positive pairs: {n_same_family:,}")
    print(f"Cross-family same-superfamily positive pairs: {n_cross_family_positive:,}")
    print(f"Different-superfamily negative pairs: {n_negative:,}")

    family_included = {}
    family_excluded = {}
    superfamily_included = {}
    superfamily_excluded = {}
    protein_included = np.full(n, np.nan)
    protein_excluded = np.full(n, np.nan)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as temp:
        same_family_global = _make_memmap(os.path.join(temp, "same_family.dat"), n_same_family)
        cross_family_global = _make_memmap(os.path.join(temp, "cross_family.dat"), n_cross_family_positive)
        negative_global = _make_memmap(os.path.join(temp, "negative.dat"), n_negative)

        same_family_cursor = 0
        cross_family_cursor = 0
        negative_cursor = 0

        sf_iterator = (tqdm(superfamily_names, desc="Superfamilies") if show_progress else superfamily_names)

        for sf in sf_iterator:
            sf_mask = superfamilies == sf
            sf_indices = np.flatnonzero(sf_mask)
            sf_family_names = sorted(np.unique(families[sf_mask]))

            sf_pos_included_parts = []
            sf_pos_excluded_parts = []
            sf_neg_parts = []

            sf_col_position = np.full(n, -1, dtype=np.int32)
            sf_col_position[sf_indices] = np.arange(len(sf_indices), dtype=np.int32)

            for family in sf_family_names:
                family_indices = np.flatnonzero(families == family)
                family_mask = families == family

                family_pos_included_parts = []
                family_pos_excluded_parts = []
                family_neg_parts = []

                for start in range(0, len(family_indices), batch_size):
                    anchor_indices = family_indices[start:start + batch_size]

                    D = _distance_block(X[anchor_indices], X, distance_metric, X_sq)

                    pos_included = D[:, sf_mask].copy()
                    pos_included[np.arange(len(anchor_indices)), sf_col_position[anchor_indices]] = np.nan
                    pos_included = pos_included[np.isfinite(pos_included)]
                    pos_excluded = D[:, sf_mask & ~family_mask].ravel()

                    negatives = D[:, ~sf_mask].ravel()

                    family_pos_included_parts.append(pos_included)
                    family_pos_excluded_parts.append(pos_excluded)
                    family_neg_parts.append(negatives)

                    sf_pos_included_parts.append(pos_included)
                    sf_pos_excluded_parts.append(pos_excluded)
                    sf_neg_parts.append(negatives)

                    for row, i in enumerate(anchor_indices):
                        # Per-protein AUROC against all other proteins
                        p_inc = D[row, sf_mask].copy()
                        p_inc[sf_col_position[i]] = np.nan
                        p_inc = p_inc[np.isfinite(p_inc)]

                        p_exc = D[row, sf_mask & ~family_mask]
                        p_neg = D[row, ~sf_mask]

                        protein_included[i] = _distance_auc_parts([p_inc], [p_neg])
                        protein_excluded[i] = _distance_auc_parts([p_exc], [p_neg])
                        if i == n - 1:
                            continue

                        d_tail = D[row, i + 1:]
                        family_tail = families[i + 1:]
                        sf_tail = superfamilies[i + 1:]

                        same_family_mask = (family_tail == families[i])
                        same_sf_mask = (sf_tail == superfamilies[i])

                        cross_family_mask = (same_sf_mask & ~same_family_mask)
                        negative_mask = ~same_sf_mask
                        vals = d_tail[same_family_mask]

                        same_family_global[same_family_cursor:same_family_cursor + len(vals)] = vals
                        same_family_cursor += len(vals)
                        vals = d_tail[cross_family_mask]

                        cross_family_global[cross_family_cursor:cross_family_cursor + len(vals)] = vals
                        cross_family_cursor += len(vals)
                        vals = d_tail[negative_mask]

                        negative_global[negative_cursor:negative_cursor + len(vals)] = vals
                        negative_cursor += len(vals)
                    del D

                family_included[family] = _distance_auc_parts(family_pos_included_parts, family_neg_parts)
                family_excluded[family] = _distance_auc_parts(family_pos_excluded_parts, family_neg_parts)

            superfamily_included[sf] = _distance_auc_parts(sf_pos_included_parts, sf_neg_parts)
            superfamily_excluded[sf] = _distance_auc_parts(sf_pos_excluded_parts, sf_neg_parts)

        if same_family_cursor != n_same_family:
            raise RuntimeError(f"Expected {n_same_family} same-family pairs, wrote {same_family_cursor}")

        if cross_family_cursor != n_cross_family_positive:
            raise RuntimeError(f"Expected {n_cross_family_positive} cross-family positive pairs, wrote {cross_family_cursor}")

        if negative_cursor != n_negative:
            raise RuntimeError(f"Expected {n_negative} negative pairs, wrote {negative_cursor}")
        
        for arr in (same_family_global, cross_family_global, negative_global):
            if isinstance(arr, np.memmap):
                arr.flush()

        print("Computing overall AUROCs...")
        overall_included = _distance_auc_parts([same_family_global, cross_family_global], [negative_global])
        overall_excluded = _distance_auc_parts([cross_family_global], [negative_global])

    return {
        "overall_included": overall_included,
        "overall_excluded": overall_excluded,
        "protein_ids": [r["id"] for r in records],
        "protein_aurocs_included": protein_included.tolist(),
        "protein_aurocs_excluded": protein_excluded.tolist(),
        "family_names": family_names,
        "family_aurocs_included": [family_included[x] for x in family_names],
        "family_aurocs_excluded": [family_excluded[x] for x in family_names],
        "superfamily_names": superfamily_names,
        "superfamily_aurocs_included": [superfamily_included[x] for x in superfamily_names],
        "superfamily_aurocs_excluded": [superfamily_excluded[x] for x in superfamily_names],
    }

def parse_astral(fasta_path):
    records = []
    for r in SeqIO.parse(fasta_path, "fasta"):
        parts = r.description.split()
        domain_id, scop_code = parts[:2]
        c, f, sf, fam = scop_code.split(".")
        records.append({
            "id": domain_id, "seq": str(r.seq).upper(), "scop_code": scop_code,
            "class": c, "fold": f"{c}.{f}",
            "superfamily": f"{c}.{f}.{sf}", "family": f"{c}.{f}.{sf}.{fam}",
        })
    return records

# outputs/astral_benchmark/embeddings_astral40/
#    -> model
#       -> layer
#          -> domain_id
#             -> embedding_N{N}_fmax{fmax}.pkl
def make_do_embedding(method, layer, n_freqs, maxF, emb_model):
    path = os.path.join(COMPRESSED_ROOT, emb_model, f"layer_{layer}", method, f"{method_tag(method, n_freqs, maxF)}.pkl")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary: {path}")

    def do_embedding(record):
        try:
            emb = data[record["id"]]
        except KeyError:
            raise KeyError(f"Missing compressed embedding for {record['id']}: {path}") from None

        emb = np.asarray(emb, dtype=np.float32).ravel()

        if method != "2d_idct" and emb.shape != (PLM_dim[emb_model],):
            raise ValueError(f"Unexpected shape {emb.shape}: {record['id']}")

        if emb.size == 0 or not np.isfinite(emb).all():
            raise ValueError(f"Invalid embedding: {record['id']}")
        return emb

    return do_embedding


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ASTRAL40 embeddings without cross-validation")
    parser.add_argument("method")
    parser.add_argument("layer", type=int)
    parser.add_argument("distance_metric", choices=["l2", "cosine"])
    parser.add_argument("n_freqs", type=int)
    parser.add_argument("maxF", type=float)
    parser.add_argument("emb_model", choices=list(N_LAYERS))
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.layer <= N_LAYERS[args.emb_model]):
        parser.error(f"layer must be 0..{N_LAYERS[args.emb_model]} for {args.emb_model}")
    if args.emb_model in ("prottrans", "carp-640M") and args.layer != N_LAYERS[args.emb_model]:
        parser.error(f"Only final-layer embeddings are cached for {args.emb_model}")
    if args.method == CLS_METHOD and not args.emb_model.startswith("esm2-"):
        parser.error("CLS pooling is only supported for ESM models")
    if args.method == PROST_METHOD:
        if args.n_freqs < 2 or PLM_dim[args.emb_model] % args.n_freqs != 0:
            parser.error("2d_idct requires N >= 2 and N must divide the embedding dimension")
    elif args.method in SPECTRAL_METHODS:
        if args.n_freqs < 2:
            parser.error("Spectral methods require N >= 2")
        if args.method in PHYSICAL_METHODS and not (0 < args.maxF <= 0.5):
            parser.error("Physical methods require 0 < maxF <= 0.5")

    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    tag = method_tag(args.method, args.n_freqs, args.maxF)

    if args.method in POOL_METHODS or args.method == CLS_METHOD:
        method_name = args.method
    else:
        method_name = f"{args.method}_{tag}"

    out_dir = os.path.join(ASTRAL_ROOT, f"a40_aurocs_{args.distance_metric}_{args.emb_model}", f"layer_{args.layer}")
    out_path = os.path.join(out_dir, f"{method_name}.pkl")

    print("Model:", args.emb_model, "Method:", args.method, "Layer:", args.layer)
    print("Metric:", args.distance_metric, "N:", args.n_freqs, "maxF:", args.maxF)
    print("Embedding filename:", f"{tag}.pkl")
    print("Output:", out_path)

    if os.path.exists(out_path) and not args.overwrite:
        print("Already exists; skipping. Use --overwrite to recompute.")
        return

    records = parse_astral(ASTRAL_FASTA)

    results = evaluate_astral40(records, make_do_embedding(args.method, args.layer, args.n_freqs, args.maxF, args.emb_model), distance_metric=args.distance_metric, batch_size=args.batch_size, tmp_dir=args.tmp_dir)

    os.makedirs(out_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".partial_", suffix=".pkl", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, out_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
