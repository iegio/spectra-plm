from pathlib import Path

# Input datasets
UR50_FASTA_PATH = Path("datasets/uniref50_042021.fasta.gz")
UR50_SAMPLE_FASTA = Path("datasets/SCOPe/randomly_chosen_proteins_UR50.fasta")
ASTRAL_FASTA = Path("datasets/SCOPe/astral40.fa")

# Output directories
OUTPUT_ROOT = Path("outputs")
ASTRAL_ROOT = OUTPUT_ROOT/"astral_benchmark"
COMPRESSED_ROOT = ASTRAL_ROOT/"compress_astral40"
FITS_ROOT = OUTPUT_ROOT/"pca_tools"/"fits"
UR50_EMBED_ROOT = OUTPUT_ROOT/"pca_tools"/"embeddings_extractions"

# Compression methods
POOL_METHODS = {"mean_pool", "mean_pool_standardized"}
CLS_METHOD = "cls_pool"
PROST_METHOD = "2d_idct"

RELATIVE_METHODS = {"dct_relative", "dft_relative"}
PHYSICAL_METHODS = {"dct_physical", "dft_physical"}
SPECTRAL_METHODS = RELATIVE_METHODS | PHYSICAL_METHODS

COMPRESS_METHODS = POOL_METHODS | SPECTRAL_METHODS | {PROST_METHOD}
EVALUATION_METHODS = COMPRESS_METHODS | {CLS_METHOD}

def method_tag(method, N=None, maxF=None):
    if method in POOL_METHODS or method == CLS_METHOD:
        return method

    if N is None:
        raise ValueError(f"{method} requires N")

    if method in PHYSICAL_METHODS:
        if maxF is None:
            raise ValueError(f"{method} requires maxF")
        return f"N{N}_fmax{maxF:g}".replace(".", "p")

    if method in RELATIVE_METHODS or method == PROST_METHOD:
        return f"N{N}"

    raise ValueError(f"Unknown method: {method}")
