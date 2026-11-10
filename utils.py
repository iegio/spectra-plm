import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm, trange
import random, os, gc, re, sys, requests
from Bio import SeqIO
from io import StringIO
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import dct, idct
from scipy.interpolate import interp1d
import gzip 
import glob 

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = '/scratch/gpfs/MONA/ml1875/data'
ARTIFACT_ROOT = f'{REPO_ROOT}/artifact'
ARTIFACT_DIR = {
    'dpi': f'{ARTIFACT_ROOT}/exp1',
    'ppi': f'{ARTIFACT_ROOT}/exp2',
    'fluorescence': f'{ARTIFACT_ROOT}/exp3',
    'stability': f'{ARTIFACT_ROOT}/exp4',
    'scope': f'{ARTIFACT_ROOT}/exp5',
    'beta_lactamase': f'{ARTIFACT_ROOT}/exp6',
    'solubility': f'{ARTIFACT_ROOT}/exp7',
}
DEEPPROTEIN_DIR = f'{ARTIFACT_ROOT}/deepprotein'

eps = 1e-12
def mean_pool(X):
    return np.mean(X, axis=0, dtype=np.float64).astype(np.float32)

def prost_idct(X, n_freqs, maxF=None):
    # Consistent with original PROST
    minmax = True
    quantize = True

    X = np.asarray(X, dtype=np.float32)
    L, D = X.shape
    N = int(n_freqs)
    M = D // N

    if D % N != 0:
        raise ValueError(f"Embedding dimension {D} must be divisible by N={N}")

    # Pad short protein sequences to N positions.
    if L < N:
        X = np.pad(X, ((0, N - L), (0, 0)), mode="constant")

    def scale(v):
        vmin, vmax = np.min(v), np.max(v)
        return (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)

    def idct_compress(v, n):
        coeff = dct(v, type=2, axis=0, norm="ortho")[:n]
        compressed = idct(coeff, type=2, axis=0, norm="ortho")

        if minmax:
            for j in range(compressed.shape[1]):
                compressed[:, j] = scale(compressed[:, j])

        return compressed

    Y = idct_compress(X, N)
    Y = idct_compress(Y.T, M).T
    out = Y.reshape(-1)

    if quantize:
        if not minmax:
            raise ValueError("Quantization requires minmax normalization.")
        out = np.trunc(out * 127).astype(np.int8).astype(np.float32)

    return out

def relative_dct(X, n_freqs, maxF=None):
    L, D = X.shape

    # Compute all available DCT coefficients
    coeffs = dct(X, type=2, axis=0, norm="ortho")

    trajectory = np.zeros((n_freqs, D), dtype=np.float32)

    # Number of available coefficients
    n_available = min(n_freqs, L)

    if n_available > 0:
        trajectory[:n_available] = coeffs[:n_available]

    # Coefficient transformation
    trajectory[1:] = np.abs(trajectory[1:])

    # Remove DC: mean pooling is handled by the independent PCA branch.
    trajectory = trajectory[1:]

    # Spectral normalization
    trajectory /= (np.linalg.norm(trajectory, axis=1, keepdims=True) + eps)

    assert trajectory.shape == (n_freqs - 1, D)
    return trajectory.reshape(-1).astype(np.float32)

def relative_dft(X, n_freqs, maxF):
    # For magnitude_zpad, append zeros to double the boundary-adjusted sequence length.
    zpad = False

    L, D = X.shape
    fft_len = 2 * X.shape[0] if zpad else X.shape[0]
    F = np.fft.rfft(X, n=fft_len, axis=0, norm="ortho")
    if zpad: F = F[:-1]  # L coefficients: omit the Nyquist bin

    # Number of available coefficients
    assert F.shape[0] == (X.shape[0] if zpad else X.shape[0] // 2 + 1)
    n_available = min(n_freqs, F.shape[0])
    F = F[:n_available]

    # Coefficient transform
    trajectory = np.abs(F)
    trajectory[0] = F[0].real

    # Zero-pad to n_freqs
    padded = np.zeros((n_freqs, D), dtype=np.float32)
    padded[:n_available] = trajectory
    trajectory = padded

    # Remove DC: mean pooling is handled by the independent PCA branch.
    trajectory = trajectory[1:]

    # Spectral normalization
    trajectory /= (np.linalg.norm(trajectory, axis=1, keepdims=True) + eps)

    return trajectory.reshape(-1).astype(np.float32)

def physical_dct(X, n_freqs, maxF):
    L, D = X.shape

    # Compute all available DCT coefficients
    C = dct(X, type=2, axis=0, norm="ortho")
    C = np.abs(C)

    freqs = np.arange(L, dtype=np.float64) / (2.0 * L)
    freq_grid = np.linspace(0.0, maxF, n_freqs, endpoint=False, dtype=np.float64)
    profile = interp1d(freqs, C, kind="linear", axis=0, bounds_error=False,
                       fill_value=0.0, assume_sorted=True)(freq_grid).astype(np.float32)

    # Remove DC: mean pooling is handled by the independent PCA branch.
    profile = profile[1:]

    # Spectral normalization
    profile /= (np.linalg.norm(profile, axis=1, keepdims=True) + eps)
    assert profile.shape == (n_freqs - 1, D)
    return profile.reshape(-1).astype(np.float32)

def physical_dft(X, n_freqs, maxF):
    # For magnitude_zpad, append zeros to double the boundary-adjusted sequence length.
    zpad = False

    _, D = X.shape
    fft_len = 2 * X.shape[0] if zpad else X.shape[0]
    F = np.fft.rfft(X, n=fft_len, axis=0, norm="ortho")

    # Physical-frequency coordinates
    freqs = np.fft.rfftfreq(fft_len, d=1.0)[:len(F)]
    freq_grid = np.linspace(0.0, maxF, n_freqs, endpoint=False, dtype=np.float64)

    # Coefficient transformation before interpolation
    trajectory = np.abs(F)

    # Interpolate onto common physical-frequency grid
    profile = interp1d(freqs, trajectory, kind="linear", axis=0, bounds_error=False,
                       fill_value=0.0, assume_sorted=True)(freq_grid).astype(np.float32)

    # Remove DC: mean pooling is handled by the independent PCA branch.
    profile = profile[1:]

    profile /= np.linalg.norm(profile, axis=1, keepdims=True) + eps

    assert profile.shape == (n_freqs - 1, D)
    return profile.reshape(-1).astype(np.float32)

def make_artifact_dirs():
    for d in ARTIFACT_DIR.values():
        os.makedirs(d, exist_ok=True)

# PyTorch utilities
def freeze_module(module, unfreeze=False):
    for param in module.parameters():
        param.requires_grad = unfreeze

def set_seed(seed=2603):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def inverse_sigmoid(x, temp):
    return torch.log(x / (1. - x)) / temp

def count_params(net):
    return sum(p.numel() for p in net.parameters())

# String processing utilities
def write_fasta(seq_dict, fasta_file, chunk=50):
    with open(fasta_file, 'w') as f:
        for seq_name, seq in seq_dict.items():
            chunk_seq = '\n'.join([seq[i:i+chunk] for i in range(0, len(seq), chunk)])
            f.write(f'>{seq_name}\n')
            f.write(f'{chunk_seq}\n')

def extract_kmer(s, k, o=None, plm_format=True):
    o = k - 1 if o is None else o
    kmers = [s[j:j + k] for j in range(0, len(s) - k + 1, k - o)]
    if plm_format:
        return [' '.join(kmer) for kmer in kmers]
    return kmers

def fetch_single_uid(uid):
    base_url="http://www.uniprot.org/uniprot"
    try:
        seq = StringIO(''.join(requests.post(f'{base_url}/{uid}.fasta').text))
        seq = list(SeqIO.parse(seq,'fasta'))
        return (uid, str(seq[0].seq))
    except Exception:
        print(f'failed to retrieve uid {uid}')
        return (uid, None)

def fetch_single_pid(pid):
    base_url = "https://www.ebi.ac.uk/pdbe/entry/pdb"
    try:
        seqs = StringIO(''.join(requests.post(f'{base_url}/{pid}/fasta').text))
        seqs = list(SeqIO.parse(seqs,'fasta'))
        if len(seqs) > 0:
            return (pid, seqs)
        else:
            return (pid, None)
    except Exception as e:
        print(e)
        return (pid, None)

def download_uniprot(uid_list):
    with ThreadPoolExecutor() as executor:
        retrieved_seqs = list(tqdm(executor.map(fetch_single_uid, uid_list), total=len(uid_list)))

    return {uid: seq for (uid, seq) in retrieved_seqs if seq is not None}

def download_pdb(pid_list):
    with ThreadPoolExecutor(max_workers=len(pid_list) // 50) as executor:
        retrieved_seqs = list(tqdm(executor.map(fetch_single_pid, pid_list), total=len(pid_list)))

    success = {pid: seq for (pid, seq) in retrieved_seqs if seq is not None}
    failure = [pid for (pid, seq) in retrieved_seqs if seq is None]
    return success, failure

def read_fasta_gz(fn, return_headers=False):
    seq_list = []
    header_list = []

    with gzip.open(fn, "rt") as f:
        line = True
        header = None
        running_seq = ""
        entry_counter = 0 
        while line:
            line = f.readline().replace("\n", "")
            if line is None or len(line) == 0:
                header_list.append(header.replace(">", ""))
                seq_list.append(running_seq)
                running_seq = ""
                continue

            if line[0] == ">":
                if header is not None:
                    seq_list.append(running_seq)
                    header_list.append(header.replace(">", ""))
                    running_seq = ""
                    header = line
                else:
                    header = line
                    entry_counter += 1

            else:
                running_seq = running_seq + line
    f.close()

    if return_headers:
        return header_list, seq_list
    return seq_list

def read_fasta(fn, return_headers=False):
    if ".gz" in fn:
        if return_headers:
            header_list, seq_list = read_fasta_gz(fn, return_headers=return_headers)
            return header_list, seq_list
        else:
            seq_list = read_fasta_gz(fn, return_headers=return_headers)
            return seq_list

    seq_list = []
    header_list = []

    with open(fn, "r") as f:
        line = True
        header = None
        running_seq = ""
        while line:
            line = f.readline().replace("\n", "")
            if line is None or len(line) == 0:
                header_list.append(header.replace(">", ""))
                seq_list.append(running_seq)
                running_seq = ""
                continue

            if line[0] == ">":
                if header is not None:
                    seq_list.append(running_seq)
                    header_list.append(header.replace(">", ""))
                    running_seq = ""
                    header = line
                else:

                    header = line

            else:
                running_seq = running_seq + line

    if return_headers:
        return header_list, seq_list
    return seq_list

def parse_astral(fasta_path, n=None):
    # Returns list of dicts: {id, seq, scop_code, class, fold, superfamily, family}
    records = []

    for i, r in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if n is not None and i >= n:
            break

        parts = r.description.split()
        domain_id = parts[0]
        scop_code = parts[1]

        c, f, sf, fam = scop_code.split(".")

        records.append({"id": domain_id, "seq": str(r.seq).upper(), "scop_code": scop_code, "class": c, "fold": f"{c}.{f}", "superfamily": f"{c}.{f}.{sf}", "family": f"{c}.{f}.{sf}.{fam}"})

    return records
