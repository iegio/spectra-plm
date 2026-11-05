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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get('BOM_DATA_ROOT', '/scratch/gpfs/MONA/ml1875/data')
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