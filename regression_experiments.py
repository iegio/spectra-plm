from utils import *
from plm_wrapper import *
from nets import *
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np

import torch.nn.functional as F

DEEP_PROTEIN_ROOT ='datasets/DeepProtein'
sys.path.append(os.path.dirname(DEEP_PROTEIN_ROOT))
artifact_dir = "outputs/deepprotein/artifact"
os.makedirs(artifact_dir, exist_ok=True)

from DeepProtein.dataset import *

# SPECTRAL FUNCTIONS
configs_funcs = {
    'dct_relative': relative_dct,
    'dft_relative': relative_dft,
    'dct_physical': physical_dct,
    'dft_physical': physical_dft,
    'prost_idct': prost_idct,
}

PCA_SEED = 42
def fit_branched_pca(train_mean, train_spectral, emb_dim, seed=PCA_SEED):
    pca_dim = emb_dim // 2

    mean_scaler = StandardScaler()
    mean_pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=seed)
    mean_pca.fit(mean_scaler.fit_transform(train_mean))

    spectral_scaler = StandardScaler()
    spectral_pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=seed)
    spectral_pca.fit(spectral_scaler.fit_transform(train_spectral))

    return mean_scaler, mean_pca, spectral_scaler, spectral_pca

def transform_branched_pca(mean_features, spectral_features, mean_scaler, mean_pca, spectral_scaler, spectral_pca):
    mean_features = mean_pca.transform(mean_scaler.transform(mean_features))
    spectral_features = spectral_pca.transform(spectral_scaler.transform(spectral_features))
    return torch.tensor(np.concatenate([mean_features, spectral_features], axis=1), dtype=torch.float32)

class LLMPredictor(nn.Module):
    def __init__(self, **kwargs):
        super(LLMPredictor, self).__init__()
        self.task = kwargs['task']
        D = PLM_dim[kwargs['emb_model']]
        fc_input_dim = 2 * D if self.task == 'ppi_affinity' else D
        self.fc = MLP([fc_input_dim, 512, 256, 1]).to('cuda')

    def forward(self, x, x2=None):
        if self.task == 'ppi_affinity':
            assert x2 is not None
            x = torch.cat([x, x2], dim=1)
        return self.fc(x)

@torch.no_grad()
def evaluate_model(model, data, fold='test', task=None):
    model.eval()

    eval_label = data[f'{fold}_y']
    n_test = len(eval_label)
    batch_size = 128

    bar = trange(0, n_test, batch_size)
    bar.set_description_str(f'Evaluating on {fold} fold:')
    
    y_pred = []
    for i in bar:
        end = min(i + batch_size, n_test)
        if task == 'ppi_affinity':
            x1 = data[f'{fold}_x1'][i:end].to('cuda')
            x2 = data[f'{fold}_x2'][i:end].to('cuda')
            y_pred.append(model(x1, x2))
            del x1, x2
        else:
            x = data[f'{fold}_x'][i:end].to('cuda')
            y_pred.append(model(x))
            del x
        torch.cuda.empty_cache()

    y_pred = torch.cat(y_pred).flatten().cpu()
    eval_label = eval_label.flatten().cpu()

    score, _ = spearmanr(y_pred.numpy(), eval_label.numpy())

    return y_pred, float(score)

def format_float_for_filename(x):
    return f"{x:g}".replace(".", "p")

def experiment_tag(method, layer_ix, N, maxF):
    if method in ['mean', 'mean_standardized', 'cls']:
        return f'{method}_layer{layer_ix}'
    return f'{method}_layer{layer_ix}_N{N}_fmax{format_float_for_filename(maxF)}'

def train_model(task, emb_model, method, layer_ix, N, maxF, seed, **kwargs):
    set_seed(seed)

    tag = experiment_tag(method, layer_ix, N, maxF)
    exp_name = f'dp_{task}_{emb_model}_{tag}'

    save_path = f'{artifact_dir}/{exp_name}_seed{seed}.pt'
    checkpoint_path = f'{artifact_dir}/{exp_name}_seed{seed}_best.pt'

    data = load_data(task, emb_model, method, layer_ix, N, maxF, seed)

    print(f"Train: {len(data['train_y']):,}")
    print(f"Valid: {len(data['valid_y']):,}")
    print(f"Test:  {len(data['test_y']):,}")

    set_seed(seed)
    model = LLMPredictor(emb_model=emb_model, task=task)

    opt = AdamW(model.parameters(), lr=kwargs.get('lr', 1e-4))
    scheduler = ReduceLROnPlateau(opt, 'min', min_lr=1e-6, factor=0.95, patience=1)

    n_train = len(data['train_y'])
    batch_size = kwargs.get('batch_size', 256)
    n_epochs = kwargs.get('n_epochs', 400)
    interval = kwargs.get('interval', 1)

    metric_name = 'rho'

    history = {
        'y_pred': [],
        'rho': [],
        'pr_auc': [],
        'valid_rho': [],
        'valid_pr_auc': [],
        'training_time': [],
        'eval_epochs': [],
        'metric': metric_name,
        'best_epoch': None,
        'best_valid_score': -np.inf,
        'test_score': None,
        'final_epoch': None,
        'final_test_score': None,
        'best_test_score': None,
    }

    best_score = -np.inf
    best_epoch = None

    def validate_and_save(epoch):
        nonlocal best_score, best_epoch

        model.eval()
        _, valid_score = evaluate_model(model, data, fold='valid', task=task)

        if not np.isfinite(valid_score):
            raise ValueError(f'Non-finite validation {metric_name} at epoch {epoch}')

        history['eval_epochs'].append(epoch)

        history['valid_rho'].append(valid_score)
        print(f'Epoch {epoch}: valid Spearman rho = {valid_score:.5f}')

        scheduler.step(-valid_score)

        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch

            history['best_epoch'] = best_epoch
            history['best_valid_score'] = best_score

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'valid_score': valid_score,
                'metric': metric_name,
                'task': task,
                'seed': seed,
            }, checkpoint_path)

            print(f'New best checkpoint: epoch {epoch}')

        torch.save(history, save_path)

    # Evaluate the untrained model at epoch 0, as in your original code.
    validate_and_save(0)

    for ep in range(n_epochs):
        start_time = time.time()
        model.train()

        idx = torch.randperm(n_train)
        bar = trange(0, n_train, batch_size)
        bar.set_description_str(f'Epoch {ep + 1}:')

        for i in bar:
            opt.zero_grad()

            idx_batch = idx[i:min(i + batch_size, n_train)]
            y_batch = data['train_y'][idx_batch].to('cuda').flatten().float()

            if task == 'ppi_affinity':
                x1 = data['train_x1'][idx_batch].to('cuda')
                x2 = data['train_x2'][idx_batch].to('cuda')
                y_pred = model(x1, x2).flatten()
            else:
                x = data['train_x'][idx_batch].to('cuda')
                y_pred = model(x).flatten()

            loss = F.huber_loss(y_pred, y_batch)

            bar.set_postfix_str(f'Batch Loss = {loss.item():.5f}')

            loss.backward()
            opt.step()

            if task == 'ppi_affinity':
                del x1, x2
            else:
                del x

        ep_time = time.time() - start_time
        history['training_time'].append(ep_time)

        print(f'Ep {ep + 1} training time: {ep_time:.2f}s')

        # Evaluate after training, including the final epoch.
        if (ep + 1) % interval == 0 or ep == n_epochs - 1:
            validate_and_save(ep + 1)

    # Save the model after exactly n_epochs completed training epochs.
    final_checkpoint_path = f'{artifact_dir}/{exp_name}_seed{seed}_final.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': n_epochs,
        'metric': metric_name,
        'task': task,
        'seed': seed,
    }, final_checkpoint_path)

    # Evaluate the final-epoch model on the test set.
    final_pred, final_score = evaluate_model(model, data, fold='test', task=task)

    history['final_epoch'] = n_epochs
    history['final_test_score'] = float(final_score)

    # Restore the model selected by validation performance.
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate the validation-selected model on the test set.
    best_pred, best_score_test = evaluate_model(model, data, fold='test', task=task)

    history['best_epoch'] = int(checkpoint['epoch'])
    history['best_valid_score'] = float(checkpoint['valid_score'])
    history['best_test_score'] = float(best_score_test)

    # Keep test_score as the best-validation-selected test result.
    history['test_score'] = float(best_score_test)
    history['test_epoch'] = int(checkpoint['epoch'])
    history['evaluation_protocol'] = 'best_validation'

    history['y_pred'].append(best_pred)

    history['rho'].append(float(best_score_test))

    print(f'Final epoch {n_epochs}: test Spearman rho = {final_score:.5f}')
    print(f'Best validation epoch {checkpoint["epoch"]}: test Spearman rho = {best_score_test:.5f}')

    print(f'Best validation score: {checkpoint["valid_score"]:.5f}')

    torch.save(history, save_path)

    del data, model
    torch.cuda.empty_cache()

    return history

def load_data(task, emb_model_name, method, layer_ix, N, maxF):
    def collate_ppi(dataset):
        x1, x2, y = [], [], []
        for i in range(len(dataset)):
            seq1, seq2, target = dataset[i]
            x1.append(seq1)
            x2.append(seq2)
            y.append(float(target))
        return x1, x2, torch.tensor(y, dtype=torch.float32)
    deep_prot_data_dir = f'{DEEP_PROTEIN_ROOT}/data'
    tag = experiment_tag(method, layer_ix, N, maxF)
    exp_name = f'dp_{task}_{emb_model_name}_{tag}'
    save_path = f'{DATA_ROOT}/{task}/{exp_name}.pt'
    print(exp_name, save_path)

    if os.path.exists(save_path):
        data = torch.load(save_path, weights_only=False)
        if task == 'ppi_affinity':
            assert 'train_x1' in data and 'train_x2' in data
            assert 'valid_x1' in data and 'valid_x2' in data
            assert 'test_x1' in data and 'test_x2' in data
        return data
    
    if task == 'fluorescence':
        train, valid, test = FluorescenceDataset(deep_prot_data_dir, 'train'), FluorescenceDataset(deep_prot_data_dir, 'valid'), FluorescenceDataset(deep_prot_data_dir, 'test')
    elif task == 'ppi_affinity':
        train, valid, test = PPI_Affinity(deep_prot_data_dir, 'train'), PPI_Affinity(deep_prot_data_dir, 'valid'), PPI_Affinity(deep_prot_data_dir, 'test')
    else:
        raise ValueError(f'Unknown task: {task}')

    if task == 'ppi_affinity':
        train_x1, train_x2, train_y = collate_ppi(train)
        valid_x1, valid_x2, valid_y = collate_ppi(valid)
        test_x1, test_x2, test_y = collate_ppi(test)
    else:
        train_x, train_y, _ = collate_fn(train)
        valid_x, valid_y, _ = collate_fn(valid)
        test_x, test_y, _ = collate_fn(test)

    # config = configs[method][config_ix]
    # spectral_func = configs_funcs[method]
    is_avg = method in ['mean', 'mean_standardized', 'cls']
    spectral_func = None if is_avg else configs_funcs[method]

    emb_model = PLM[emb_model_name]()
    emb_model.eval()

    if is_avg:
        def embed_avg(fold_x, fold, protein_ix=None, collect_stats=False, chunk_size=1000):
            pool_dir = 'cls' if method == 'cls' else 'mean_shared'
            cache_dir = f'{DATA_ROOT}/chunks/{task}/{emb_model_name}/{pool_dir}/layer{layer_ix}'
            os.makedirs(cache_dir, exist_ok=True)

            prefix = f'{fold}_protein{protein_ix}' if protein_ix is not None else fold

            means_all = []
            residue_sum_all = None
            residue_sumsq_all = None
            n_residues_all = 0

            for start in range(0, len(fold_x), chunk_size):
                end = min(start + chunk_size, len(fold_x))
                chunk_path = f'{cache_dir}/{prefix}_{start}_{end}.npz'

                if not os.path.exists(chunk_path):
                    means = []
                    residue_sum = np.zeros(PLM_dim[emb_model_name], dtype=np.float64)
                    residue_sumsq = np.zeros(PLM_dim[emb_model_name], dtype=np.float64)
                    n_residues = 0

                    with torch.no_grad():
                        for seq in tqdm(fold_x[start:end]):
                            x_emb = emb_model([seq], hidden_layer=layer_ix, return_cls=True) if method == 'cls' else emb_model([seq], hidden_layer=layer_ix)
                            X = x_emb[0].detach().cpu().numpy().astype(np.float64)

                            assert X.shape == (1 if method == 'cls' else len(seq), PLM_dim[emb_model_name])
                            
                            means.append(X.mean(axis=0))
                            residue_sum += X.sum(axis=0)
                            residue_sumsq += np.square(X).sum(axis=0)
                            n_residues += X.shape[0]

                    tmp_path = chunk_path + '.tmp.npz'
                    np.savez(
                        tmp_path,
                        means=np.stack(means),
                        residue_sum=residue_sum,
                        residue_sumsq=residue_sumsq,
                        n_residues=n_residues
                    )
                    os.replace(tmp_path, chunk_path)

                    print(f'Saved {chunk_path}')
                else:
                    print(f'Skipping completed chunk: {chunk_path}')

                with np.load(chunk_path) as chunk:
                    means_all.append(torch.from_numpy(chunk['means']))
                    if collect_stats:
                        chunk_sum = chunk['residue_sum']
                        chunk_sumsq = chunk['residue_sumsq']

                        if residue_sum_all is None:
                            residue_sum_all = np.zeros_like(chunk_sum)
                            residue_sumsq_all = np.zeros_like(chunk_sumsq)

                        residue_sum_all += chunk_sum
                        residue_sumsq_all += chunk_sumsq
                        n_residues_all += int(chunk['n_residues'])

            means_all = torch.cat(means_all, dim=0)

            return means_all
        
        if task == 'ppi_affinity':
            train_emb1 = embed_avg(train_x1, 'train', protein_ix=1, collect_stats=True)
            train_emb2 = embed_avg(train_x2, 'train', protein_ix=2, collect_stats=True)

            valid_emb1 = embed_avg(valid_x1, 'valid', protein_ix=1)
            valid_emb2 = embed_avg(valid_x2, 'valid', protein_ix=2)

            test_emb1 = embed_avg(test_x1, 'test', protein_ix=1)
            test_emb2 = embed_avg(test_x2, 'test', protein_ix=2)

            if method == 'mean_standardized':
                train_all = torch.cat([train_emb1, train_emb2])
                mu = train_all.mean(0)
                sigma = train_all.std(0, unbiased=False)

        else:
            train_emb = embed_avg(train_x, 'train', collect_stats=True)

            valid_emb = embed_avg(valid_x, 'valid')
            test_emb = embed_avg(test_x, 'test')

            if method == 'mean_standardized':
                mu = train_emb.mean(0)
                sigma = train_emb.std(0, unbiased=False)

        if method == 'mean_standardized':
            sigma = torch.clamp(sigma, min=1e-12)
            standardize = lambda x: ((x - mu) / sigma).float()

            if task == 'ppi_affinity':
                train_emb1, train_emb2 = standardize(train_emb1), standardize(train_emb2)
                valid_emb1, valid_emb2 = standardize(valid_emb1), standardize(valid_emb2)
                test_emb1, test_emb2 = standardize(test_emb1), standardize(test_emb2)
            else:
                train_emb, valid_emb, test_emb = standardize(train_emb), standardize(valid_emb), standardize(test_emb)

        if method in ['mean', 'cls']:
            if task == 'ppi_affinity':
                train_emb1, train_emb2 = train_emb1.float(), train_emb2.float()
                valid_emb1, valid_emb2 = valid_emb1.float(), valid_emb2.float()
                test_emb1, test_emb2 = test_emb1.float(), test_emb2.float()
            else:
                train_emb, valid_emb, test_emb = train_emb.float(), valid_emb.float(), test_emb.float()

    def embed_spectral(fold_x, fold, protein_ix=None, chunk_size=10000):
        cache_dir = f'{DATA_ROOT}/chunks/{task}/{emb_model_name}/{method}/layer{layer_ix}/N{N}/fmax{maxF}'
        os.makedirs(cache_dir, exist_ok=True)

        prefix = f'{fold}_protein{protein_ix}' if protein_ix is not None else fold

        mean_features, spectral_features = [], []

        for start in range(0, len(fold_x), chunk_size):
            end = min(start + chunk_size, len(fold_x))
            chunk_path = f'{cache_dir}/{prefix}_{start}_{end}.npz'

            if os.path.exists(chunk_path):
                print(f'Skipping completed chunk: {chunk_path}')
                continue

            chunk_mean, chunk_spectral = [], []

            with torch.no_grad():
                for seq in tqdm(fold_x[start:end]):
                    x_emb = emb_model([seq], hidden_layer=layer_ix)
                    X = x_emb[0].detach().cpu().numpy()

                    assert X.shape == (len(seq), PLM_dim[emb_model_name])

                    if method != 'prost_idct':
                        chunk_mean.append(X.mean(0))

                    chunk_spectral.append(spectral_func(X, n_freqs=N, maxF=maxF))

            tmp_path = chunk_path + '.tmp.npz'
            np.savez(
                tmp_path,
                mean=np.stack(chunk_mean) if chunk_mean else np.empty((0,)),
                spectral=np.stack(chunk_spectral)
            )
            os.replace(tmp_path, chunk_path)
            

            print(f'Saved {chunk_path}')

        for start in range(0, len(fold_x), chunk_size):
            end = min(start + chunk_size, len(fold_x))
            chunk_path = f'{cache_dir}/{prefix}_{start}_{end}.npz'

            with np.load(chunk_path) as chunk:
                if method != 'prost_idct':
                    mean_features.append(chunk['mean'])
                spectral_features.append(chunk['spectral'])

        mean_features = np.concatenate(mean_features) if mean_features else None
        spectral_features = np.concatenate(spectral_features)

        return mean_features, spectral_features
    if not is_avg:
        if task == 'ppi_affinity':
            print('Embedding train protein 1...')
            train_mean1, train_spectral1 = embed_spectral(train_x1, 'train', protein_ix=1)
            print('Embedding train protein 2...')
            train_mean2, train_spectral2 = embed_spectral(train_x2, 'train', protein_ix=2)
            print('Embedding valid protein 1...')
            valid_mean1, valid_spectral1 = embed_spectral(valid_x1, 'valid', protein_ix=1)
            print('Embedding valid protein 2...')
            valid_mean2, valid_spectral2 = embed_spectral(valid_x2, 'valid', protein_ix=2)
            print('Embedding test protein 1...')
            test_mean1, test_spectral1 = embed_spectral(test_x1, 'test', protein_ix=1)
            print('Embedding test protein 2...')
            test_mean2, test_spectral2 = embed_spectral(test_x2, 'test', protein_ix=2)

            if method == 'prost_idct':
                train_emb1 = torch.tensor(train_spectral1, dtype=torch.float32)
                train_emb2 = torch.tensor(train_spectral2, dtype=torch.float32)
                valid_emb1 = torch.tensor(valid_spectral1, dtype=torch.float32)
                valid_emb2 = torch.tensor(valid_spectral2, dtype=torch.float32)
                test_emb1 = torch.tensor(test_spectral1, dtype=torch.float32)
                test_emb2 = torch.tensor(test_spectral2, dtype=torch.float32)
            else:
                train_mean_all = np.concatenate([train_mean1, train_mean2], axis=0)
                train_spectral_all = np.concatenate([train_spectral1, train_spectral2], axis=0)
                mean_scaler, mean_pca, spectral_scaler, spectral_pca = fit_branched_pca(train_mean_all, train_spectral_all, PLM_dim[emb_model_name])
                transform = lambda m, s: transform_branched_pca(m, s, mean_scaler, mean_pca, spectral_scaler, spectral_pca)

                train_emb1, train_emb2 = transform(train_mean1, train_spectral1), transform(train_mean2, train_spectral2)
                valid_emb1, valid_emb2 = transform(valid_mean1, valid_spectral1), transform(valid_mean2, valid_spectral2)
                test_emb1, test_emb2 = transform(test_mean1, test_spectral1), transform(test_mean2, test_spectral2)
        else:
            print('Embedding train...')
            train_mean, train_spectral = embed_spectral(train_x, 'train')
            print('Embedding valid...')
            valid_mean, valid_spectral = embed_spectral(valid_x, 'valid')
            print('Embedding test...')
            test_mean, test_spectral = embed_spectral(test_x, 'test')

            if method == 'prost_idct':
                train_emb = torch.tensor(train_spectral, dtype=torch.float32)
                valid_emb = torch.tensor(valid_spectral, dtype=torch.float32)
                test_emb = torch.tensor(test_spectral, dtype=torch.float32)
            else:
                mean_scaler, mean_pca, spectral_scaler, spectral_pca = fit_branched_pca(train_mean, train_spectral, PLM_dim[emb_model_name])
                transform = lambda m, s: transform_branched_pca(m, s, mean_scaler, mean_pca, spectral_scaler, spectral_pca)
                train_emb, valid_emb, test_emb = transform(train_mean, train_spectral), transform(valid_mean, valid_spectral), transform(test_mean, test_spectral)

    del emb_model
    torch.cuda.empty_cache()

    if task == 'ppi_affinity':
        assert train_emb1.shape[1] == PLM_dim[emb_model_name] and train_emb2.shape[1] == PLM_dim[emb_model_name]
        data = {'train_x1': train_emb1, 'train_x2': train_emb2, 'train_y': train_y,
                'valid_x1': valid_emb1, 'valid_x2': valid_emb2, 'valid_y': valid_y,
                'test_x1': test_emb1, 'test_x2': test_emb2, 'test_y': test_y}
    else:
        assert train_emb.shape[1] == PLM_dim[emb_model_name]
        assert valid_emb.shape[1] == PLM_dim[emb_model_name]
        assert test_emb.shape[1] == PLM_dim[emb_model_name]
        data = {'train_x': train_emb, 'train_y': train_y, 'valid_x': valid_emb, 'valid_y': valid_y, 'test_x': test_emb, 'test_y': test_y}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    return data

# python file.py TASK METHOD LAYER_IX N MAXF EMB_MODEL SEED
if __name__ == '__main__':
    task = sys.argv[1]
    method = sys.argv[2]
    LAYER_IX = int(sys.argv[3])
    N = int(sys.argv[4])
    MAXF = float(sys.argv[5])
    emb_model = sys.argv[6]
    seed = int(sys.argv[7])

    assert task in ['fluorescence', 'ppi_affinity']
    assert method in ['dct_relative', 'dft_relative', 'dct_physical', 'dft_physical', 'prost_idct', 'mean', 'mean_standardized', 'cls']
    assert method != 'cls' or emb_model.startswith('esm2-')
    assert emb_model in ['esm2-650M', 'prottrans']

    kwargs = {'n_epochs': 100, 'interval': 1, 'batch_size': 48, 'lr': 1e-4}

    
    print("TASK:", task)
    print("METHOD:", method)
    print("LAYER:", LAYER_IX)
    print("N_FREQS:", N)
    print("MAXF:", MAXF)
    print("MODEL:", emb_model)
    print("SEED:", seed)

    train_model(task, emb_model, method, LAYER_IX, N, MAXF, seed, **kwargs)
    