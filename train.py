import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

from loss import MSEBCELoss, HuberBCELoss
from model.fsrgnn import FSRGNN
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from data.make_dataset import make_dataset_one_group
from utils import get_feature_stats_filepath, get_train_test_indices, get_wet_idx


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(area_name: str, mode: str, identifier: str, val_group: int, epochs_to_do: int, seed: int = 42):
    DEVICE = torch.device('cuda')
    NUM_EPOCHS = epochs_to_do
    CHECKPOINT_DIR = f'model/{area_name}'
    CONFIG_PATH = f'model/configs/{identifier}_config.json'
    LATEST_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/{identifier}_Val_{val_group}_latest.pt'
    BEST_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/{identifier}_Val_{val_group}_best.pt'
    LOSS_STATS_PATH = f'{CHECKPOINT_DIR}/{identifier}_Val_{val_group}_loss_stats.json'

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(seed)

    # ==============================
    # config and checkpoint loading
    # ==============================

    if mode == 'new':
        print(f'Training new model on {area_name} on validation group {val_group} using config {CONFIG_PATH}.')

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            model_config = config['model_config']
            loss_func_name = config['loss_func']
            loss_config = config['loss_config']
            batch_size = config['batch_size']
            learning_rate = config['learning_rate']
            weight_decay = config['weight_decay']
            sch_factor = config['sch_factor']
            sch_patience = config['sch_patience']
            min_lr = config['min_lr']
            t_interval = config['t_interval']

        start_epoch = 1
        best_val_loss = float('inf')
        loss_stats = {
            'loss_func': loss_func_name,
            'epoch': [],
            'train_loss': [],
            'train_loss_components': {},
            'val_loss': [],
            'val_loss_components': {}
        }

    elif mode == 'continue':
        print(f'Resuming training of model on {area_name} on validation group {val_group} using config {CONFIG_PATH}.')

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            model_config = config['model_config']
            loss_func_name = config['loss_func']
            loss_config = config['loss_config']
            batch_size = config['batch_size']
            learning_rate = config['learning_rate']
            weight_decay = config['weight_decay']
            sch_factor = config['sch_factor']
            sch_patience = config['sch_patience']
            min_lr = config['min_lr']
            t_interval = config['t_interval']

        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        model_state_dict = checkpoint['model_state_dict']
        optimiser_state_dict = checkpoint['optimiser_state_dict']
        scheduler_state_dict = checkpoint['scheduler_state_dict']
        random_state = checkpoint['random_state']

        with open(LOSS_STATS_PATH, 'r') as f:
            loss_stats = json.load(f)
            assert loss_stats['loss_func'] == loss_func_name, \
                f'Loss function mismatch between config ({loss_stats['loss_func']}) and checkpoint ({loss_func_name}).'
        
    else:
        raise Exception(f'Unknown training mode: {mode}, should be "new" or "continue".')

    # ==============================
    # create dataset and dataloaders
    # ==============================

    dataset = make_dataset_one_group(area_name=area_name, val_group=val_group, mode='train')
    wet_idx = torch.from_numpy(get_wet_idx(area_name=area_name, val_group=val_group)).long().to(DEVICE)
    idx_train, idx_val = get_train_test_indices(area_name=area_name, val_group=val_group)
    assert idx_train.shape[0] + idx_val.shape[0] == len(dataset), \
        f'Dataset size {len(dataset)} does not match number of indices {idx_train.shape[0] + idx_val.shape[0]} in train-test split.'

    # optionally filter indices to only use every <t_interval> timesteps from each event
    assert t_interval >= 1, f'Invalid t_interval of {t_interval}, should be >=1.'
    if t_interval > 1:
        event_ends = np.cumsum(np.asarray(dataset.num_timesteps, dtype=np.int32))
        idx_train = filter_timesteps(idx_train, event_ends, t_interval)
        idx_val = filter_timesteps(idx_val, event_ends, t_interval)
    print(f'----- Using timestep interval {t_interval}: Training samples: {idx_train.shape}, Validation samples: {idx_val.shape} -----')

    train_dataset = Subset(dataset, idx_train)
    val_dataset = Subset(dataset, idx_val)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        generator=generator)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True)

    # ==============================
    # create model, loss, optimiser and scheduler
    # ==============================

    print('Creating model...')
    model = FSRGNN(**model_config, feat_stats_file=get_feature_stats_filepath(area_name, val_group))
    model.init_static_features(dataset.STATIC_FEATURES_PATH)
    model.to(DEVICE)
    num_hf_nodes = model.num_hf_nodes

    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=sch_factor, patience=sch_patience, min_lr=min_lr)

    if loss_func_name == 'mse':
        loss_func = nn.MSELoss()
        loss_component_keys = {'mse'}
    elif loss_func_name == 'mse_bce':
        loss_func = MSEBCELoss(**loss_config)
        loss_component_keys = loss_func.LOSS_COMPONENT_KEYS
    else:
        raise Exception(f'Unknown loss function {loss_func_name}.')

    if mode == 'new':
        for key in loss_component_keys:
            loss_stats['train_loss_components'][key] = []
            loss_stats['val_loss_components'][key] = []

    elif mode == 'continue':
        for key in loss_component_keys:
            assert key in loss_stats['train_loss_components'], f'Loss component {key} not found in loss stats file.'
            assert key in loss_stats['val_loss_components'], f'Loss component {key} not found in loss stats file.'

        model.load_state_dict(model_state_dict)
        optimiser.load_state_dict(optimiser_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        random.setstate(random_state['python'])
        np.random.set_state(random_state['numpy'])
        torch.set_rng_state(random_state['torch'])
        torch.cuda.set_rng_state_all(random_state['cuda'])
        generator.set_state(random_state['generator'])

    # ==============================
    # training and validation loops
    # ==============================

    print('Starting training...')
    
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):

        # ----- training loop -----
        model.train()
        total_train_loss = 0.0
        total_train_loss_components = {
            key: 0.0 for key in loss_component_keys
        }
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_residual_pred = model(
                    lf_wd = batch.lf_dyn_node,
                    ups_wd = batch.ups_wd,
                    batch_size = batch.num_graphs
                )
            y_pred = batch.ups_wd + y_residual_pred.float()
            train_loss, train_loss_components = masked_loss(
                y_t = batch.hf_wd,
                y_pred = y_pred,
                num_graphs = batch.num_graphs,
                num_hf_nodes = num_hf_nodes,
                wet_idx = wet_idx,
                loss_func = loss_func
            )
            train_loss.backward()
            optimiser.step()
            total_train_loss += train_loss.detach()
            for key in loss_component_keys:
                total_train_loss_components[key] += train_loss_components[key]

        # ----- validation loop -----
        model.eval()
        total_val_loss = 0.0
        total_val_loss_components = {
            key: 0.0 for key in loss_component_keys
        }
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y_residual_pred = model(
                        lf_wd = batch.lf_dyn_node,
                        ups_wd = batch.ups_wd,
                        batch_size = batch.num_graphs
                    )
                y_pred = batch.ups_wd + y_residual_pred.float()
                val_loss, val_loss_components = masked_loss(
                    y_t = batch.hf_wd,
                    y_pred = y_pred,
                    num_graphs = batch.num_graphs,
                    num_hf_nodes = num_hf_nodes,
                    wet_idx = wet_idx,
                    loss_func = loss_func
                )
                total_val_loss += val_loss.detach()
                for key in loss_component_keys:
                    total_val_loss_components[key] += val_loss_components[key]
            
        # ---------- save loss metrics ----------

        avg_train_loss = (total_train_loss / len(train_loader)).item()
        avg_train_loss_components = {
            k: (v / len(train_loader)).item()
            for k, v in total_train_loss_components.items()
        }

        avg_val_loss = (total_val_loss / len(val_loader)).item()
        avg_val_loss_components = {
            k: (v / len(val_loader)).item()
            for k, v in total_val_loss_components.items()
        }

        scheduler.step(avg_val_loss)
        loss_stats['epoch'].append(epoch)
        loss_stats['train_loss'].append(avg_train_loss)
        loss_stats['val_loss'].append(avg_val_loss)
        for key in loss_component_keys:
            loss_stats['train_loss_components'][key].append(avg_train_loss_components[key])
            loss_stats['val_loss_components'][key].append(avg_val_loss_components[key])
        
        with open(LOSS_STATS_PATH, 'w') as f:
            json.dump(loss_stats, f, indent=4)
        print(f'Epoch {epoch}, Train loss {avg_train_loss:.6f}, Validation loss {avg_val_loss:.6f}')

        # ---------- checkpointing ----------

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'model_config': model_config,
                'model_state_dict': model.state_dict()
            }
            torch.save(best_checkpoint, BEST_CHECKPOINT_PATH)
            print(f'Saved new best model checkpoint at epoch {epoch}.')

        latest_checkpoint = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'random_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
                'generator': generator.get_state()
            }
        }
        torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)


def masked_loss(y_t, y_pred, num_graphs, num_hf_nodes, wet_idx, loss_func):
    y_pred_grid = y_pred.view(num_graphs, num_hf_nodes)
    y_t_grid = y_t.view(num_graphs, num_hf_nodes)

    y_pred_grid = y_pred_grid[:, wet_idx]
    y_t_grid = y_t_grid[:, wet_idx]

    loss = loss_func(y_pred_grid, y_t_grid)
    if isinstance(loss, tuple):
        total_loss, loss_components = loss[0], loss[1]
    else: # for MSE loss only
        total_loss, loss_components = loss, {'mse': loss.detach()}

    return total_loss, loss_components


def filter_timesteps(indices, event_ends, t_interval):
    """
    Converts global timestep indices to event-local indices and filters by t_interval.
    e.g. if t_interval=2, take every even indexed timestep per event.
    """
    event_ids = np.searchsorted(event_ends, indices, side='right')
    event_starts = np.concatenate(([0], event_ends[:-1]))
    local_timesteps = indices - event_starts[event_ids]
    return indices[(local_timesteps % t_interval) == 0]


if __name__ == '__main__':
    print('----- Burnett, with MSE-BCE loss -----')
    train(area_name='BurnettRV', mode='new', identifier='2026-10-20b', val_group=1, epochs_to_do=100, seed=42)
