import json
import numpy as np
import os
import torch
import torch.nn as nn

from model.fsrgnn import FSRGNN
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from data.make_dataset import make_carlisle_dataset

def train(mode: str, identifier: str, val_group: int):
    DEVICE = torch.device('cuda')
    NUM_EPOCHS = 10 # num epochs to add, not do until this num of epochs
    CHECKPOINT_DIR = 'model/Carlisle'
    CONFIG_PATH = f'{CHECKPOINT_DIR}/{identifier}_config.json'
    LATEST_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/{identifier}_Val_{val_group}_latest.pt'
    BEST_CHECKPOINT_PATH = f'{CHECKPOINT_DIR}/{identifier}_Val_{val_group}_best.pt'
    LOSS_STATS_PATH = f'{CHECKPOINT_DIR}/{identifier}_Val_{val_group}_loss_stats.json'

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if mode == 'new':

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            model_config = config['model_config']
            batch_size = config['batch_size']
            learning_rate = config['learning_rate']
            weight_decay = config['weight_decay']
            sch_factor = config['sch_factor']
            sch_patience = config['sch_patience']
            t_interval = config['t_interval']

        start_epoch = 1
        best_val_loss = float('inf')
        loss_stats = {
            'epoch': [],
            'train_loss': [],
            'val_loss': []
        }
        print(f'Training new model on Carlisle on validation group {val_group} using config {CONFIG_PATH}.')

    elif mode == 'continue':

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            model_config = config['model_config']
            batch_size = config['batch_size']
            learning_rate = config['learning_rate']
            weight_decay = config['weight_decay']
            sch_factor = config['sch_factor']
            sch_patience = config['sch_patience']
            t_interval = config['t_interval']

        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=DEVICE)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        # model_config = checkpoint['model_config']
        model_state_dict = checkpoint['model_state_dict']
        optimiser_state_dict = checkpoint['optimiser_state_dict']
        scheduler_state_dict = checkpoint['scheduler_state_dict']

        with open(LOSS_STATS_PATH, 'r') as f:
            loss_stats = json.load(f)

    else:
        assert False, f'Unknown training mode: {mode}, should be "new" or "continue"'

    # ---------- prep datasets/loaders ----------
    dataset = make_carlisle_dataset(validation_group=val_group, mode='train')
    ttsplit = np.load(f'../dataset/Carlisle/Train_test_split_data/Train_test_split_ValidateOnGrp_{val_group}.npz')

    # optionally filter indices to only use every <t_interval> timesteps from each event
    event_ends = np.cumsum(np.asarray(dataset.num_timesteps, dtype=np.int32))
    idx_train = filter_timesteps(ttsplit['idx_train'], event_ends, t_interval)
    idx_val = filter_timesteps(ttsplit['idx_test'], event_ends, t_interval)
    print(f'----- DEBUG INFO train samples {idx_train.shape}, validation samples {idx_val.shape} -----')

    train_dataset = Subset(dataset, idx_train)
    val_dataset = Subset(dataset, idx_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True) # already shuffled in ttsplit
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    # constants for this fold/validation group
    wet_idx = dataset.wet_idx.to(DEVICE)
    num_hf_nodes = dataset.hf_coords.shape[0]

    # ---------- make model ----------
    print('Creating model...')
    model = FSRGNN(**model_config).to(DEVICE)

    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=sch_factor, patience=sch_patience)
    loss_func = nn.MSELoss()

    if mode == 'continue':
        model.load_state_dict(model_state_dict)
        optimiser.load_state_dict(optimiser_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    # ---------- training/validation loops ----------
    print('Starting training...')
    
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        # training loop
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad()
            y_residual_pred = model(
                lf_x = batch.lf_x,
                lf_coords = batch.lf_coords,
                lf_edge_index = batch.lf_edge_index,
                lf_edge_attr = batch.lf_edge_attr,
                hf_x = batch.hf_x,
                hf_coords = batch.hf_coords,
                hf_edge_index = batch.hf_edge_index,
                hf_edge_attr = batch.hf_edge_attr
            )
            y_pred = batch.upsampled_water_depth + y_residual_pred
            # y_pred = torch.clamp(batch.upsampled_water_depth + y_residual_pred, min=0.0)
            loss = masked_loss(
                y_t = batch.hf_water_depth,
                y_pred = y_pred,
                num_graphs = batch.num_graphs,
                num_hf_nodes = num_hf_nodes,
                wet_idx = wet_idx,
                loss_func = loss_func
            )
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                y_residual_pred = model(
                    lf_x = batch.lf_x,
                    lf_coords = batch.lf_coords,
                    lf_edge_index = batch.lf_edge_index,
                    lf_edge_attr = batch.lf_edge_attr,
                    hf_x = batch.hf_x,
                    hf_coords = batch.hf_coords,
                    hf_edge_index = batch.hf_edge_index,
                    hf_edge_attr = batch.hf_edge_attr
                )
                y_pred = batch.upsampled_water_depth + y_residual_pred
                # y_pred = torch.clamp(batch.upsampled_water_depth + y_residual_pred, min=0.0)
                val_loss = masked_loss(
                    y_t = batch.hf_water_depth,
                    y_pred = y_pred,
                    num_graphs = batch.num_graphs,
                    num_hf_nodes = num_hf_nodes,
                    wet_idx = wet_idx,
                    loss_func = loss_func
                )
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        print(f'Epoch {epoch}, Train loss {avg_train_loss:.6f}, Validation loss {avg_val_loss:.6f}')

        loss_stats['epoch'].append(epoch)
        loss_stats['train_loss'].append(avg_train_loss)
        loss_stats['val_loss'].append(avg_val_loss)
        
        with open(LOSS_STATS_PATH, 'w') as f:
            json.dump(loss_stats, f, indent=4)

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

        # always save latest model to use when resuming training
        latest_checkpoint = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)


def masked_loss(y_t, y_pred, num_graphs, num_hf_nodes, wet_idx, loss_func):
    # unbatch into 2D shape
    y_pred_grid = y_pred.view(num_graphs, num_hf_nodes)
    y_t_grid = y_t.view(num_graphs, num_hf_nodes)

    # filter
    y_pred_grid = y_pred_grid[:, wet_idx]
    y_t_grid = y_t_grid[:, wet_idx]

    return loss_func(y_t_grid, y_pred_grid)


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
    print('----- Fixed negative upsampled water depth values, removed clamping during training -----')
    train(mode='new', identifier='2026-10-02', val_group=1)
