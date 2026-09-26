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

def train(val_group: int, identifier: str):
    DEVICE = torch.device('cuda')
    NUM_EPOCHS = 50 # 100?
    CHECKPOINT_DIR = 'model/Carlisle'

    model_config = {
        'lf_static_node_features': 3,
        'lf_dynamic_node_features': 1,
        'lf_static_edge_features': 1,
        'hf_static_node_features': 2, # no mannings
        'hf_static_edge_features': 1,
        'hidden_features': 32,
        'output_features': 1,
        'encoder_layers': 2,
        'lfgnn_layers': 1,
        'lfgnn_mlp_layers': 2,
        'hfgnn_layers': 1,
        'hfgnn_mlp_layers': 2,
        'decoder_layers': 2
    }

    config = {
        'model_config': model_config,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'sch_factor': 0.5,
        'sch_patience': 5
    }

    # save json for quick reading, not for setting configs
    with open(os.path.join(CHECKPOINT_DIR, f'{identifier}_config.json')):
        json.dump(config, f, indent=4)

    optimiser = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=config['sch_factor'], patience=config['sch_patience'])
    loss_func = nn.MSELoss()

    print(f"Training on Carlisle with validation group {val_group}.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------- prep datasets/loaders ----------
    dataset = make_carlisle_dataset(validation_group=val_group, mode='train')
    ttsplit = np.load(f'../dataset/Carlisle/Train_test_split_data/Train_test_split_ValidateOnGrp_{val_group}.npz', mmap_mode='r')
    idx_train = ttsplit['idx_train']
    idx_val = ttsplit['idx_test']
    train_dataset = Subset(dataset, idx_train)
    val_dataset = Subset(dataset, idx_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4) # already shuffled in ttsplit
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # constants for this fold/validation group
    wet_idx = dataset.wet_idx.to(DEVICE)
    num_hf_nodes = dataset.hf_coords.shape[0]

    # ---------- make model ----------
    print('Creating model...')
    model = FSRGNN(**model_config).to(DEVICE)

    # ---------- training/validation loops ----------
    print('Starting training...')
    best_val_loss = float('inf')
    loss_stats = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
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
            y_pred = torch.clamp(batch.upsampled_water_depth + y_residual_pred, min=0.0)
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
                y_pred = torch.clamp(batch.upsampled_water_depth + y_residual_pred, min=0.0)
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
        
        with open(os.path.join(CHECKPOINT_DIR, f'{identifier}_Validation_{val_group}_loss_stats.json'), 'w') as f:
            json.dump(loss_stats, f, indent=4)

        # model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'model_config': model_config,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'{identifier}_Validation_{val_group}.pt'))
            print(f'Saved new model checkpoint at epoch {epoch}.')


def masked_loss(y_t, y_pred, num_graphs, num_hf_nodes, wet_idx, loss_func):
    # unbatch into 2D shape
    y_pred_grid = y_pred.view(num_graphs, num_hf_nodes)
    y_t_grid = y_t.view(num_graphs, num_hf_nodes)

    # filter
    y_pred_grid = y_pred_grid[:, wet_idx]
    y_t_grid = y_t_grid[:, wet_idx]

    return loss_func(y_t_grid, y_pred_grid)


if __name__ == '__main__':
    train(val_group=1, identifier='featnorm')
