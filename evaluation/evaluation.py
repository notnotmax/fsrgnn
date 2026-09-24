import json
import math
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from data.dataset import FloodEventDataset
from model.fsrgnn import FSRGNN
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from data.make_dataset import make_carlisle_dataset

def eval(val_group, model_file: str = None):
    print(f"Evaluating model from file {model_file} on validation group {val_group}.")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_file is None:
        model_file = f'model/Carlisle/Validation_{val_group}.pt'

    # ---------- prep datasets/loaders ----------
    test_dataset = make_carlisle_dataset(validation_group=val_group, mode='eval')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # ---------- make model ----------
    print('Creating model...')
    model = FSRGNN(
            lf_static_node_features = 3,
            lf_dynamic_node_features = 1,
            lf_static_edge_features = 1,
            hf_static_node_features = 2, # no mannings
            hf_static_edge_features = 1,
            hidden_features = 32, # 64
            output_features = 1,
            encoder_layers = 2,
            lfgnn_layers = 1,
            lfgnn_mlp_layers = 2,
            hfgnn_layers = 1,
            hfgnn_mlp_layers = 2,
            decoder_layers = 2
        ).to(DEVICE)
        
    state_dict = torch.load(model_file, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    wet_idx = test_dataset.wet_idx.to(DEVICE)
    all_y_pred = []
    all_y_upsampled = [] # upsampled from LF without passing through model
    all_y_t = []

    print('Evaluating...')
    with torch.no_grad():
        num_hf_nodes = test_dataset.hf_coords.shape[0]

        for batch in test_loader:
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
            y_upsampled = batch.upsampled_water_depth
            y_t = batch.hf_water_depth

            # unbatch into 2D shape
            y_pred_grid = y_pred.view(batch.num_graphs, num_hf_nodes)
            y_upsampled_grid = y_upsampled.view(batch.num_graphs, num_hf_nodes)
            y_t_grid = y_t.view(batch.num_graphs, num_hf_nodes)

            # filter
            y_pred_grid = y_pred_grid[:, wet_idx]
            y_upsampled_grid = y_upsampled_grid[:, wet_idx]
            y_t_grid = y_t_grid[:, wet_idx]

            all_y_pred.append(y_pred_grid.cpu())
            all_y_upsampled.append(y_upsampled_grid.cpu())
            all_y_t.append(y_t_grid.cpu())
    
    # collect into full matrices, shape (T, N_hf_wet)
    y_pred = torch.cat(all_y_pred, dim=0).numpy()
    y_upsampled = torch.cat(all_y_upsampled, dim=0).numpy()
    y_t = torch.cat(all_y_t, dim=0).numpy()

    rmse_upsampled = np.mean(np.sqrt(np.mean(np.square(y_upsampled - y_t), axis=0)))
    rmse_pred = np.mean(np.sqrt(np.mean(np.square(y_pred - y_t), axis=0)))

    print(f'Upsampled RMSE: {rmse_upsampled:.6f}')
    print(f'Prediction RMSE: {rmse_pred:.6f}')

if __name__ == '__main__':
    eval(val_group=1)
