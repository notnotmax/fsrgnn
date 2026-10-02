import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from data.dataset import FloodEventDataset
from model.fsrgnn import FSRGNN
from sklearn.metrics import r2_score
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from data.make_dataset import make_carlisle_dataset

def eval(identifier: str, val_group: int):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_PATH = f'model/Carlisle/{identifier}_Val_{val_group}_best.pt'
    METRICS_PATH = f'model/Carlisle/{identifier}_Val_{val_group}_metrics.json'
    
    print(f"Evaluating model from file {CHECKPOINT_PATH} on validation group {val_group}.")

    # ---------- prep datasets/loaders ----------
    test_dataset = make_carlisle_dataset(validation_group=val_group, mode='eval')
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model_config = checkpoint['model_config']
    model_state_dict = checkpoint['model_state_dict']

    # ---------- make model ----------
    print('Creating model...')
    model = FSRGNN(**model_config).to(DEVICE)
    model.load_state_dict(model_state_dict)
    model.eval()

    wet_idx = test_dataset.wet_idx.to(DEVICE)
    n_wet = wet_idx.numel()

    # sum_sq_error_upsampled = torch.zeros(n_wet, device=DEVICE) # baseline error w/o using model
    sum_sq_error_pred = torch.zeros(n_wet, device=DEVICE)
    total_t = 0
    all_y_pred = []
    all_y_t = []

    print('Evaluating...')
    with torch.inference_mode():
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

            # y_upsampled = batch.upsampled_water_depth
            y_pred = torch.clamp(batch.upsampled_water_depth + y_residual_pred, min=0.0)
            y_t = batch.hf_water_depth

            # unbatch into 2D shape and filter
            # y_upsampled_grid = y_upsampled.view(batch.num_graphs, num_hf_nodes)[:, wet_idx]
            y_pred_grid = y_pred.view(batch.num_graphs, num_hf_nodes)[:, wet_idx]
            y_t_grid = y_t.view(batch.num_graphs, num_hf_nodes)[:, wet_idx]

            # sum_sq_error_upsampled += ((y_upsampled_grid - y_t_grid) ** 2).sum(dim=0)
            sum_sq_error_pred += ((y_pred_grid - y_t_grid) ** 2).sum(dim=0)
            total_t += batch.num_graphs

            all_y_pred.append(y_pred_grid.cpu().numpy())
            all_y_t.append(y_t_grid.cpu().numpy())

    # ---------- metrics ----------

    hf_wd = np.concatenate(all_y_t, axis=0)
    pred_wd = np.concatenate(all_y_pred, axis=0)
    
    # 1. maximum flood extent (CSI)
    hf_extent_max = convert2binary(hf_wd).max(axis=0)
    pred_extent_max = convert2binary(pred_wd).max(axis=0)
    pred_pod, pred_rfa, pred_csi = pod_rfa_1tstep(hf_extent_max, pred_extent_max)
    print(f'Maximum flood extent CSI: {pred_csi:.6f}')

    # 2. peak water depth R2 and avg diff
    hf_max_wd = np.max(hf_wd, axis=0)
    pred_max_wd = np.max(pred_wd, axis=0)
    max_wd_r2 = r2_score(hf_max_wd, pred_max_wd)
    avg_peak_diff = np.mean(hf_max_wd - pred_max_wd)
    print(f'Max water depth R2: {max_wd_r2:.6f}')
    print(f'Average peak water depth difference: {avg_peak_diff:.6f}')

    # 3. RMSE
    pred_mse_per_node = (sum_sq_error_pred / total_t).cpu().numpy()
    pred_rmse = np.mean(np.sqrt(pred_mse_per_node))

    # upsampled_mse_per_node = (sum_sq_error_upsampled / total_t).cpu().numpy()
    # upsampled_rmse = np.mean(np.sqrt(upsampled_mse_per_node))
    
    # print(f'Upsampled RMSE: {rmse_upsampled:.6f}')
    print(f'Prediction RMSE: {pred_rmse:.6f}')

    # 4. fidelity index
    pred_fi = fidelity_index(true=hf_wd, pred=pred_wd)
    print(f'Fidelity Index: {pred_fi:.6f}')

    metrics = {
        'max_csi': pred_csi,
        'max_wd_r2': max_wd_r2,
        'avg_peak_diff': avg_peak_diff,
        'rmse': pred_rmse,
        'fi': pred_fi
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

# ---------- metric helper functions ---------- adapted from https://doi.org/10.26188/24312658

def convert2binary(waterdepth_data, threshold_flood = 0.03):
    # Threshold for flooding
    data_bin = np.where(waterdepth_data >= threshold_flood, 1, 0)
    return data_bin


def pod_rfa_1tstep(true_labels, pred_labels):
    true_pos_all = np.sum(np.where((pred_labels > 0) & (true_labels > 0), 1, 0))
    true_neg_all = np.sum(np.where((pred_labels == 0) & (true_labels == 0), 1, 0))
    false_pos_all = np.sum(np.where((pred_labels > 0) & (true_labels == 0), 1, 0))
    false_neg_all = np.sum(np.where((pred_labels == 0) & (true_labels > 0), 1, 0))
    # ConfusionMatrix_sum = True_pos + True_neg + False_pos + False_neg

    precision_all = true_pos_all / (true_pos_all + false_pos_all)
    recall_all = true_pos_all / (true_pos_all + false_neg_all)
    fscore_all = (2 * precision_all * recall_all) / (precision_all + recall_all)

    # Rate of false alarm and probability of detection
    pod_all = recall_all
    rfa_all = (false_pos_all / (true_pos_all + false_pos_all))
    
    csi_all = 1/((1/(1-rfa_all)) + (1/(pod_all)-1))
    
    return pod_all, rfa_all, csi_all


def fidelity_index(true, pred, wd_tol=0.05, time_tol=0.05, time_tol_as_timesteps=False):
    #Check if number of timesteps to shift or fraction of total timesteps
    if time_tol_as_timesteps: 
        total_shifts = time_tol
    else: # Calculate number of timesteps to shift from fraction of total timesteps
        total_shifts = round(time_tol * len(true))
    
    n_predictions = len(true[total_shifts:-total_shifts, :].ravel())
    
    diff_min = np.full_like(true[total_shifts:-total_shifts, :], np.inf)
    for i in range(-total_shifts, total_shifts):
        diff = np.abs(pred[total_shifts:-total_shifts, :] - np.roll(true, i, axis=0)[total_shifts:-total_shifts, :])

        diff_min = np.minimum(diff_min, diff)
        
    FI = np.sum(diff_min < wd_tol) / n_predictions
    
    return FI

# ---------- visualisation functions ----------

def visualise_loss_curves(identifier: str, val_group: int):
    JSON_PATH = f'model/Carlisle/{identifier}_Val_{val_group}_loss_stats.json'
    IMG_PATH = f'model/Carlisle/{identifier}_Val_{val_group}_loss_curves.png'

    with open(JSON_PATH, 'r') as f:
        loss_stats = json.load(f)
        epochs = loss_stats['epoch']
        train_loss = loss_stats['train_loss']
        val_loss = loss_stats['val_loss']
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.ylim(0.03, 0.06) # loss axis
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()

    plt.savefig(IMG_PATH, dpi=300)
    print(f'Created loss curve image for {JSON_PATH}.')
    plt.close()


if __name__ == '__main__':
    # visualise_loss_curves(identifier='2026-10-01b', val_group=1)
    eval(identifier='2026-10-01b', val_group=1)

