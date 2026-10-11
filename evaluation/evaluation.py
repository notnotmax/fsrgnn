import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from data.dataset import FloodEventDataset
from data.make_dataset import make_dataset
from data.visuals import plot_line
from model.fsrgnn import FSRGNN
from sklearn.metrics import r2_score
from torch.utils.data import Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader


def eval_upsampled(area_name: str):
    METRICS_PATH = f'model/{area_name}/upsampled_metrics.json'
    ev_summary = pd.read_csv(f'../dataset/{area_name}/{area_name}_event_summary.csv')
    metrics = []
    for event_idx in range(len(ev_summary)):
        if area_name == 'BurnettRV':
            event_id = ev_summary['No'][event_idx]
            val_group = ev_summary['Group'][event_idx]
        cat_data_path = f'../dataset/{area_name}/HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_{val_group}.npz'
        wet_idx = np.load(cat_data_path, allow_pickle=True)['wet_idx']
        event_metrics = eval_upsampled_one_event(area_name, event_id=1, wet_idx=wet_idx)
        metrics.append(event_metrics)
        break
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)


def eval_upsampled_one_event(area_name: str, event_id: str, wet_idx):
    print(f'Evaluating upsampled metrics for {area_name} event {event_id}.')
    ups_wd = np.load(f'data/{area_name}/processed/upsampled_water_depth_event_{event_id}.npy', mmap_mode='r')[:, wet_idx]
    hf_wd = np.load(f'data/{area_name}/processed/hf_water_depth_event_{event_id}.npy', mmap_mode='r')[:, wet_idx]
    event_length = hf_wd.shape[0]

    hf_bin = convert2binary(hf_wd)
    ups_bin = convert2binary(ups_wd)

    # CSI and RMSE over time
    csi_per_t = csi_all_t(hf_bin, ups_bin)
    squared_error = (ups_wd - hf_wd) ** 2
    rmse_per_t = np.sqrt(np.mean(squared_error, axis=1))

    t_range = np.arange(int(event_length))
    plot_line(t_range, csi_per_t, title=f'Water Depth CSI ({area_name}, event {event_id})', xlabel='Timestep', ylabel='CSI',
        ylim_bottom=0, ylim_top=1, output_path=f'model/{area_name}/upsampled_Ev_{event_id}_csi_t.png')
    plot_line(t_range, rmse_per_t, title=f'Water Depth RMSE ({area_name}, event {event_id})', xlabel='Timestep', ylabel='RMSE',
        ylim_bottom=0, output_path=f'model/{area_name}/upsampled_Ev_{event_id}_rmse_t.png')

    # maximum flood extent (CSI)
    hf_extent_max = hf_bin.max(axis=0)
    ups_extent_max = ups_bin.max(axis=0)
    _, _, ups_csi = pod_rfa_1tstep(hf_extent_max, ups_extent_max)

    # peak water depth R2 and avg diff
    hf_max_wd = np.max(hf_wd, axis=0)
    ups_max_wd = np.max(ups_wd, axis=0)
    max_wd_r2 = r2_score(hf_max_wd, ups_max_wd)
    avg_peak_diff = np.mean(hf_max_wd - ups_max_wd)

    # AvgRMSE (RMSE over time, averaged over nodes)
    ups_rmse = np.mean(np.sqrt(np.mean(squared_error, axis=0)))

    # fidelity index
    ups_fi = fidelity_index(true=hf_wd, pred=ups_wd)

    event_metrics = {
        "event_id": str(event_id),
        "num_timesteps": int(event_length),
        "max_csi": float(ups_csi),
        "max_wd_r2": float(max_wd_r2),
        "avg_peak_diff": float(avg_peak_diff),
        "rmse": float(ups_rmse),
        "fi": float(ups_fi),
    }

    print(
        f'Event {event_id}:\n'
        f'Maximum flood extent CSI: {ups_csi:.6f}\n'
        f'Max water depth R2: {max_wd_r2:.6f}\n'
        f'Average peak water depth difference: {avg_peak_diff:.6f}\n'
        f'Prediction RMSE: {ups_rmse:.6f}\n'
        f'Fidelity Index: {ups_fi:.6f}\n'
    )

    return event_metrics


def eval(area_name: str, identifier: str, val_group: int, one_event_only: bool = False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_PATH = f'model/{area_name}/{identifier}_Val_{val_group}_best.pt'
    METRICS_PATH = f'model/{area_name}/{identifier}_Val_{val_group}_metrics.json'

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    epoch = checkpoint['epoch']
    model_config = checkpoint['model_config']
    model_state_dict = checkpoint['model_state_dict']
    
    print(f"Evaluating model from file {CHECKPOINT_PATH} on validation group {val_group}. Model was saved at {epoch} epochs.")

    # ---------- prep datasets/loaders ----------
    test_dataset = make_dataset(area_name=area_name, val_group=val_group, mode='eval')
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    # ---------- make model ----------
    print('Creating model...')
    model = FSRGNN(**model_config)
    model.init_static_features(test_dataset.STATIC_FEATURES_PATH)
    model.to(DEVICE)
    model.load_state_dict(model_state_dict)
    model.eval()

    wet_idx = test_dataset.wet_idx.to(DEVICE)
    n_wet = wet_idx.numel()

    all_y_pred = []
    all_y_t = []

    print('Evaluating...')

    with torch.inference_mode():
        num_hf_nodes = test_dataset.num_hf_nodes

        counter = 0
        for batch in test_loader:
            batch = batch.to(DEVICE)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_residual_pred = model(
                    lf_dyn = batch.lf_dyn,
                    batch_size = batch.num_graphs
                )

            y_pred = torch.clamp(batch.upsampled_water_depth + y_residual_pred, min=0.0)
            y_t = batch.hf_water_depth

            # unbatch into 2D shape and filter
            y_pred_grid = y_pred.view(batch.num_graphs, num_hf_nodes)[:, wet_idx]
            y_t_grid = y_t.view(batch.num_graphs, num_hf_nodes)[:, wet_idx]

            all_y_pred.append(y_pred_grid.cpu().numpy())
            all_y_t.append(y_t_grid.cpu().numpy())

            counter += batch_size
            if counter > 400:
                break

    # ---------- metrics ----------

    hf_all = np.concatenate(all_y_t, axis=0)
    pred_all = np.concatenate(all_y_pred, axis=0)

    event_lengths = np.asarray(test_dataset.num_timesteps, dtype=np.int32)

    assert event_lengths.sum() == hf_all.shape[0], \
        f'Mismatch between expected number of timesteps {event_lengths.sum()} and actual collected timesteps {hf_all.shape[0]}'

    event_ids = test_dataset.event_ids
    assert len(event_ids) == len(event_lengths)

    metrics = {
        'epoch': int(epoch),
        'events': []
    }

    # run evaluation per event in this validation group
    start_t = 0
    for event_idx, (event_id, event_length) in enumerate(zip(event_ids, event_lengths)):

        # extract timesteps from this event only
        end_t = start_t + int(event_length)
        hf_wd = hf_all[start_t:end_t]
        pred_wd = pred_all[start_t:end_t]
        start_t = end_t

        hf_bin = convert2binary(hf_wd)
        pred_bin = convert2binary(pred_wd)

        # CSI and RMSE over time
        csi_per_t = csi_all_t(hf_bin, pred_bin)
        squared_error = (pred_wd - hf_wd) ** 2
        rmse_per_t = np.sqrt(np.mean(squared_error, axis=1))

        t_range = np.arange(int(event_length))
        plot_line(t_range, csi_per_t, title=f'Water Depth CSI ({area_name}, event {event_id})', xlabel='Timestep', ylabel='CSI',
            ylim_bottom=0, ylim_top=1, output_path=f'model/{area_name}/{identifier}_Ev_{event_id}_csi_t.png')
        plot_line(t_range, rmse_per_t, title=f'Water Depth RMSE ({area_name}, event {event_id})', xlabel='Timestep', ylabel='RMSE',
            ylim_bottom=0, output_path=f'model/{area_name}/{identifier}_Ev_{event_id}_rmse_t.png')
    
        # maximum flood extent (CSI)
        hf_extent_max = hf_bin.max(axis=0)
        pred_extent_max = pred_bin.max(axis=0)
        _, _, pred_csi = pod_rfa_1tstep(hf_extent_max, pred_extent_max)

        # peak water depth R2 and avg diff
        hf_max_wd = np.max(hf_wd, axis=0)
        pred_max_wd = np.max(pred_wd, axis=0)
        max_wd_r2 = r2_score(hf_max_wd, pred_max_wd)
        avg_peak_diff = np.mean(hf_max_wd - pred_max_wd)

        # AvgRMSE (RMSE over time, averaged over nodes)
        pred_rmse = np.mean(np.sqrt(np.mean(squared_error, axis=0)))

        # fidelity index
        pred_fi = fidelity_index(true=hf_wd, pred=pred_wd)

        event_metrics = {
            "event_id": str(event_id),
            "num_timesteps": int(event_length),
            "max_csi": float(pred_csi),
            "max_wd_r2": float(max_wd_r2),
            "avg_peak_diff": float(avg_peak_diff),
            "rmse": float(pred_rmse),
            "fi": float(pred_fi),
        }
        metrics["events"].append(event_metrics)

        print(
            f'Event {event_id}:\n'
            f'Maximum flood extent CSI: {pred_csi:.6f}\n'
            f'Max water depth R2: {max_wd_r2:.6f}\n'
            f'Average peak water depth difference: {avg_peak_diff:.6f}\n'
            f'Prediction RMSE: {pred_rmse:.6f}\n'
            f'Fidelity Index: {pred_fi:.6f}\n'
        )

        if one_event_only:
            break

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


def csi_all_t(true, pred):
    true = true > 0
    pred = pred > 0
    true_pos = (true & pred).sum(axis=1)
    false_pos = (~true & pred).sum(axis=1)
    false_neg = (true & ~pred).sum(axis=1)
    denom = true_pos + false_pos + false_neg
    csi_all = np.where(denom > 0, true_pos / denom, 1) # set CSI 1 if nothing to predict
    return csi_all


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

def visualise_loss_curves(area_name: str, identifier: str, val_group: int):
    JSON_PATH = f'model/{area_name}/{identifier}_Val_{val_group}_loss_stats.json'
    IMG_PATH = f'model/{area_name}/{identifier}_Val_{val_group}_loss_curves.png'

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
    # plt.ylim(0.02, 0.07) # loss axis
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()

    plt.savefig(IMG_PATH, dpi=300)
    print(f'Created loss curve image for {JSON_PATH}.')
    plt.close()


def read_stats(area_name: str, identifier: str, val_group: int):
    LATEST_CHECKPOINT_PATH = f'model/{area_name}/{identifier}_Val_{val_group}_latest.pt'
    checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location="cpu")
    opt_state_dict = checkpoint['optimiser_state_dict']
    for i, group in enumerate(opt_state_dict['param_groups']):
        lr = group['lr']
        print(f'Param group {i} learning rate: {lr:.6e}')


if __name__ == '__main__':
    print('---------- Evaluating ----------')
    # visualise_loss_curves(area_name='Carlisle', identifier='2026-10-09', val_group=1)
    # eval(area_name='Carlisle', identifier='2026-10-09', val_group=1)
    # print('---------- Evaluating 2 ----------')
    # visualise_loss_curves(area_name='Chowilla', identifier='2026-10-09', val_group=1)
    # eval(area_name='Chowilla', identifier='2026-10-09', val_group=1)
    # eval(area_name='BurnettRV', identifier='2026-10-09', val_group=1, one_event_only=True)
    eval_upsampled('BurnettRV')
    # read_stats('2026-10-02', val_group=1)
    
    # key format: [index, model]
    # 0: LSG
    # 1: 1dCNN
    # 2: LSTM-SRR
    # 3: GP-EOF
    # 4: LSTM-EOF
    # np.set_printoptions(precision=4, suppress=True)
    # lsg_results = np.load(f'../dataset/Chowilla/Result_data/Validation_results.npz')
    # lsg_csi = lsg_results['CSI'][:, 0]
    # lsg_max_wd_r2 = lsg_results['MaxWD_R2'][:, 0]
    # lsg_peak_diff = lsg_results['peak_diff'][:, 0]
    # lsg_rmse = lsg_results['RMSE'][:, 0]
    # lsg_fi = lsg_results['FI'][:, 0]
    # print(f'----- LSG model evaluation metrics: -----')
    # print(f'CSI: {lsg_csi}')
    # print(f'Max WD R2: {lsg_max_wd_r2}')
    # print(f'Peak diff: {lsg_peak_diff}')
    # print(f'RMSE: {lsg_rmse}')
    # print(f'FI: {lsg_fi}')

