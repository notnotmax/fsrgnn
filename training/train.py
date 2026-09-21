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

def make_carlisle_dataset(validation_group):

    print("Making Carlisle dataset.")
    DATASET_PATH = '../dataset/Carlisle'
    EVENT_SUMMARY_PATH = os.path.join(DATASET_PATH, 'Carlisle_event_summary.csv')
    event_summary = pd.read_csv(EVENT_SUMMARY_PATH)

    num_events = len(event_summary)
    num_groups = np.max(event_summary['Group'])
    EVENT_NUM_TIMESTEPS = [266, 242, 198, 253, 318, 199, 266, 312, 316] # static
    print(f'Total: {num_events} events across {num_groups} groups.')


    event_ids = []
    group_ids = []
    lf_filepaths = []
    hf_filepaths = []
    num_timesteps = []

    for event_idx in range(num_events):
        event_id = event_summary['No'][event_idx]
        group_id = event_summary['Group'][event_idx]

        if group_id == validation_group: # leave one out
            continue

        lf_run_name = event_summary['HEC_RAS_plan'][event_idx]
        lf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/Low-fidelity/Carlisle_LFmodelA.{lf_run_name}.hdf')
        hf_run_name = event_summary['Lisflood'][event_idx]
        hf_filepath = os.path.join(DATASET_PATH, f'HD_model_data/High-fidelity/{hf_run_name}_alltimesteps.npz')
        # hf_data = np.load(hf_filepath, mmap_mode='r') # slow
        # event_num_timesteps = hf_data['wse_data'].shape[0]
        event_num_timesteps = EVENT_NUM_TIMESTEPS[event_idx]

        event_ids.append(event_id)
        group_ids.append(group_id)
        lf_filepaths.append(lf_filepath)
        hf_filepaths.append(hf_filepath)
        num_timesteps.append(event_num_timesteps)

        print(f'Added event (event id {event_id}, group id {group_id}) to train split. \
                Event has {event_num_timesteps} timesteps.')

    print(f'Making dataset for train split {validation_group}.')

    # triggers the preprocessing
    dataset = FloodEventDataset(
        root_dir = 'data/Carlisle',
        area_name = 'Carlisle',
        lf_geometry_path = 'data/Carlisle/LF_geometry_data.npz',
        hf_geometry_path = 'data/Carlisle/HF_geometry_data.npz',
        lf_hecras_paths = lf_filepaths,
        hf_paths = hf_filepaths,
        hf_filetype = 'npz',
        event_ids = event_ids,
        group_ids = group_ids,
        num_timesteps = num_timesteps,
        previous_timesteps = 0
    )

    print(f'Made Carlisle dataset, contains total {len(dataset)} timesteps.')

    return dataset

def train():
    DEVICE = torch.device('cuda')
    NUM_EPOCHS = 1 # TODO make this bigger
    LEARNING_RATE = 1e-3
    VAL_GROUP = 1
    CHECKPOINT_DIR = 'model/Carlisle'

    print("Training on Carlisle.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---------- prep datasets/loaders ----------
    dataset = make_carlisle_dataset(validation_group=VAL_GROUP)
    ttsplit = np.load('../dataset/Carlisle/Train_test_split_data/Train_test_split_ValidateOnGrp_1.npz', mmap_mode='r')
    idx_train = ttsplit['idx_train']
    idx_val = ttsplit['idx_test']
    train_dataset = Subset(dataset, idx_train)
    val_dataset = Subset(dataset, idx_val)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # already shuffled in ttsplit
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

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

    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)
    loss_func = nn.MSELoss(reduction='none') # no reduction to allow for wet masking

    # ---------- training/validation loops ----------
    print('Starting training...')
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # training loop
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad()
            y_pred = model(
                lf_x = batch.lf_x,
                lf_coords = batch.lf_coords,
                lf_edge_index = batch.lf_edge_index,
                lf_edge_attr = batch.lf_edge_attr,
                hf_x = batch.hf_x,
                hf_coords = batch.hf_coords,
                hf_edge_index = batch.hf_edge_index,
                hf_edge_attr = batch.hf_edge_attr
            )
            loss = masked_loss(
                y=batch.hf_residual_targets,
                y_pred=y_pred,
                wet_mask=batch.hf_wet_mask,
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
                y_pred = model(
                    lf_x = batch.lf_x,
                    lf_coords = batch.lf_coords,
                    lf_edge_index = batch.lf_edge_index,
                    lf_edge_attr = batch.lf_edge_attr,
                    hf_x = batch.hf_x,
                    hf_coords = batch.hf_coords,
                    hf_edge_index = batch.hf_edge_index,
                    hf_edge_attr = batch.hf_edge_attr
                )
                val_loss = masked_loss(
                    y=batch.hf_residual_targets,
                    y_pred=y_pred,
                    wet_mask=batch.hf_wet_mask,
                    loss_func = loss_func
                )
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        print(f'Epoch {epoch}, Train loss {avg_train_loss:.6f}, Validation loss {avg_val_loss:.6f}')

        # model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'Validation_{VAL_GROUP}.pt'))
            print(f'Saved new model checkpoint at epoch {epoch}.')

def masked_loss(y, y_pred, wet_mask, loss_func):
    loss = loss_func(y, y_pred)
    wet_loss = loss[wet_mask]
    # edge case if no wet cells, return 0
    if wet_loss.numel() == 0:
        return y_pred.sum() * 0.0
    return wet_loss.mean()

if __name__ == '__main__':
    train()
