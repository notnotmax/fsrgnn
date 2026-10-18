import numpy as np
import pandas as pd

def get_event_summary(area_name: str):
    event_summary = pd.read_csv(f'../dataset/{area_name}/{area_name}_event_summary.csv')
    return event_summary


def get_static_features_filepath(area_name: str):
    return f'data/{area_name}/processed/static_features.npz'


def get_feature_stats_filepath(area_name: str, val_group: int):
    return f'data/{area_name}/processed/feature_stats_fold_{val_group}.npz'


def get_lf_dyn_node_filepath(area_name: str, event_id: str):
    return f'data/{area_name}/processed/lf_dyn_node_event_{event_id}.npy'


def get_ups_wd_filepath(area_name: str, event_id: str):
    return f'data/{area_name}/processed/ups_wd_event_{event_id}.npy'


def get_hf_wd_filepath(area_name: str, event_id: str):
    return f'data/{area_name}/processed/hf_wd_event_{event_id}.npy'


def get_train_test_indices(area_name: str, val_group: int):
    ttsplit = np.load(f'../dataset/{area_name}/Train_test_split_data/Train_test_split_ValidateOnGrp_{val_group}.npz')
    return ttsplit['idx_train'], ttsplit['idx_test']


def get_wet_idx(area_name: str, val_group: int):
    wet_idx = np.load(f'../dataset/{area_name}/HF_EOF_analysis/Categories_HFdata_ValidateOnGrp_{val_group}.npz')['wet_idx']
    return wet_idx
