import numpy as np
import os
import torch

from data.get_feature_stats import get_feature_stats
from data.hecras_data_retrieval import get_min_cell_elevation
from data.HECRAS_functions import filter_dry_areas, get_2d_hecras_results
from scipy.spatial import KDTree
from torch_geometric.data import Data, Dataset
from utils import get_lf_dyn_node_filepath, get_ups_wd_filepath, get_hf_wd_filepath, get_static_features_filepath

DEPTH_CAP = {'Carlisle': 9.0, 'Chowilla': 15.0, 'BurnettRV': 27.0}

class FloodEventDataset(Dataset):
    """
    Static lf node features: area, elevation, roughness
    Dynamic lf node features: LF water depth
    Static lf edge features: edge length
    Static hf node features: area, elevation
    Target: residual water depth at hf nodes
    """

    def __init__(
        self,
        mode: str,
        root_dir: str,
        area_name: str,
        lf_geometry_path: str,
        hf_geometry_path: str,
        lf_hecras_paths: list[str],
        hf_paths: list[str],
        event_ids: list,
        group_ids: list,
        num_timesteps: list):
        self.mode = mode

        # filepaths
        self.root = root_dir
        self.area_name = area_name
        self.lf_geometry_path = lf_geometry_path
        self.hf_geometry_path = hf_geometry_path
        self.lf_hecras_paths = lf_hecras_paths
        self.hf_paths = hf_paths

        # event data
        self.event_ids = event_ids
        self.group_ids = group_ids
        self.num_timesteps = num_timesteps

        # preset filenames to save into in processed dir
        self.STATIC_FEATURES_PATH = get_static_features_filepath(self.area_name)

        self.LF_DYN_NODE_PATHS = [get_lf_dyn_node_filepath(area_name, event_id) for event_id in self.event_ids]
        self.UPS_WD_PATHS = [get_ups_wd_filepath(area_name, event_id) for event_id in self.event_ids]
        self.HF_WD_PATHS = [get_hf_wd_filepath(area_name, event_id) for event_id in self.event_ids]

        self.dynamic_data_per_event = {} # caching

        super().__init__(self.root, transform = None, pre_transform = None, pre_filter = None)

        if self.mode == 'all':
            return # 'all' mode is for preprocessing, not training or evaluation

        static_features = np.load(self.STATIC_FEATURES_PATH)
        self.num_lf_nodes = static_features['lf_coords'].shape[0]
        self.num_hf_nodes = static_features['hf_coords'].shape[0]


    @property
    def raw_file_names(self):
        # pass filepaths into the constructor
        return []


    @property
    def processed_file_names(self):
        # call process_all manually
        return []


    def download(self):
        # no downloading
        pass

    
    def process(self):
        # run process_all below
        pass


    def process_all(self):
        assert self.mode == 'all', 'Not preprocessing all npy files at once. Ensure this is intentional.'

        # ----- create static file -----
        lf_geom = self._get_lf_geometry()
        lf_static_node_features = self._get_lf_static_node_features()
        lf_static_edge_features = self._get_lf_static_edge_features()
        hf_geom = self._get_hf_geometry()
        hf_static_node_features = self._get_hf_static_node_features()
        hf_static_edge_features = self._get_hf_static_edge_features()
        
        save_path = os.path.join(self.processed_dir, self.STATIC_FEATURES_FILE)
        np.savez(save_path,
                 lf_coords = lf_geom['cell_coordinates'],
                 lf_edge_index = lf_geom['edge_index'],
                 lf_static_node_features = lf_static_node_features,
                 lf_static_edge_features = lf_static_edge_features,
                 hf_coords = hf_geom['cell_coordinates'],
                 hf_edge_index = hf_geom['edge_index'],
                 hf_static_node_features = hf_static_node_features,
                 hf_static_edge_features = hf_static_edge_features)
        print(f'Saved constant values to {save_path}')
        del lf_geom, lf_static_node_features, lf_static_edge_features, hf_geom, hf_static_node_features, hf_static_edge_features

        # ----- create dynamic files -----
        for i, event_id in enumerate(self.event_ids):
            print(f'Getting dynamic values for event {i+1} / {len(self.event_ids)}, event ID: {event_id}')
            lf_dyn_node = self._get_lf_dynamic_node_features(event_idx=i)
            hf_wd, ups_wd = self._get_hf_dynamic_data(event_idx=i)

            assert lf_dyn_node.shape[0] == self.num_timesteps[i], \
                f'LF timestep mismatch for event {event_id}. Expected {self.num_timesteps[i]}, got {lf_dyn_node.shape[0]}.'
            assert hf_wd.shape[0] == self.num_timesteps[i], \
                f'HF timestep mismatch for event {event_id}. Expected {self.num_timesteps[i]}, got {hf_wd.shape[0]}.'
            assert ups_wd.shape == hf_wd.shape, \
                f'Mismatch in shape between upsampled and HF water depth. ups: {ups_wd.shape}, HF: {hf_wd.shape}'

            np.save(self.LF_DYN_NODE_PATHS[i], lf_dyn_node.astype(np.float32))
            np.save(self.UPS_WD_PATHS[i], ups_wd.astype(np.float32))
            np.save(self.HF_WD_PATHS[i], hf_wd.astype(np.float32))

            print(f'Saved dynamic values for event {event_id}.')
            del lf_dyn_node, ups_wd, hf_wd

        VAL_GROUPS = {'Carlisle': 9, 'Chowilla': 10, 'BurnettRV': 4}
        assert self.area_name in VAL_GROUPS, f'Loading of dynamic feature statistics for {self.area_name} not set.'

        for val in range(1, VAL_GROUPS[self.area_name] + 1):
            get_feature_stats(self.area_name, val)


    def len(self):
        return sum(self.num_timesteps)


    def _get_dynamic_data(self, event_idx: int):
        if event_idx not in self.dynamic_data_per_event:
            self.dynamic_data_per_event[event_idx] = {
                "lf_dyn_node": np.load(
                    self.LF_DYN_NODE_PATHS[event_idx], mmap_mode="r"
                ),
                "ups_wd": np.load(
                    self.UPS_WD_PATHS[event_idx], mmap_mode="r"
                ),
                "hf_wd": np.load(
                    self.HF_WD_PATHS[event_idx], mmap_mode="r"
                ),
            }
        return self.dynamic_data_per_event[event_idx]


    def get(self, idx):

        # find the event this index belongs to
        event_idx = -1
        timestep_in_event = -1
        timesteps_counter = 0
        for i, event_num_timesteps in enumerate(self.num_timesteps):
            if timesteps_counter + event_num_timesteps > idx:
                event_idx = i
                timestep_in_event = idx - timesteps_counter
                break
            timesteps_counter += event_num_timesteps
        assert 0 <= event_idx < len(self.event_ids), f"Error: event index {event_idx} is out of bounds."
        assert 0 <= timestep_in_event < self.num_timesteps[event_idx], \
            f"Error: timestep {timestep_in_event} out of bounds for event with {self.num_timesteps[event_idx]} timesteps."

        # load dynamic data
        dynamic_data = self._get_dynamic_data(event_idx)

        lf_dyn_node = dynamic_data['lf_dyn_node'][timestep_in_event]
        lf_dyn_node = torch.from_numpy(lf_dyn_node).float().unsqueeze(-1)

        ups_wd = dynamic_data['ups_wd'][timestep_in_event]
        ups_wd = torch.from_numpy(ups_wd).float().unsqueeze(-1)

        hf_wd = dynamic_data['hf_wd'][timestep_in_event]
        hf_wd = torch.from_numpy(hf_wd).float().unsqueeze(-1)

        data = Data(
            lf_dyn_node = lf_dyn_node,
            ups_wd = ups_wd,
            hf_wd = hf_wd,

            # for PyG
            num_nodes = self.num_hf_nodes,
            lf_num_nodes = self.num_lf_nodes,
            hf_num_nodes = self.num_hf_nodes
        )
        
        return data


    def _get_num_timesteps(self, event_idx):
        return self.num_timesteps[event_idx]

    # ---------- Low-fidelity ----------

    def _get_lf_geometry(self):
        return np.load(self.lf_geometry_path)


    def _get_lf_static_node_features(self):
        """
        LF static node features: cell area, elevation, roughness (Manning's)
        """
        lf_geom = self._get_lf_geometry()
        lf_area = lf_geom['cell_area']
        lf_elevation = lf_geom['cell_elevation']
        lf_roughness = lf_geom['cell_roughness']

        static_node_features = np.array([lf_area, lf_elevation, lf_roughness]).transpose()
        return static_node_features


    def _get_lf_static_edge_features(self):
        """
        LF edge features: edge length
        """
        lf_geom = self._get_lf_geometry()
        lf_coords = lf_geom['cell_coordinates']
        lf_edge_index = lf_geom['edge_index']
        n1_coords, n2_coords = lf_coords[lf_edge_index[0]], lf_coords[lf_edge_index[1]]
        lf_edge_lengths = np.linalg.norm(n1_coords - n2_coords, axis = 1)

        static_edge_features = np.array([lf_edge_lengths]).transpose()
        return static_edge_features


    def _get_lf_dynamic_node_features(self, event_idx: int):
        """
        LF dynamic node features: water depth
        """
        if self.area_name == 'Carlisle':
            _, lf_water_depth = self._get_lf_wse_depth_carlisle(event_idx)
        elif self.area_name == 'Chowilla':
            _, lf_water_depth = self._get_lf_wse_depth_chowilla(event_idx)
        elif self.area_name == 'BurnettRV':
            _, lf_water_depth = self._get_lf_wse_depth_burnett(event_idx)
        else:
            raise NotImplementedError(f'Raw data extraction for {self.area_name} not implemented.')

        return lf_water_depth


    def _get_lf_wse_depth_carlisle(self, event_idx: int):
        """
        Get LF water level and depth from raw HECRAS file with ghost cells filtered out.
        For Carlisle, use warmup period of 1.9 hours.
        (With each timestep being 15min apart, trims 8 timesteps.)
        """
        assert self.area_name == 'Carlisle', 'Mismatch between area name and function.'
        
        lf_path = self.lf_hecras_paths[event_idx]
        lf_elevation = get_min_cell_elevation(lf_path, 'Carlisle')
        ghost_idx = np.where(np.isnan(lf_elevation))[0]
        lf_elevation = np.delete(lf_elevation, ghost_idx)

        # get LF wse from hecras file
        lf_water_level = get_2d_hecras_results(lf_path, 'Carlisle', warmupperiod="1.9 hours", remove_ghost_cells=True)[0]
        lf_water_level = filter_dry_areas(lf_water_level, lf_elevation)
        lf_water_depth = np.clip(lf_water_level - lf_elevation, a_min=0.0, a_max=DEPTH_CAP[self.area_name])

        return lf_water_level, lf_water_depth

    
    def _get_lf_wse_depth_chowilla(self, event_idx: int):
        """
        Get LF water level and depth from raw HECRAS file with ghost cells filtered out.
        For Chowilla, use the default warmup period of 10 hours.
        Trims 41 timesteps.
        """
        assert self.area_name == 'Chowilla', 'Mismatch between area name and function.'

        lf_path = self.lf_hecras_paths[event_idx]
        lf_elevation = get_min_cell_elevation(lf_path, 'Chowilla')
        ghost_idx = np.where(np.isnan(lf_elevation))[0]
        lf_elevation = np.delete(lf_elevation, ghost_idx)

        # get LF wse from hecras file
        lf_water_level = get_2d_hecras_results(lf_path, 'Chowilla', remove_ghost_cells=True)[0]
        lf_water_level = filter_dry_areas(lf_water_level, lf_elevation)
        lf_water_depth = np.clip(lf_water_level - lf_elevation, a_min=0.0, a_max=DEPTH_CAP[self.area_name])

        return lf_water_level, lf_water_depth


    def _get_lf_wse_depth_burnett(self, event_idx: int):
        """
        Get LF water level and depth from raw HECRAS file with ghost cells filtered out.
        For Burnett, use warmup period of 47.5 hours and trim to match HF timesteps.
        """
        assert self.area_name == 'BurnettRV', 'Mismatch between area name and function.'

        lf_path = self.lf_hecras_paths[event_idx]
        lf_elevation = get_min_cell_elevation(lf_path, 'BurnettRV_region')
        ghost_idx = np.where(np.isnan(lf_elevation))[0]
        lf_elevation = np.delete(lf_elevation, ghost_idx)

        # get LF wse from hecras file
        lf_water_level = get_2d_hecras_results(lf_path, 'BurnettRV_region', warmupperiod="47.5 hours", remove_ghost_cells=True)[0]
        lf_water_level = filter_dry_areas(lf_water_level, lf_elevation)
        lf_water_depth = np.clip(lf_water_level - lf_elevation, a_min=0.0, a_max=DEPTH_CAP[self.area_name])

        # trim to match HF timesteps (Burnett only)
        hf_timesteps = self.num_timesteps[event_idx]
        lf_water_depth = lf_water_depth[:hf_timesteps, :]
        lf_water_level = lf_water_level[:hf_timesteps, :]

        return lf_water_level, lf_water_depth

    # ---------- High-fidelity ----------

    def _get_hf_geometry(self):
        return np.load(self.hf_geometry_path)


    def _get_hf_static_node_features(self):
        """
        HF static node features: cell area, elevation
        """
        hf_geom = self._get_hf_geometry()
        hf_area = hf_geom['cell_area']
        hf_elevation = hf_geom['cell_elevation']

        static_node_features = np.array([hf_area, hf_elevation]).transpose()
        return static_node_features


    def _get_hf_static_edge_features(self):
        """
        HF static edge features: edge length
        """
        hf_geom = self._get_hf_geometry()
        hf_coords = hf_geom['cell_coordinates']
        hf_edge_index = hf_geom['edge_index']
        n1_coords, n2_coords = hf_coords[hf_edge_index[0]], hf_coords[hf_edge_index[1]]
        hf_edge_lengths = np.linalg.norm(n1_coords - n2_coords, axis = 1)

        static_edge_features = np.array([hf_edge_lengths]).transpose()
        return static_edge_features


    def _get_hf_dynamic_data(self, event_idx: int):
        """
        Extracts HF water depth (target) and LF-to-HF upsampled water depth (used to calculate residual).

        Upsampling is done by finding the WSE of the nearest LF neighbour for each HF cell and
        subtracting the HF cell elevation, then clamping to be nonnegative.
        """

        # get LF water level (not depth!) and HF water depth
        if self.area_name == 'Carlisle':
            lf_water_level, _ = self._get_lf_wse_depth_carlisle(event_idx)
            hf_water_depth = self._get_hf_water_depth_carlisle(event_idx)
        elif self.area_name == 'Chowilla':
            lf_water_level, _ = self._get_lf_wse_depth_chowilla(event_idx)
            hf_water_depth = self._get_hf_water_depth_chowilla(event_idx)
        elif self.area_name == 'BurnettRV':
            lf_water_level, _ = self._get_lf_wse_depth_burnett(event_idx)
            hf_water_depth = self._get_hf_water_depth_burnett(event_idx)
        else:
            raise NotImplementedError

        # upsample
        hf_elevation = self._get_hf_geometry()['cell_elevation']
        ups_water_level = self._upsample_water_level(lf_water_level, hf_elevation)
        ups_water_depth = np.clip(ups_water_level - hf_elevation, a_min=0.0, a_max=DEPTH_CAP[self.area_name])

        return hf_water_depth, ups_water_depth


    def _upsample_water_level(self, lf_water_level, hf_elevation):
        lf_geom = self._get_lf_geometry()
        lf_coords = lf_geom['cell_coordinates']
        hf_geom = self._get_hf_geometry()
        hf_coords = hf_geom['cell_coordinates']

        lf_kdtree = KDTree(lf_coords)
        dists, nearest_lf_indices = lf_kdtree.query(hf_coords)
        ups_water_level = lf_water_level[:, nearest_lf_indices]
        ups_water_level = filter_dry_areas(ups_water_level, hf_elevation)

        return ups_water_level


    def _get_hf_water_depth_carlisle(self, event_idx: int):
        """
        Get HF water depth from npz file (Lisflood).
        """
        assert self.area_name == 'Carlisle', 'Mismatch between area name and function.'
        hf_path = self.hf_paths[event_idx]
        hf_data = np.load(hf_path)
        hf_geom = self._get_hf_geometry()
        hf_elevation = hf_geom['cell_elevation']

        hf_water_level = hf_data['wse_data']
        assert hf_water_level.shape[0] == self._get_num_timesteps(event_idx), 'HF water depth timestep mismatch with input timesteps.'
        hf_water_level = filter_dry_areas(hf_water_level, hf_elevation)
        hf_water_depth = np.maximum(0.0, hf_water_level - hf_elevation)

        return hf_water_depth


    def _get_hf_water_depth_chowilla(self, event_idx: int):
        """
        Get HF water depth from raw HECRAS file with ghost cells filtered out.
        For Chowilla, use the default warmup period of 10 hours.
        """
        assert self.area_name == 'Chowilla', 'Mismatch between area name and function.'
        hf_path = self.hf_paths[event_idx]
        hf_elevation = get_min_cell_elevation(hf_path, 'Chowilla')
        ghost_idx = np.where(np.isnan(hf_elevation))[0]
        hf_elevation = np.delete(hf_elevation, ghost_idx)

        hf_water_level = get_2d_hecras_results(hf_path, 'Chowilla', remove_ghost_cells=True)[0]  # Open dataset
        assert hf_water_level.shape[0] == self._get_num_timesteps(event_idx), 'HF water depth timestep mismatch with input timesteps.'
        hf_water_level = filter_dry_areas(hf_water_level, hf_elevation)
        hf_water_depth = np.maximum(0.0, hf_water_level - hf_elevation)

        return hf_water_depth

    
    def _get_hf_water_depth_burnett(self, event_idx: int):
        """
        Get HF water depth from npz file (Tuflow).
        For Burnett, need to preprocess nan values.
        """
        assert self.area_name == 'BurnettRV', 'Mismatch between area name and function.'
        hf_path = self.hf_paths[event_idx]
        hf_data = np.load(hf_path)
        hf_geom = self._get_hf_geometry()
        hf_elevation = hf_geom['cell_elevation']

        hf_water_level = np.nan_to_num(hf_data['wl_data'][48:,:], nan=-999) # Tuflow can have nans
        assert hf_water_level.shape[0] == self._get_num_timesteps(event_idx), 'HF water depth timestep mismatch with input timesteps.'
        hf_water_level = filter_dry_areas(hf_water_level, hf_elevation)
        hf_water_depth = np.maximum(0.0, hf_water_level - hf_elevation)

        return hf_water_depth
