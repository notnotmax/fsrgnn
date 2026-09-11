import numpy as np

from data.hecras_data_retrieval import get_cell_coordinates, get_min_cell_elevation,\
        get_cell_area, get_roughness, get_faces_cell_indexes
from scipy.spatial import KDTree

def get_lf_geometry(area_name: str, input_path: str, output_path: str):
    
    # LF geometry assumed to be from HEC-RAS
    print(f"Processing LF geometry of {area_name} from {input_path}.")
    cell_coordinates = get_cell_coordinates(input_path, area_name)
    cell_elevation = get_min_cell_elevation(input_path, area_name)
    cell_area = get_cell_area(input_path, area_name)
    cell_roughness = get_roughness(input_path, area_name)
    faces_cell_indexes = get_faces_cell_indexes(input_path, area_name, int)

    # ---------- filter out ghost cells ----------
    real_cell_ids = np.where(~np.isnan(cell_elevation))[0]
    cell_id_map = {old_id: new_id for new_id, old_id in enumerate(real_cell_ids)}

    cell_coordinates = cell_coordinates[real_cell_ids]
    cell_elevation = cell_elevation[real_cell_ids]
    cell_area = cell_area[real_cell_ids]
    cell_roughness = cell_roughness[real_cell_ids]

    # ---------- construct edges ----------
    valid_c0 = np.isin(faces_cell_indexes[:, 0], real_cell_ids)
    valid_c1 = np.isin(faces_cell_indexes[:, 1], real_cell_ids)
    internal_face_mask = valid_c0 & valid_c1
    internal_faces = faces_cell_indexes[internal_face_mask]
    edges_src = np.array([cell_id_map[c] for c in internal_faces[:, 0]], dtype=np.int64)
    edges_dst = np.array([cell_id_map[c] for c in internal_faces[:, 1]], dtype=np.int64)
    
    edge_index = np.array([ # make bidirectional edge
        np.concatenate([edges_src, edges_dst]),
        np.concatenate([edges_dst, edges_src])
    ], dtype=np.int64)

    # ---------- save ----------
    np.savez_compressed(
        output_path,
        cell_coordinates=cell_coordinates,
        cell_elevation=cell_elevation,
        cell_area=cell_area,
        cell_roughness=cell_roughness,
        edge_index=edge_index
    )

    print(f"Saved LF geometry data to {output_path}.")

def get_hf_geometry_from_hecras(area_name, input_path: str, output_path: str):

    print(f"Processing HF geometry of {area_name} from HEC-RAS file from {input_path}.")
    cell_coordinates = get_cell_coordinates(input_path, area_name)
    cell_elevation = get_min_cell_elevation(input_path, area_name)
    cell_area = get_cell_area(input_path, area_name)
    cell_roughness = get_roughness(input_path, area_name)
    faces_cell_indexes = get_faces_cell_indexes(input_path, area_name, int)

    # ---------- filter out ghost cells ----------
    real_cell_ids = np.where(~np.isnan(cell_elevation))[0]
    cell_id_map = {old_id: new_id for new_id, old_id in enumerate(real_cell_ids)}

    cell_coordinates = cell_coordinates[real_cell_ids]
    cell_elevation = cell_elevation[real_cell_ids]
    cell_area = cell_area[real_cell_ids]
    cell_roughness = cell_roughness[real_cell_ids]

    # ---------- construct edges ----------
    valid_c0 = np.isin(faces_cell_indexes[:, 0], real_cell_ids)
    valid_c1 = np.isin(faces_cell_indexes[:, 1], real_cell_ids)
    internal_face_mask = valid_c0 & valid_c1
    internal_faces = faces_cell_indexes[internal_face_mask]
    edges_src = np.array([cell_id_map[c] for c in internal_faces[:, 0]], dtype=np.int64)
    edges_dst = np.array([cell_id_map[c] for c in internal_faces[:, 1]], dtype=np.int64)
    
    edge_index = np.array([ # make bidirectional edge
        np.concatenate([edges_src, edges_dst]),
        np.concatenate([edges_dst, edges_src])
    ], dtype=np.int64)

    # ---------- save ----------
    np.savez_compressed(
        output_path,
        cell_coordinates=cell_coordinates,
        cell_elevation=cell_elevation,
        cell_area=cell_area,
        cell_roughness=cell_roughness,
        edge_index=edge_index
    )

    print(f"Saved HF geometry data to {output_path}.")

def get_hf_geometry_from_npz(input_path: str, output_path: str):

    print(f"Processing HF geometry from {input_path}.")
    # We assume edges are not provided in the npz file
    hf_geom = np.load(input_path)
    print('Available keys in npz file', hf_geom.keys())
    # using the keys given in the dataset
    cell_coordinates = hf_geom['XY_coor']
    cell_elevation = hf_geom['Z_coor']
    cell_area = hf_geom['Area']

    # --------- construct edges with kNN ----------
    k = 4
    num_nodes = cell_coordinates.shape[0]
    tree = KDTree(cell_coordinates)
    _, neighbor_indices = tree.query(cell_coordinates, k=k+1) # k+1 to get k non-self neighbours
    source_nodes = np.repeat(np.arange(num_nodes)[:, None], k + 1, axis=1)
    src = source_nodes.ravel()
    dst = neighbor_indices.ravel()
    
    # remove edges to self
    mask = (src != dst)
    src = src[mask]
    dst = dst[mask]
    edge_index = np.vstack([src, dst]).astype(np.int64)

    np.savez_compressed(
        output_path,
        cell_coordinates=cell_coordinates,
        cell_elevation=cell_elevation,
        cell_area=cell_area,
        edge_index=edge_index
    )

    print(f"Saved HF geometry data to {output_path}.")

if __name__ == '__main__':

    # we assume geometry to be static across runs, so we take from one known file
    CARLISLE_NAME = 'Carlisle'
    CARLISLE_LF_INPUT_PATH = '../dataset/Carlisle/HD_model_data/Low-fidelity/Carlisle_LFmodelA.p01.hdf'
    CARLISLE_LF_OUTPUT_PATH = 'data/Carlisle/LF_geometry_data.npz'
    CARLISLE_HF_INPUT_PATH = '../dataset/Carlisle/Geometry_data/Lisflood_Geometry_data.npz'
    CARLISLE_HF_OUTPUT_PATH = 'data/Carlisle/HF_geometry_data.npz'

    CHOWILLA_NAME = 'Chowilla'
    CHOWILLA_LF_INPUT_PATH = '../dataset/Chowilla/HD_model_data/Low-fidelity/Chow_LF_modelG.p01.hdf'
    CHOWILLA_LF_OUTPUT_PATH = 'data/Chowilla/LF_geometry_data.npz'
    CHOWILLA_HF_INPUT_PATH = '../dataset/Chowilla/HD_model_data/High-fidelity/Chow_HF.p01.hdf'
    CHOWILLA_HF_OUTPUT_PATH = 'data/Chowilla/HF_geometry_data.npz'

    BURNETT_NAME = 'BurnettRV_region'
    BURNETT_LF_INPUT_PATH = '../dataset/BurnettRV/HD_model_data/Low-fidelity/BurnettRV_LFmodelB.p01.hdf'
    BURNETT_LF_OUTPUT_PATH = 'data/BurnettRV/LF_geometry_data.npz'
    BURNETT_HF_INPUT_PATH = '../dataset/BurnettRV/Geometry_data/Tuflow_Geometry_data.npz'
    BURNETT_HF_OUTPUT_PATH = 'data/BurnettRV/HF_geometry_data.npz'

    get_lf_geometry(CARLISLE_NAME, CARLISLE_LF_INPUT_PATH, CARLISLE_LF_OUTPUT_PATH)
    lf_data = np.load(CARLISLE_LF_OUTPUT_PATH, allow_pickle=True)

    # get_hf_geometry_from_npz(CARLISLE_HF_INPUT_PATH, CARLISLE_HF_OUTPUT_PATH)
    # # get_HF_geometry_from_hecras(CHOWILLA_NAME, CHOWILLA_HF_INPUT_PATH, CHOWILLA_HF_OUTPUT_PATH)
    
