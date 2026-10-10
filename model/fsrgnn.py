import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch_cluster import knn
from model.model_utils import make_mlp, make_gnn

class FSRGNN(Module):

    def __init__(self,
        lf_node_features: int,
        lf_edge_features: int,
        hf_node_features: int,
        hf_edge_features: int,
        hidden_features: int,
        output_features: int,
        mlp_layers: int = 2,
        lfgnn_layers: int = 6,
        hfgnn_layers: int = 6,
        mlp_activation: str = 'relu',
        layer_norm: bool = True,
        residual: bool = True,
        upsampler_k: int = 4):

        super().__init__()

        self.lf_node_encoder = make_mlp(
            input_size = lf_node_features,
            output_size = hidden_features,
            hidden_size = hidden_features,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = layer_norm
        )

        self.lf_edge_encoder = make_mlp(
            input_size = lf_edge_features,
            output_size = hidden_features,
            hidden_size = hidden_features,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = layer_norm
        )

        self.hf_node_encoder = make_mlp(
            input_size = hf_node_features,
            output_size = hidden_features,
            hidden_size = hidden_features,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = layer_norm
        )

        self.hf_edge_encoder = make_mlp(
            input_size = hf_edge_features,
            output_size = hidden_features,
            hidden_size = hidden_features,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = layer_norm
        )

        self.lfgnn = make_gnn(
            input_node_size=hidden_features,
            input_edge_size=hidden_features,
            output_node_size=hidden_features,
            output_edge_size=hidden_features,
            hidden_features=hidden_features,
            num_gnn_layers=lfgnn_layers,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            layer_norm=layer_norm,
            residual=residual)

        self.hfgnn = make_gnn(
            input_node_size=hidden_features * 2, # using concatenated lf and hf
            input_edge_size=hidden_features,
            output_node_size=hidden_features,
            output_edge_size=hidden_features,
            hidden_features=hidden_features,
            num_gnn_layers=hfgnn_layers,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            layer_norm=layer_norm,
            residual=residual)

        self.node_decoder = make_mlp(
            input_size = hidden_features,
            output_size = output_features,
            hidden_size = hidden_features,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = False
        )

        self.upsampler_k = upsampler_k
        self.num_lf_nodes = None
        self.num_hf_nodes = None
        self._edge_cache = {}

        self.register_buffer('lf_coords', None, persistent=False)
        self.register_buffer('lf_edge_index', None, persistent=False)
        self.register_buffer('lf_static_x', None, persistent=False)
        self.register_buffer('lf_edge_attr', None, persistent=False)

        self.register_buffer('hf_coords', None, persistent=False)
        self.register_buffer('hf_edge_index', None, persistent=False)
        self.register_buffer('hf_static_x', None, persistent=False)
        self.register_buffer('hf_edge_attr', None, persistent=False)

        self.register_buffer('upsampler_neighbour_indices', None, persistent=False)
        self.register_buffer('upsampler_weights', None, persistent=False)


    def init_static_features(self, static_features_path: str):
        if self.lf_coords is not None:
            assert False, 'Attempted to initialise static features twice.'
        static_features = np.load(static_features_path)

        self.lf_coords = torch.from_numpy(static_features['lf_coords']).float()
        self.num_lf_nodes = self.lf_coords.size(0)
        self.lf_edge_index = torch.from_numpy(static_features['lf_edge_index']).long()
        self.lf_static_x = self._standardise(
            torch.from_numpy(static_features['lf_static_node_features']).float())
        self.lf_edge_attr = self._standardise(
            torch.from_numpy(static_features['lf_static_edge_features']).float())

        self.hf_coords = torch.from_numpy(static_features['hf_coords']).float()
        self.num_hf_nodes = self.hf_coords.size(0)
        self.hf_edge_index = torch.from_numpy(static_features['hf_edge_index']).long()
        self.hf_static_x = self._standardise(
            torch.from_numpy(static_features['hf_static_node_features']).float())
        self.hf_edge_attr = self._standardise(
            torch.from_numpy(static_features['hf_static_edge_features']).float())

        self._init_upsampler(self.lf_coords, self.hf_coords)


    def _standardise(self, x: Tensor, eps: float=1e-8):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + eps)
    

    def _init_upsampler(self, lf_coords, hf_coords, eps: float=1e-8):
        if self.upsampler_neighbour_indices is not None or self.upsampler_weights is not None:
            assert False, 'Attempted to initialise upsampler weights twice.'

        hf_indices, lf_indices = knn(x=lf_coords, y=hf_coords, k=self.upsampler_k)
        dists = torch.norm(hf_coords[hf_indices] - lf_coords[lf_indices], p=2, dim=-1)
        w = 1.0 / (dists ** 2 + eps)
        w = w.view(-1, self.upsampler_k)
        lf_indices = lf_indices.view(-1, self.upsampler_k)
        w = w / (w.sum(dim=-1, keepdim=True) + eps)

        self.upsampler_neighbour_indices = lf_indices
        self.upsampler_weights = w


    def _batched_edges(self, name: str, num_nodes: int, batch_size: int):
        key = (name, batch_size)
        if key not in self._edge_cache:
            edge_index = getattr(self, name)
            off = (torch.arange(batch_size, device=edge_index.device) * num_nodes).view(batch_size, 1, 1)
            self._edge_cache[key] = (edge_index.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1)
        return self._edge_cache[key]


    def forward(self, lf_dyn: Tensor, batch_size: int) -> Tensor:
        lf_x = torch.cat([self.lf_static_x.repeat(batch_size, 1), lf_dyn], dim=-1)
        x_lf = self.lf_node_encoder(lf_x)
        e_lf = self.lf_edge_encoder(self.lf_edge_attr).repeat(batch_size, 1)
        x_lf, _ = self.lfgnn(
            x_lf,
            self._batched_edges('lf_edge_index', self.num_lf_nodes, batch_size),
            e_lf
        )
        x_ups = self.upsample(x_lf, batch_size)
        x_hf = self.hf_node_encoder(self.hf_static_x).repeat(batch_size, 1)
        e_hf = self.hf_edge_encoder(self.hf_edge_attr).repeat(batch_size, 1)
        x_hf, _ = self.hfgnn(
            torch.cat([x_ups, x_hf], dim=-1),
            self._batched_edges('hf_edge_index', self.num_hf_nodes, batch_size),
            e_hf
        )
        y_pred = self.node_decoder(x_hf)
        return y_pred


    def upsample(self, x_lf: Tensor, num_graphs: int):
        device = x_lf.device

        # offset each graph's local lf indices into the batch indexing
        offsets = (torch.arange(num_graphs, device=device) * self.num_lf_nodes).view(num_graphs, 1, 1)
        idx = (self.upsampler_neighbour_indices.unsqueeze(0) + offsets).reshape(-1, self.upsampler_k) # [B * N_hf, k]
        w = self.upsampler_weights.unsqueeze(0).expand(num_graphs, -1, -1).reshape(-1, self.upsampler_k) # [B * N_hf, k]

        gathered = x_lf[idx] # [B * N_hf, k, h]
        x_hf_upsampled = torch.sum(gathered * w.unsqueeze(-1), dim=1) # [B * N_hf, h]
        return x_hf_upsampled
