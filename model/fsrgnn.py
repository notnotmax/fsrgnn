# Adapted from https://github.com/acostacos/dual_flood_gnn

import torch
from torch import Tensor
from torch.nn import Module
from torch_cluster import knn
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from typing import Tuple
from utils.model_utils import make_mlp

class FSRGNN(Module):

    def __init__(self,
        lf_static_node_features: int,
        lf_dynamic_node_features: int,
        lf_static_edge_features: int,
        hf_static_node_features: int,
        hf_static_edge_features: int,
        hidden_features: int,
        output_features: int,

        # number of layers
        encoder_layers: int,
        lfgnn_layers: int,
        lfgnn_mlp_layers: int,
        hfgnn_layers: int,
        hfgnn_mlp_layers: int,
        decoder_layers: int,

        # activations
        encoder_activation: str = 'relu',
        lfgnn_activation: str = 'relu',
        hfgnn_activation: str = 'relu',
        decoder_activation: str = 'relu',
        
        # misc
        upsampler_k: int = 4,
        mlp_norm: str = None,
        residual: bool = False,
        bias: bool = False):

        super().__init__()

        lf_node_features = lf_static_node_features + lf_dynamic_node_features
        encoder_decoder_hidden = hidden_features * 2

        self.node_encoder = make_mlp(
            input_size=lf_node_features,
            output_size=hidden_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=encoder_layers,
            activation=encoder_activation,
            norm=mlp_norm,
            bias=bias)

        self.edge_encoder = make_mlp(
            input_size=lf_static_edge_features,
            output_size=hidden_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=encoder_layers,
            activation=encoder_activation,
            norm=mlp_norm,
            bias=bias)

        self.hf_node_encoder = make_mlp(
            input_size=hf_static_node_features,
            output_size=hidden_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=encoder_layers,
            activation=encoder_activation,
            norm=mlp_norm,
            bias=bias)

        self.hf_edge_encoder = make_mlp(
            input_size=hf_static_edge_features,
            output_size=hidden_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=encoder_layers,
            activation=encoder_activation,
            norm=mlp_norm,
            bias=bias)

        self.lfgnn = self._make_gnn(
            input_node_size=hidden_features,
            input_edge_size=hidden_features,
            output_node_size=hidden_features,
            output_edge_size=hidden_features,
            hidden_features=hidden_features,
            num_gnn_layers=lfgnn_layers,
            mlp_layers=lfgnn_mlp_layers,
            mlp_norm=mlp_norm,
            activation=lfgnn_activation,
            residual=residual,
            bias=bias)

        self.hfgnn = self._make_gnn(
            input_node_size=hidden_features * 2, # using concatenated lf and hf
            input_edge_size=hidden_features,
            output_node_size=hidden_features,
            output_edge_size=hidden_features,
            hidden_features=hidden_features,
            num_gnn_layers=hfgnn_layers,
            mlp_layers=hfgnn_mlp_layers,
            mlp_norm=mlp_norm,
            activation=hfgnn_activation,
            residual=residual,
            bias=bias)
        
        self.node_decoder = make_mlp(
            input_size=hidden_features,
            output_size=output_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=decoder_layers,
            activation=decoder_activation,
            norm=mlp_norm,
            bias=bias)

        self.upsampler_k = upsampler_k
        self.num_lf_nodes = None
        self.num_hf_nodes = None
        self.register_buffer('upsampler_neighbour_indices', None, persistent=False)
        self.register_buffer('upsampler_weights', None, persistent=False)
    
    def init_upsampler(self, lf_coords, hf_coords):
        if self.upsampler_neighbour_indices is not None or self.upsampler_weights is not None:
            assert False, 'Attempted to initialise upsampler weights twice.'

        EPS = 1e-8
        hf_indices, lf_indices = knn(x=lf_coords, y=hf_coords, k=self.upsampler_k)
        dists = torch.norm(hf_coords[hf_indices] - lf_coords[lf_indices], p=2, dim=-1)
        w = 1.0 / (dists ** 2 + EPS)
        w = w.view(-1, self.upsampler_k)
        lf_indices = lf_indices.view(-1, self.upsampler_k)
        w = w / (w.sum(dim=-1, keepdim=True) + EPS)

        self.num_lf_nodes = lf_coords.shape[0]
        self.num_hf_nodes = hf_coords.shape[0]
        self.upsampler_neighbour_indices = lf_indices
        self.upsampler_weights = w

    def forward(self, lf_x: Tensor, lf_edge_index: Tensor, lf_edge_attr: Tensor,
        hf_x: Tensor, hf_edge_index: Tensor, hf_edge_attr: Tensor, num_graphs: int) -> Tensor:
        x_lf = self.node_encoder(lf_x)
        e_lf = self.edge_encoder(lf_edge_attr)

        x_lf, _ = self.lfgnn(x_lf, lf_edge_index, e_lf) # ignore edge features because they are not used in upscaling

        x_lf_upsampled = self.upsample(x_lf, num_graphs)

        x_hf = self.hf_node_encoder(hf_x)

        x_hf_combined = torch.cat([x_lf_upsampled, x_hf], dim=-1)

        e_hf = self.hf_edge_encoder(hf_edge_attr)
        x_hf, _ = self.hfgnn(x_hf_combined, hf_edge_index, e_hf)

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
        
    def _make_gnn(self,
        input_node_size: int,
        input_edge_size: int,
        output_node_size: int,
        output_edge_size: int,
        hidden_features: int,
        num_gnn_layers: int,
        mlp_layers: int,
        activation: str,
        mlp_norm: str,
        residual: bool,
        bias: bool):

        layers = []

        if num_gnn_layers == 1:
            first_layer_residual = residual and (input_node_size == output_node_size) and (input_edge_size == output_edge_size)
            layers.append((
                NodeEdgeConv(
                    node_in_channels=input_node_size,
                    edge_in_channels=input_edge_size,
                    node_out_channels=output_node_size,
                    edge_out_channels=output_edge_size,
                    hidden_size=hidden_features,
                    num_layers=mlp_layers,
                    activation=activation,
                    mlp_norm=mlp_norm,
                    residual=first_layer_residual,
                    bias=bias),
                    'x, edge_index, edge_attr -> x, edge_attr'
            ))

        else: # multilayer GNN
            layers = []
            first_layer_residual = residual and (input_node_size == hidden_features) and (input_edge_size == hidden_features)
            layers.append(( # first layer
                NodeEdgeConv(
                    node_in_channels=input_node_size,
                    edge_in_channels=input_edge_size,
                    node_out_channels=hidden_features,
                    edge_out_channels=hidden_features,
                    hidden_size=hidden_features,
                    num_layers=mlp_layers,
                    activation=activation,
                    mlp_norm=mlp_norm,
                    residual=first_layer_residual,
                    bias=bias),
                    'x, edge_index, edge_attr -> x, edge_attr'
            ))
            for _ in range(num_gnn_layers - 2):
                layers.append(( # middle layers
                    NodeEdgeConv(
                        node_in_channels=hidden_features,
                        edge_in_channels=hidden_features,
                        node_out_channels=hidden_features,
                        edge_out_channels=hidden_features,
                        hidden_size=hidden_features,
                        num_layers=mlp_layers,
                        activation=activation,
                        mlp_norm=mlp_norm,
                        residual=residual,
                        bias=bias),
                    'x, edge_index, edge_attr -> x, edge_attr'
                ))
            layers.append(( # last layer
                NodeEdgeConv(
                    node_in_channels=hidden_features,
                    edge_in_channels=hidden_features,
                    node_out_channels=output_node_size,
                    edge_out_channels=output_edge_size,
                    hidden_size=hidden_features,
                    num_layers=mlp_layers,
                    activation=activation,
                    mlp_norm=mlp_norm,
                    residual=residual,
                    bias=bias),
                'x, edge_index, edge_attr -> x, edge_attr'
            ))

        return PygSequential('x, edge_index, edge_attr', layers)

class NodeEdgeConv(MessagePassing):
    """
    Message = m_ij = MLP(x_i, e_ij, x_j)
    Aggregate = a_i = sum_j(m_ij)
    Node Update = MLP(a_i)
    Edge Update = Message
    """
    def __init__(self,
        node_in_channels: int,
        edge_in_channels: int,
        node_out_channels: int,
        edge_out_channels: int,
        hidden_size: int,
        num_layers: int = 2,
        activation: str = 'relu',
        mlp_norm: str = None,
        residual: bool = False,
        bias: bool = False):

        super().__init__(aggr='sum')

        self.residual = residual
        
        self.msg_mlp = make_mlp(
            input_size=node_in_channels * 2 + edge_in_channels,
            output_size=edge_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            norm=mlp_norm,
            bias=bias)
        
        self.node_mlp = make_mlp(
            input_size=edge_out_channels,
            output_size=node_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            norm=mlp_norm,
            bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        row, col = edge_index
        x_i, x_j = x[col], x[row]

        # message logic here
        cat_feats = torch.cat([x_i, edge_attr, x_j], dim=-1)
        edge_update = self.msg_mlp(cat_feats)

        if self.residual:
            edge_update = edge_update + edge_attr

        node_update = self.propagate(edge_index, msg=edge_update, x=x)
        return node_update, edge_update

    def message(self, msg: Tensor) -> Tensor:
        return msg

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        out = self.node_mlp(aggr)
        if self.residual:
            out = out + x
        return out
