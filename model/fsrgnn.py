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
        
        mlp_norm: str = None):

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
            bias=False)

        self.edge_encoder = make_mlp(
            input_size=lf_static_edge_features,
            output_size=hidden_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=encoder_layers,
            activation=encoder_activation,
            norm=mlp_norm,
            bias=False)

        self.hf_edge_encoder = make_mlp(
            input_size=hf_static_edge_features,
            output_size=hidden_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=encoder_layers,
            activation=encoder_activation,
            norm=mlp_norm,
            bias=False)

        self.lfgnn = self._make_gnn(
            input_node_size=hidden_features,
            input_edge_size=hidden_features,
            output_node_size=hidden_features,
            output_edge_size=hidden_features,
            hidden_features=hidden_features,
            num_gnn_layers=lfgnn_layers,
            mlp_layers=lfgnn_mlp_layers,
            mlp_norm=mlp_norm,
            activation=lfgnn_activation)

        self.hfgnn = self._make_gnn(
            input_node_size=hidden_features,
            input_edge_size=hidden_features,
            output_node_size=hidden_features,
            output_edge_size=hidden_features,
            hidden_features=hidden_features,
            num_gnn_layers=hfgnn_layers,
            mlp_layers=hfgnn_mlp_layers,
            mlp_norm=mlp_norm,
            activation=hfgnn_activation)
        
        self.node_decoder = make_mlp(
            input_size=hidden_features,
            output_size=output_features,
            hidden_size=encoder_decoder_hidden,
            num_layers=decoder_layers,
            activation=decoder_activation,
            norm=mlp_norm,
            bias=False)

        # no edge decoder because we are only interested in node-level regression

    def forward(self, lf_x: Tensor, lf_coords: Tensor, lf_edge_index: Tensor, lf_edge_attr: Tensor,
        hf_x: Tensor, hf_coords: Tensor, hf_edge_index: Tensor, hf_edge_attr: Tensor) -> Tensor:
        x_lf = self.node_encoder(lf_x)
        e_lf = self.edge_encoder(lf_edge_attr)

        x_lf, _ = self.lfgnn(x_lf, lf_edge_index, e_lf) # ignore edge features because they are not used in upscaling

        x_hf = self.upsample(x_lf, lf_coords, hf_coords, k = 4)

        # TODO use HF static node features, either concatenate before hfgnn or put into MLP for attention style weights

        e_hf = self.hf_edge_encoder(hf_edge_attr)
        x_hf, _ = self.hfgnn(x_hf, hf_edge_index, e_hf)

        y_pred = self.node_decoder(x_hf)
        return y_pred

    def upsample(self,
        x_lf: Tensor, # [N_lf, hidden features]
        lf_coords: Tensor, # [N_lf, 2]
        hf_coords: Tensor, # [N_hf, 2]
        k: int = 4):

        EPS = 1e-8
        # for each hf node, find k lf neighbours and upsample their hidden features
        hf_indices, lf_indices = knn(x=lf_coords, y=hf_coords, k=k)

        # inverse distance weighting
        dists = torch.norm(hf_coords[hf_indices] - lf_coords[lf_indices], p=2, dim=-1)
        w = 1.0 / (dists ** 2 + EPS)

        # reshape to [N_hf, k]
        w = w.view(-1, k)
        lf_indices = lf_indices.view(-1, k)

        # normalise weights by their sum
        w = w / (w.sum(dim=-1, keepdim=True) + EPS)

        x_hf = torch.sum(x_lf[lf_indices] * w.unsqueeze(-1), dim=1)
        return x_hf # [N_hf, hidden_features]
        
    def _make_gnn(self,
        input_node_size: int,
        input_edge_size: int,
        output_node_size: int,
        output_edge_size: int,
        hidden_features: int,
        num_gnn_layers: int,
        mlp_layers: int,
        activation: str,
        mlp_norm: str):

        layers = []

        if num_gnn_layers == 1:
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
                    bias=False),
                    'x, edge_index, edge_attr -> x, edge_attr'
            ))

        else: # multilayer GNN
            layers = []
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
                    bias=False),
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
                        bias=False),
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
                    bias=False),
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
        bias: bool = False):

        super().__init__(aggr='sum')
        
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
        i, j = edge_index
        x_i, x_j = x[i], x[j]

        # message logic here
        cat_feats = torch.cat([x_i, edge_attr, x_j], dim=-1)
        edge_update = self.msg_mlp(cat_feats)

        node_update = self.propagate(edge_index, msg=edge_update)
        return node_update, edge_update

    def message(self, msg: Tensor) -> Tensor:
        return msg

    def update(self, aggr: Tensor) -> Tensor:
        out = self.node_mlp(aggr)
        return out
