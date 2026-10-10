import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing, Sequential as PygSequential


def get_activation_func(name: str) -> Module:
    if name == 'relu':
        return ReLU()
    raise Exception(f'Activation function {name} is not implemented.')


def make_mlp(
    input_size: int,
    output_size: int,
    hidden_size: int,
    num_layers: int = 2,
    activation: str = 'relu',
    layer_norm: str = True) -> Module:
    
    assert num_layers >= 2, 'Number of layers should be at least 2.'

    layers = []
    
    layers.append(Linear(input_size, hidden_size))
    layers.append(get_activation_func(activation))

    for _ in range(num_layers - 2): # middle layers
        layers.append(Linear(hidden_size, hidden_size))
        layers.append(get_activation_func(activation))

    layers.append(Linear(hidden_size, output_size))
    if layer_norm:
        layers.append(LayerNorm(output_size))

    return Sequential(*layers)


def make_gnn(
    input_node_size: int,
    input_edge_size: int,
    output_node_size: int,
    output_edge_size: int,
    hidden_features: int,
    num_gnn_layers: int,
    mlp_layers: int = 2,
    mlp_activation: str = 'relu',
    layer_norm: bool = True,
    residual: bool = True):

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
                mlp_layers = mlp_layers,
                mlp_activation = mlp_activation,
                layer_norm = layer_norm,
                residual = first_layer_residual),
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
                mlp_layers = mlp_layers,
                mlp_activation = mlp_activation,
                layer_norm = layer_norm,
                residual = first_layer_residual),
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
                    mlp_layers = mlp_layers,
                    mlp_activation = mlp_activation,
                    layer_norm = layer_norm,
                    residual = residual),
                'x, edge_index, edge_attr -> x, edge_attr'
            ))
        layers.append(( # last layer
            NodeEdgeConv(
                node_in_channels=hidden_features,
                edge_in_channels=hidden_features,
                node_out_channels=output_node_size,
                edge_out_channels=output_edge_size,
                hidden_size=hidden_features,
                mlp_layers = mlp_layers,
                mlp_activation = mlp_activation,
                layer_norm = layer_norm,
                residual = residual),
            'x, edge_index, edge_attr -> x, edge_attr'
        ))

    return PygSequential('x, edge_index, edge_attr', layers)


class NodeEdgeConv(MessagePassing):
    """
    Message = m_ij = MLP(x_i, e_ij, x_j)
    Aggregate = a_i = sum_j(m_ij)
    Node Update = MLP(x_i, a_i)
    Edge Update = Message
    """
    def __init__(self,
        node_in_channels: int,
        edge_in_channels: int,
        node_out_channels: int,
        edge_out_channels: int,
        hidden_size: int,
        mlp_layers: int = 2,
        mlp_activation: str = 'relu',
        layer_norm: bool = True,
        residual: bool = True):

        super().__init__(aggr='sum')

        self.msg_mlp = make_mlp(
            input_size = node_in_channels * 2 + edge_in_channels,
            output_size = edge_out_channels,
            hidden_size = hidden_size,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = layer_norm
        )

        self.node_mlp = make_mlp(
            input_size = node_in_channels + edge_out_channels,
            output_size = node_out_channels,
            hidden_size = hidden_size,
            num_layers = mlp_layers,
            activation = mlp_activation,
            layer_norm = layer_norm
        )
        
        if residual:
            assert node_in_channels == node_out_channels, 'Unable to perform residual on mismatched node sizes'
            assert edge_in_channels == edge_out_channels, 'Unable to perform residual on mismatched edge sizes'

        self.residual = residual

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        row, col = edge_index
        x_i, x_j = x[col], x[row]

        edge_update = self.msg_mlp(torch.cat([x_i, edge_attr, x_j], dim=-1))
        if self.residual:
            edge_update = edge_attr + edge_update

        node_update = self.propagate(edge_index, msg=edge_update, x=x)
        return node_update, edge_update

    def message(self, msg: Tensor) -> Tensor:
        return msg

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        out = self.node_mlp(torch.cat([x, aggr], dim=-1))
        if self.residual:
            out = out + x
        return out


def make_mlp_old(
    input_size: int,
    output_size: int,
    hidden_size: int = None,
    num_layers: int = 1,
    activation: str = None,
    norm: str = None,
    bias: bool = True,
    device: str = 'cpu') -> Module:

    layers = []

    if num_layers == 1:
        if norm is not None:
            layers.append(get_norm_layer(norm, input_size, device))
        layers.append(LinearLayer(input_size, output_size, activation, bias, device))
    else:
        hidden_size = hidden_size if hidden_size is not None else (input_size * 2)

        # input layer
        if norm is not None:
            layers.append(get_norm_layer(norm, input_size, device))
        layers.append(LinearLayer(input_size, hidden_size, activation, bias, device))

        # hidden layers
        for _ in range(num_layers-2):
            if norm is not None:
                layers.append(get_norm_layer(norm, hidden_size, device))
            layers.append(LinearLayer(hidden_size, hidden_size, activation, bias, device))
        
        # output layer
        if norm is not None:
            layers.append(get_norm_layer(norm, hidden_size, device))
        layers.append(LinearLayer(hidden_size, output_size, None, bias, device))

    return Sequential(*layers) if len(layers) > 1 else layers[0]


class LinearLayer(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = None,
        bias: bool = True,
        device: str = 'cpu'):
        super().__init__()
        self.in_features = in_features # For GINE compatibility
        self.out_features = out_features
        self.linear = Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)
        if activation is not None:
            self.activation = get_activation_func(activation, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        return x

    def reset_parameters(self):
        self.linear.reset_parameters()
        if hasattr(self, 'activation') and hasattr(self.activation, 'reset_parameters'):
            self.activation.reset_parameters()