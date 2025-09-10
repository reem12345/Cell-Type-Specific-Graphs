from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SGConv, Linear, sequential, TransformerConv, global_mean_pool
import torch
import torch_geometric.utils
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import *
import torch
import pickle


class MLP(torch.nn.Module):
    def __init__(self, sizes, mid_layer_act, batch_norm=False, last_layer_act="linear"):
        """
        Multi-layer perceptron (MLP) implementation
        :param sizes: list containing the sizes of each layer
        :param mid_layer_act: activation function for the middle layers ('Sigmoid' or 'Softplus')
        :param batch_norm: whether to include batch normalization after hidden layers
        :param last_layer_act: activation function for the last layer (default is "linear")
        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            # Add linear layer from sizes[s] -> sizes[s+1]
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                # Optionally add BatchNorm if enabled (skip for last layer)
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                # Add activation function (choose Sigmoid or Softplus)
                torch.nn.Sigmoid() if mid_layer_act == 'Sigmoid' else torch.nn.Softplus()
            ]

        # Remove None layers if batch_norm=False and drop last activation
        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act  # store last activation (not directly applied here)
        self.network = torch.nn.Sequential(*layers)  # build sequential MLP

    def forward(self, x):
        # Forward pass through the network
        return self.network(x)


class GNN(torch.nn.Module):
    def __init__(self, total_genes, num_perts, hidden_channels, in_head,
                 output_channels=1, act_fct=None, multi_pert=True):
        """
        Graph Neural Network (GNN) model
        :param total_genes: number of genes (input size)
        :param num_perts: number of perturbations
        :param hidden_channels: hidden dimension size
        :param in_head: number of attention heads
        :param output_channels: output dimension (default=1)
        :param act_fct: activation function for MLPs
        :param multi_pert: whether to include perturbation embeddings
        """
        super().__init__()
        torch.manual_seed(42)  # fix random seed for reproducibility

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert

        # First convolution: Transformer-based convolution layer
        self.conv1 = TransformerConv(-1, hidden_channels, heads=self.in_head)
        # Linear layer for projection after conv1
        self.lin1 = Linear(-1, hidden_channels * self.in_head)

        # Second convolution: Graph Attention (GAT) layer
        self.conv2 = GATConv(-1, hidden_channels, heads=self.in_head,
                             add_self_loops=True, concat=False)
        # Linear projection after conv2
        self.lin2 = Linear(-1, hidden_channels)

        # Perturbation embedding network (fixed size [124 -> 124])
        self.embd_pert = MLP([124, 124], self.act_fct)

        # Final prediction network (depends on multi_pert flag)
        self.lin_predict = None
        if multi_pert == True:
            # Input includes genes + hidden features + perturbation embedding
            self.lin_predict = MLP([self.total_genes + hidden_channels + 124,
                                    1024, self.total_genes], self.act_fct)
        else:
            # Input includes genes + hidden features only
            self.lin_predict = MLP([self.total_genes + (hidden_channels * self.in_head),
                                    1024, self.total_genes], self.act_fct)

    def forward(self, x, edge_index, cell_line, cell_type_keys, ctrl, pert, pos):
        """
        Forward pass of the GNN
        :param x: dict of node feature tensors per cell type
        :param edge_index: dict of edge indices per cell type
        :param cell_line: list of cell types present in batch
        :param cell_type_keys: all available cell type keys
        :param ctrl: control input features
        :param pert: perturbation input
        :param pos: positional encodings (currently unused)
        """
        for key in cell_type_keys:
            edge_index[key] = edge_index[key]  # keep edge structure
            # Apply first conv + linear projection
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            # Aggregate node features with max pooling
            x[key] = torch.max(x[key], dim=0)[0].unsqueeze(0)

        # Collect features for all cell lines in batch
        cell_type_fet = [x[c] for c in cell_line]
        cell_type_fet = torch.cat(cell_type_fet, dim=0)

        if self.multi_pert == False:
            # Concatenate control and cell-type features
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            # Predict gene expression
            x = self.lin_predict(x)
        if self.multi_pert == True:
            # Concatenate control and cell-type features
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            # Add perturbation embedding
            x = torch.cat([x, self.embd_pert(pert.to(torch.float32))], dim=1)
            # Predict gene expression
            x = self.lin_predict(x)

        return x
        
