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

    def __init__(self, sizes,mid_layer_act, batch_norm=False, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.Sigmoid() if mid_layer_act == 'Sigmoid' else torch.nn.Softplus()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class GNN(torch.nn.Module):
    def __init__(self, total_genes,num_perts, hidden_channels,in_head, output_channels = 1, act_fct = None, multi_pert = True):
        super().__init__()
        torch.manual_seed(42) 

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert
        # GNN layers
        self.conv1 = TransformerConv(-1, hidden_channels, heads=self.in_head) 
        self.lin1 = Linear(-1, hidden_channels*self.in_head)
        self.conv2 = GATConv(-1, hidden_channels, heads=self.in_head,  add_self_loops=True, concat = False)  
        self.lin2 = Linear(-1, hidden_channels)

        self.embd_pert = MLP([124, 124], self.act_fct)
        self.lin_predict = None
        if multi_pert == True: 
            self.lin_predict = MLP([self.total_genes+hidden_channels + 124, 1024, self.total_genes], self.act_fct)
        else:
            self.lin_predict = MLP([self.total_genes+(hidden_channels*self.in_head), 1024, self.total_genes], self.act_fct)

        
    def forward(self,x, edge_index, cell_line, cell_type_keys,ctrl, pert, pos):
        for key in cell_type_keys:
            edge_index[key] = edge_index[key] 
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            x[key] = torch.max(x[key], dim = 0)[0].unsqueeze(0)
            
        cell_type_fet = [x[c] for c in cell_line]
        cell_type_fet = torch.cat(cell_type_fet, dim = 0)
        if self.multi_pert == False:
            x = torch.cat([ ctrl, cell_type_fet], dim = 1)
            x = self.lin_predict(x)
        if self.multi_pert == True:
            x = torch.cat([ ctrl, cell_type_fet], dim = 1)
            x = torch.cat([x, self.embd_pert(pert.to(torch.float32))], dim = 1) 
            x = self.lin_predict(x)
        return x
        