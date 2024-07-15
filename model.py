from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SGConv, Linear, sequential
import torch
import torch_geometric.utils
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import *

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
    def __init__(self, total_genes,num_perts, hidden_channels,in_head, output_channels = 1,all_feat = True, act_fct = 'Sigmoid'):
        super().__init__()
        torch.manual_seed(42) 

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.all_feat = all_feat
        self.act_fct = act_fct
        print(hidden_channels)
        # GNN layers
        self.conv1 = GATConv(-1, hidden_channels, heads=self.in_head,  add_self_loops=True, concat = False) 
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv(-1, 1, heads=self.in_head,  add_self_loops=True, concat = False)  
        self.lin2 = Linear(-1, 1)

        self.embd_pert = MLP([600, 128, hidden_channels], self.act_fct)
        self.gene_emed = torch.nn.Embedding(total_genes, hidden_channels)

    
        self.lin_predict = MLP([self.total_genes+hidden_channels, 1024, self.total_genes], self.act_fct)
        self.lin_predict_single = MLP([self.total_genes, 1024, self.total_genes], self.act_fct)
        self.Sigmoid = torch.nn.Sigmoid()

        
    def forward(self,x, edge_index, cell_line, cell_type_keys,ctrl, pert, pos, multi_pert):
        for key in cell_type_keys:
            pos[key] = self.gene_emed(pos[key])
            edge_index[key] = edge_index[key] 
            if self.all_feat:
                x[key] = torch.cat([x[key] , pos[key]], dim = 1) 
            else: 
                x[key] = pos[key]
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            x[key] = self.Sigmoid(x[key])
            x[key] = self.conv2(x[key], edge_index[key]) + self.lin2(x[key])
            x[key] = self.Sigmoid(x[key])
            x[key] = to_dense_batch(x[key] , batch_size = 1 ,fill_value = 0, max_num_nodes = self.total_genes)[0].squeeze(2)
            
            
        cell_type_fet = [x[c] for c in cell_line]
        cell_type_fet = torch.cat(cell_type_fet, dim = 0)
        # Perturbation + cell type features
        if multi_pert == False:
            x = torch.cat([cell_type_fet + ctrl], dim = 1)
            x = self.lin_predict_single(x)
        if multi_pert == True:
            x = torch.cat([cell_type_fet + ctrl , self.embd_pert(pert)], dim = 1)
            x = self.lin_predict(x)
        return x
     
