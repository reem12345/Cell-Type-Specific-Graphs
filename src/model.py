from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SGConv, Linear, sequential
import torch


class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=False, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)

class GNN(torch.nn.Module):
    def __init__(self, total_genes,num_perts, hidden_channels, output_channels = 1,all_feat = True):
        super().__init__()
        self.all_feat = all_feat
        self.in_head = 1
        self.conv1 = GATConv(-1, hidden_channels, heads=self.in_head,  add_self_loops=True, concat = False) 
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv(-1, hidden_channels, heads=self.in_head,  add_self_loops=True, concat = False)  
        self.lin2 = Linear(-1, hidden_channels)
        self.embd_pert = torch.nn.Embedding(num_perts, hidden_channels)
        self.lin_predict = MLP([2*hidden_channels+1, hidden_channels//2 ,output_channels])
        self.gene_emed = torch.nn.Embedding(total_genes, hidden_channels, max_norm=True)

    def forward(self,x, edge_index, edge_attr, cell_line, cell_type_keys, pert, ctrl, pos):
        for key in cell_type_keys:
            pos[key] = self.gene_emed(pos[key])
            edge_index[key] = edge_index[key] 
            if self.all_feat:
                x[key] = torch.cat([x[key] , pos[key]], dim = 1) 
            else: 
                x[key] = pos[key]
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            x[key] = x[key].relu()
            x[key] = self.conv2(x[key], edge_index[key]) + self.lin2(x[key])
            
        x = [x[c] for c in cell_line]
        x = torch.cat(x, dim = 0)
        x = torch.cat([x , ctrl, self.embd_pert(pert)], dim = 1)
        x = self.lin_predict(x).relu()
        x = x.squeeze(1)
        return x
    