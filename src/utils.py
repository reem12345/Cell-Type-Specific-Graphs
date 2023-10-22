from torchmetrics import R2Score, PearsonCorrCoef
from torchmetrics.functional import r2_score
import scanpy as sc
import numpy as np 
import pandas as pd
import networkx as nx
import anndata
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import torch_geometric
import tqdm.notebook as tq
import torch
import seaborn as sns
from scipy.sparse import csr_matrix
sns.set_style("darkgrid")
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, HeteroData, Batch
from torch_geometric.utils import *
import torch
from torch import nn
device = 'cuda'


def loss_fct(pred, y, perts,mask, ctrl = None, gamma = 0):
    direction_lambda = 0.005
    perts = np.array(perts)
    mse_loss = torch.nn.MSELoss(reduction = 'mean')
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in (set(perts)):
        pert_idx = np.where(perts == p)[0]
        y_p = y[pert_idx]
        pred_p = pred[pert_idx]
        mask_p = mask[pert_idx]
        losses = losses + mse_loss(y_p[mask_p], pred_p[mask_p])
    return losses/(len(set(perts)))

#--------------------------------------------------------------------------------------------------
def HVG_per_celltype(adata, num_genes, cell_type_key):
    """
    return a dictionary of the highly variable genes per cell types and anndata object with the union of the HVGs.
    """
    hv_genes = []
    hv_genes_cells = {}
    for cell_line in adata.obs[cell_type_key].unique():
        ad = adata[adata.obs[cell_type_key] == cell_line].copy()
        sc.pp.highly_variable_genes(ad, subset=True, n_top_genes = num_genes)
        hv_genes.append(ad.var.index.values.tolist())
        hv_genes_cells[cell_line] = ad.var.index.values.tolist()
    flatten_list = sum(hv_genes, [])
    adata = adata[:, np.unique(flatten_list)]
    adata.var['highly_variable'] = True
    return hv_genes_cells, adata


def Correlation_matrix(adata, cell_type, perturbation_key, cell_type_key,
                       hv_genes_cells = None, union_HVGs = False):
    # pairwise correlation
    if union_HVGs:
        ad = adata[(adata.obs[perturbation_key] == 'control') & (adata.obs[cell_type_key] == cell_type), :].copy()
    else:
        ad = adata[(adata.obs[perturbation_key] == 'control') & (adata.obs[cell_type_key] == cell_type), hv_genes_cells[cell_type] ].copy()

    X = ad.X.A
    genes = ad.var.index.values.tolist()
    
    out = np.corrcoef(X, rowvar= False)
    out[np.isnan(out)] = 0.0
    
    values = (out[np.triu_indices(len(genes), k = 1)].flatten())
    
    print(len(genes), X.shape, out.shape)
    
    out = pd.DataFrame((out), index = genes, columns = genes)
    
    #stack the upper part of the matrix
    out = out.mask(np.triu(np.ones(out.shape)).astype(bool)).stack().reset_index()
    return out

#--------------------------------------------------------------------------------------------------

def create_coexpression_graph(adata, co_expr_net, cell_type, threshold,
                              gene_key, perturbation_key = 'perturbation', celltype_key = 'cell_line'):

    co_expr_net[0] = np.abs(co_expr_net[0])
    co_expr_net = co_expr_net.loc[co_expr_net[0] >= threshold]
    co_expr_net = co_expr_net.loc[co_expr_net.level_0 != co_expr_net.level_1]
    
    co_expr_net = nx.from_pandas_edgelist(co_expr_net, source='level_0', target='level_1', edge_attr=0)
    cn = [len(c) for c in sorted(nx.connected_components(co_expr_net), key=len, reverse=True)]
    print(cn)
    # largest_cc = max(nx.connected_components(co_expr_net), key=len)
    # co_expr_net = co_expr_net.subgraph(largest_cc)
    
    nodes_list = adata.var.reset_index()
    nodes_list = nodes_list.loc[nodes_list[gene_key].isin(list(co_expr_net.nodes))]
    nodes_list = pd.DataFrame({'gene_loc': nodes_list.index.values, 'gene_id': nodes_list[gene_key].values})
    
    dic_nodes = dict(zip(nodes_list.gene_id, nodes_list.index))
    edges = nx.to_pandas_edgelist(co_expr_net)
    edges['source'] = edges['source'].map(dic_nodes)
    edges['target'] = edges['target'].map(dic_nodes)
    edges.sort_values(['source', 'target'], inplace = True)
    
    # Get the control samples as basal gene expression in the selected cell type
    ctrl = adata[adata.obs[perturbation_key] == 'control'].copy()
    #ctrl = random_sample(ctrl, "cell_line")
    ctrl = ctrl[(ctrl.obs[celltype_key] == cell_type)].copy()
    ctrl = ctrl[:, nodes_list.gene_loc]
    x = torch.tensor(ctrl.X.A).float()
    G = Data(x=x.T, edge_index=torch.tensor(edges[['source', 'target']].to_numpy().T), pos= list(nodes_list.gene_loc.values), edge_attr = torch.tensor(edges[0]))
    return G



def train(model, adata,ctrl_data, num_epochs,
          lr, weight_decay, cell_type_network, train_loader, loss_key, 
          max_nodes, gamma = 0, cell_typekey ='cell_line'):
    
    print('Training Starts')
    mse_loss = torch.nn.MSELoss(reduction = 'mean')
    device = 'cuda'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device).float()
    num_epochs = num_epochs
    for epoch in tq.tqdm(range(num_epochs), leave=False):
        running_loss = 0.0
        train_epoch_loss = 0.0
        count = 0
        for sample in tq.tqdm(train_loader, leave=False):
            model.train()
            optimizer.zero_grad()
            sample = sample.to(device)
            cell_line = sample.cell_line
            ctrl = sample.x
            pert_label = sample.pert_label
            batch = sample.batch
            y = sample.y
            cell_graphs_x = {Cell: cell_type_network[Cell].x.to(device) for Cell in np.unique(cell_line)}
            cell_graphs_pos = {Cell: cell_type_network[Cell].pos.to(device) for Cell in np.unique(cell_line)}
            cell_graphs_edges = {Cell: cell_type_network[Cell].edge_index.to(device) for Cell in np.unique(cell_line)}
            cell_graphs_edges_attr = {Cell: cell_type_network[Cell].edge_attr.to(device) for Cell in np.unique(cell_line)}
            out = model(cell_graphs_x, cell_graphs_edges, cell_graphs_edges_attr, 
                        cell_line, cell_graphs_edges.keys(), pert_label, ctrl, cell_graphs_pos)
            
            out = to_dense_batch(out, batch = batch ,batch_size = len(cell_line)
                                 ,fill_value = 0, max_num_nodes = max_nodes)[0]
           
            ctrl = to_dense_batch(ctrl, batch = batch ,batch_size = len(cell_line)
                                 ,fill_value = 0, max_num_nodes = max_nodes)[0]
            
            y, mask = to_dense_batch(y, batch = batch, batch_size = len(cell_line)
                               ,fill_value = 0, max_num_nodes = max_nodes)
            loss = loss_fct(out, y.squeeze(2), sample.cov_drug, mask) 
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()
        train_epoch_loss = running_loss / len(train_loader)
        print('Epoch', epoch, ':', train_epoch_loss)
    return model



def Inference(cell_type_network,ctrl_data, model, save_path_res,
              ood_loader, cell_type,adata,testing_drug, degs = False, device = 'cuda', overlap_list = None):
    pred = []
    truth = []
    pos_genes = cell_type_network[cell_type].pos
    print(len(pos_genes))
    with torch.no_grad():
        model.eval()
        treat = adata[adata.obs.cov_drug == cell_type+'_'+testing_drug, pos_genes.tolist()].copy()
        ctrl_adata = adata[adata.obs.cov_drug == cell_type+'_control', pos_genes.tolist()].copy()
        eval_data = treat.concatenate(ctrl_adata)
        mapping_genes_indices = dict(zip(eval_data.var.index.values, list(range(0,len(eval_data.var)))))
        sc.tl.rank_genes_groups(eval_data, groupby = 'perturbation',rankby_abs = True,
                                reference = 'control',n_genes = int(len(pos_genes.tolist())*0.10))
        DEGs_name = pd.Series(eval_data.uns["rank_genes_groups"]["names"][testing_drug])
        DEGs = DEGs_name.map(mapping_genes_indices).values
        DEGs_name = DEGs_name.values
            
        for sample in tq.tqdm(ood_loader, leave=False):
            sample = sample.to(device)
            cell_line = sample.cell_line
            ctrl = sample.x
            pert_label = sample.pert_label
            batch = sample.batch
            y = sample.y
            cell_graphs_x = {Cell: cell_type_network[Cell].x.to(device) for Cell in np.unique(cell_line)}
            cell_graphs_pos = {Cell: cell_type_network[Cell].pos.to(device) for Cell in np.unique(cell_line)}
            cell_graphs_edges = {Cell: cell_type_network[Cell].edge_index.to(device) for Cell in np.unique(cell_line)}
            cell_graphs_edges_attr = {Cell: cell_type_network[Cell].edge_attr.to(device) for Cell in np.unique(cell_line)}
            out = model(cell_graphs_x,cell_graphs_edges, cell_graphs_edges_attr, cell_line, 
                        cell_graphs_edges.keys(), pert_label, ctrl, cell_graphs_pos)
            out = out.reshape(len(cell_line), len(pos_genes))
            y = y.reshape(len(cell_line), len(pos_genes))
            pred.extend(out)
            truth.extend(y)
   
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    # R2 
    pred = (pred).cpu().numpy()
    truth = (truth).cpu().numpy()
    x = np.mean(truth, axis = 0) 
    y = np.mean(pred, axis = 0) 
    
    x_coeff=0.35
    r2_all = metrics.r2_score(x, y)
    r2_DEGs = metrics.r2_score(x[DEGs], y[DEGs])

    # Scatter plot
    fig, ax =plt.subplots(figsize = (6,6))
    sns.regplot(x, y, ci = None, color="#6661c7")
    y_coeff=0.8
    print("R2_all: ", r2_all)
    ax.text( x.max() -x.max() * x_coeff, y.max() - y_coeff * y.max(),
            r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= '+ f"{r2_all:.4f}",fontsize = 'large'
        )
    y_coeff=0.9
    print("R2_DEGs: ", r2_DEGs)
    ax.text( x.max() -x.max() * x_coeff, y.max() - y_coeff * y.max(),
           r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 10\% \ DEGs}}}}$= ' + f"{r2_DEGs:.4f}",fontsize = 'large'
        )
    #if degs:
    n = treat.var.index.values
    for i, txt in enumerate(n):
        if txt in DEGs_name[0:5]:
            ax.scatter(x[i], y[i], color = "red")
            ax.annotate(txt, (x[i], y[i]))
    plt.xlabel('Real expression')
    plt.ylabel('Predicted expression')
    plt.savefig(save_path_res+"_"+cell_type+"_R2.pdf", bbox_inches='tight')
    plt.show()

    return r2_all, r2_DEGs, DEGs_name



