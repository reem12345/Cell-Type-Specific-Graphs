from scipy import stats
import scanpy as sc
import numpy as np 
import pandas as pd
import networkx as nx
import anndata
import copy
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import torch_geometric
import tqdm.notebook as tq
from numpy.random import RandomState
from scipy import sparse
import torch
import seaborn as sns
import matplotlib.cm as cm
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import StepLR
import anndata as ad
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, Batch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, HeteroData, Batch
from torch_geometric.utils import *
import torch
import ot
from torch import nn
device = 'cuda'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

    
def loss_fct(pred,y, perts):
    """
    EMD losses to train the model.
    """
    perts = np.array(perts)
    losses = torch.tensor(0.0).to(pred.device)
    for p in (set(perts)):
        pert_idx = np.where(perts == p)[0]
        y_p = y[pert_idx]
        pred_p = pred[pert_idx]
        ab = torch.ones(y_p.shape[0]) / y_p.shape[0]
        M = ot.dist(pred_p, y_p, metric = 'euclidean').to(pred.device)
        loss = ot.lp.emd2(ab.to(pred.device), ab, M)
        losses = losses + loss
        del M
    return losses / (len(set(perts)))

#--------------------------------------------------------------------------------------------------------------

def Correlation_matrix(adata, cell_type, cell_type_key,
                       hv_genes_cells = None, union_HVGs = False):
    # pairwise correlation
    if union_HVGs:
        ad = adata[ (adata.obs[cell_type_key] == cell_type), :].copy()
    else:
        ad = adata[ (adata.obs[cell_type_key] == cell_type), hv_genes_cells[cell_type] ].copy()
        
    X = ad.X.A
    genes = ad.var.index.values.tolist()
    
    out = np.corrcoef(X, rowvar= False)
    out[np.isnan(out)] = 0.0
    
    values = (out[np.triu_indices(len(genes), k = 1)].flatten())
    
    out = pd.DataFrame((out), index = genes, columns = genes)
    
    out = out.stack().reset_index()
    return out

#--------------------------------------------------------------------------------------------------
 
def create_coexpression_graph(adata, co_expr_net, cell_type, threshold,
                              gene_key = 'gene_name',  celltype_key = 'cell_type'):

    co_expr_net[0] = np.abs(co_expr_net[0])
    co_expr_net = co_expr_net.loc[co_expr_net[0] >= threshold]
    co_expr_net = co_expr_net.loc[co_expr_net.level_0 != co_expr_net.level_1]
    co_expr_net = nx.from_pandas_edgelist(co_expr_net, source='level_0', target='level_1', edge_attr=0)
    
    nodes_list = adata.var.reset_index()
    nodes_list = nodes_list.loc[nodes_list[gene_key].isin(list(co_expr_net.nodes))]
    nodes_list = pd.DataFrame({'gene_loc': nodes_list.index.values, 'gene_id': nodes_list[gene_key].values})
    dic_nodes = dict(zip(nodes_list.gene_id, nodes_list.index))
    edges = nx.to_pandas_edgelist(co_expr_net)
    edges['source'] = edges['source'].map(dic_nodes)
    edges['target'] = edges['target'].map(dic_nodes)
    edges.sort_values(['source', 'target'], inplace = True)
    
    # Get the control samples as basal gene expression in the selected cell type
    ctrl = adata[(adata.obs[celltype_key] == cell_type)].copy()
    ctrl = ctrl[:, nodes_list.gene_loc]
    x = torch.tensor(ctrl.X.A).float()
    G = Data(x=x.T, edge_index=torch.tensor(edges[['source', 'target']].to_numpy().T)
             , pos= list(nodes_list.gene_loc.values), edge_attr = torch.tensor(edges[0]))
    return G   

#-----------------------------------------------------------------------------------------------------

def create_cells(stim_data, cell_type_network, canonical_smiles):
    cells = []
    print(stim_data.obs.cell_type.unique(), stim_data.obs.condition.unique())
    for cov_drug in tq.tqdm(stim_data.obs.cov_drug.unique()):
        cell_type = cov_drug.split("_")[0]
        genes = cell_type_network[cell_type].pos.tolist()
        adata_cov_drug = stim_data[stim_data.obs.cov_drug == cov_drug, :].copy()
        drug = cov_drug.split("_")[1]            
        for sample in tq.tqdm(adata_cov_drug, leave = False):
            x = torch.tensor(sample.layers['ctrl_x'])
            if canonical_smiles is None:
                cell = Data(x = x, y = torch.tensor(sample.X.A), #pert_label = pert, 
                        cell_type = cell_type, cov_drug = cov_drug, drug = drug)
            else: 
                pert =  torch.tensor(canonical_smiles[sample.obs['condition'].values[0]])
                cell = Data(x = x, y = torch.tensor(sample.X.A), pert_label = pert, 
                            cell_type = cell_type, cov_drug = cov_drug, drug = drug)
            cells.append(cell)
    return cells

#-------------------------------------------------------------------------------------------------------------

def rank_genes(dedf): 
    dedf['abs_logfoldchanges'] = dedf['logfoldchanges'].abs()
    # from the most significant to the less significant 
    dedf["Rank_pvals_adj"] = dedf["pvals_adj"].rank(method = 'dense')
    dedf["Rank_abs_logfoldchanges"] = dedf["abs_logfoldchanges"].rank(method = 'dense', ascending = False)
    dedf['Final_rank'] = (dedf["Rank_pvals_adj"] * dedf["Rank_abs_logfoldchanges"]) ** (1/2)
    dedf = dedf.sort_values('Final_rank')
    # display(dedf.head(20))
    num_genes = 100
    DEGs_name = dedf.head(num_genes).names.values #
    return list(DEGs_name)

#--------------------------------------------------------------------------------------------------------

def balance_subsample(data, labels, total_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    # To subset the remainder from the largest group
    sorted_indices = np.argsort(class_counts)
    unique_labels = unique_labels[sorted_indices]
    samples_per_class = np.floor_divide(total_samples, len(unique_labels))
    rem = total_samples % len(unique_labels)
    total = total_samples
    balanced_data = []
    for count, label in enumerate(unique_labels):
        prng = RandomState(1234567890)
        indices = np.where(labels == label)[0]
        if int(samples_per_class) <= len(indices): 
            selected_indices = prng.choice(indices, int(samples_per_class), replace=False)
        else:
            selected_indices = prng.choice(indices, int(samples_per_class), replace=True)
        balanced_data.extend(data[selected_indices])
        total = total - int(samples_per_class)
        if count == (len(unique_labels)-1):
            if rem <= len(indices): 
                selected_indices = prng.choice(indices, rem, replace=False)
            else: 
                selected_indices = prng.choice(indices, rem, replace=True)
            balanced_data.extend(data[selected_indices])
    return balanced_data

#---------------------------------------------------------------------------------------------------------

def train(model, num_epochs, lr, weight_decay, cell_type_network, train_loader, multi_pert = True):
    """
    The training function
    """
    
    print('Training Starts')
    mse_loss = torch.nn.MSELoss(reduction = 'mean')
    device = 'cuda'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay= weight_decay)
    model = model.to(device).float()
    num_epochs = num_epochs
    for epoch in tq.tqdm(range(num_epochs), leave=False):
        running_loss = 0.0
        train_epoch_loss = 0.0
        count = 0
        for sample in tq.tqdm(train_loader, leave=False):
            model.train()
            sample = sample.to(device)
            cell_type = sample.cell_type
            ctrl = sample.x
            if multi_pert:
                pert_label = sample.pert_label
            else: 
                pert_label = None
            batch = sample.batch
            y = sample.y
            cell_graphs_x = {Cell: cell_type_network[Cell].x.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_pos = {Cell: cell_type_network[Cell].pos.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_edges = {Cell: cell_type_network[Cell].edge_index.to(device) for Cell in np.unique(cell_type)}
            out = model(cell_graphs_x, cell_graphs_edges, 
                        cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos, multi_pert)
            loss = loss_fct(out,y, sample.cov_drug) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()
        train_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}, train loss: {train_epoch_loss}")
    return model

#-----------------------------------------------------------------------------------------------------------------------------------------------


def Inference(cell_type_network, model, save_path_res,
              ood_loader, cell_type, adata, testing_drug, degs_dict,
              device = 'cuda', mean_or_std = True, plot = True, multi_pert = True):
    """
    The Inference function
    """
    pred = []
    truth = []
    pos_genes = cell_type_network[cell_type].pos
    with torch.no_grad():
        model.eval()
        treat = adata[adata.obs.cov_drug == cell_type+'_'+testing_drug, pos_genes.tolist()].copy()
        ctrl_adata = adata[adata.obs.cov_drug == cell_type+'_control', pos_genes.tolist()].copy()
        eval_data = sc.concat([treat,ctrl_adata], join = 'outer')
        mapping_genes_indices = dict(zip(eval_data.var.index.values, list(range(0,len(eval_data.var)))))
        DEGs_name = pd.Series(eval_data[:, degs_dict].copy().var.index.values)
        DEGs = DEGs_name.map(mapping_genes_indices).values
        DEGs_name = DEGs_name.values
        for sample in tq.tqdm(ood_loader, leave=False):
            sample = sample.to(device)
            cell_type = sample.cell_type
            ctrl = sample.x
            if multi_pert:
                pert_label = sample.pert_label
            else: 
                pert_label = None
            batch = sample.batch
            y = sample.y
            cell_graphs_x = {Cell: cell_type_network[Cell].x.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_pos = {Cell: cell_type_network[Cell].pos.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_edges = {Cell: cell_type_network[Cell].edge_index.to(device) for Cell in np.unique(cell_type)}
            out = model(cell_graphs_x, cell_graphs_edges, 
                        cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos, multi_pert)
            pred.extend(out)
            truth.extend(y)
   
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    # R2 
    pred = (pred).cpu().numpy()[:, pos_genes.tolist()] 
    truth = (truth).cpu().numpy()[:, pos_genes.tolist()]
    sign_degs = np.mean(truth, axis = 0) - ctrl_adata.X.mean(0)
    std_problem = np.where( np.std(pred, axis = 0) == 0.0)[0]
    pred_adata = pred
    if mean_or_std:
        x = np.mean(truth, axis = 0)
        y = np.mean(pred, axis = 0) 
    else: 
        x = np.std(truth, axis = 0) 
        y = np.std(pred, axis = 0) 

    sns.set_style("darkgrid")
    x_coeff = 0.35
    r2_all = metrics.r2_score(x, y)
    r2_DEGs = metrics.r2_score(x[DEGs], y[DEGs])
    print("R2 top 20 DEGs: ", metrics.r2_score(x[DEGs[0:20]], y[DEGs[0:20]]))
    print("R2 top 50 DEGs: ", metrics.r2_score(x[DEGs[0:50]], y[DEGs[0:50]]))

    if plot:
        # Scatter plot
        fig, ax =plt.subplots(figsize = (6,6))
        sns.regplot(x = x, y = y, ci = None, color="#1C2E54")
        y_coeff=0.8
        print("R2 all genes: ", r2_all)
        ax.text( x.max() -x.max() * x_coeff, y.max() - y_coeff * y.max(),
                r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= '+ f"{r2_all:.4f}",fontsize = 'large',
            )
        y_coeff=0.9
        print("R2 top 100 DEGs: ", r2_DEGs)
        ax.text( x.max() -x.max() * x_coeff, y.max() - y_coeff * y.max(),
               r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100 \ DEGs}}}}$= ' + f"{r2_DEGs:.4f}",fontsize = 'large',
            )
        n = eval_data.var.index.values
        print(len(n))
        for i, txt in enumerate(n):
            if i in std_problem:
                ax.scatter(x[i], y[i], color = "#868386")
        #if degs:
        for i, txt in enumerate(n):
            if txt in DEGs_name[0:20]:
                if sign_degs[0,i] >= 0:
                    ax.scatter(x[i], y[i], color = "#B5345C")
                elif sign_degs[0,i] < 0:
                    ax.scatter(x[i], y[i], color = "green")
            if txt in DEGs_name[0:10]:
                ax.annotate(txt, (x[i], y[i]), color = "black")
        plt.xlabel('Real expression')
        plt.ylabel('Predicted expression')
        plt.savefig(save_path_res+"_"+cell_type[0]+'_'+testing_drug+"_R2.pdf", bbox_inches='tight')
        plt.show()
            
        sns.set_style("whitegrid")
        treat_pred = treat.copy()
        treat_pred.X = pred_adata.copy()
        treat_pred.obs['condition'] = 'pred_'+ testing_drug
        dot_adata = ad.concat([treat_pred, treat.copy(), ctrl_adata.copy()])
        plt.figure()
        color_map = sns.light_palette("#1C2E54", as_cmap=True)
        sc.pl.dotplot(dot_adata,  DEGs_name[0:20], groupby='condition', dendrogram=True, cmap = color_map, show=False)
        plt.savefig(save_path_res+"_"+cell_type[0]+"_dotplot.pdf", bbox_inches='tight')
        plt.show()
        return r2_all, r2_DEGs, DEGs_name, ad.concat([treat_pred, treat.copy()])
    else: 
        return r2_all, r2_DEGs, DEGs_name



