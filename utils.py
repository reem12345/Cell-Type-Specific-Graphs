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
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, HeteroData, Batch
from torch_geometric.utils import *
import torch
import ot
from torch import nn
device = 'cuda'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
    
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
    try: 
        X = ad.X.A
    except: 
        X = ad.X
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
    print('number_of_edges before the threshold (5000 x 5000): ',len(co_expr_net))
    co_expr_net = co_expr_net.loc[co_expr_net[0] >= threshold]
    print('number_of_edges after the threshold: ',len(co_expr_net))
    co_expr_net = co_expr_net.loc[co_expr_net.level_0 != co_expr_net.level_1]
    print('number_of_edges after removing self loops: ',len(co_expr_net))
    co_expr_net = nx.from_pandas_edgelist(co_expr_net,source = 'level_0', target = 'level_1', edge_attr=0, create_using=nx.DiGraph())
    print('final number_of_edges: ', co_expr_net.number_of_edges())
    connected_components = nx.weakly_connected_components(co_expr_net)

    largest_component = max(connected_components, key=len)
    
  #  co_expr_net = co_expr_net.subgraph(largest_component).copy()
    
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
    try: 
        x = torch.tensor(ctrl.X.A).float()
    except: 
        x = torch.tensor(ctrl.X).float()
    G = Data(x=x.T, edge_index=torch.tensor(edges[['source', 'target']].to_numpy().T)
             , pos= list(nodes_list.gene_loc.values), edge_attr = torch.tensor(edges[0]))
    return G   

#-----------------------------------------------------------------------------------------------------

def create_cells(stim_data, cell_type_network, canonical_smiles):
    cells = []
    obs = stim_data.obs
    print(obs.cell_type.unique(), obs.condition.unique())

    # Group the data by cov_drug to avoid filtering repeatedly
    cov_drug_groups = {
        cov_drug: stim_data[obs.cov_drug == cov_drug, :].copy()
        for cov_drug in obs.cov_drug.unique()
    }

    for cov_drug, adata_cov_drug in tq.tqdm(cov_drug_groups.items(), desc="Processing cov_drugs"):
        try:
            cell_type, drug = cov_drug.split("_", 1)
        except ValueError:
            # If splitting fails, skip this group
            continue

        # Iterate over each sample in the current group
        for sample in tq.tqdm(adata_cov_drug, leave=False, desc=f"Processing {cov_drug} samples"):
            # Convert the control layer to a tensor
            x = torch.tensor(sample.layers['ctrl_x'])
            y = torch.tensor(sample.X.A)

            if canonical_smiles is None:
                cell = Data(
                    x=x,
                    y=y,
                    cell_type=cell_type,
                    cov_drug=cov_drug,
                    drug=drug
                )
            else:
                # Retrieve the condition from the sample's observation and convert the corresponding SMILES fingerprint to tensor
                condition = sample.obs['condition'].values[0]
                pert = torch.tensor(canonical_smiles[condition]).unsqueeze(0)
                cell = Data(
                    x=x,
                    y=y,
                    pert_label=pert,
                    cell_type=cell_type,
                    cov_drug=cov_drug,
                    drug=drug
                )
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
    num_genes = 100
    DEGs_name = dedf.head(num_genes).names.values 
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
                        cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos)
            loss = loss_fct(out,y, sample.cov_drug) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()
        train_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}, train loss: {train_epoch_loss}")
    return model
#-----------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scanpy as sc
import torch

def create_anndata(pred_p, truth_p, adata, cell_type_network, p):
    # Extract cell type (c) and drug (d) from the input parameter `p`
    c = p.split('_')[0]
    d = p.split('_')[1]

    # Retrieve the positions of the genes for the specified cell type
    pos_genes = cell_type_network[c].pos


    # Get control data from the AnnData object
    ctrl_p = adata[adata.obs.cov_drug == c + '_control', pos_genes.tolist()].X.A

    # Combine the data (truth, reaction, control)
    combined_data = np.vstack([truth_p, pred_p, ctrl_p])

    # Create observation DataFrame with cell type and condition
    cell_type = np.array([c] * combined_data.shape[0])

    # Assign conditions
    condition_truth = np.array([d] * truth_p.shape[0])  # For truth, condition = d
    condition_pred = np.array(['pred_' + d] * pred_p.shape[0])  # For prediction, condition = 'pred_<d>'
    condition_ctrl = np.array(['control'] * ctrl_p.shape[0])  # For control, condition = 'control'

    # Combine all conditions
    condition = np.concatenate([condition_truth, condition_pred, condition_ctrl])

    # Create observation DataFrame
    obs_df = pd.DataFrame({
        'cell_type': np.concatenate([cell_type[:truth_p.shape[0]], cell_type[:pred_p.shape[0]], cell_type[:ctrl_p.shape[0]]]),
        'condition': condition
    })

    # Create the AnnData object
    adata_combined = sc.AnnData(X=combined_data, obs=obs_df)

    # You can also assign additional information, such as `var` (genes), if necessary
    adata_combined.var_names = pos_genes.tolist()

    return adata_combined


def Inference_multi_pert(cell_type_network, model, save_path_res,
              ood_loader, adata, degs_dict, device = 'cuda', mean_or_std = True, plot = True, multi_pert = True):
    """
    The Inference function
    """
    pred = []
    truth = []
    with torch.no_grad():
        model.eval()
        cov_drugs = []
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
                        cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos)
            pred.extend(out)
            truth.extend(y)
            cov_drugs.extend(sample.cov_drug)
   
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    pred = (pred).cpu().numpy()
    truth = (truth).cpu().numpy()
    
    perts = np.array(cov_drugs)
    for p in (set(perts)):
        c = p.split('_')[0]
        d = p.split('_')[1]
        pos_genes = cell_type_network[c].pos
        p_pred = pred[:, pos_genes.tolist()] 
        p_truth = truth[:, pos_genes.tolist()]
        pert_idx = np.where(perts == p)[0]
        y_p = p_truth[pert_idx]
        pred_p = p_pred[pert_idx]
        Ann_Data = create_anndata(pred_p, y_p, adata, cell_type_network, p)
        DEGs = degs_dict[p]
        Ann_Data.uns['DEGs'] = DEGs
        Ann_Data.write(save_path_res+p+'_pred.h5ad')
        mse = np.mean((y_p - pred_p) ** 2)
        print(p, " mse: ", mse)
        if mean_or_std:
            x = np.mean(y_p, axis = 0)
            y = np.mean(pred_p, axis = 0) 
            r2_all = metrics.r2_score(x, y)
            print(f"R² value for predicting the **mean** expression of all genes for perturbation '{p}': {r2_all:.4f}")
            r2_DEGs = metrics.r2_score(x[DEGs], y[DEGs])
            print(f"R² value for predicting the **mean** of the top 100 DEGs for perturbation '{p}': {r2_DEGs:.4f}")
        else: 
            x = np.std(y_p, axis = 0) 
            y = np.std(pred_p, axis = 0) 
            data_to_plot = np.vstack([x, y])
            r2_all = metrics.r2_score(x, y)
            print(f"R² value for predicting the **standard deviation** expression of all genes for perturbation '{p}': {r2_all:.4f}")
            r2_DEGs = metrics.r2_score(x[DEGs], y[DEGs])
            print(f"R² value for predicting the **standard deviation** of the top 100 DEGs for perturbation '{p}': {r2_DEGs:.4f}")
    
        sns.set_style("darkgrid")
        x_coeff = 0.35
        
        if plot:
            # Scatter plot
            fig, ax =plt.subplots(figsize = (6,6))
            sns.regplot(x = x, y = y, ci = None, color="#1C2E54")
            y_coeff=0.8
            ax.text( x.max() -x.max() * x_coeff, y.max() - y_coeff * y.max(),
                    r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= '+ f"{r2_all:.4f}",fontsize = 'large',
                )
            y_coeff=0.9
            ax.text( x.max() -x.max() * x_coeff, y.max() - y_coeff * y.max(),
                   r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100 \ DEGs}}}}$= ' + f"{r2_DEGs:.4f}",fontsize = 'large',
                )