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

def loss_fct(pred, y, perts):
    """
    EMD (Earth Mover’s Distance) loss to train the model.
    Computes distributional distance between predicted and true values,
    grouped by perturbations.
    """
    perts = np.array(perts)  # Convert perturbation labels to NumPy array
    losses = torch.tensor(0.0).to(pred.device)  # Initialize total loss on the same device as predictions
    
    # Loop over each unique perturbation
    for p in set(perts):
        pert_idx = np.where(perts == p)[0]   # Indices of samples for this perturbation
        y_p = y[pert_idx]                    # True values for this perturbation
        pred_p = pred[pert_idx]              # Predicted values for this perturbation
        
        # Uniform weights over samples in this perturbation
        ab = torch.ones(y_p.shape[0]) / y_p.shape[0]
        
        # Compute pairwise cost (Euclidean distance) between predicted and true samples
        M = ot.dist(pred_p, y_p, metric='euclidean').to(pred.device)
        
        # Compute Earth Mover’s Distance (optimal transport cost)
        loss = ot.lp.emd2(ab.to(pred.device), ab, M)
        
        # Accumulate loss across perturbations
        losses = losses + loss
        
        # Free memory by deleting cost matrix
        del M
    
    # Average loss over number of unique perturbations
    return losses / len(set(perts))
    
#--------------------------------------------------------------------------------------------------------------
def Correlation_matrix(adata, cell_type, cell_type_key,
                       hv_genes_cells = None, union_HVGs = False):
    # compute pairwise gene-gene correlation matrix for a given cell type
    
    # subset AnnData object to the given cell type
    if union_HVGs:
        ad = adata[ (adata.obs[cell_type_key] == cell_type), :].copy()  # use all genes
    else:
        ad = adata[ (adata.obs[cell_type_key] == cell_type), hv_genes_cells[cell_type] ].copy()  # use HVGs for this cell type
    
    # extract expression matrix (dense or sparse)
    try: 
        X = ad.X.A  # if sparse, convert to dense array
    except: 
        X = ad.X    # if already dense
    
    genes = ad.var.index.values.tolist()  # gene names
    
    # compute correlation matrix across genes
    out = np.corrcoef(X, rowvar= False)
    out[np.isnan(out)] = 0.0  # replace NaNs with zeros
    
    # flatten upper triangle values (optional, not used later)
    values = (out[np.triu_indices(len(genes), k = 1)].flatten())
    
    # store correlation matrix in DataFrame with gene names
    out = pd.DataFrame((out), index = genes, columns = genes)
    
    # reshape to long format (gene1, gene2, correlation)
    out = out.stack().reset_index()
    return out  # return correlation matrix in long-format DataFrame

#--------------------------------------------------------------------------------------------------
 
def create_coexpression_graph(adata, co_expr_net, cell_type, threshold,
                              gene_key = 'gene_name',  celltype_key = 'cell_type'):

    # Take absolute values of co-expression weights (to ignore sign of correlation)
    co_expr_net[0] = np.abs(co_expr_net[0])
    print('number_of_edges before the threshold (5000 x 5000): ', len(co_expr_net))

    # Keep only edges above the given threshold
    co_expr_net = co_expr_net.loc[co_expr_net[0] >= threshold]
    print('number_of_edges after the threshold: ', len(co_expr_net))

    # Remove self-loops (edges where source == target)
    co_expr_net = co_expr_net.loc[co_expr_net.level_0 != co_expr_net.level_1]
    print('number_of_edges after removing self loops: ', len(co_expr_net))

    # Convert filtered dataframe into a directed graph with edge weights
    co_expr_net = nx.from_pandas_edgelist(
        co_expr_net,
        source='level_0', target='level_1',
        edge_attr=0, create_using=nx.DiGraph()
    )
    print('final number_of_edges: ', co_expr_net.number_of_edges())

    # Get all weakly connected components in the graph
    connected_components = nx.weakly_connected_components(co_expr_net)

    # Identify the largest connected component
    largest_component = max(connected_components, key=len)
    
    # Optional: restrict graph to the largest connected component
    # co_expr_net = co_expr_net.subgraph(largest_component).copy()
    
    # Map gene names to their positions in adata.var
    nodes_list = adata.var.reset_index()
    nodes_list = nodes_list.loc[nodes_list[gene_key].isin(list(co_expr_net.nodes))]
    nodes_list = pd.DataFrame({
        'gene_loc': nodes_list.index.values,
        'gene_id': nodes_list[gene_key].values
    })

    # Dictionary: gene_id → index in nodes_list
    dic_nodes = dict(zip(nodes_list.gene_id, nodes_list.index))

    # Convert networkx graph back to edge list dataframe
    edges = nx.to_pandas_edgelist(co_expr_net)
    edges['source'] = edges['source'].map(dic_nodes)
    edges['target'] = edges['target'].map(dic_nodes)
    edges.sort_values(['source', 'target'], inplace=True)

    # Select control samples for the given cell type
    ctrl = adata[(adata.obs[celltype_key] == cell_type)].copy()
    ctrl = ctrl[:, nodes_list.gene_loc]

    # Convert control expression matrix to PyTorch tensor
    try: 
        x = torch.tensor(ctrl.X.A).float()   # if sparse matrix
    except: 
        x = torch.tensor(ctrl.X).float()     # if dense matrix

    # Build PyTorch Geometric graph object
    G = Data(
        x=x.T,  # features: gene expression (genes × cells → transposed)
        edge_index=torch.tensor(edges[['source', 'target']].to_numpy().T),
        pos=list(nodes_list.gene_loc.values),
        edge_attr=torch.tensor(edges[0])  # edge weights
    )
    return G

#-----------------------------------------------------------------------------------------------------

def create_cells(stim_data, cell_type_network, canonical_smiles):
    cells = []
    obs = stim_data.obs
    print(obs.cell_type.unique(), obs.condition.unique())

    # Group the AnnData object by cov_drug (cell_type + drug) to avoid repeated filtering
    cov_drug_groups = {
        cov_drug: stim_data[obs.cov_drug == cov_drug, :].copy()
        for cov_drug in obs.cov_drug.unique()
    }

    # Iterate over each cov_drug group
    for cov_drug, adata_cov_drug in tq.tqdm(cov_drug_groups.items(), desc="Processing cov_drugs"):
        try:
            # Split cov_drug string into cell_type and drug
            cell_type, drug = cov_drug.split("_", 1)
        except ValueError:
            # Skip this group if splitting fails
            continue

        # Iterate over each sample (row) in the current group
        for sample in tq.tqdm(adata_cov_drug, leave=False, desc=f"Processing {cov_drug} samples"):
            # Convert control expression layer to tensor
            x = torch.tensor(sample.layers['ctrl_x'])
            # Convert perturbed expression to tensor (dense if sparse)
            y = torch.tensor(sample.X.A)

            if canonical_smiles is None:
                # If no drug fingerprints are provided, store only cell/drug info
                cell = Data(
                    x=x,
                    y=y,
                    cell_type=cell_type,
                    cov_drug=cov_drug,
                    drug=drug
                )
            else:
                # Retrieve condition and get its drug fingerprint (SMILES → tensor)
                condition = sample.obs['condition'].values[0]
                pert = torch.tensor(canonical_smiles[condition]).unsqueeze(0)
                # Include fingerprint in the graph data object
                cell = Data(
                    x=x,
                    y=y,
                    pert_label=pert,
                    cell_type=cell_type,
                    cov_drug=cov_drug,
                    drug=drug
                )
            # Append the constructed cell graph to the list
            cells.append(cell)

    return cells



#-------------------------------------------------------------------------------------------------------------

def rank_genes(dedf): 
    # Compute absolute log fold changes (magnitude of change regardless of direction)
    dedf['abs_logfoldchanges'] = dedf['logfoldchanges'].abs()

    # Rank genes by adjusted p-values (smaller p-value = higher rank)
    dedf["Rank_pvals_adj"] = dedf["pvals_adj"].rank(method='dense')

    # Rank genes by absolute log fold changes (larger change = higher rank)
    dedf["Rank_abs_logfoldchanges"] = dedf["abs_logfoldchanges"].rank(method='dense', ascending=False)

    # Combine ranks using geometric mean of the two ranks
    dedf['Final_rank'] = (dedf["Rank_pvals_adj"] * dedf["Rank_abs_logfoldchanges"]) ** (1/2)

    # Sort genes by the final combined ranking score
    dedf = dedf.sort_values('Final_rank')

    # Select the top 100 genes as DEGs
    num_genes = 100
    DEGs_name = dedf.head(num_genes).names.values 

    # Return the list of top-ranked gene names
    return list(DEGs_name)


#--------------------------------------------------------------------------------------------------------

def balance_subsample(data, labels, total_samples, seed=None):
    # Set random seed for reproducibility (if provided)
    if seed is not None:
        np.random.seed(seed)

    # Get unique class labels and their counts
    unique_labels, class_counts = np.unique(labels, return_counts=True)

    # Sort labels by their class size (smallest to largest)
    # This ensures remainder samples are assigned starting from the largest group
    sorted_indices = np.argsort(class_counts)
    unique_labels = unique_labels[sorted_indices]

    # Base number of samples to draw per class
    samples_per_class = np.floor_divide(total_samples, len(unique_labels))

    # Remainder after equal distribution across classes
    rem = total_samples % len(unique_labels)

    total = total_samples
    balanced_data = []

    # Iterate over each class
    for count, label in enumerate(unique_labels):
        # Initialize deterministic random generator
        prng = RandomState(1234567890)

        # Get indices of all samples belonging to this class
        indices = np.where(labels == label)[0]

        # Sample from class (without replacement if enough samples, else with replacement)
        if int(samples_per_class) <= len(indices): 
            selected_indices = prng.choice(indices, int(samples_per_class), replace=False)
        else:
            selected_indices = prng.choice(indices, int(samples_per_class), replace=True)

        # Add sampled data to the balanced dataset
        balanced_data.extend(data[selected_indices])

        # Update remaining sample count
        total = total - int(samples_per_class)

        # For the last class, handle the remainder distribution
        if count == (len(unique_labels)-1):
            if rem <= len(indices): 
                selected_indices = prng.choice(indices, rem, replace=False)
            else: 
                selected_indices = prng.choice(indices, rem, replace=True)
            balanced_data.extend(data[selected_indices])

    # Return the final balanced dataset
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
