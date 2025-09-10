# Cell-Type-Specific-Graphs
[![DOI](https://zenodo.org/badge/699833834.svg)](https://doi.org/10.5281/zenodo.15241234)

* [Summary](#summary)
* [Installation](#installation)
* [Directories](#directories)
* [Reproduciblity](#reproduciblity)

## Summary 
**PrePR-CT** is a graph-based deep learning method designed to predict transcriptional responses to chemical perturbations in single-cell data. This method utilizes Graph Attention Network (GAT) layers to encode cell-type graphs from batches of training samples. These encoded graphs are then integrated with control gene expression data and predefined perturbation embeddings. The combined data is processed through Multi-Layer Perceptrons (MLPs) to accurately predict gene expression responses.

![Graphical Abstract](PrePR-CT.png)

## Installation

1- Create a conda environment with Python (recommended version 3.8.19):
```
conda create -n preprct python=3.8.19
conda activate preprct
```

2- Install the necesarry libraries (from `requeriments.txt`):
```
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install jupyterlab
pip install torch_geometric==2.5.3
pip install ipywidgets --upgrade
```

3- After previous steps are completed, PrePR-CT is simply installed by cloning the repository:
```
git clone https://github.com/reem12345/Cell-Type-Specific-Graphs.git
cd Cell-Type-Specific-Graphs/ 
```

## Directories

### Data_Notebooks
This directory contains the pre-processing notebooks for each dataset, starting from the raw counts. The pre-processing steps are described in detail in the paper (section Methods/Datasets).
* [Kang dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)
* [NeurIPS dataset](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations)
* [McFarland and Chang datasets](http://projects.sanderlab.org/scperturb/datavzrd/scPerturb_vzrd_v1/dataset_info/index_1.html)
* [Nault dataset](https://github.com/BhattacharyaLab/scVIDR/tree/main)
* [LINCS dataset](https://f003.backblazeb2.com/file/chemCPA-datasets/lincs_full.h5ad.gz)

### Graphs 
This directory contains the cell-type-specific graphs for each dataset, which are required to reproduce the results.

### Training
This directory includes the training notebooks required to reproduce the figures for each dataset. While results may slightly differ from those reported in the paper, these variations do not affect the overall conclusions.

## Reproduciblity

### 1- Setting up
Create necessary directories:
```
mkdir model_checkpoints
mkdir Data
```

### 2- Data
To reproduce the results for a specific dataset, download the preprocessed data _.h5ad_ and _.pkl_ files (named after the dataset) from the following links, and place them in the `Data` folder created during the setup step:

- All datasets are here: https://figshare.com/s/7beaf41998af17bdbe33  
- NeurIPS dataset: https://figshare.com/s/b7f07ac5c522db3ba3af

In case the user wants to generate the above preprocessed data from the raw data, run the corresponding notebook located in the `Data_Notebooks` directory.

### 3- Usage

#### 3.1- Update the configuration
 * Modify the `config_train.yaml` file, located in the `training` folder, with the appropriate settings according to the dataset you want to use.
 * Below is an example of parameters used for `Kang` dataset to reproduce the results:

```yaml
# Kang dataset
# Path to the project root directory (modify as needed)
project_dir: "../Cell-Type-Specific-Graphs/"

# Folder where the input data files (.h5ad, .pkl) are stored
data_path: "Data/"

# Directory where result outputs will be saved
save_path_results: "Results/"

# Directory where model checkpoints will be saved during training
save_path_models: "model_checkpoints/"

# Filename of the AnnData (.h5ad) dataset to use
dataset_h5ad: "Kang.h5ad"

# Directory containing the precomputed graphs
graphs_path: "graphs/"

# Name of the graph file or prefix corresponding to the dataset (without extension)
graphs_data: "Kang"

# CSV file containing SMILES-based features for all drugs
SMILES_feat: "SMILES_feat_all_datasets.csv"

params:
  # Number of hidden channels in the GAT layers
  hidden_channels: 64

  # Weight decay (L2 regularization) applied to the optimizer
  weight_decay: 0.00001

  # Number of attention heads in the input GAT layer
  in_head: 1

  # Learning rate for the optimizer, usually set as 10^learning_rate (e.g., -3 → 0.001)
  learning_rate: -3

  # Number of training epochs
  num_epochs: 100

  # Batch size used during training
  batch_size: 256

# List of cell types to hold out for testing (out-of-distribution evaluation)
testing_cell_type: ['CD4 T cells']

# List of drugs or conditions to hold out for testing
testing_drugs: ['stimulated']

# Whether the dataset includes multiple unique perturbations (e.g., several different drugs or conditions)
# Set to True if the dataset includes more than one perturbation across cells.
multi_pert: False
```
#### 3.2- Run training and testing
 * Launch the [training_testing_demo.ipynb](https://github.com/reem12345/Cell-Type-Specific-Graphs/blob/main/training/training_testing_demo.ipynb) notebook (located in the `training` folder).
 * This will train and evaluate the model using your previous configuration.

#### 3.3- Output
 * The `Results` directory will contain an AnnData object (_.h5ad_) for each predicted (cell type, perturbation) pair.
 * Each file includes both the observed and predicted gene expression profiles for the corresponding condition.
