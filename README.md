# Cell-Type-Specific-Graphs

**PrePR-CT** is a graph-based deep learning method designed to predict transcriptional responses to chemical perturbations in single-cell data. This method utilizes Graph Attention Network (GAT) layers to encode cell-type graphs from batches of training samples. These encoded graphs are then integrated with control gene expression data and predefined perturbation embeddings. The combined data is processed through Multi-Layer Perceptrons (MLPs) to accurately predict gene expression responses.

![Graphical Abstract](PrePR-CT.png)

## Installation

Create a conda environment using the following packages:
<pre>
conda create -n preprct python=3.8.19

conda activate preprct

pip install -r requirements.txt

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install jupyterlab
pip install torch_geometric==2.5.3
pip install ipywidgets --upgrade

git clone https://github.com/reem12345/Cell-Type-Specific-Graphs.git
cd Cell-Type-Specific-Graphs/ 

</pre>

## Directories

### Data_Notebooks
This directory includes the pre-processing notebooks for each dataset, starting from the raw counts. The pre-processing steps are explained in the paper.
* [Kang dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)
* [NeurIPS dataset](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations)
* [McFarland and Chang datasets](http://projects.sanderlab.org/scperturb/datavzrd/scPerturb_vzrd_v1/dataset_info/index_1.html)
* [Nault dataset](https://github.com/BhattacharyaLab/scVIDR/tree/main)

### graphs 
This directory contains the cell-type-specific graphs for each dataset, which are required to reproduce the results.

### Training

This directory contains the training notebooks required to reproduce the figures for each dataset. While results may slightly differ from those reported in the paper, these variations do not affect the overall conclusions.

## Reproduciblity
### Setting up
<pre>
mkdir model_checkpoints
mkdir Data
</pre> 

### Data
To reproduce the results for a specific dataset, download the corresponding `.h5ad` and `.pkl` files (named after the dataset) from the following links, and place them in the `Data` folder you created during the installation steps: 

- https://figshare.com/s/7beaf41998af17bdbe33  
- https://figshare.com/s/b7f07ac5c522db3ba3af

### Running
The next step is to update the `config_train.yaml` file with the appropriate settings for the selected dataset, and then run the demo notebook `training_testing_demo.ipynb` to train and test the model.Below is an example of how to run different datasets in the paper. For new datasets, we are working on improving reusability.


<pre>
# Kang dataset
# modify the path
project_dir: "../PrePR-CT/"
data_path: "Data/"
save_path_results: "Results/"
save_path_models: "model_checkpoints/"
dataset_h5ad: "Kang.h5ad"
graphs_path: "graphs/"
graphs_data: "Kang"
SMILES_feat: "SMILES_feat_all_datasets.csv"
params:
  hidden_channels: 64
  weight_decay: 0.00001
  in_head: 1
  learning_rate: -3
  num_epochs: 100
  batch_size: 256
testing_cell_type: ['CD4 T cells']
testing_drugs: ['stimulated']
multi_pert: False

</pre>


<pre>
# NeurIPS dataset
# modify the path
project_dir: "../PrePR-CT/"
data_path: "Data/"
save_path_results: "Results/"
save_path_models: "model_checkpoints/"
dataset_h5ad: "Neurips_Data_Processed.h5ad"
graphs_path: "graphs/"
graphs_data: "NeurIPS"
SMILES_feat: "SMILES_feat_all_datasets.csv"
params:
  hidden_channels: 128
  weight_decay: 0.00001
  in_head: 1
  learning_rate: -3
  num_epochs: 100
  batch_size: 512
testing_cell_type: ["Myeloid cells", "NK cells", "B cells", "T cells"]
testing_drugs: ['Scriptaid', 'Ketoconazole', 'Dactolisib'] 
multi_pert: True

</pre>


<pre>
# McFarland dataset
# modify the path
project_dir: "../PrePR-CT/"
data_path: "Data/"
save_path_results: "Results/"
save_path_models: "model_checkpoints/"
dataset_h5ad: "McFarland_processed.h5ad"
graphs_path: "graphs/"
graphs_data: "McFarland"
SMILES_feat: "SMILES_feat_all_datasets.csv"
params:
  hidden_channels: 128
  weight_decay: 0.00001
  in_head: 1
  learning_rate: -3
  num_epochs: 200
  batch_size: 256
testing_cell_type: ['COLO680N', 'TEN', 'RCC10RGB', 'LNCAPCLONEFGC', 'BICR31']
testing_drugs: ['JQ1', 'Bortezomib']
multi_pert: True

</pre>


### Output
