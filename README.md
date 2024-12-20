# Cell-Type-Specific-Graphs

**PrePR-CT** is a graph-based deep learning method designed to predict transcriptional responses to chemical perturbations in single-cell data. This method utilizes Graph Attention Network (GAT) layers to encode cell-type graphs from batches of training samples. These encoded graphs are then integrated with control gene expression data and predefined perturbation embeddings. The combined data is processed through Multi-Layer Perceptrons (MLPs) to accurately predict gene expression responses.

The preprint is available [here].

![Graphical Abstract](PrePR-CT.png)


## Directories

### Data_Notebooks
This directory includes the pre-processing notebooks for each dataset, starting from the raw counts. The pre-processing steps are explained in the preprint.
* [Kang dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)
* [NeurIPS dataset](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations)
* [McFarland and Chang datasets](http://projects.sanderlab.org/scperturb/datavzrd/scPerturb_vzrd_v1/dataset_info/index_1.html)
* [Nault dataset](https://github.com/BhattacharyaLab/scVIDR/tree/main)


### Training
This directory contains the training notebooks needed to reproduce the figures for each dataset. While results may slightly differ from the paper, they do not affect the overall conclusions.


## Required Packages

Create a conda environment using the following packages:
```yaml
conda create --prefix python=3.8.19
conda activate --prefix
pip install -r requeriments.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install jupyterlab
pip install torch_geometric==2.5.3
pip install optuna
pip install ipywidgets --upgrade
mkdir model_checkpoints
mkdir Data
