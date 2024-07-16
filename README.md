# Cell-Type-Specific-Graphs

**PrePR-CT** is a graph-based deep learning method designed to predict transcriptional responses to chemical perturbations in single-cell data. This method utilizes Graph Attention Network (GAT) layers to encode cell-type graphs from batches of training samples. These encoded graphs are then integrated with control gene expression data and predefined perturbation embeddings. The combined data is processed through Multi-Layer Perceptrons (MLPs) to accurately predict gene expression responses.

The preprint is available [here].

![Graphical Abstract](PrePR-CT.png)

## Required Packages

Create a conda environment using the following packages:
```yaml
channels:
  - conda-forge
  - defaults
dependencies:
  - scipy=1.10.1
  - numpy=1.23.5
  - pandas=2.0.3
  - networkx=3.0
  - anndata=0.8.0
  - matplotlib=3.7.3
  - scikit-learn=1.2.2
  - pytorch=1.13.1
  - torchvision
  - torchaudio
  - pytorch-cuda=11.7
  - seaborn=0.13.2
  - pip
  - pip:
      - SEACells==0.3.3
      - torch_geometric==2.5.3
      - scanpy==1.9.8
      - ot==0.9.1

## Directories

### Data_Notebooks
This directory includes the pre-processing notebooks for each dataset, starting from the raw counts. The pre-processing steps are explained in the preprint.
* [Kang dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)
* [NeurIPS dataset](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations)
* [McFarland and Chang datasets](http://projects.sanderlab.org/scperturb/datavzrd/scPerturb_vzrd_v1/dataset_info/index_1.html)
* [Nault dataset](https://github.com/BhattacharyaLab/scVIDR/tree/main)


### Training
This directory contains the training notebooks needed to reproduce the figures for each dataset. While results may slightly differ from the paper, they do not affect the overall conclusions.


