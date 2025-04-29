# SAGE-net

This repository contains the software package described in the paper  "A scalable approach to investigating sequence-to-expression prediction from personal genomes" (https://doi.org/10.1101/2025.02.21.639494) and the scripts to reproduce the paper's analyses.  
For descriptions of each file and instructions on how to reproduce our analyses and run your own, see the READMEs in each directory.    
For an example notebook that walks through how to use SAGE-net, see `SAGEnet_usage.ipynb`.  

## Installation 
To get started, clone the repository and install the required dependencies:
```bash
git clone git@github.com:mostafavilabuw/SAGEnet.git
mamba env create -f environment.yml
mamba activate SAGEnet
```

Mamba is a faster implementation of the conda package manager. If you don't already have mamba installed, you can install it using conda (https://anaconda.org/conda-forge/mamba) or use conda instead. 

Then, install the SAGEnet editable package: 
```bash
pip install -e .
```

When you import SAGEnet at the top of any file, you can access functions from the source code within the SAGEnet package (i.e., files within the SAGEnet directory: attributions.py, data.py, enformer.py, models.py, nn.py, plot.py, tools.py).      
By installing SAGEnet as an editable package, you can make your own changes to the source code, and these changes will be reflected in your project without having to reinstall the package.   

## Repository Structure 
- `SAGEnet/` - package source code 
  - `model.py` - model architectures (rSAGEnet, pSAGEnet)  
  - `data.py` - custom datasets (PersonalGenomeDataset, ReferenceGenomeDataset, VariantDataset)
  - `enformer.py` - use the Enformer model 
  - `attributions.py` - save model attributions, identify seqlets, match identified seqlets to motif database 
  - `nn.py` - model components used in models.py
  - `plot.py` - plotting functions
  - `tools.py` - miscelanious helper functions used throughout the SAGEnet package

- `input_data/` - input data used to run our analyses
- `plot_figs/` - code to create the figures in the paper
- `results_data/` - results from our analyses 
- `script/` - scripts used to run our analyses 
  - `baselines/` - scripts to train the baselines used (r-SAGE-net, Enformer, PrediXcan) 
  - `data_preprocessing/` - script to preprocess the input data used in our analyses 
  - `eval_model/` - scripts to evaluate models (p-SAGE-net, r-SAGE-net, Enformer)
  - `train_model/` - scripts to train p-SAGE-net models on personal sequence and expression 

For more in-depth descriptions of each file, see the READMEs in each directory. 

## Abstract

_A key promise of sequence-to-function (S2F) models is their ability to evaluate arbitrary sequence inputs, providing a robust framework for understanding genotype-phenotype relationships. However, despite strong performance across genomic loci , S2F models struggle with inter-individual variation. Training a model to make genotype-dependent predictions at a single locus—an approach we call personal genome training—offers a potential solution. We introduce SAGE-net, a scalable framework and software package for training and evaluating S2F models using personal genomes. Leveraging its scalability, we conduct extensive experiments on model and training hyperparameters, demonstrating that training on personal genomes improves predictions for held-out individuals. However, the model achieves this by identifying predictive variants rather than learning a cis-regulatory grammar that generalizes across loci. This failure to generalize persists across a range of hyperparameter settings. These findings highlight the need for further exploration to unlock the full potential of S2F models in decoding the regulatory grammar of personal genomes. Scalable software and infrastructure development will be critical to this progress._
