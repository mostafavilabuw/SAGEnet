# SAGE-net

This repository contains the software package described in the paper  "A scalable approach to investigating sequence-to-expression prediction from personal genomes" [[1]](#1) and the scripts to reproduce the paper's analyses.  
For descriptions of each file and instructions on how to reproduce our analyses and run your own, see the READMEs in each directory.    

For a notebook tutorials, see `example_notebooks/`. `GEUVADIS_example.ipynb` is a fully runnable notebook that goes through dataset creation and model training with the publicly available GEUVADIS dataset [[2]](#2), using expression data preprocessed by Rastogi et al. [[3]](#3). `example_usage.ipynb` uses toy data provided in this repo to describe the inputs required to create your own datset. It details important parameters to change to make SAGEnet fit your needs, and walks through the process we use to train models on the ROSMAP dataset.  

## Getting started  
First, clone the repository and install the required dependencies:
```bash
git clone git@github.com:mostafavilabuw/SAGEnet.git
mamba env create -f environment.yml
mamba activate SAGEnet
```

Mamba is a faster implementation of the conda package manager. If you don't already have mamba installed, you can install it using conda (https://anaconda.org/conda-forge/mamba) or use conda instead. 

Then, install the `SAGEnet` editable package: 
```bash
pip install -e .
```

When you import SAGEnet at the top of any file, you can access functions from the source code (i.e., files within the SAGEnet directory: attributions.py, data.py, enformer.py, models.py, nn.py, plot.py, tools.py).      
By installing SAGEnet as an editable package, you can make your own changes to the source code, and these changes will be reflected in your project without having to reinstall the package.  

This pacakge also uses attribution analysis functions from the editable package `drg_tools` (https://github.com/sasselab/DRG/tree/main). After cloning this repo, you can install the `drg_tools` editable package in the same way you installed SAGEnet: by running:
```bash
pip install -e .
```
From the folder that contains the `setup.py` file.

Throughout our scripts and example notebooks, we use Weights & Biases (https://wandb.ai/home) to track model training. We recommend setting up a free account so that you can do the same, but if you prefer not to, you can simply delete these lines from your code.

## Repository Structure 
- `SAGEnet/` - package source code 
  - `model.py` - model architectures (rSAGEnet, pSAGEnet)  
  - `data.py` - custom datasets (PersonalGenomeDataset, ReferenceGenomeDataset, VariantDataset)
  - `enformer.py` - use the Enformer model 
  - `attributions.py` - save model attributions, identify seqlets, cluster seqlets, and match cluster to database
  - `nn.py` - model components used in models.py
  - `plot.py` - plotting functions
  - `tools.py` - miscelanious helper functions used throughout the SAGEnet package

- `input_data/` - input data used to run our analyses
- `plot_figs/` - code to create the figures in the paper
- `results_data/` - results from our analyses 
- `script/` - scripts used to run our analyses
  - `attributions/` - script to do ISM on a given region with specified variant inserted or to use the functions in `SAGEnet/attributions.py` to perform a global motif analyis
  - `baselines/` - scripts to train the baselines used (r-SAGE-net, Enformer, PrediXcan) 
  - `data_preprocessing/` - script to preprocess the input data used in our analyses 
  - `eval_model/` - scripts to evaluate models (p-SAGE-net, r-SAGE-net, Enformer)
  - `train_model/` - scripts to train p-SAGE-net models on personal sequence and expression 

For more in-depth descriptions of each file, see the READMEs in each directory. 

## Abstract

_Sequence-to-function (S2F) models hold the promise of evaluating arbitrary DNA sequences,  providing a powerful framework for linking genotype to phenotype. Yet, despite strong  performance across genomic loci, these models often struggle to capture inter-individual  variation in gene expression. To address this, we propose personal genome training—training  models to make genotype-specific predictions at a single locus. We introduce SAGE-net, a  scalable framework and software package for training and evaluating S2F models using personal  genomes. Using SAGE-net, we systematically explore model architectures and training regimes,  showing that personal genome training improves gene expression prediction accuracy for  held-out individuals. However, performance gains arise primarily from identifying predictive  variants, rather than learning a cis-regulatory grammar that generalizes across loci. This lack of  generalization persists across a wide range of hyperparameters. In contrast, when applied to  DNA methylation (DNAm), personal genome training enables improved generalization to unseen  individuals in unseen genomic regions. This suggests that S2F models may more readily capture  the sequence-level determinants of inter-individual variation in epigenomic traits. These  findings highlight the need for further exploration to unlock the full potential of S2F models in  decoding the regulatory grammar of personal genomes. Scalable software and infrastructure  development will be critical to this progress._

## References

<a id="1">[1]</a> 
Spiro AE\*, Tu X\*, Sheng Y, Sasse A, Hosseini R, Chikina M, et al. A scalable approach to investigating sequence-to-expression prediction from personal genomes [Internet]. bioRxiv; 2025 [cited 2025 Feb 26]. p. 2025.02.21.639494. Available from: https://www.biorxiv.org/content/10.1101/2025.02.21.639494v2, \* These authors contributed equally

<a id="2">[2]</a>
Lappalainen T, Sammeth M, Friedländer MR, ‘t Hoen PAC, Monlong J, Rivas MA, et al. Transcriptome and genome sequencing uncovers functional variation in humans. Nature. 2013 Sep;501(7468):506–11. 

<a id="2">[3]</a>
Rastogi R\*, Reddy AJ\*, Chung R, Ioannidis NM. Fine-tuning sequence-to-expression models on personal genome and transcriptome data [Internet]. 2024 [cited 2024 Oct 17]. Available from: http://biorxiv.org/lookup/doi/10.1101/2024.09.23.614632, \* These authors contributed equally



