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

Then, install the editable package: 
```bash
pip install -e .
```

## Example Use 
After these steps, you can import SAGEnet to access code from the files within the SAGEnet package (attributions.py, data.py, enformer.py, models.py, nn.py, plot.py, tools.py).    

For example, to initailize a PersonalGenomeDataset:   
You first need to download the hg38 reference genome as `'input_data/hg38.fa'` using `input_data/download_genome.sh`.  
We provide toy data in `'input_data/example_data/'` to demonstrate required data formats.  

Using this directory, you can define your paths:    
`hg38_file_path='input_data/hg38.fa'`  
`example_vcf_file_path='input_data/example_data/example_vcf.vcf.gz'`    

Load your data:   
`example_individuals = np.loadtxt('input_data/example_data/example_individuals.csv',delimiter=',',dtype=str)` (list of sample names as they appear in the VCF)     
`example_expression_data = pd.read_csv('input_data/example_data/example_expression.csv',index_col=0)` (DataFrame of expression data indexed by gene names, with sample names as columns)   
`gene_meta_info = pd.read_csv(tss_data_path, sep='\t')`  (DataFrame of gene metadata containing the columns 'chr', 'tss', and 'strand')    

And select gene meta information for an example gene (for which variant data is provided in `example_vcf_file_path`):   
`example_gene_meta_info=gene_meta_info[gene_meta_info['gene_id']=='ENSG00000013573']`

After this, you can initialize a PersonalGenomeDataset with:
```
import SAGEnet.data  
personal_dataset = SAGEnet.data.PersonalGenomeDataset(gene_metadata=example_gene_meta_info, vcf_file_path=example_vcf_file_path, hg38_file_path=hg38_file_path, sample_list=example_individuals, expr_data=example_expression_data)
```

Each item in `personal_dataset` will be a tuple containing:   
- One-hot-encoded tensor of genomic sequence of shape `[2,8,40000]`. This contains personal genomic sequence from each haplotype (maternal: `[0,:4,:]`, paternal: `[0,4:,:]`) and reference sequence (`[1,:,:]`). 
- Expression tensor containing [mean expression, personal difference from mean expression]  
- Gene index 
- Sample index      

Note that these outputs (for example, sequence length, how to return personal expression) are customisable, see `SAGEnet/data.py` documentation for details.
There should be no lag time when iterating through the PersonalGenomeDataset.  

For more examples, see the READMEs in each directory. 

## Abstract

_A key promise of sequence-to-function (S2F) models is their ability to evaluate arbitrary sequence inputs, providing a robust framework for understanding genotype-phenotype relationships. However, despite strong performance across genomic loci , S2F models struggle with inter-individual variation. Training a model to make genotype-dependent predictions at a single locus—an approach we call personal genome training—offers a potential solution. We introduce SAGE-net, a scalable framework and software package for training and evaluating S2F models using personal genomes. Leveraging its scalability, we conduct extensive experiments on model and training hyperparameters, demonstrating that training on personal genomes improves predictions for held-out individuals. However, the model achieves this by identifying predictive variants rather than learning a cis-regulatory grammar that generalizes across loci. This failure to generalize persists across a range of hyperparameter settings. These findings highlight the need for further exploration to unlock the full potential of S2F models in decoding the regulatory grammar of personal genomes. Scalable software and infrastructure development will be critical to this progress._
