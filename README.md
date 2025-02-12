# SAGE-net

This repository contains the software package described in the paper  "A scalable approach to investigating sequence-to-expression prediction from personal genomes" and the scripts to reproduce the paper's analyses. 

## Installation 
To get started, clone the repository and install the required dependencies:
```bash
git clone git@github.com:mostafavilabuw/SAGEnet.git
cd SAGEnet
conda env create -f environment.yml
conda activate SAGEnet
```
Then, install the editable package: 
```bash
cd SAGEnet
pip install -e .
```
After these steps, you can import SAGEnet and access code from the files within the SAGEnet package (attributions.py, data.py, enformer.py, models.py, nn.py, plot.py, tools.py). For example, once you define the relevant inputs (see /scripts for examples), you can initialize a PersonalGenomeDataset by running: 
```bash
import SAGEnet
personal_dataset = SAGEnet.data.PersonalGenomeDataset(gene_metadata=your_gene_metadata, vcf_file_path=your_vcf_file_path, hg38_file_path=your_hg38_file_path, sample_list=your_sample_list)
```

