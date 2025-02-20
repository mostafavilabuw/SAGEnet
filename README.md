# SAGE-net

This repository contains the software package described in the paper  "A scalable approach to investigating sequence-to-expression prediction from personal genomes" and the scripts to reproduce the paper's analyses. For descriptions of each file and instructions on how to reproduce our analyses and run your own, see the READMEs in each directory. 

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
After these steps, you can import SAGEnet to access code from the files within the SAGEnet package (attributions.py, data.py, enformer.py, models.py, nn.py, plot.py, tools.py).    

For example,    

After defining your paths: 
`<hg38_file_path>,<vcf_file_path>`   

And loading in your:  
`<sample_list>` (list of sample names as they appear in the VCF),  
`<gene_metadata>` (DataFrame of gene metadata containing the columns "chr", "tss", and "strand"),    
`<expr_data>` (DataFrame of expression data indexed by gene names, with sample names as columns),  

You can initialize a PersonalGenomeDataset by running:  
```
import SAGEnet.data  
personal_dataset = SAGEnet.data.PersonalGenomeDataset(gene_metadata=<gene_metadata>, vcf_file_path=<vcf_file_path>, hg38_file_path=<hg38_file_path>, sample_list=<sample_list>, expr_data=<expr_data>)
```

For more examples, see the READMEs in each directory. 

