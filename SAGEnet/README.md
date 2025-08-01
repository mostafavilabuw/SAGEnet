This directory contains the SAGEnet editable package. 

## attributions.py 
- Save model attributions, identify seqlets, cluster seqlets, and match clusters to database. For usage, see READme in `/script/attributions/`

## data.py 
- Initialize PersonalGenomeDataset (from reference genome, WGS data, and y data), ReferenceGenomeDataset (from reference genome, y data), or VariantDataset (from reference genome, variant information). Y data can be any output measured per-individaul -- we use gene expression and DNA methylation in our analyses.

First, import SAGEnet.data: 
`import SAGEnet.data`

#### PersonalGenomeDataset 
After defining your paths: 
`<hg38_file_path>,<vcf_file_path>` 

And loading in your:  
`<sample_list>` (list of sample names as they appear in the VCF),  
`<metadata>` (DataFrame containing genome region-related information, specifically the columns 'chr', 'pos', and, optionally 'strand'. 'pos' should be the center of the region -- i.e., TSS for gene expression.      
`<y_data>` (DataFrame with y data, indexed by region names, with sample names as columns),  

Initialize the PersonalGenomeDataset with:   
`personal_dataset = SAGEnet.data.PersonalGenomeDataset(metadata=<metadata>, vcf_file_path=<vcf_file_path>, hg38_file_path=<hg38_file_path>, sample_list=<sample_list>, y_data=<y_data>)`

#### ReferenceGenomeDataset 
After defining your paths: 
`<hg38_file_path>`

And loading in your:  
`<metadata>` (DataFrame containing genome region-related information, specifically the columns 'chr', 'pos', and, optionally 'strand'. 'pos' should be the center of the region -- i.e., TSS for gene expression.      
`<y_data>` (DataFrame with y data, indexed by region names, with sample names as columns),  

Initialize the ReferenceGenomeDataset with:   
`reference_dataset = SAGEnet.data.ReferenceGenomeDataset(metadata=<metadata>, hg38_file_path=<hg38_file_path>, y_data=<y_data>)`

#### VariantDataset 
After defining your paths: 
`<hg38_file_path>`

And loading in your:  
`<metadata>` (DataFrame containing genome region-related information, specifically the columns 'chr', 'pos', and, optionally 'strand'. 'pos' should be the center of the region -- i.e., TSS for gene expression.      
`<variant_info>` (DataFrame of variant info containing the columns "region_id", "chr", "pos", "ref", "alt"),  

Initialize the ReferenceGenomeDataset with:   
`variant_dataset = SAGEnet.data.VariantDataset(metadata=<metadata>, hg38_file_path=<hg38_file_path>, variant_info=<variant_info>)`

## enformer.py 
- Initialize the Enformer model from TF Hub and use to predict from PersonalGenomeDataset, ReferenceGenomeDataset, or VariantDataset. For example use, see `SAGEnet/script/eval_model/eval_model.py`.

## models.py 
- Initialize rSAGEnet or pSAGEnet model. For training rSAGEnet, see `SAGEnet/script/baselines/train_rsagenet.py`. For training pSAGEnet, see `SAGEnet/script/train_model/train_model.py`. For evaluating both models, see `SAGEnet/script/eval_model/eval_model.py`.

## nn.py 
- Contains model components used in `models.py`.

## plot.py 
- Contains plotting functions. For example use, see `SAGEnet/plot_figs`. 

## tools.py 
- Contians misc functions used throughout SAGEnet package. 









