This directory contains the SAGEnet editable package. 

## attributions.py 
- Save attributions (gradients or ISM) and use tangermeme functions to identify seqlets and match identified seqlets to motif database.

#### ISM example (Fig. S3-S5) 
After defining your paths: 
`<results_save_dir>,<ckpt_path>,<variant_info_path>,<hg38_file_path>,<tss_data_path>,<finetuned_weights_dir>`  

Run:

p-SAGE-net, insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type psagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --variant_info_path <variant_info_path>`   

p-SAGE-net, do not insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type psagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 

r-SAGE-net, insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type rsagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --variant_info_path <variant_info_path>`   

r-SAGE-net, do not insert variant:       
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type rsagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 

Enformer, insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type enformer --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --variant_info_path <variant_info_path> --finetuned_weights_dir <finetuned_weights_dir>`   

Enformer, do not insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type enformer --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --finetuned_weights_dir <finetuned_weights_dir>` 

#### Seqlet analysis (Fig. S6) 
After defining your paths: 
`<ckpt_path>,<results_save_dir>,<hg38_file_path>,<tss_data_path>,<predixcan_res_path>,<gene_list_path>,<motif_database_path>` 

Run: 

p-SAGE-net save gradients:   
`$python attributions.py --which_fn save_ref_seq_gradients --ckpt_path <ckpt_path> --results_save_dir <results_save_dir> --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path> --model_type psagenet --predixcan_res_path <predixcan_res_path>`

r-SAGE-net save gradients:   
`$python attributions.py --which_fn save_ref_seq_gradients --ckpt_path <ckpt_path> --results_save_dir <results_save_dir> --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path> --model_type rsagenet --predixcan_res_path <predixcan_res_path>`

p-SAGE-net annotate seqlets:   
`$python attributions.py --which_fn mult_gene_save_annotated_seqlets --gene_list_path <gene_list_path> --attrib_path <results_save_dir>psagenet_model/gradients/personal_seq_1_idx_grads.npy --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 

r-SAGE-net annotate seqlets:   
`$python attributions.py --which_fn mult_gene_save_annotated_seqlets --gene_list_path <gene_list_path> --attrib_path <results_save_dir>rsagenet_model/gradients/grads.npy --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 

## data.py 
- Initialize PersonalGenomeDataset (from reference genome, WGS data, and expression data), ReferenceGenomeDataset (from reference genome, expression data), or VariantDataset (from reference genome, variant information).

First, import SAGEnet.data: 
`import SAGEnet.data`

#### PersonalGenomeDataset 
After defining your paths: 
`<hg38_file_path>,<vcf_file_path>` 

And loading in your:  
`<sample_list>` (list of sample names as they appear in the VCF),  
`<gene_metadata>` (DataFrame of gene metadata containing the columns "chr", "tss", and "strand"),    
`<expr_data>` (DataFrame of expression data indexed by gene names, with sample names as columns),  

Initialize the PersonalGenomeDataset with:   
`personal_dataset = SAGEnet.data.PersonalGenomeDataset(gene_metadata=<gene_metadata>, vcf_file_path=<vcf_file_path>, hg38_file_path=<hg38_file_path>, sample_list=<sample_list>, expr_data=<expr_data>)`

#### ReferenceGenomeDataset 
After defining your paths: 
`<hg38_file_path>`

And loading in your:  
`<gene_metadata>` (DataFrame of gene metadata containing the columns "chr", "tss", and "strand"),    
`<expr_data>` (DataFrame of expression data indexed by gene names, with sample names as columns),  

Initialize the ReferenceGenomeDataset with:   
`reference_dataset = SAGEnet.data.ReferenceGenomeDataset(gene_metadata=<gene_metadata>, hg38_file_path=<hg38_file_path>, expr_data=<expr_data>)`

#### VariantDataset 
After defining your paths: 
`<hg38_file_path>`

And loading in your:  
`<gene_metadata>` (DataFrame of gene metadata containing the columns "chr", "tss", and "strand"),    
`<variant_info>` (DataFrame of variant info containing the columns "gene", "chr", "pos", "ref", "alt"),  

Initialize the ReferenceGenomeDataset with:   
`variant_dataset = SAGEnet.data.VariantDataset(gene_metadata=<gene_metadata>, hg38_file_path=<hg38_file_path>, variant_info=<variant_info>)`

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









