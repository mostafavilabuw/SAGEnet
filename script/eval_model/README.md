This directory contains code to evaluate p-SAGE-net, r-SAGE-net, and Enformer. 

## evaluation_sh_scripts
- Contains scripts to evaluate p-SAGE-net models with different combinations of hyperparameters. Each calls `run_eval_model.sh`.

## eval_model.py 
- Evaluate p-SAGE-net, r-SAGE-net, or Enformer model on personal sequence or reference sequence.

To evaluate a model, define your paths:  
 `<ckpt_path>,<results_save_dir>,<tss_data_path>,<hg38_file_path>,<predixcan_res_path>,<vcf_file_path>,<sub_data_dir>`  

If evaluating Enformer, specify path to fine-tuned weights and how to transform Enformer output: 
 `<enformer_finetuned_weights_dir>,<enformer_save_mode>`  
 
Specify your gene set:  
`<num_genes>,<rand_genes>,<gene_idx_start>` 

Specify the individuals to evaluate on: 
`<rosmap_or_gtex>,<train_val_test>`

Specify your model type: 
`<model_type>`

(Optionally) specify a MAF threshold above which to insert variants: 
`<maf_threshold>`

And run:  
`$python eval_model.py --ckpt_path <ckpt_path> --results_save_dir <results_save_dir> --tss_data_path <tss_data_path> --hg38_file_path <hg38_file_path> --predixcan_res_path <predixcan_res_path> --vcf_file_path <vcf_file_path> --sub_data_dir <sub_data_dir> --enformer_finetuned_weights_dir <enformer_finetuned_weights_dir> --enformer_save_mode <enformer_save_mode> --num_genes <num_genes> --rand_genes <rand_genes> --gene_idx_start <gene_idx_start> --rosmap_or_gtex <rosmap_or_gtex> --train_val_test <train_val_test> --model_type <model_type> --maf_threshold <maf_threshold>`

To evaluate model on personal sequence.   
To instead evaluate on reference sequence, add `--eval_on_ref_seq True`

## run_eval_model.sh
- Run `eval_model.py` to evaluate p-SAGE-net model with specified set of hyperparameters. Called by scripts in `evaluation_sh_scripts`.

 
  
