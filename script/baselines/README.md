This directory contains code for the baselines used: Enformer, PrediXcan, r-SAGE-net. 

## enformer/finetune_enformer.py 
- Save weights to use to fine-tune Enformer predictions by training an elastic net model on all track predictions and mean GTEx expression data for Enformer train genes. 

## enformer/save_enformer_gene_sets.py 
- Save Enformer's train, validation, and test sets based on data from `https://console.cloud.google.com/storage/browser/basenji_barnyard/data`

## predixcan/predixcan.py 
- Train PrediXcan (elastic net) model on ROSMAP training individuals, evaluate on ROSMAP train/validation/test individuals and GTEx individuals.

  After defining your paths: 
`<results_save_dir>,<gene_list_path>,<sub_list_dir>,<rosmap_vcf_path>,<gtex_vcf_path>,<hg38_file_path>,<rosmap_expr_data_path>,<tss_data_path>`  

Run:

`$python predixcan.py --results_save_dir <results_save_dir> --gene_list_path <gene_list_path> --sub_list_dir <sub_list_dir> --rosmap_vcf_path <rosmap_vcf_path>  --gtex_vcf_path <gtex_vcf_path> --hg38_file_path <hg38_file_path> --rosmap_expr_data_path <rosmap_expr_data_path> --tss_data_path <tss_data_path>`   

To save PrediXcan predictions and coefficients. 

## rsagenet/hyperparameter_search.sh 
- Submit jobs to train rSAGEnet models with different combinations of hyperparameters (from defined grid search). Calls `run_train_rsagenet.sh`. 

## rsagenet/run_train_rsagenet.sh 
- Run `train_rsagenet.py` with specified set of hyperparameters. Called by `hyperparameter_search.sh`.

## rsagenet/train_rsagenet.py
- Train r-SAGE-net model with specified set of hyperparameters. Called by `run_train_rsagenet.sh`.

