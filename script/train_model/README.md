This directory contains code to train p-SAGE-net. 

## training_experiment_sh_scripts
- Contains scripts to train p-SAGE-net models with different combinations of hyperparameters. Each calls `run_train_model.sh`.

## run_eval_model.sh
- Run `train_model.py` to train p-SAGE-net model with specified set of hyperparameters. Called by scripts in `training_experiment_sh_scripts`.

## train_model.py 
- Train p-SAGE-net on personal sequence for ROSMAP training individuals.

First, add your WANDB (`https://wandb.ai/home`) key to the top of the script (`os.environ["WANDB_API_KEY"] = 'your_key'`) to track model training. 

To train a model, define your paths:  
`<model_save_dir>,<tss_data_path>,<hg38_file_path>,<predixcan_res_path>,<vcf_file_path>,<sub_data_dir>,<expr_data_path>`  

If initializing p-SAGE-net model from r-SAGE-net model, define path to r-SAGE-net model: 
`<ref_model_ckpt_path>`
 
Specify your gene set:  
`<num_genes>,<rand_genes>,<gene_idx_start>` 

Specify number of training individuals: 
`<num_training_subs>`

And run:  
`$python train_model.py --model_save_dir <model_save_dir> --tss_data_path <tss_data_path> --hg38_file_path <hg38_file_path> --predixcan_res_path <predixcan_res_path> --vcf_file_path <vcf_file_path> --vcf_file_path <vcf_file_path> --sub_data_dir <sub_data_dir> --expr_data_path <expr_data_path> --ref_model_ckpt_path <ref_model_ckpt_path> --num_genes <num_genes> --rand_genes <rand_genes> --gene_idx_start <gene_idx_start> --num_training_subs <num_training_subs>`   
To train model on personal sequence and personal expression.   

