#!/bin/bash

"""
Used for running train_model.py on UW's hyak commpute cluster. 
This script is called by a seperate .sh script specifying command-line arguments for train_models.py. 
"""

#SBATCH --job-name=test_training # update job-name for each training run 
#SBATCH --output=./output/test_training/%j.out
#SBATCH --error=./output/test_training/%j.err
#SBATCH --partition=ckpt-all
#SBATCH --account=mostafavilab
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --exclude=g3007

source ~/.bashrc
source /gscratch/mostafavilab/aspiro17/micromamba/etc/profile.d/micromamba.sh
micromamba activate PersonalGenome # activate conda environment to use 

# define constant paths that do not change between model training runs 
TSS_DATA_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/gene-ids-and-positions.tsv
EXPR_DATA_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/vcf_match_covariate_adjusted_log_tpm.csv
SUB_DATA_DIR=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/sub_lists/
PREDIXCAN_RES_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/predixcan_40k_MAF=0.01_pearson_corr.csv
HG38_FILE_PATH=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/Genome/hg38.fa
VCF_FILE_PATH=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/ROSMAP/wholeGenomeSeq/chrAll.phased.vcf.gz
MODEL_SAVE_DIR=/gscratch/mostafavilab/aspiro17/personal_genome_res/${exp_label}/

# if job_id is provided (to resume model training), use this provided id instead of the SLURM job id 
if [ "${use_manual_job_id}" = "1" ]; then
    echo "using provided job id: ${manual_job_id}"
    curr_job_id=${manual_job_id}
else
    echo "using slurm job id: ${SLURM_JOB_ID}"
    curr_job_id=${SLURM_JOB_ID}
fi

# run train_model.py with the inputs provided, either in this .sh script or in the .sh script that calls this one 
srun python train_model.py \
    --model_save_dir ${MODEL_SAVE_DIR} \
    --tss_data_path ${TSS_DATA_PATH} \
    --expr_data_path ${EXPR_DATA_PATH} \
    --sub_data_dir ${SUB_DATA_DIR} \
    --predixcan_res_path ${PREDIXCAN_RES_PATH} \
    --hg38_file_path ${HG38_FILE_PATH} \
    --vcf_file_path ${VCF_FILE_PATH} \
    --wandb_project ${exp_label} \
    --lam_diff ${lam_diff} \
    --lam_ref ${lam_ref} \
    --ref_model_ckpt_path ${ref_model_ckpt_path} \
    --wandb_job_name ${curr_job_id} \
    --start_from_ref ${start_from_ref} \
    --num_nodes ${SLURM_NNODES} \
    --zscore ${zscore} \
    --num_top_train_genes ${num_top_train_genes} \
    --num_top_val_genes ${num_top_val_genes} \
    --only_snps ${only_snps} \
    --split_expr ${split_expr} \
    --num_training_subs ${num_training_subs} \
    --save_all_epochs ${save_all_epochs} \
    --rand_genes ${rand_genes} \
    --input_len ${input_len} \
    --max_epochs ${max_epochs} \
    --gene_idx_start ${gene_idx_start} \
    --allow_reverse_complement ${allow_reverse_complement} \
    --block_type ${block_type} \
    --first_layer_kernel_number ${first_layer_kernel_number} \
    --int_layers_kernel_number ${int_layers_kernel_number} \
    --h_layers ${h_layers} \
    --batch_size ${batch_size} 


    

