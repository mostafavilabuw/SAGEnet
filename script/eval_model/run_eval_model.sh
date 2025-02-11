#!/bin/bash

"""
Used for running eval_model.py on UW's hyak commpute cluster. 
This script is called by a seperate .sh script specifying command-line arguments for eval_models.py. 
"""

#SBATCH --job-name=eval_models
#SBATCH --output=./output/eval_models/%j.out
#SBATCH --error=./output/eval_models/%j.err
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

# define constant paths that do not change between model evals
TSS_DATA_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/gene-ids-and-positions.tsv
PREDIXCAN_RES_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/predixcan_40k_MAF=0.01_pearson_corr.csv
HG38_FILE_PATH=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/Genome/hg38.fa
SUB_DATA_DIR=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/sub_lists/
VCF_FILE_PATH=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/ROSMAP/wholeGenomeSeq/chrAll.phased.vcf.gz
BATCH_SIZE=6

srun python eval_model.py \
    --tss_data_path ${TSS_DATA_PATH} \
    --hg38_file_path ${HG38_FILE_PATH} \
    --batch_size ${BATCH_SIZE} \
    --predixcan_res_path ${PREDIXCAN_RES_PATH} \
    --sub_data_dir ${SUB_DATA_DIR} \
    --vcf_file_path ${VCF_FILE_PATH} \
    --train_subs_vcf_file_path ${VCF_FILE_PATH} \
    --train_subs_expr_data_path ${TRAIN_SUBS_EXPR_DATA_PATH} \
    --ckpt_path ${ckpt_path} \
    --num_genes ${num_genes} \
    --rand_genes ${rand_genes} \
    --only_snps ${only_snps} \
    --max_epochs ${max_epochs} \
    --maf_threshold ${maf_threshold} \
    --best_ckpt_metric ${best_ckpt_metric} \
    --gene_idx_start ${gene_idx_start} \
    --input_len ${input_len} \
    --allow_reverse_complement ${allow_reverse_complement}



