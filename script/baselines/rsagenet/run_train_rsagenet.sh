#!/bin/bash

"""
Used for running train_rsagenet.py on UW's hyak commpute cluster. 
This script is called by a seperate .sh script specifying command-line arguments for train_rsagenet.py. 
"""

#SBATCH --job-name=train_rsagenet
#SBATCH --output=./output/train_rsagenet%j.out
#SBATCH --error=./output/train_rsagenet%j.err
#SBATCH --partition=gpu-a40
#SBATCH --account=mostafavilab
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --exclude=g3007

source ~/.bashrc
source /gscratch/mostafavilab/aspiro17/micromamba/etc/profile.d/micromamba.sh
micromamba activate PersonalGenome

# define constant paths that do not change between model training runs 
MODEL_SAVE_DIR=/gscratch/mostafavilab/aspiro17/personal_genome_res/ref_cnn_models/test_arch
TSS_DATA_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/gene-ids-and-positions.tsv
EXPR_DATA_PATH=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/vcf_match_covariate_adjusted_log_tpm.csv
SUB_DATA_DIR=/gscratch/mostafavilab/aspiro17/PersonalGenomeExpression-dev/data/ROSMAP/sub_lists/
HG38_FILE_PATH=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/Genome/hg38.fa


# run train_model.py with the inputs provided, either in this .sh script or in the .sh script that calls this one 
srun python train_model.py \
    --num_nodes ${SLURM_NNODES} \
    --model_save_dir ${MODEL_SAVE_DIR} \
    --tss_data_path ${TSS_DATA_PATH} \
    --expr_data_path ${EXPR_DATA_PATH} \
    --sub_data_dir ${SUB_DATA_DIR} \
    --hg38_file_path ${HG38_FILE_PATH} \
    --first_layer_kernel_number ${first_layer_kernel_number} \
    --int_layers_kernel_number ${int_layers_kernel_number} \
    --first_layer_kernel_size ${first_layer_kernel_size} \
    --int_layers_kernel_size ${int_layers_kernel_size} \
    --n_conv_blocks ${n_conv_blocks} \
    --pooling_size ${pooling_size} \
    --pooling_type ${pooling_type} \
    --n_dilated_conv_blocks ${n_dilated_conv_blocks} \
    --dropout ${dropout} \
    --h_layers ${h_layers} \
    --increasing_dilation ${increasing_dilation} \
    --batch_norm ${batch_norm} \
    --hidden_size ${hidden_size} \
    --learning_rate ${learning_rate} \

