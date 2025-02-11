#!/bin/bash

num_nodes=8
zscore=1
start_from_ref=1
num_top_val_genes=1000
num_top_train_genes=1000
only_snps=0
split_expr=1
save_all_epochs=1
rand_genes=0
use_manual_job_id=0
max_epochs=10
input_len=40000
manual_job_id=placeholder
allow_reverse_complement=1
ref_model_ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/ref_cnn_models/rc_neg_strand/job0/epoch=12-step=32045.ckpt
gene_idx_start=0
block_type=conv
int_layers_kernel_number=256
first_layer_kernel_number=900
h_layers=1
batch_size=6
lam_ref=1
lam_diff=10
num_training_subs=689
base_exp_label=num_gpu

num_nodes=(2 4 8)
for num_node in "${num_nodes[@]}"; do
    exp_label="${base_exp_label}_${num_node}"
    sbatch --nodes=${num_node} --export=ALL,exp_label=${exp_label} run_train_model.sh
done 

