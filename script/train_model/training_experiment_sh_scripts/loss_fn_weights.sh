#!/bin/bash

num_nodes=8
zscore=1
start_from_ref=1
num_top_train_genes=1000
num_top_val_genes=1000
only_snps=0
split_expr=1
num_training_subs=689
save_all_epochs=1
rand_genes=0
use_manual_job_id=0
max_epochs=10
input_len=40000
allow_reverse_complement=0
ref_model_ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/causal_analysis/epoch=16-step=15000.ckpt
gene_idx_start=0
block_type=conv
int_layers_kernel_number=256
first_layer_kernel_number=900
h_layers=1
batch_size=6
lam_ref=1
exp_label=loss_fn_weights

lam_diffs=(1 10 100)
for lam_diff in "${lam_diffs[@]}"; do
    sbatch --nodes=${num_nodes} --export=ALL,lam_diff=${lam_diff} run_train_model.sh
done 

