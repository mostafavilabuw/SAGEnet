#!/bin/bash

num_jobs=10
num_genes=1000
rand_genes=0
only_snps=0
max_epochs=10
best_ckpt_metric=train_gene_gene
gene_idx_start=4000
input_len=40000
allow_reverse_complement=1
ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/harder_num_training_subs_save_all_epochs/23317290/

for maf_threshold in "${maf_thresholds[@]}"; do
    for i in $(seq 1 ${num_jobs}); do
        sbatch --export=ALL,maf_threshold=${maf_threshold} run_eval_model.sh
    done
done 


