#!/bin/bash

num_jobs=10
num_genes=1000
rand_genes=0
only_snps=0
max_epochs=10
maf_threshold=-1
best_ckpt_metric=train_gene_gene
gene_idx_start=0
input_len=40000
allow_reverse_complement=0
ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/no_expr_split_no_subtract/23941856/

for i in $(seq 1 ${num_jobs}); do
    sbatch --export=ALL run_eval_model.sh
done 
