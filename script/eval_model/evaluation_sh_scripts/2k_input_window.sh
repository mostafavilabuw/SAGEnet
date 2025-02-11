#!/bin/bash

num_jobs=10
num_genes=1000
rand_genes=0
only_snps=0
max_epochs=10
maf_threshold=-1
best_ckpt_metric=train_gene_gene
gene_idx_start=0
input_len=2000
allow_reverse_complement=1
ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/1k_input_2k_no_rc_fix_err/23149453/

for i in $(seq 1 ${num_jobs}); do
    sbatch --export=ALL run_eval_model.sh
done 

