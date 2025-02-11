#!/bin/bash

num_jobs=10
num_genes=1000
rand_genes=0
only_snps=0
split_expr=1
max_epochs=10
maf_threshold=-1
best_ckpt_metric=train_gene_gene
gene_idx_start=0
allow_reverse_complement=1
base_ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/test_weights_save_all_epochs/

diff_weights=(1 10 100)

for weight in "${diff_weights[@]}"; do

    if [ "${weight}" -eq 1 ]; then
    job_id=22528530
    fi

    if [ "${weight}" -eq 10 ]; then
    job_id=22528531
    fi

    if [ "${weight}" -eq 100 ]; then
    job_id=22528533
    fi
    
    ckpt_path="${base_ckpt_path}${job_id}/"

    for i in $(seq 1 ${num_jobs}); do
        sbatch --export=ALL,ckpt_path=${ckpt_path} run_eval_model.sh
    done

done 

