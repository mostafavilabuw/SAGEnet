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
base_ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/larger_gene_sets_save_all_epochs/

train_gene_sets=(1000 2000 3000 4000 5000)

for train_gene_set in "${train_gene_sets[@]}"; do

    if [ "${train_gene_set}" -eq 1000 ]; then
    job_id=22527202
    num_jobs=10
    fi

    if [ "${train_gene_set}" -eq 2000 ]; then
    job_id=22527203
    num_jobs=20
    fi

    if [ "${train_gene_set}" -eq 3000 ]; then
    job_id=22527204
    num_jobs=30
    fi

    if [ "${train_gene_set}" -eq 4000 ]; then
    job_id=22527205
    num_jobs=40
    fi
    
    if [ "${train_gene_set}" -eq 5000 ]; then
    job_id=22527206
    num_jobs=50
    fi

    ckpt_path="${base_ckpt_path}${job_id}/"

    for i in $(seq 1 ${num_jobs}); do
        sbatch --export=ALL,ckpt_path=${ckpt_path} run_eval_model.sh
    done

done 


