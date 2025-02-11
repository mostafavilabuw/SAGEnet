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
allow_reverse_complement=1
base_ckpt_path=/gscratch/mostafavilab/aspiro17/personal_genome_res/harder_num_training_subs_save_all_epochs/23317290/

num_training_subs=(5 50 200 400)
for num in "${num_training_subs[@]}"; do

    if [ "${num}" -eq 5 ]; then
    job_id=22982304
    fi

    if [ "${num}" -eq 50 ]; then
    job_id=22982305
    fi

    if [ "${num}" -eq 200 ]; then
    job_id=22982306
    fi
    
    if [ "${num}" -eq 400 ]; then
    job_id=22982307
    fi
    
    ckpt_path="${base_ckpt_path}${job_id}/"

    for i in $(seq 1 ${num_jobs}); do
        sbatch --export=ALL,ckpt_path=${ckpt_path} run_eval_model.sh
    done
done
