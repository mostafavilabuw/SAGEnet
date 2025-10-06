#!/bin/bash

#SBATCH --job-name=eval_dnam_fig_2c
#SBATCH --output=./output/eval_dnam_fig_2c_%j.out
#SBATCH --error=./output/eval_dnam_fig_2c_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00 

source ~/.bashrc
source /gscratch/mostafavilab/aspiro17/micromamba/etc/profile.d/micromamba.sh
micromamba activate SAGEnet

metadata_path=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/data/ROSMAP_DNAm/dnam_meta_hg38.csv
enet_res_path="/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/enet/dnam/summarized_res/rosmap/input_len_10000/maf_filter_0.01/pearson_corrs.csv"
new_chr_split=0
num_eval_regions=5000
device=0

hg38_file_path=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/Genome/hg38.fa
vcf_path="/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/ROSMAP/wholeGenomeSeq/chrAll.phased.vcf.gz"
sub_data_dir='/gscratch/mostafavilab/aspiro17/DNAm_and_expression/input_data/individual_lists/'

# C 
train_val_test_subs=test

# psagenet 
model_type=psagenet
psagenet_base_dir=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/revisions_pt_2/psagenet/dnam/rosmap/dnam_version_of_fig_2/

#for region_split in train test; do 
#    for region_idx_start in 0 20000; do # easier set vs. harder set 
#        for n_training_subs in 5 50 200 400; do 
#            model_ckpt=${psagenet_base_dir}panel_c_region_idx_start_${region_idx_start}_n_training_subs_${n_training_subs}/epoch=0.ckpt
#            srun python /gscratch/mostafavilab/aspiro17/DNAm_and_expression/script/eval/eval_model.py --model_type ${model_type} --ckpt_path ${model_ckpt} --eval_on_ref_seq 0 --train_val_test_regions ${region_split} --num_eval_regions ${num_eval_regions} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --enet_res_path ${enet_res_path} --device ${device} --train_val_test_subs ${train_val_test_subs} --hg38_file_path ${hg38_file_path} --vcf_path ${vcf_path} --sub_data_dir ${sub_data_dir} --region_idx_start ${region_idx_start}
#        done 
#    done 
#done 

for region_split in train test; do 
    for region_idx_start in 20000; do # easier set vs. harder set 
        for n_training_subs in 5 50 200 400; do 
            model_ckpt=${psagenet_base_dir}panel_c_region_idx_start_${region_idx_start}_n_training_subs_${n_training_subs}/epoch=0.ckpt
            srun python /gscratch/mostafavilab/aspiro17/DNAm_and_expression/script/eval/eval_model.py --model_type ${model_type} --ckpt_path ${model_ckpt} --eval_on_ref_seq 0 --train_val_test_regions ${region_split} --num_eval_regions ${num_eval_regions} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --enet_res_path ${enet_res_path} --device ${device} --train_val_test_subs ${train_val_test_subs} --hg38_file_path ${hg38_file_path} --vcf_path ${vcf_path} --sub_data_dir ${sub_data_dir} --region_idx_start ${region_idx_start}
        done 
    done 
done 