#!/bin/bash

#SBATCH --job-name=dnam_version_of_fig_2_d
#SBATCH --output=./output/dnam_version_of_fig_2_d%j.out
#SBATCH --error=./output/dnam_version_of_fig_2_d%j.err
#SBATCH --partition=gpu-a100
#SBATCH --account=cse
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=15G
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00 
#SBATCH --exclude=g3007,g3052

source ~/.bashrc
source /gscratch/mostafavilab/aspiro17/micromamba/etc/profile.d/micromamba.sh
micromamba activate SAGEnet

n_devices=4

wandb_project=dnam_version_of_fig_2
#model_save_dir=/data/aspiro17/DNAm_and_expression/psagenet/
model_save_dir=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/revisions_pt_2/psagenet/

#ref_model_ckpt_path=/data/aspiro17/DNAm_and_expression/rsagenet/dnam/rosmap/input_len_10000/epoch=14-step=108183.ckpt
ref_model_ckpt_path=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/rsagenet/dnam/rosmap/frac_regions_1.0/24849171/epoch=14-step=108183.ckpt

#metadata_path=/data/aspiro17/DNAm_and_expression/data/ROSMAP/DNAm/dnam_meta_hg38_incl_non_cg.csv
metadata_path=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/data/ROSMAP_DNAm/dnam_meta_hg38.csv

hg38_file_path=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/Genome/hg38.fa
y_data_path="/gscratch/mostafavilab/aspiro17/DNAm_and_expression/data/ROSMAP_DNAm/methylationSNMnorm.csv"
enet_res_path="/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/enet/dnam/summarized_res/rosmap/input_len_10000/maf_filter_0.01/pearson_corrs.csv"
vcf_path="/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/ROSMAP/wholeGenomeSeq/chrAll.phased.vcf.gz"
sub_data_dir='/gscratch/mostafavilab/aspiro17/DNAm_and_expression/input_data/individual_lists/'

model_type=psagenet
input_len=10000
seed=0
new_chr_split=0
lam_diff=100
lam_ref=1
include_test_regions_test_subs_dataloader=0
num_val_regions=5000
patience=10

# D 
max_epochs=1
region_idx_start=0

# top 
#rand_regions=0
#for num_train_regions in 10000 15000 25000; do
#    wandb_job_name=chelan_panel_d_rand_regions_${rand_regions}_n_training_regions_${num_train_regions}
#    python /homes/gws/aspiro17/DNAm_and_expression/script/psagenet/train_psagenet.py --wandb_project ${wandb_project} --model_save_dir ${model_save_dir} --ref_model_ckpt_path ${ref_model_ckpt_path} --n_devices ${n_devices} --model_type ${model_type} --input_len ${input_len} --num_train_regions ${num_train_regions} --num_val_regions ${num_val_regions} --wandb_job_name ${wandb_job_name} --include_test_regions_test_subs_dataloader ${include_test_regions_test_subs_dataloader} --seed ${seed} --lam_diff ${lam_diff} --lam_ref ${lam_ref} --max_epochs ${max_epochs} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --region_idx_start ${region_idx_start} --rand_regions ${rand_regions}
#done  

# rand 
rand_regions=1
for num_train_regions in 5000 10000 15000 20000 25000; do
    wandb_job_name=hyak_panel_d_rand_regions_${rand_regions}_n_training_regions_${num_train_regions}
    python /gscratch/mostafavilab/aspiro17/DNAm_and_expression/script/psagenet/train_psagenet.py --wandb_project ${wandb_project} --model_save_dir ${model_save_dir} --ref_model_ckpt_path ${ref_model_ckpt_path} --n_devices ${n_devices} --model_type ${model_type} --input_len ${input_len} --num_train_regions ${num_train_regions} --num_val_regions ${num_val_regions} --wandb_job_name ${wandb_job_name} --include_test_regions_test_subs_dataloader ${include_test_regions_test_subs_dataloader} --seed ${seed} --lam_diff ${lam_diff} --lam_ref ${lam_ref} --max_epochs ${max_epochs} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --region_idx_start ${region_idx_start} --rand_regions ${rand_regions}  --hg38_file_path ${hg38_file_path} --y_data_path ${y_data_path} --enet_res_path ${enet_res_path} --vcf_path ${vcf_path} --sub_data_dir ${sub_data_dir} --patience ${patience}
done  
