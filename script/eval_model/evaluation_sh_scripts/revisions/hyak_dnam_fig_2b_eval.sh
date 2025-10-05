#!/bin/bash

#SBATCH --job-name=eval_dnam_fig_2b
#SBATCH --output=./output/eval_dnam_fig_2b_%j.out
#SBATCH --error=./output/eval_dnam_fig_2b_%j.err
#SBATCH --partition=gpu-a100
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
input_len=10000
new_chr_split=0
num_eval_regions=5000
model_type=psagenet
device=0

hg38_file_path=/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/Genome/hg38.fa
vcf_path="/gscratch/mostafavilab/tuxm/project/PersonalGenomeExpression-dev/data/ROSMAP/wholeGenomeSeq/chrAll.phased.vcf.gz"
sub_data_dir='/gscratch/mostafavilab/aspiro17/DNAm_and_expression/input_data/individual_lists/'

# B 

# eval train regions + val regions, val individuals 
#region_split=train
train_val_test_subs=test

easier_set_psagenet_model_ckpt_dir=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/revisions_pt_2/psagenet/dnam/rosmap/dnam_version_of_fig_2/panel_a_train_mult_epochs/epoch=0.ckpt 
harder_set_psagenet_model_ckpt_dir=/gscratch/mostafavilab/aspiro17/DNAm_and_expression/results/revisions_pt_2/psagenet/dnam/rosmap/dnam_version_of_fig_2/panel_b_harder_set/epoch=0.ckpt

for region_split in train test; do 
    for maf_min in -1 0.05 0.1 0.2 0.3 0.4; do  
        #for model_ckpt in ${easier_set_psagenet_model_ckpt_dir} ${harder_set_psagenet_model_ckpt_dir}; do 
        for model_ckpt in ${harder_set_psagenet_model_ckpt_dir}; do 
            if [[ "$model_ckpt" == "$easier_set_psagenet_model_ckpt_dir" ]]; then
                region_idx_start=0
            else
                region_idx_start=20000
            fi
            echo "running model: ${model_ckpt}, region_idx_start=${region_idx_start}, maf_min=${maf_min}"
            srun python /gscratch/mostafavilab/aspiro17/DNAm_and_expression/script/eval/eval_model.py --model_type ${model_type} --ckpt_path ${model_ckpt} --eval_on_ref_seq 0 --train_val_test_regions ${region_split} --num_eval_regions ${num_eval_regions} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --enet_res_path ${enet_res_path} --device ${device} --train_val_test_subs ${train_val_test_subs} --maf_min ${maf_min} --hg38_file_path ${hg38_file_path} --vcf_path ${vcf_path} --sub_data_dir ${sub_data_dir} --region_idx_start ${region_idx_start}
        done 
    done 
done