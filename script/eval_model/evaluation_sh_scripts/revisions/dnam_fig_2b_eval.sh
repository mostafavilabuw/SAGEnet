metadata_path=/data/aspiro17/DNAm_and_expression/data/ROSMAP/DNAm/dnam_meta_hg38_incl_non_cg.csv
enet_res_path=/data/aspiro17/DNAm_and_expression/enet_res/dnam/summarized_res/rosmap/input_len_10000/maf_filter_0.01/pearson_corrs.csv
input_len=10000
new_chr_split=0
num_eval_regions=5000
device=2
model_type=psagenet

# B 

# eval train regions + val regions, val individuals 
#region_split=train
train_val_test_subs=test

easier_set_psagenet_model_ckpt_dir=/data/aspiro17/DNAm_and_expression/psagenet/dnam/rosmap/dnam_version_of_fig_2/panel_a_train_mult_epochs/epoch=0.ckpt
harder_set_psagenet_model_ckpt_dir=/data/aspiro17/DNAm_and_expression/psagenet/dnam/rosmap/dnam_version_of_fig_2/panel_b_harder_set/epoch=0.ckpt

#for region_split in train test; do 
#    #for maf_min in -1 0.05 0.1 0.2 0.3 0.4; do  
#    for maf_min in 0.4; do  
#        for model_ckpt in ${easier_set_psagenet_model_ckpt_dir} ${harder_set_psagenet_model_ckpt_dir}; do 
#            python /homes/gws/aspiro17/DNAm_and_expression/script/eval/eval_model.py --model_type ${model_type} --ckpt_path ${model_ckpt} --eval_on_ref_seq 0 --train_val_test_regions ${region_split} --num_eval_regions ${num_eval_regions} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --enet_res_path ${enet_res_path} --device ${device} --train_val_test_subs ${train_val_test_subs} --maf_min ${maf_min}
#        done 
#    done 
#done

# run for harder set, change region_idx_start 
region_idx_start=20000
model_ckpt=${harder_set_psagenet_model_ckpt_dir}
for region_split in train test; do 
    for maf_min in -1 0.05 0.1 0.2 0.3 0.4; do  
        python /homes/gws/aspiro17/DNAm_and_expression/script/eval/eval_model.py --model_type ${model_type} --ckpt_path ${model_ckpt} --eval_on_ref_seq 0 --train_val_test_regions ${region_split} --num_eval_regions ${num_eval_regions} --new_chr_split ${new_chr_split} --metadata_path ${metadata_path} --enet_res_path ${enet_res_path} --device ${device} --train_val_test_subs ${train_val_test_subs} --maf_min ${maf_min} --region_idx_start ${region_idx_start}
    done 
done 
