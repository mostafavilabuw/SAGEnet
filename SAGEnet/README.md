This directory contains the SAGEnet editable package. 

## attributions.py 
save attributions (gradients or ISM) and use tangermeme functions to identify seqlets and match identified seqlest to motif database.

#### ISM example (Fig. S3-S5) 
After defining your
`<results_save_dir>,<ckpt_path>,<variant_info_path>,<hg38_file_path>,<tss_data_path>` \n 
Run:  \n 
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --ism_win_size 150 --variant_info_path <variant_info_path> --gene ENSG00000134202 --model_type psagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 


