## analyze_attribs.py
- Analyze model attributions:
  - Perform a global motif analyis
  - Do ISM on a given region with specified variant inserted
 
#### Global motif analysis (Fig. 3c)
First, we save attributions and identify seqlets from the two models we are analyzing (r-SAGE-net and p-SAGE-net). 

After defining your paths: 
<rsagenet_model_ckpt_path>,<psagenet_model_ckpt_path>

Run the function `get_ref_seqlets` for each model:
`python analyze_attribs.py --ckpt_path <psagenet_model_ckpt_path> --model_type psagenet --num_eval_regions 5000 --attrib_type grad --train_val_test_regions test --which_fn get_ref_seqlets`
`python analyze_attribs.py --ckpt_path <rsagenet_model_ckpt_path> --model_type rsagenet --num_eval_regions 5000 --attrib_type grad --train_val_test_regions test --which_fn get_ref_seqlets`

Next, we jointly cluser the seqlets identified by each model, and match these seqlets to our motif database. 

Define these paths based on where the results from the previous step are saved (results will be saved in the same directory as each model checkpoint): 
`<rsagenet_attrib_save_dir>,<psagenet_attrib_save_dir>`

And define a path to save the clustering and annotation results: 
`<combined_clustering_annotation_save_dir>`

Then run:
`python analyze_attribs.py --num_eval_regions 5000 --attrib_type grad --train_val_test_regions test --device 0 --which_fn cluster_and_annotate  --results_save_dir_a <rsagenet_attrib_save_dir> --a_label rsagenet --results_save_dir_b <psagenet_attrib_save_dir> --b_label psagenet --results_save_dir <combined_clustering_annotation_save_dir>`

To save the clustering and annotation results to that directory. 
To see how to go from the results saved in this step to the final Fig. 3c, see the section "DNAm" in `plot_figs/plot_attribution_figs.ipynb`. 

#### ISM example (Fig. S4-S6) 
After defining your paths: 
`<results_save_dir>,<ckpt_path>,<variant_info_path>,<hg38_file_path>,<tss_data_path>,<finetuned_weights_dir>`  

Run:

p-SAGE-net, insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type psagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --variant_info_path <variant_info_path>`   

p-SAGE-net, do not insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type psagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 

r-SAGE-net, insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type rsagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --variant_info_path <variant_info_path>`   

r-SAGE-net, do not insert variant:       
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type rsagenet --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>` 

Enformer, insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type enformer --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --variant_info_path <variant_info_path> --finetuned_weights_dir <finetuned_weights_dir>`   

Enformer, do not insert variant:     
`$python attributions.py --which_fn save_gene_ism --results_save_dir <results_save_dir> --ckpt_path <ckpt_path> --ism_center_genome_pos 109731286 --gene ENSG00000134202 --model_type enformer --hg38_file_path <hg38_file_path> --tss_data_path <tss_data_path>  --finetuned_weights_dir <finetuned_weights_dir>` 
