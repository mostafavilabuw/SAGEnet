import numpy as np 
import pandas as pd 
import argparse
import os
import time
import SAGEnet.attributions
from SAGEnet.models import rSAGEnet,pSAGEnet
import SAGEnet.tools
from SAGEnet.data import ReferenceGenomeDataset, VariantDataset

def save_gene_ism(gene, results_save_dir, ckpt_path, ism_center_genome_pos, ism_win_size,hg38_file_path,tss_data_path,input_len,device,model_type,allow_reverse_complement,finetuned_weights_dir,variant_info_path,enformer_input_len=393216):
    """
    Given a gene (with optional variant inserted), perform ISM by mutating each base within a specified window around a specified center position. 
    
    Parameters: 
    - gene: String gene ENSG id. 
    - results_save_dir: String path to directory in which to save ISM res.  
    - ckpt_path: String ckpt path to model to use for ISM. Only used if model_type!=enformer. 
    - ism_center_genome_pos: Integer center position (coordinate in genome) around which to perform ISM. 
    - ism_win_size: Integer window size in which to do ISM, centered on ism_center_genome_pos. 
    - hg38_file_path: String path to the human genome (hg38) reference file.
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - input_len: Integer, size of the genomic window for model input. 
    - device: Integer, GPU index. 
    - model_type: String type of model to evaluate, from {'psagenet, rsagenet, enformer'}. 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand. 
    - finetuned_weights_dir: String of directory containing 'coef.npy' and 'intercept.npy' used to finetune Enformer predictions. 
    - variant_info_path: String path to DataFrame containing variant information, specifially the columns 'gene', 'chr', 'pos', and 'alt'. If None, no variant is inserted. 
    - enformer_input_len: Integer input length to use for dataset to Enformer model. 
    
    Saves: Numpy array of attributions (zero-centered), shape (4, input_len). 
    """
    model_type=model_type.lower()
    if model_type=='enformer':
        input_len=enformer_input_len

    # determine ISM start and stop idxs in sequence 
    start_of_interest = ism_center_genome_pos-ism_win_size//2
    end_of_interest = ism_center_genome_pos+ism_win_size//2
    start_ism_idx = SAGEnet.tools.get_pos_idx_in_seq(gene, start_of_interest,tss_data_path,input_len,allow_reverse_complement=allow_reverse_complement)
    end_ism_idx = SAGEnet.tools.get_pos_idx_in_seq(gene, end_of_interest,tss_data_path,input_len,allow_reverse_complement=allow_reverse_complement)
    start_ism_idx=min(start_ism_idx,end_ism_idx)
    end_ism_idx=start_ism_idx+ism_win_size

    # name results_save_dir 
    if results_save_dir=='':
        results_save_dir = os.path.dirname(ckpt_path) # by default, save results in the directory containing model ckpt 
    results_save_dir=f'{results_save_dir}/{model_type}_model/ism/'
    os.makedirs(results_save_dir, exist_ok=True)

    # load model 
    if model_type=='rsagenet':
        print('rsagenet model')
        model = rSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path,using_personal_dataset=False,predict_from_personal=False)
        single_seq=True
    elif model_type=='psagenet':
        print('psagenet model')
        model = pSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path)
        single_seq=False
    elif model_type=='enformer':
        os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
        model = Enformer(finetuned_weights_dir=finetuned_weights_dir)
        single_seq=True
    if model_type!='enformer':
        model=model.to(device).eval()

    gene_meta_info = pd.read_csv(tss_data_path, sep="\t",index_col='region_id')
    selected_genes_meta = gene_meta_info.loc[[gene]]
    if variant_info_path is None: 
        print('no variant info provided, ISM on reference sequence')
        dataset = ReferenceGenomeDataset(metadata=selected_genes_meta,hg38_file_path=hg38_file_path,allow_reverse_complement=allow_reverse_complement,input_len=input_len,single_seq=single_seq)
        insert_variant=0
    else: 
        print('variant info provided, ISM on reference sequence with variant inserted')
        variant_info = pd.read_csv(variant_info_path, sep="\t")
        if variant_info.iloc[0]['gene']!=gene: 
            raise ValueError(f"gene in variant info ({variant_info.iloc[0]['gene']}) does not match gene provided ({gene})")
        dataset = VariantDataset(metadata=selected_genes_meta,hg38_file_path=hg38_file_path,variant_info=variant_info, allow_reverse_complement=allow_reverse_complement, input_len=input_len,single_seq=single_seq,insert_variants=True)
        insert_variant=1
    x = dataset[0][0].unsqueeze(0).numpy()
    
    if model_type == 'psagenet': 
        ism_res = np.zeros((2,4,ism_win_size,2)) # the 0th dim is for ref or personal, 1st is nucleotide, 2nd is ISM window size,  3rd is for first model output or 2nd model output 
        for ref_or_personal_idx in [0,1]: # which sequence we are mutating 
            for len_idx in range(ism_win_size):
                pos_idx = len_idx+start_ism_idx
                for nuc_idx in range(4): 
                    mutated_seq = x.copy()
                    mutated_seq[0,ref_or_personal_idx,:4,pos_idx] = np.zeros(4)
                    mutated_seq[0,ref_or_personal_idx,nuc_idx,pos_idx]=1
                    mutated_seq=torch.from_numpy(mutated_seq).to(device)
                    ism_res[ref_or_personal_idx,nuc_idx,len_idx,:] = model(mutated_seq)[0].detach().cpu().numpy()
    else: # rSAGEnet or enformer 
        ism_res = np.zeros((4,ism_win_size)) 
        for len_idx in range(ism_win_size):
            print(f'len_idx:{len_idx}')
            pos_idx = len_idx+start_ism_idx
            for nuc_idx in range(4): 
                mutated_seq = x.copy()
                mutated_seq[0,:,pos_idx] = np.zeros(4)
                mutated_seq[0,nuc_idx,pos_idx]=1
                mutated_seq=torch.from_numpy(mutated_seq)
                if model_type=='enformer': 
                    ism_res[nuc_idx,len_idx] = model.predict_on_batch(mutated_seq,save_mode='finetuned')
                else: 
                    mutated_seq=mutated_seq.to(device)
                    ism_res[nuc_idx,len_idx] = model(mutated_seq)[0].detach().cpu().numpy() 
                
    ism_res = SAGEnet.tools.zero_center_attributions(ism_res)
    if insert_variant:         
        np.save(f"{results_save_dir}{gene}_{start_of_interest}_to_{end_of_interest}_pos_{variant_info.iloc[0]['pos']}_{variant_info.iloc[0]['ref']}_to_{variant_info.iloc[0]['alt']}", ism_res)
    else: 
        np.save(f'{results_save_dir}{gene}_{start_of_interest}_to_{end_of_interest}',ism_res)

def get_dataset(metadata_path, eval_res_path, num_eval_regions, train_val_test_regions,enet_res_path, hg38_file_path, input_len,allow_reverse_complement, single_seq):
    """
    Create a ReferenceGenomeDataset, either using regions ranked by model performance (when eval_res_path is not '') or regions ranked by prediXcan performance.   
    """
    metadata = pd.read_csv(metadata_path, sep='\t', index_col='region_id')

    if len(eval_res_path)>0: 
        eval_res = pd.read_csv(eval_res_path,index_col=0)
        eval_res = eval_res.sort_values(by='pearson', ascending=False)
        region_list=eval_res.index[:num_eval_regions]
        print(f'analyzing attributions for regions from the performance-ranked top {num_eval_regions} regions')

    else: 
        region_list=SAGEnet.tools.select_region_set(enet_path=enet_res_path, ran_regions=0, num_regions=num_eval_regions,metadata=metadata)
        print(f'analyzing attributions for regions from the e-net ranked top {num_eval_regions} regions')
    
    train_regions, val_regions, test_regions = SAGEnet.tools.get_train_val_test_regions(region_list,metadata)
    if train_val_test_regions=='train':
        regions=train_regions
    elif train_val_test_regions=='val':
        regions=val_regions
    elif train_val_test_regions=='test':
        regions=test_regions
    print(f'analyzing attributions for {len(regions)} {train_val_test_regions} regions')

    selected_regions_meta=metadata[metadata.index.isin(regions)]
    
    dataset = ReferenceGenomeDataset(metadata=selected_regions_meta, hg38_file_path=hg38_file_path,input_len=input_len,allow_reverse_complement=allow_reverse_complement,single_seq=single_seq)
    return dataset


def get_ref_seqlets(ckpt_path, results_save_dir, eval_res_path, train_val_test_regions, metadata_path, attrib_type, hg38_file_path,input_len,model_type,allow_reverse_complement,best_ckpt_metric,max_epochs,additional_flanks,identify_seqlet_threshold,device,enet_res_path,num_eval_regions,save_dir_label):
    """
    Given a model checkpoint, use save_ref_attribs and identify_seqlets to save attributions (from reference sequence) and identify seqlets from these attributions. 
    If the model type is pSAGEnet, attributions will be saved seperately from the 0 idx and 1 idx model outputs. 
    Attributions and seqlet information will be saved per-region. 
    """
    model_type=model_type.lower()

    if os.path.isdir(ckpt_path):
        ckpt_path=SAGEnet.tools.select_ckpt_path(ckpt_path,best_ckpt_metric=best_ckpt_metric,max_epochs=max_epochs)
    ckpt_label = ckpt_path.split('/')[-1]
    print(f'ckpt_label:{ckpt_label}')

    # name results_save_dir 
    if results_save_dir=='':
        results_save_dir = os.path.dirname(ckpt_path) # by default, save results in the directory containing model ckpt 

    if save_dir_label!='':
        results_save_dir=f'{results_save_dir}/attribs/{save_dir_label}/{ckpt_label}/{attrib_type}/'
    else:
        results_save_dir=f'{results_save_dir}/attribs/{ckpt_label}/{attrib_type}/'
    os.makedirs(results_save_dir, exist_ok=True)

    if model_type=='rsagenet':
        print('rsagenet model')
        model = rSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path,using_personal_dataset=False,predict_from_personal=False,map_location=f'cuda:{device}')
        single_seq=True
    elif model_type=='psagenet':
        model = pSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path,map_location=f'cuda:{device}')
        single_seq=False
    model.eval()

    if attrib_type=='grad' or attrib_type=='ism':
        zero_center_attribs=1
        print('zero centering attribs')
    else: 
        zero_center_attribs=0
        print('not zero centering attribs')

    dataset = get_dataset(
        metadata_path=metadata_path,
        eval_res_path=eval_res_path,
        num_eval_regions=num_eval_regions,
        train_val_test_regions=train_val_test_regions,
        enet_res_path=enet_res_path,
        hg38_file_path=hg38_file_path,
        input_len=input_len,
        allow_reverse_complement=allow_reverse_complement,
        single_seq=single_seq
    )
        
    single_ref_dataset = get_dataset(
        metadata_path=metadata_path,
        eval_res_path=eval_res_path,
        num_eval_regions=num_eval_regions,
        train_val_test_regions=train_val_test_regions,
        enet_res_path=enet_res_path,
        hg38_file_path=hg38_file_path,
        input_len=input_len,
        allow_reverse_complement=allow_reverse_complement,
        single_seq=True
    )

    # save attributions (per region)
    print('saving attributions')
    start = time.time()
    SAGEnet.attributions.save_ref_attribs(model, dataset,model_type=model_type,results_save_dir=results_save_dir,attrib_type=attrib_type)
    end = time.time()
    print(f'save attributions time:{end-start}')
 
    # save seqlets (per region)
    print('identifying seqlets')
    start = time.time()
    if model_type=='psagenet':
        SAGEnet.attributions.identify_seqlets(single_ref_dataset,results_save_dir=f'{results_save_dir}output_idx_0/',zero_center_attribs=zero_center_attribs,additional_flanks=additional_flanks,threshold=identify_seqlet_threshold)
        SAGEnet.attributions.identify_seqlets(single_ref_dataset,results_save_dir=f'{results_save_dir}output_idx_1/',zero_center_attribs=zero_center_attribs,additional_flanks=additional_flanks,threshold=identify_seqlet_threshold)
    else: 
        SAGEnet.attributions.identify_seqlets(single_ref_dataset,results_save_dir=results_save_dir,zero_center_attribs=zero_center_attribs,additional_flanks=additional_flanks,threshold=identify_seqlet_threshold)
    end = time.time()
    print(f'save seqlets time:{end-start}')


def cluster_and_annotate(results_save_dir,results_save_dir_a,results_save_dir_b, a_label, b_label,metadata_path,eval_res_path,num_eval_regions,train_val_test_regions,enet_res_path,hg38_file_path,input_len,allow_reverse_complement,model_type,cluster_seqlet_threshold,batch_size,database_path,n_top_motif_matches,linkage,distance_threshold,cluster_metric,zero_center_attribs_pre_cluster,device):
    """
    Uses cluster_seqlets and annotate_seqlets to cluster and annotate seqlets from previously-saved attributions.  
    """
    single_ref_dataset = get_dataset(
        metadata_path=metadata_path,
        eval_res_path=eval_res_path,
        num_eval_regions=num_eval_regions,
        train_val_test_regions=train_val_test_regions,
        enet_res_path=enet_res_path,
        hg38_file_path=hg38_file_path,
        input_len=input_len,
        allow_reverse_complement=allow_reverse_complement,
        single_seq=True
    )

    print('clustering seqlets')
    start = time.time()
    SAGEnet.attributions.cluster_seqlets(dataset=single_ref_dataset,comb_results_save_dir=results_save_dir,results_save_dir_a=results_save_dir_a,results_save_dir_b=results_save_dir_b,a_label=a_label,b_label=b_label,zero_center_attribs=zero_center_attribs_pre_cluster,cluster_seqlet_threshold=cluster_seqlet_threshold,device=device,batch_size=batch_size,linkage=linkage,distance_threshold=distance_threshold,cluster_metric=cluster_metric)
    end = time.time()
    print(f'cluster seqlets:{end-start}')
   
    print('matching clusters to motif database')
    start = time.time()
    SAGEnet.attributions.annotate_seqlets(cluster_seqlet_dir=results_save_dir,database_path=database_path,n_top_motif_matches=n_top_motif_matches)
    end = time.time()
    print(f'annotate seqlets:{end-start}')
    

if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_val_test_regions", type=str, default='test')
    parser.add_argument("--attrib_type", default='grad')
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--input_len", default=10000, type=int)
    parser.add_argument("--model_type", default='rsagenet')
    parser.add_argument("--allow_reverse_complement", type=int, default=1)
    parser.add_argument("--best_ckpt_metric", default='val_region_region')
    parser.add_argument("--num_eval_regions", default=5000, type=int)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--zero_center_attribs_pre_cluster", default=0, type=int)
    parser.add_argument("--additional_flanks", default=0, type=int)
    parser.add_argument("--identify_seqlet_threshold", default=0.05, type=float)
    parser.add_argument("--cluster_seqlet_threshold", default=0.005, type=float)
    parser.add_argument("--database_path", default='/data/mostafavilab/personal_genome_expr/data/H12CORE_meme_format.meme')
    parser.add_argument("--n_top_motif_matches", default=5, type=int)
    parser.add_argument("--linkage", default='complete')
    parser.add_argument("--distance_threshold", default=0.05, type=float)
    parser.add_argument("--cluster_metric", default='correlation_pvalue')
    parser.add_argument("--eval_res_path", default='')
    parser.add_argument("--enet_res_path", default='/data/aspiro17/DNAm_and_expression/enet_res/dnam/summarized_res/rosmap/input_len_10000/maf_filter_0.01/pearson_corrs.csv')
    parser.add_argument("--metadata_path", default='/data/aspiro17/DNAm_and_expression/data/ROSMAP_DNAm/dnam_meta_hg38.csv')
    parser.add_argument("--hg38_file_path", default='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa')
    parser.add_argument("--ckpt_path")
    parser.add_argument("--results_save_dir", default='', type=str)
    parser.add_argument("--results_save_dir_a", default='', type=str)
    parser.add_argument("--results_save_dir_b", default='', type=str)
    parser.add_argument("--a_label", default='', type=str)
    parser.add_argument("--b_label", default='', type=str)
    parser.add_argument("--save_dir_label", default='', type=str)

    # only used for save_gene_ism
    parser.add_argument("--gene")
    parser.add_argument("--ism_center_genome_pos",type=int)
    parser.add_argument("--ism_win_size",default=150,type=int)
    parser.add_argument('--finetuned_weights_dir',default='/data/mostafavilab/personal_genome_expr/final_results/enformer/ref_seq_all_tracks/')
    parser.add_argument("--variant_info_path",default=None)

    parser.add_argument("--which_fn")

    args = parser.parse_args()

    if args.which_fn=='get_ref_seqlets':
        get_ref_seqlets(
        ckpt_path=args.ckpt_path,
        results_save_dir=args.results_save_dir,
        eval_res_path=args.eval_res_path,
        train_val_test_regions=args.train_val_test_regions,
        metadata_path=args.metadata_path,
        attrib_type=args.attrib_type,
        hg38_file_path=args.hg38_file_path,
        input_len=args.input_len,
        model_type=args.model_type,
        allow_reverse_complement=args.allow_reverse_complement,
        best_ckpt_metric=args.best_ckpt_metric,
        max_epochs=args.max_epochs,
        additional_flanks=args.additional_flanks,
        identify_seqlet_threshold=args.identify_seqlet_threshold,
        device=args.device,
        enet_res_path=args.enet_res_path,
        num_eval_regions=args.num_eval_regions,
        save_dir_label=args.save_dir_label
    )
        
    if args.which_fn=='cluster_and_annotate':
        cluster_and_annotate(
        results_save_dir=args.results_save_dir,
        results_save_dir_a=args.results_save_dir_a,
        results_save_dir_b=args.results_save_dir_b,
        a_label=args.a_label,
        b_label=args.b_label,
        metadata_path=args.metadata_path,
        eval_res_path=args.eval_res_path,
        num_eval_regions=args.num_eval_regions,
        train_val_test_regions=args.train_val_test_regions,
        enet_res_path=args.enet_res_path,
        hg38_file_path=args.hg38_file_path,
        input_len=args.input_len,
        allow_reverse_complement=args.allow_reverse_complement,
        model_type=args.model_type,
        cluster_seqlet_threshold=args.cluster_seqlet_threshold,
        batch_size=args.batch_size,
        database_path=args.database_path,
        n_top_motif_matches=args.n_top_motif_matches,
        linkage=args.linkage,
        distance_threshold=args.distance_threshold,
        cluster_metric=args.cluster_metric,
        zero_center_attribs_pre_cluster=args.zero_center_attribs_pre_cluster,
        device=args.device
        )
            
    if args.which_fn == 'save_gene_ism': 
        save_gene_ism(
        gene=args.gene,
        results_save_dir=args.results_save_dir,
        ckpt_path=args.ckpt_path,
        ism_center_genome_pos=args.ism_center_genome_pos,
        ism_win_size=args.ism_win_size,
        hg38_file_path=args.hg38_file_path,
        tss_data_path=args.metadata_path,
        input_len=args.input_len,
        device=args.device,
        model_type=args.model_type,
        allow_reverse_complement=args.allow_reverse_complement,
        finetuned_weights_dir=args.finetuned_weights_dir,
        variant_info_path=args.variant_info_path
    )
    