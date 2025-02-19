import numpy as np
import pandas as pd 
import argparse
import os
import torch
import logging
import tangermeme.io
import tangermeme.seqlet
import tangermeme.annotate
import SAGEnet.tools
from SAGEnet.data import ReferenceGenomeDataset, VariantDataset
from SAGEnet.enformer import Enformer
from SAGEnet.models import pSAGEnet,rSAGEnet

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

def summarize_seqlet_annotations(results_save_dir,gene_list,motif_match_threshold=0.05,motif_database_path='/data/mostafavilab/personal_genome_expr/data/H12CORE_meme_format.meme',seqlet_threshold=None):
    """
    Given a directory containing per-gene files of seqlets and annotations, summarize these reuslts across genes. 
    
    Paramters: 
    - results_save_dir: String path to directory containing per-gene seqlet annotations. 
    - gene_list: List of genes (as strings) for which to summarize seqlet annotations. 
    - motif_match_threshold: Float threshold for match between seqlets and motifs from database. 
    - motif_database_path: String path to motif database in .meme format used to annotate seqlets. 
    - seqlet_threshold: Float threshold for seqlet identification. If None, all seqlets in annotation dataframes are used. 
    
    Returns: lists of across-gene summaries for all seqlets and all seqlets that match below motif_match_threshold to a known motif. 
    """
        
    motifs = tangermeme.io.read_meme(motif_database_path)
    motif_names = list(motifs.keys())
    
    num_seqlets = 0
    signif_match_num_seqlets = 0
    starts = []
    signif_match_starts = []
    motifs = []
    signif_match_motifs = []
    per_gene_matches= []

    for gene in gene_list: 
        res = pd.read_csv(f'{results_save_dir}{gene}.csv',index_col=0)
        if len(res)>0: 
            if seqlet_threshold is not None: 
                res=res[res['p-value']<seqlet_threshold]
            res['corrected_pvals'] = res['match_rank_0_pval'] * len(res) * len(motif_names) # Bonferroni correction 
            signif_res = res[res['corrected_pvals']<motif_match_threshold]
            num_seqlets+=len(res)
            signif_match_num_seqlets+=len(signif_res)
            per_gene_matches.append(len(signif_res))
            starts.append(res['start'])
            signif_match_starts.append(signif_res['start'])
            motifs.append([item.split('.')[0] for item in res['match_rank_0']])
            signif_match_motifs.append([item.split('.')[0] for item in signif_res['match_rank_0']])            
        else: 
            per_gene_matches.append(0)

    return [num_seqlets, signif_match_num_seqlets, np.concatenate(starts),np.concatenate(signif_match_starts), np.concatenate(motifs),np.concatenate(signif_match_motifs),per_gene_matches]


def save_ref_seq_gradients(ckpt_path, results_save_dir, num_genes, tss_data_path, hg38_file_path,device,input_len,model_type,predixcan_res_path,best_ckpt_metric,max_epochs,identify_best_ckpt,rand_genes,top_genes_to_consider,seed,allow_reverse_complement,gene_idx_start):
    """
    Save gradients for model (pSAGEnet or rSAGEnet) evaluated on reference sequence. 
    If model is pSAGEnet, save gradients for (personal seq input, model idx 1 output) and (reference seq input, model 0 output). 
    
    Parameters: 
    - ckpt_path: String path to either model ckpt to evaluate (if identify_best_ckpt==False) or directory containing model ckpt (if identify_best_ckpt==True). 
    - results_save_dir: String path to directory in which to save gradients. 
    - num_genes: Integer number of genes to evaluate. 
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - hg38_file_path: String path to the human genome (hg38) reference file.
    - device: Integer, GPU index. 
    - input_len: Integer, size of the genomic window for model input. 
    - model_type: String type of model to evaluate, from {'psagenet', 'rsagenet'}. 
    - predixcan_res_path: String path to predixcan results DataFrame, to be used to construct ranked gene sets. Must contain the column "val_pearson". 
    - best_ckpt_metric: Metric used to select best model from ckpt dir, if identify_best_ckpt==True. Can be one of {'train_gene_gene', 'train_gene_sample', 'val_gene_gene', 'val_gene_sample'}. 
    - rand_genes: Boolean indicating whether or not to randomly select genes (from top_genes_to_consider gene set) to use in model evaluation. If False, select gene set from top-prediXcan ranked genes. 
    - top_genes_to_consider: Integer, length of prediXcan-ranked top gene set to consider when randomly selecting genes (only relevant if rand_genes==True). 
    - seed: Integer seed to determine random shuffling of gene set. 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
    - gene_idx_start: Integer index in prediXcan-ranked gene list of first gene to use in model evaluation.
    - max_epochs: Integer, maximum number of epochs to consider when selecting best model ckpt. Only relevant if identify_best_ckpt==True. 
    - identify_best_ckpt: Boolean, whether or not to use best_ckpt_metric to select best model ckpt within ckpt_path. If False, ckpt_path is used as best model ckpt path. 
    
    Saves: Numpy array of (zero-centered) gradients of shape (num_genes, 4, input_len). 
    """
    # load model 
    model_type=model_type.lower()
    
    # identify best ckpt from directory based on metric provided 
    if identify_best_ckpt:
        ckpt_path = SAGEnet.tools.select_ckpt_path(ckpt_path,max_epochs=max_epochs,best_ckpt_metric=best_ckpt_metric)
    
    if model_type=='rsagenet':
        print('rsagenet model')
        model = rSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path,using_personal_dataset=False,predict_from_personal=False)
    elif model_type=='psagenet':
        print('psagenet model')
        model = pSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path)
    model=model.to(device).eval()

    # name results_save_dir 
    if results_save_dir=='':
        results_save_dir = os.path.dirname(ckpt_path) # by default, save results in the directory containing model ckpt 
    results_save_dir=f'{results_save_dir}/{model_type}_model/gradients/'
    os.makedirs(results_save_dir, exist_ok=True)
    
    # select gene set 
    gene_list = SAGEnet.tools.select_gene_set(predixcan_res_path=predixcan_res_path, rand_genes=rand_genes, top_genes_to_consider=top_genes_to_consider,seed=seed, num_genes=num_genes,gene_idx_start=gene_idx_start) 
    print(f"n genes={len(gene_list)}")
    np.save(results_save_dir+'gene_list',gene_list)
    
    gene_meta_info = pd.read_csv(tss_data_path, sep="\t")
    selected_genes_meta = gene_meta_info.set_index('ensg', drop=False).loc[gene_list]

    # initialize arrays for gradient 
    if model_type=='psagenet': 
        ref_seq_0_idx_attribs = np.zeros((len(gene_list),4,input_len))
        personal_seq_1_idx_attribs = np.zeros((len(gene_list),4,input_len))
        single_seq=False
    
    elif model_type=='rsagenet':
        attribs = np.zeros((len(gene_list),4,input_len))
        single_seq=True

    dataset = ReferenceGenomeDataset(gene_metadata=selected_genes_meta,hg38_file_path=hg38_file_path,allow_reverse_complement=allow_reverse_complement, input_len=input_len,single_seq=single_seq)

    for i, data in enumerate(dataset):
        x, *others=data
        print(i)
        x = x.unsqueeze(0).to(device)
        x.requires_grad_()
        model_output = model(x)[0]
        
        if model_type=='rsagenet': 
            model_output.backward()
            attribs[i,:,:] =x.grad.clone().cpu().numpy()
            x.grad.zero_()

        elif model_type=='psagenet': 
            # ref seq, 0 idx output 
            model_output[0].backward(retain_graph=True)
            ref_seq_0_idx_attribs[i,:,:] = x.grad.clone().cpu().numpy()[:,0,:4,:]
            x.grad.zero_()

            # personal seq, 1 idx output 
            model_output[1].backward()
            personal_seq_1_idx_attribs[i,:,:] = x.grad.clone().cpu().numpy()[:,1,:4,:]
            x.grad.zero_()

    print('saving centered grads.npy')
    if model_type == 'psagenet': 
        ref_seq_0_idx_attribs = SAGEnet.tools.zero_center_attributions(ref_seq_0_idx_attribs)
        personal_seq_1_idx_attribs = SAGEnet.tools.zero_center_attributions(personal_seq_1_idx_attribs)
        np.save(f'{results_save_dir}ref_seq_0_idx_grads',ref_seq_0_idx_attribs)
        np.save(f'{results_save_dir}personal_seq_1_idx_grads',personal_seq_1_idx_attribs)

    elif model_type == 'rsagenet': 
        attribs = SAGEnet.tools.zero_center_attributions(attribs)
        np.save(f'{results_save_dir}grads',attribs)
    
    
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

    gene_meta_info = pd.read_csv(tss_data_path, sep="\t")
    selected_genes_meta = gene_meta_info.set_index('ensg', drop=False).loc[[gene]]
    if variant_info_path is None: 
        print('no variant info provided, ISM on reference sequence')
        dataset = ReferenceGenomeDataset(gene_metadata=selected_genes_meta,hg38_file_path=hg38_file_path,allow_reverse_complement=allow_reverse_complement,input_len=input_len,single_seq=single_seq)
        insert_variant=0
    else: 
        print('variant info provided, ISM on reference sequence with variant inserted')
        variant_info = pd.read_csv(variant_info_path, sep="\t")
        if variant_info.iloc[0]['gene']!=gene: 
            raise ValueError(f"gene in variant info ({variant_info.iloc[0]['gene']}) does not match gene provided ({gene})")
        dataset = VariantDataset(gene_metadata=selected_genes_meta,hg38_file_path=hg38_file_path,variant_info=variant_info, allow_reverse_complement=allow_reverse_complement, input_len=input_len,single_seq=single_seq,insert_variants=True)
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


def get_annotated_seqlets(arr,seq,additional_flanks=2,threshold=.05,motif_database_path="/data/mostafavilab/personal_genome_expr/data/H12CORE_meme_format.meme",n_nearest=5):
    """
    Given an attribution array (zero-centered) and sequence for that array, identify seqlets (and annotate seqlets if motif database is provided). 
    
    Paramters: 
    - arr: Numpy array of attributions (zero-centered), shape (4, input_len). 
    - seq: Numpy array of sequence to multiply by attributions , shape (4, input_len). 
    - additional_flanks: Integer additional_flanks input to tangermeme.seqlet.recursive_seqlets. 
    - threshold: Float pvalue threshold input to tangermeme.seqlet.recursive_seqlets. 
    - motif_database_path: String path to motif database in .meme format to use to annotate seqlets.
    - n_nearest: Integer input to tangermeme.annotate.annotate_seqlets specifying number of top annotations to provide. 
    
    Returns: Dataframe of seqlets and annotations, or empty DataFrame if error occurs. 
    """
    try: 
        arr=np.sum(arr*seq,axis=0) # get attributions at reference sequence 
        motifs = tangermeme.io.read_meme(motif_database_path)
        motif_names = list(motifs.keys())
        seqlets = tangermeme.seqlet.recursive_seqlets(arr[np.newaxis, :], additional_flanks=additional_flanks,threshold=threshold)

        # annotate seqlets
        if motif_database_path: 
            motif_idxs, motif_pvalues = tangermeme.annotate.annotate_seqlets(seq[np.newaxis,:], seqlets, motif_database_path,n_nearest=n_nearest)
            for match_rank in range(n_nearest): # create df based on annotated seqlets 
                curr_matches = []
                curr_match_pvals = []
                for i in range(len(seqlets)): 
                    match_idx = motif_idxs[i,match_rank]
                    curr_match_pvals.append(float(motif_pvalues[i,match_rank]))
                    curr_matches.append(motif_names[match_idx])
                seqlets[f'match_rank_{match_rank}'] = curr_matches
                seqlets[f'match_rank_{match_rank}_pval'] = curr_match_pvals
        return seqlets 

    except Exception as e:
        print(f"error in get_annotated_seqlets: {e}")
        return pd.DataFrame()


def mult_gene_save_annotated_seqlets(attrib_path,gene_list_path, hg38_file_path,tss_data_path,input_len,additional_flanks,motif_database_path,n_nearest,threshold,allow_reverse_complement):
    """
    Run get_annotated_seqlets for genes in a specified list, given their corresponding attributions. 
    Assumes that a file called gene_list exists in the same directory as attrib_path and specifies the gene order of attributions in attrib_path. 
    
    Paramters: 
    - attrib_path: String path to numpy array containing attributions of shape (num_genes, 4, input_len). 
    - gene_list_path: String path to gene list for which to save annotated seqlets. 
    - hg38_file_path: String path to the human genome (hg38) reference file.
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - input_len: Integer, size of the genomic window model input. 
    - additional_flanks: Integer additional_flanks input to tangermeme.seqlet.recursive_seqlets. 
    - motif_database_path: String path to motif database in .meme format to use to annotate seqlets. 
    - n_nearest: Integer input to tangermeme.annotate.annotate_seqlets specifying number of top annotations to provide. 
    - threshold: Float pvalue threshold input to tangermeme.seqlet.recursive_seqlets. 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand. Should match what was used to save attributions. 

    Saves: seqlet and annotation Dataframe per gene within a directory created inside of the same directory as attrib_path. 
    """
    gene_meta_info = pd.read_csv(tss_data_path, sep="\t")

    attribs_save_dir = f'{os.path.dirname(attrib_path)}/'
    attribs = np.load(attrib_path)
    attribs_label = attrib_path.split('/')[-1].split('.')[0] # label for directory in which to save annotations -- filename of attributions 

    attribs_gene_list = np.load(f'{attribs_save_dir}gene_list.npy',allow_pickle=True)
    use_gene_list = np.load(gene_list_path,allow_pickle=True)
    use_gene_list_idxs = [np.where(attribs_gene_list == value)[0][0] for value in use_gene_list]
    
    attrib_analysis_save_dir = f'{attribs_save_dir}{attribs_label}_seqlet_analysis/additional_flanks={additional_flanks}/'
    os.makedirs(attrib_analysis_save_dir, exist_ok=True)
    
    for use_gene_idx in use_gene_list_idxs: 
        gene = attribs_gene_list[use_gene_idx]
        print(gene)
        attrib = attribs[use_gene_idx,:]
        selected_genes_meta = gene_meta_info.set_index('ensg', drop=False).loc[[gene]]
        ref_dataset = ReferenceGenomeDataset(gene_metadata=selected_genes_meta,hg38_file_path=hg38_file_path,allow_reverse_complement=allow_reverse_complement,input_len=input_len,single_seq=True)
        ref_seq = ref_dataset[0][0].numpy()
        annotated_seqlets =  get_annotated_seqlets(attrib,ref_seq,additional_flanks=additional_flanks,threshold=threshold,motif_database_path=motif_database_path,n_nearest=n_nearest)
        annotated_seqlets.to_csv(f'{attrib_analysis_save_dir}{gene}.csv')
              

if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path")
    parser.add_argument("--results_save_dir")
    parser.add_argument("--model_type")
    parser.add_argument("--gene")
    parser.add_argument("--variant_info_path")
    parser.add_argument("--attrib_path")
    parser.add_argument("--gene_list_path")
    parser.add_argument("--ism_center_genome_pos",type=int)
    
    parser.add_argument("--ism_win_size",default=150,type=int)
    parser.add_argument("--num_genes",default=1000,type=int)
    parser.add_argument('--device', default=0,type=int)
    parser.add_argument('--input_len', type=int, default=40000)
    parser.add_argument("--best_ckpt_metric",default='train_gene_gene',type=str)
    parser.add_argument("--max_epochs",default=10,type=int)
    parser.add_argument("--identify_best_ckpt",default=1,type=int)
    parser.add_argument("--rand_genes",type=int,default=0)
    parser.add_argument("--top_genes_to_consider",type=int,default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--allow_reverse_complement",type=int,default=1)
    parser.add_argument("--gene_idx_start",type=int,default=0)
    parser.add_argument('--additional_flanks', type=int, default=2)
    parser.add_argument('--n_nearest', type=int, default=5)
    parser.add_argument('--threshold', type=int, default=.05)
    
    parser.add_argument('--hg38_file_path', default='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa')
    parser.add_argument('--predixcan_res_path', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/results_data/predixcan/rosmap_pearson_corr.csv')
    parser.add_argument('--tss_data_path', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/gene-ids-and-positions.tsv')
    parser.add_argument('--finetuned_weights_dir',default='/data/mostafavilab/personal_genome_expr/final_results/enformer/ref_seq_all_tracks/')
    parser.add_argument('--motif_database_path', default='/data/mostafavilab/personal_genome_expr/data/H12CORE_meme_format.meme')

    parser.add_argument("--which_fn")

    args = parser.parse_args()
        
    if args.which_fn == 'mult_gene_save_annotated_seqlets': 
        mult_gene_save_annotated_seqlets(
        attrib_path=args.attrib_path,
        gene_list_path=args.gene_list_path,
        hg38_file_path=args.hg38_file_path,
        tss_data_path=args.tss_data_path,
        input_len=args.input_len,
        additional_flanks=args.additional_flanks,
        motif_database_path=args.motif_database_path,
        n_nearest=args.n_nearest,
        threshold=args.threshold,
        allow_reverse_complement=args.allow_reverse_complement
    )
        
    if args.which_fn == 'save_ref_seq_gradients': 
        save_ref_seq_gradients(
        ckpt_path=args.ckpt_path,
        results_save_dir=args.results_save_dir,
        num_genes=args.num_genes,
        tss_data_path=args.tss_data_path,
        hg38_file_path=args.hg38_file_path,
        device=args.device,
        input_len=args.input_len,
        model_type=args.model_type,
        predixcan_res_path=args.predixcan_res_path,
        best_ckpt_metric=args.best_ckpt_metric,
        max_epochs=args.max_epochs,
        identify_best_ckpt=args.identify_best_ckpt,
        rand_genes=args.rand_genes,
        top_genes_to_consider=args.top_genes_to_consider,
        seed=args.seed,
        allow_reverse_complement=args.allow_reverse_complement,
        gene_idx_start=args.gene_idx_start
    )
        
    if args.which_fn == 'save_gene_ism': 
        save_gene_ism(
        gene=args.gene,
        results_save_dir=args.results_save_dir,
        ckpt_path=args.ckpt_path,
        ism_center_genome_pos=args.ism_center_genome_pos,
        ism_win_size=args.ism_win_size,
        hg38_file_path=args.hg38_file_path,
        tss_data_path=args.tss_data_path,
        input_len=args.input_len,
        device=args.device,
        model_type=args.model_type,
        allow_reverse_complement=args.allow_reverse_complement,
        finetuned_weights_dir=args.finetuned_weights_dir,
        variant_info_path=args.variant_info_path
    )