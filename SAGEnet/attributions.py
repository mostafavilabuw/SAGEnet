import numpy as np
import pandas as pd 
import argparse
import os
import torch
import logging
import tangermeme.io
import tangermeme.seqlet
import tangermeme.annotate
import pytorch_lightning as pl
from statsmodels.stats.multitest import multipletests
from torch.utils.data import TensorDataset, DataLoader
import drg_tools.motif_analysis
import drg_tools.io_utils
import pickle 
from sklearn.cluster import AgglomerativeClustering
from memelite import tomtom
import SAGEnet.tools

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

def cwm_to_ppm(cwm):
    """
    Converts a Contriubtion Weight Matrix (CWM) (i.e., from model attributions) to a Position Probabilty Matrix (PPM). 
    Input can be numpy array or torch Tensor, first dimension should correspond to the 4 nucleoties. 
    """
    is_torch=True
    if cwm.shape[0]!=4: 
        raise ValueError(f"cwm.shape[0] must be 4, but it is {cwm.shape[0]}")
    if not isinstance(cwm, torch.Tensor):
        is_torch=False
        cwm = torch.from_numpy(cwm)

    cwm[cwm != cwm] = 0 # deal with any nan values 
    abs_cwm = torch.abs(cwm)
    sum_cwm = torch.sum(abs_cwm, axis=0, keepdim=True)
    ppm = abs_cwm / sum_cwm

    if not is_torch:
        ppm=ppm.detach().cpu().numpy()
    return ppm

def annotate_seqlets(cluster_seqlet_dir, database_path='/data/mostafavilab/personal_genome_expr/data/H12CORE_meme_format.meme', n_top_motif_matches=5, pval_thresh=0.05):
    """
    Use tomtom to match clusters in the provided directory to a specified motif database. Transform cluster CWMs to PPMs before matching. 
    Save the top 5 most significant (BH-corrected) matches (or all significant if more than 5 are significant).
    """
    my_motifs = tangermeme.io.read_meme(f'{cluster_seqlet_dir}cluster_cwms.meme')
    my_motif_names = list(my_motifs.keys())

    for motif_name in my_motif_names:
        my_motifs[motif_name] = cwm_to_ppm(my_motifs[motif_name])
    my_motif_values = [my_motifs[name] for name in my_motif_names]

    database_motifs = tangermeme.io.read_meme(database_path)
    database_motif_names = list(database_motifs.keys())
    database_motif_values = [database_motifs[name] for name in database_motif_names]

    print(f'aligning {len(my_motif_values)} provided motifs with {len(database_motif_values)} from {database_path}')
    p, scores, offsets, overlaps, strands = tomtom(my_motif_values, database_motif_values)

    raw_p_df = pd.DataFrame(index=my_motif_names, columns=database_motif_names, data=p)
    _, pvals_corrected, _, _ = multipletests(p.flatten(), method='fdr_bh')
    pvals_corrected = pvals_corrected.reshape(p.shape[0], p.shape[1])
    bh_p_df = pd.DataFrame(index=my_motif_names, columns=database_motif_names, data=pvals_corrected)

    all_rows = []
    all_indices = []

    for i, idx in enumerate(raw_p_df.index):
        raw_row = raw_p_df.loc[idx]
        corrected_row = bh_p_df.loc[idx]

        # get top k hits
        top_hits = raw_row.nsmallest(n_top_motif_matches)

        # get all hits with bh-corrected p < threshold
        significant_hits = corrected_row[corrected_row < pval_thresh]

        # combine and deduplicate matches
        combined_matches = pd.concat([top_hits, significant_hits]).drop_duplicates()

        row_data = []
        for motif_match in combined_matches.index:
            raw_pval = raw_row[motif_match]
            bh_pval = corrected_row[motif_match]
            db_idx = database_motif_names.index(motif_match)
            strand = int(strands[i, db_idx])
            row_data.append({
                "cluster_idx": idx,
                "motif_match": motif_match,
                "raw_pval": raw_pval,
                "bh_corrected_pval": bh_pval,
                "strand": strand
            })

        for row in row_data:
            all_rows.append(row)
            all_indices.append(idx)

    result_df = pd.DataFrame(all_rows, index=all_indices).drop_duplicates()
    result_df.to_csv(f'{cluster_seqlet_dir}annotations.csv')


def cluster_seqlets(dataset,comb_results_save_dir,results_save_dir_a,results_save_dir_b,a_label,b_label,cluster_seqlet_threshold,device,batch_size,zero_center_attribs=False,linkage='average',distance_threshold=.05,cluster_metric='correlation_pvalue'):
    """
    Use torch_compute_similarity_motifs and AgglomerativeClustering to first compute a metric of similarity between the motifs (saved as CWMs)
        and then cluster using that metric. Using comb_results_save_dir,results_save_dir_a,results_save_dir_b, you can specify if you are clustering 
        motifs saved in a single directory (i.e. for single model analysis) or two directories (i.e. combined model analysis).
    Saves cluster assignments as numpy array, cluster CWMs as meme. 
    """
    if comb_results_save_dir=='':
        print('clustering seqlets from single directory')
        comb_results_save_dir=results_save_dir_a
        attrib_res_dirs = [results_save_dir_a]
        attrib_res_dir_labels=[a_label]
    else: 
        print('clustering seqlets two directories')
        attrib_res_dirs = [results_save_dir_a,results_save_dir_b]
        attrib_res_dir_labels=[a_label,b_label]
    
    os.makedirs(comb_results_save_dir, exist_ok=True)
    print(f'saving cluster results to {comb_results_save_dir}')
    
    normed_seqlets = []
    seqlet_ids = []
    region_list=dataset.metadata.index

    for i, data in enumerate(dataset):
        print(i)
        ref_seq, *others=data
        ref_seq = ref_seq.numpy() # [4,seq_len]
        region = region_list[i]

        for attrib_res_dir_idx, attrib_res_dir in enumerate(attrib_res_dirs):
            
            attrib_res_dir_label=attrib_res_dir_labels[attrib_res_dir_idx]
            
            attrib = np.load(f'{attrib_res_dir}per_region_attribs/{region}.npy')[0,:,:] # from (1,4,seq_len) to (4,seq_len)
            seqlet_info=pd.read_csv(f'{attrib_res_dir}seqlet_info/{region}.csv',index_col=0)
            if zero_center_attribs: 
                attrib=SAGEnet.tools.zero_center_attributions(attrib,axis=0)
            attrib=attrib*ref_seq

            seqlet_info=seqlet_info[seqlet_info['p-value']<cluster_seqlet_threshold]
            for seqlet_idx in range(len(seqlet_info)):
                seqlet_id=f'{region}_{seqlet_idx}_{attrib_res_dir_label}'
                seqlet_ids.append(seqlet_id)
                seqlet_attrib = attrib[:,int(seqlet_info.iloc[seqlet_idx]['start']):int(seqlet_info.iloc[seqlet_idx]['end'])]
                max_abs = seqlet_attrib.flat[np.argmax(np.abs(seqlet_attrib))]
                norm=seqlet_attrib/max_abs # so that all attribs have the same weight in combine, and motifs with opposite directions can be combined 
                normed_seqlets.append(norm.T) # norm motif takes shape [seq_len,4]
    
    print(f'running torch_compute_similarity_motifs on {len(normed_seqlets)} seqlets')
    motif_distance, offsets, revcomp_matrix = drg_tools.motif_analysis.torch_compute_similarity_motifs(normed_seqlets, normed_seqlets,device=device,batchsize=batch_size,return_alignment=True,reverse_complement=True,padding=0,metric = cluster_metric, bk_freq=0) # padding=0 bc we're clustering seqlets 
    
    with open(f'{comb_results_save_dir}normed_seqlets.pkl', 'wb') as f:
        pickle.dump(normed_seqlets, f)

    print('saving res')
    np.save(f'{comb_results_save_dir}motif_distance',motif_distance)
    np.save(f'{comb_results_save_dir}offsets',offsets)
    np.save(f'{comb_results_save_dir}revcomp_matrix',revcomp_matrix)
    np.save(f'{comb_results_save_dir}seqlet_ids',seqlet_ids)

    print('agglomerative clustering')
    clustering = AgglomerativeClustering(n_clusters = None, metric = 'precomputed', linkage = linkage, distance_threshold=distance_threshold)
                               
    clustering.fit(motif_distance)  
    clusters = clustering.labels_
    cluster_ids, n_seqlets = np.unique(clusters, return_counts=True)
    np.save(f'{comb_results_save_dir}clusters',clusters)

    cluster_cwms = drg_tools.motif_analysis.combine_pwms(normed_seqlets, clustering.labels_, 1.-motif_distance, offsets, revcomp_matrix)
    print(f'len(cluster_cwms):{len(cluster_cwms)}')
    drg_tools.io_utils.write_meme_file(cluster_cwms, cluster_ids.astype(str), 'ACGT', f'{comb_results_save_dir}cluster_cwms.meme', round = 2)


def identify_seqlets(dataset, results_save_dir,zero_center_attribs,additional_flanks=0,threshold=0.05):
    """
    Given model attributions (ex, gradients or ISM, shape [1,4,seq_len] per region), identify seqlests using tangermeme recursive_seqlets. 
    dataset should be ReferenceGenomeDataset. 
    Saves seqlet information as csv. 
    """
    seqlets_dir=f'{results_save_dir}seqlet_info/'
    attrib_dir=f'{results_save_dir}per_region_attribs/'
    os.makedirs(seqlets_dir, exist_ok=True)
    os.makedirs(attrib_dir, exist_ok=True)
    print(f'saving seqlets to {seqlets_dir}')
    
    region_list=dataset.metadata.index

    for i, data in enumerate(dataset):
        print(i)
        ref_seq, *others=data
        ref_seq = ref_seq.numpy() # [4,seq_len]
        attrib=np.load(f'{attrib_dir}{region_list[i]}.npy')[0,:,:] # from (1,4,seq_len) to (4,seq_len)
        if zero_center_attribs: 
            attrib=SAGEnet.tools.zero_center_attributions(attrib,axis=0)
        attrib_for_seqlets=np.sum(attrib*ref_seq,axis=0) # get attributions at reference sequence 
        try: 
            seqlets = tangermeme.seqlet.recursive_seqlets(attrib_for_seqlets[np.newaxis, :], additional_flanks=additional_flanks,threshold=threshold)

        except Exception as e:
            print(f"error in recursive_seqlets: {e}")
            seqlets= pd.DataFrame()
        seqlets.to_csv(f'{seqlets_dir}{region_list[i]}.csv')


def ppm_to_ic(ppm, epsilon=1e-6):
    """
    Converts a Position Probability Matrix (PPM) to an Information Content (IC) matrix.
    Input can be numpy array or torch Tensor, first dimension should correspond to the 4 nucleoties. 
    """
    is_torch=True
    if ppm.shape[0]!=4: 
        raise ValueError(f"ppm.shape[0] must be 4, but it is {ppm.shape[0]}")
    if not isinstance(ppm, torch.Tensor):
        is_torch=False
        ppm = torch.from_numpy(ppm)

    ppm = ppm + epsilon
    ppm = ppm / ppm.sum(dim=0, keepdim=True) # re normalize 
    entropy = -torch.sum(ppm * torch.log2(ppm), dim=0)
    ic_total = 2 - entropy
    ic_matrix = ppm * ic_total

    if not is_torch:
        ic_matrix=ic_matrix.detach().cpu().numpy()

    return ic_matrix


def get_top_motif_labels(annotations,cluster_ids):
    """ 
    Given a df of summarized attribution info, return motif labels for use in plotting: 
        cluster idx, motif name, q-value. 
    """
    motif_labels=[]
    for cluster_id in cluster_ids:
        rel_annotations=annotations[annotations['cluster_idx']==cluster_id].sort_values(by='bh_corrected_pval', ascending=True).iloc[0] # top match 
        motif_labels.append(f"{cluster_id}:{rel_annotations['motif_match']}:{rel_annotations['strand']}:{rel_annotations['bh_corrected_pval']:.2e}")
    return motif_labels


def load_seqlet_info(cluster_dir,model_to_attrib_dir_dict):
    """ 
    Summarize cluster results from attribution analysis. 
    Return a dataframe containing, for each seqlet, its idx, its genomic region, the ID of its cluster, which model it came from, its mean attribution, its start idx, and its end idx. 
    """
    seqlet_ids = np.load(f'{cluster_dir}seqlet_ids.npy')
    cluster_assignments = np.load(f'{cluster_dir}clusters.npy')
    seqlet_ids_to_clusters = pd.DataFrame(index=seqlet_ids)
    seqlet_ids_to_clusters['cluster_ids'] = cluster_assignments
    seqlet_ids_to_clusters['region_id'] = [item.split('_')[0] for item in seqlet_ids_to_clusters.index]
    seqlet_ids_to_clusters['seqlet_idx'] = [int(item.split('_')[1]) for item in seqlet_ids_to_clusters.index]
    seqlet_ids_to_clusters['model'] = ['_'.join(item.split('_')[2:]) for item in seqlet_ids_to_clusters.index]

    # get attribution info 
    attribution_effects = []
    starts = []
    ends = []
    for i in range(len(seqlet_ids_to_clusters)):
        curr_info=seqlet_ids_to_clusters.iloc[i]
        curr_region = curr_info['region_id']
        curr_seqlet_idx = curr_info['seqlet_idx']
        curr_model = curr_info['model']
        curr_attrib_dir=model_to_attrib_dir_dict[curr_model]

        attrib_data=np.load(f'{curr_attrib_dir}/per_region_attribs/{curr_region}.npy')
        abs_max=np.max(np.abs(attrib_data))
        all_seqlets = pd.read_csv(f'{curr_attrib_dir}seqlet_info//{curr_region}.csv',index_col=0)
        curr_seqlet=all_seqlets.iloc[curr_seqlet_idx]
        curr_seqlet_len = curr_seqlet['end']-curr_seqlet['start']
        norm_by_len=curr_seqlet['attribution']/curr_seqlet_len
        norm_by_abs_max=norm_by_len/abs_max
        attribution_effects.append(norm_by_abs_max) # normalize 
        starts.append(int(curr_seqlet['start']))
        ends.append(int(curr_seqlet['end']))

    seqlet_ids_to_clusters['attribution'] = attribution_effects
    seqlet_ids_to_clusters['starts'] = starts
    seqlet_ids_to_clusters['ends'] = ends

    return seqlet_ids_to_clusters


def load_annotations(base_dir,cluster_match_threshold=0.05,n_seqlets_threshold=10,output_suffix=''):
    """ 
    Summarize annoation results from attribution analysis. 
    Return a dataframe containing, for each cluster, annotations (including motif match, p value, strand, n_seqlets).
    """
    if output_suffix=='':
        annotations = pd.read_csv(f'{base_dir}annotations.csv',index_col=0)
    else:
        annotations = pd.read_csv(f'{base_dir}{output_suffix}/annotations.csv',index_col=0)
    cluster_assignments = np.load(f'{base_dir}{output_suffix}/clusters.npy')
    values, counts = np.unique(cluster_assignments, return_counts=True)
    curr_dict = dict(zip(values, counts))
    n_seqlets = [curr_dict[cluster_id] for cluster_id in annotations['cluster_idx']]
    annotations['n_seqlets'] = n_seqlets
    if cluster_match_threshold>0:
        annotations=annotations[annotations['bh_corrected_pval']<cluster_match_threshold]
    if n_seqlets_threshold>0:
        annotations=annotations[annotations['n_seqlets']>=n_seqlets_threshold]
    annotations=annotations.drop_duplicates()
    annotations = annotations.reset_index(drop=True)
    return annotations


def save_ref_attribs(model, dataset,model_type,results_save_dir,attrib_type='grad', device=0,batch_size=12,num_workers=8): 
    """
    Save attributions (ISM or gradient) for model (pSAGEnet or rSAGEnet) evaluated on reference sequence. 
    If model is pSAGEnet, save attributions for (personal seq input, model idx 1 output) and (reference seq input, model 0 output). 
    
    Parameters: 
    - model: pl.LightningModule to evaluate (pSAGEnet or rSAGEnet). 
    - dataset: pytorch Dataset to evaluate (ReferenceGenomeDataset, VariantDataset, or PersonalGenomeDataset).
    - model_type: String identifying the model type, "psagenet" or "rsagenet". 
    - results_save_dir: String path to directory in which to save attributions. 
    - attrib_type: String identifying type of attribution to save, "ism" or "grad" (for gradient). 
    - device: Integer, GPU index. 
    - batch_size: Integer, batch size. 
    - num_workers: Integer, number of workers. 

    Saves: Numpy array(s) of attributions. 
    """

    print(f'attrib_type:{attrib_type}')
    os.makedirs(results_save_dir, exist_ok=True)

    if model_type=='psagenet': # save attributions for each model output 
        os.makedirs(f'{results_save_dir}output_idx_0/per_region_attribs/', exist_ok=True)
        os.makedirs(f'{results_save_dir}output_idx_1/per_region_attribs/', exist_ok=True)

    print(f'saving attribs to {results_save_dir}')

    region_list=dataset.metadata.index

    for i, data in enumerate(dataset):
        print(i)
        x, *others=data
        x = x.unsqueeze(0).to(device)
        if attrib_type=='ism':
            # not yet tested 
            if model_type=='rsagenet' or model_type=='paired_psagenet': 
                # fill in with ref pred
                ref_pred = model(x)[0]
                seq_for_ism=x[0,:,:]
                attrib=np.zeros(seq_for_ism.shape)
                attrib[np.where(seq_for_ism==1)[0],np.where(seq_for_ism==1)[1]]=ref_pred

                zero_idxs = torch.where(seq_for_ism== 0) # seq_for_ism is [4,seq_len]
                nuc_idxs = zero_idxs[0]
                pos_idxs = zero_idxs[1]

                # init alt seqs 
                alt_seqs = torch.clone(seq_for_ism.expand([len(pos_idxs)] + list(seq_for_ism.size()))) # alt seqs init as current seq with shape [num_altsxlenx4x2] -- if len=10000 and channels=4, num_alts=3000

                # fill in alt seqs based on zero idxs 
                for alt_seq_idx in range(alt_seqs.shape[0]): 
                    alt_seqs[alt_seq_idx,:,pos_idxs[alt_seq_idx]] = 0 # set all nucs to 0 
                    alt_seqs[alt_seq_idx,nuc_idxs[alt_seq_idx],pos_idxs[alt_seq_idx]] = 1 # set curr nuc to 1 

                dataset = TensorDataset(alt_seqs)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
                pred = pl.trainer.predict(model,loader)
                pred = torch.cat(pred).detach().numpy()

                 # put res in correct order
                for alt_seq_idx in range(alt_seqs.shape[0]): 
                    attrib[nuc_idxs[alt_seq_idx],pos_idxs[alt_seq_idx]] = pred[alt_seq_idx] 

            elif model_type=='psagenet': 
                # personal seq, 1 idx output 
                ref_pred = model(x)[0][:,1] 
                
                seq_for_ism=x[0,1,:4,:] 
                attrib=np.zeros(seq_for_ism.shape)
                attrib[np.where(seq_for_ism==1)[0],np.where(seq_for_ism==1)[1]]=ref_pred 

                zero_idxs = torch.where(seq_for_ism== 0) 
                nuc_idxs = zero_idxs[0]
                pos_idxs = zero_idxs[1]

                # init alt seqs 
                alt_seqs = torch.clone(x[0,:,:].expand([len(pos_idxs)] + list(x[0,:,:].size())))

                # fill in alt seqs based on zero idxs 
                for alt_seq_idx in range(alt_seqs.shape[0]): 
                    alt_seqs[alt_seq_idx,:,pos_idxs[alt_seq_idx]] = 0 # set all nucs to 0 
                    alt_seqs[alt_seq_idx,nuc_idxs[alt_seq_idx],pos_idxs[alt_seq_idx]] = 1 # set curr nuc to 1 

                # alt_seqs is now (num_alts, 4, seq_len) -- but we want (num_alts, 2, 8, seq_len) to use as input 
                alt_seqs_model_input = x.repeat(alt_seqs.shape[0], 1, 1, 1)
                alt_seqs_model_input[:,1,:4,:]=alt_seqs # ISM for personal, mat seq 

                dataset = TensorDataset(alt_seqs_model_input)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
                pred = pl.trainer.predict(model,loader)
                pred = torch.cat(pred).detach().numpy()[:,1]

                 # put in correct order in res 
                for alt_seq_idx in range(alt_seqs.shape[0]): 
                    attrib[nuc_idxs[alt_seq_idx],pos_idxs[alt_seq_idx]] = pred[alt_seq_idx] 

        elif attrib_type=='grad':
            x.requires_grad_()
            model_output = model(x)[0]
            
            if model_type=='rsagenet' or model_type=='paired_psagenet': 
                model_output.backward()
                attrib =x.grad.clone().cpu().numpy()

            elif model_type=='psagenet': 
                # personal seq, 1 idx output 
                model_output[1].backward()
                attrib = x.grad.clone().cpu().numpy()[:,1,:4,:]

                np.save(f'{results_save_dir}output_idx_1/per_region_attribs/{region_list[i]}',attrib)
                x.grad.zero_()
                model.zero_grad()

                model_output = model(x)[0]
                model_output[0].backward()
                attrib = x.grad.clone().cpu().numpy()[:,0,:4,:]
                np.save(f'{results_save_dir}output_idx_0/per_region_attribs/{region_list[i]}',attrib)

            x.grad.zero_()
            model.zero_grad()

        if model_type!='psagenet':
            np.save(f'{results_save_dir}per_region_attribs/{region_list[i]}',attrib)
    
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
              
