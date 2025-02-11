import numpy as np 
import pandas as pd 
import argparse
from torch.utils.data import DataLoader
import os
import time
import torch
import SAGEnet.tools
from SAGEnet.data import PersonalGenomeDataset
import pytorch_lightning as pl
from SAGEnet.models import rSAGEnet,pSAGEnet
from SAGEnet.enformer import Enformer

ENFORMER_INPUT_LEN=393216

def eval_model(ckpt_path, results_save_dir, num_genes, tss_data_path, hg38_file_path,batch_size,num_workers,device,input_len,model_type,rosmap_or_gtex,predixcan_res_path,sub_data_dir,train_val_test,eval_on_ref_seq,vcf_file_path,best_ckpt_metric,max_epochs,identify_best_ckpt,rand_genes,top_genes_to_consider,seed,only_snps,save_per_gene,maf_threshold,train_subs_vcf_file_path,allow_reverse_complement,gene_idx_start,enformer_save_mode,enformer_finetuned_weights_dir):
    """
    Eval model (pSAGEnet, rSAGEnet, or Enformer) on WGS data. 
    
    Parameters: 
    - ckpt_path: String path to either model ckpt to evaluate (if identify_best_ckpt==False) or directory containing model ckpt (if identify_best_ckpt==True). 
    - results_save_dir: String path to directory in which to save evaluation results. 
    - num_genes: Integer number of genes to evaluate. 
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - hg38_file_path: String path to the human genome (hg38) reference file.
    - batch_size: Integer, batch size to use in model eval. 
    - num_workers: Integer, number of workers to use in model eval. 
    - device: Integer, GPU index. 
    - input_len: Integer, size of the genomic window model input. 
    - model_type: String type of model to evaluate, from {'psagenet, rsagenet, enformer'}. 
    - rosmap_or_gtex: String indicating which dataset is being evaluated. 
    - predixcan_res_path: String path to predixcan results path, to be used to construct ranked gene sets. 
    - sub_data_dir: String path to directory containing lists of individual train/validation/test splits. 
    - train_val_test: String indicating what set of individuals to evaluate on (if ROSMAP), from {'train', 'val', 'test'} 
    - eval_on_ref_seq: Boolean indicating whether to evalute on reference sequence for each gene (instead of all personal sequences) 
    - vcf_file_path: String path to the VCF file with variant information.
    - best_ckpt_metric: Metric used to select best model from ckpt dir, (if identify_best_ckpt==True). Can be one of {'train_gene_gene', 'train_gene_sample', 'val_gene_gene', 'val_gene_sample'}
    - rand_genes: Boolean indicating whether or not to randomly select genes (from top_genes_to_consider gene set) to use in model evaluation. If False, select gene set from top-prediXcan ranked genes. 
    - top_genes_to_consider: Integer, length of prediXcan-ranked top gene set to consider when randomly selecting genes (only relevant if rand_genes==True). 
    - seed: Integer seed to determine random shuffling of gene set. 
    - only_snps: Boolean indicating if only SNPs should be inserted or if all variants (including indels) should be inserted. 
    - save_per_gene: Boolean indicating whether results for each gene are saved seperately. Can be used to parallelize model evaluation. If model_type==enformer, this is set to True. 
    - maf_threshold: Float, MAF threshold to use in evaluation. 
    - train_subs_vcf_file_path: String path to the VCF file for train_subs (to use to calculate MAF). 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
    - gene_idx_start: Integer index in prediXcan-ranked gene list of first gene to use in model evaluation. 
    - enformer_save_mode
    - enformer_save_mode: String indicating how Enformer's outputs should be processed. If save_mode=='finetuned', log(predictions+1) from the 3 center bins will be transformed using weights from finetuned_weights_dir and summed to yield a single prediction value. If save_mode=='only_brain', log(predictions+1) from the 3 center bins, track 4980 ('CAGE:brain, adult') will be summed to yield a single prediction value. If save_mode=='all_tracks', prediction values from the center 3 bins from all tracks (shape 3,5313) will be saved. 
    - finetuned_weights_dir: String of directory containing 'coef.npy' and 'intercept.npy' to finetune Enformer predictions. 
    - max_epochs: Integer, maximum number of epochs to consider when selecting best model ckpt. 
    - identify_best_ckpt: Boolean, whether or not to use best_ckpt_metric to select best model ckpt within ckpt_path. If False, best model ckpt path is assumed to be the path given by ckpt_path. 
    """
    # load model 
    model_type=model_type.lower()
    
    # identify best ckpt from directory based on metric provided 
    if identify_best_ckpt:
        ckpt_path = SAGEnet.tools.select_ckpt_path(ckpt_path,max_epochs=max_epochs,best_ckpt_metric=best_ckpt_metric)
    ckpt_label = ckpt_path.split('/')[-1]
        
    if model_type=='rsagenet':
        print('rsagenet model')
        model = rSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path,using_personal_dataset=True,predict_from_personal=not eval_on_ref_seq)
        
    elif model_type=='psagenet':
        print('psagenet model')
        model = pSAGEnet.load_from_checkpoint(checkpoint_path=ckpt_path)
        
    elif model_type=='enformer':
        print('enformer model')
        input_len=ENFORMER_INPUT_LEN
        os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
        model = Enformer(finetuned_weights_dir=finetuned_weights_dir)
        
    # info for constructing dataset 
    if rosmap_or_gtex == 'rosmap':
        contig_prefix=''
    elif rosmap_or_gtex == 'gtex':
        contig_prefix='chr'
    else: 
        raise ValueError("rosmap_or_gtex must be one of {rosmap,gtex}")
    gene_meta_info = pd.read_csv(args.tss_data_path, sep="\t")
    
    # name results_save_dir 
    if results_save_dir=='':
        results_save_dir = os.path.dirname(ckpt_path) # by default, save results in the directory containing model ckpt 
    if only_snps: 
        variant_set_label='only_snps'
    else: 
        variant_set_label='snps_and_indels'
    if maf_threshold>=0: 
        maf_label=f'maf_filter_{maf_threshold}'
    else: 
        maf_label='no_maf_filter'
    if model_type=='enformer' or model_type=='rsagenet': 
        if eval_on_ref_seq:         
            results_save_dir=f'{results_save_dir}/{model_type}_model/eval_on_ref_seq/'
        else: 
            results_save_dir=f'{results_save_dir}/{model_type}_model/eval_on_individual/{variant_set_label}/{maf_label}/{rosmap_or_gtex}/{train_val_test}_subs/'
    else: 
        if eval_on_ref_seq:         
            results_save_dir=f'{results_save_dir}/{model_type}_model/metric_{best_ckpt_metric}/eval_on_ref_seq/{ckpt_label}/'
        else: 
            results_save_dir=f'{results_save_dir}/{model_type}_model/metric_{best_ckpt_metric}/eval_on_individual/{variant_set_label}/{maf_label}/{rosmap_or_gtex}/{train_val_test}_subs/{ckpt_label}/'
    os.makedirs(results_save_dir, exist_ok=True)
    
    # select gene set 
    gene_list = SAGEnet.tools.select_gene_set(predixcan_res_path=predixcan_res_path, rand_genes=rand_genes, top_genes_to_consider=top_genes_to_consider,seed=seed, num_genes=num_genes,gene_idx_start=gene_idx_start)           
    print(f"n genes={len(gene_list)}")
    np.save(results_save_dir+'gene_list',gene_list)
    selected_genes_meta = gene_meta_info.set_index('ensg', drop=False).loc[gene_list]
    
    # load sub info 
    
    # both rosmap and gtex need rosmap train subs to use in MAF calculation  
    train_subs = np.loadtxt(sub_data_dir + 'train_subs.csv',delimiter=',',dtype=str)
    if rosmap_or_gtex == 'rosmap':
        val_subs = np.loadtxt(sub_data_dir + 'val_subs.csv',delimiter=',',dtype=str)
        test_subs = np.loadtxt(sub_data_dir + 'test_subs.csv',delimiter=',',dtype=str)
        if train_val_test=='train':
            subs=train_subs
        elif train_val_test=='val':
            subs=val_subs
        elif train_val_test=='test':
            subs=test_subs
        else: 
            subs = np.concatenate((train_subs,val_subs,test_subs)) # all subs
                    
    if rosmap_or_gtex == 'gtex':
        subs = np.loadtxt(sub_data_dir + 'all_subs.csv',delimiter=',',dtype=str)
        
    if eval_on_ref_seq: 
        subs=[subs[0]]  
    np.save(results_save_dir+'sub_list', subs)
    print(f"n subs={len(subs)}")
    
    if save_per_gene or model_type=='enformer': # use to parallelize model evaluation 
        gene_res_save_dir = f'{results_save_dir}gene_res/'
        started_genes_save_dir = f'{results_save_dir}started_genes/'
        finished_genes_save_dir = f'{results_save_dir}finished_genes/'
        os.makedirs(gene_res_save_dir, exist_ok=True)
        os.makedirs(started_genes_save_dir, exist_ok=True)
        os.makedirs(finished_genes_save_dir, exist_ok=True)
        started_genes = [item.split('.')[0] for item in os.listdir(started_genes_save_dir)]
        finished_genes = [item.split('.')[0] for item in os.listdir(finished_genes_save_dir)]
        unstarted_genes = list(set(gene_list) - set(started_genes))
        unfinished_genes = list(set(gene_list) - set(finished_genes))
        while len(unfinished_genes)!=0: 
            if len(unstarted_genes)!=0:
                to_run_gene = unstarted_genes[0]
            else: 
                to_run_gene = unfinished_genes[0]
            print(f"evaluating {to_run_gene}")
            selected_genes_meta = gene_meta_info[gene_meta_info['ensg']==to_run_gene]
            dataset = PersonalGenomeDataset(gene_metadata=selected_genes_meta, vcf_file_path=vcf_file_path, hg38_file_path=hg38_file_path,sample_list=subs, input_len=input_len,contig_prefix=contig_prefix,train_subs=train_subs, only_snps=only_snps,maf_threshold=maf_threshold,train_subs_vcf_file_path=train_subs_vcf_file_path,allow_reverse_complement=allow_reverse_complement)
            np.save(started_genes_save_dir + to_run_gene, [])

            if model_type=='enformer': 
                preds = model.predict_on_dataset(dataset,save_mode=enformer_save_mode,predict_from_personal=not eval_on_ref_seq,using_personal_dataset=True)
            else: 
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                trainer = pl.Trainer(accelerator="gpu", devices=[device], precision=16,logger=False)
                preds = trainer.predict(model, dataloader)
                preds=np.concatenate(preds)
            np.save(gene_res_save_dir+to_run_gene,preds)
            np.save(finished_genes_save_dir + to_run_gene, [])
            started_genes = [item.split('.')[0] for item in os.listdir(started_genes_save_dir)]
            finished_genes = [item.split('.')[0] for item in os.listdir(finished_genes_save_dir)]
            unstarted_genes = list(set(gene_list) - set(started_genes))
            unfinished_genes = list(set(gene_list) - set(finished_genes))
            print(f"num unstarted genes = {len(unstarted_genes)}")
            print(f"num unfinished genes = {len(unfinished_genes)}")
    
    else: 
        dataset = PersonalGenomeDataset(gene_metadata=selected_genes_meta, vcf_file_path=vcf_file_path, hg38_file_path=hg38_file_path, sample_list=subs, input_len=input_len,contig_prefix=contig_prefix,train_subs=train_subs, only_snps=only_snps,maf_threshold=maf_threshold,train_subs_vcf_file_path=train_subs_vcf_file_path,allow_reverse_complement=allow_reverse_complement)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        trainer = pl.Trainer(accelerator="gpu", devices=[device], precision=16,logger=False)
        preds = trainer.predict(model, dataloader)
        preds=np.concatenate(preds)
        np.save(results_save_dir+'preds',preds)
        
    
if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt_path",default='/data/mostafavilab/personal_genome_expr/da_models/1k_with_rc_22962405/')
    parser.add_argument("--results_save_dir",default='',type=str)
    parser.add_argument("--num_genes",type=int,default=5000)
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--num_workers",default=8,type=int)
    parser.add_argument("--device",default=0,type=int)
    parser.add_argument("--input_len",default=40000,type=int)
    parser.add_argument("--model_type",default='psagenet')
    parser.add_argument("--rosmap_or_gtex",default='rosmap')
    parser.add_argument("--eval_on_ref_seq",default=0,type=int)
    parser.add_argument("--best_ckpt_metric",default='train_gene_gene',type=str)
    parser.add_argument("--max_epochs",default=10,type=int)
    parser.add_argument("--identify_best_ckpt",default=1,type=int)
    parser.add_argument("--rand_genes",type=int,default=0)
    parser.add_argument("--top_genes_to_consider",type=int,default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only_snps', default=0,type=int)
    parser.add_argument("--save_per_gene",type=int,default=1)
    parser.add_argument("--maf_threshold",type=float,default=-1)
    parser.add_argument("--allow_reverse_complement",type=int,default=1)
    parser.add_argument("--gene_idx_start",type=int,default=0)
    parser.add_argument('--train_val_test',default='test')
    parser.add_argument("--enformer_save_mode",default='finetuned')
    parser.add_argument("--tss_data_path",default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/gene-ids-and-positions.tsv')
    parser.add_argument("--hg38_file_path",default='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa')
    parser.add_argument("--predixcan_res_path",default='/data/mostafavilab/personal_genome_expr/predixcan_res/rosmap/40k/MAF_0.01/pearson_corr.csv')
    parser.add_argument('--enformer_finetuned_weights_dir',default='/data/mostafavilab/personal_genome_expr/final_results/enformer/ref_seq_all_tracks/')
    parser.add_argument('--vcf_file_path', default='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz')
    parser.add_argument('--train_subs_vcf_file_path', default='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz')
    args = parser.parse_args()
    
    eval_model(
        ckpt_path=args.ckpt_path, 
        results_save_dir=args.results_save_dir, 
        num_genes=args.num_genes, 
        tss_data_path=args.tss_data_path, 
        hg38_file_path=args.hg38_file_path, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        device=args.device, 
        input_len=args.input_len, 
        model_type=args.model_type, 
        rosmap_or_gtex=args.rosmap_or_gtex, 
        predixcan_res_path=args.predixcan_res_path, 
        sub_data_dir=args.sub_data_dir, 
        train_val_test=args.train_val_test, 
        eval_on_ref_seq=args.eval_on_ref_seq, 
        vcf_file_path=args.vcf_file_path, 
        best_ckpt_metric=args.best_ckpt_metric, 
        max_epochs=args.max_epochs, 
        identify_best_ckpt=args.identify_best_ckpt, 
        rand_genes=args.rand_genes, 
        top_genes_to_consider=args.top_genes_to_consider, 
        seed=args.seed, 
        only_snps=args.only_snps, 
        save_per_gene=args.save_per_gene, 
        maf_threshold=args.maf_threshold, 
        train_subs_vcf_file_path=args.train_subs_vcf_file_path, 
        allow_reverse_complement=args.allow_reverse_complement, 
        gene_idx_start=args.gene_idx_start, 
        enformer_save_mode=args.enformer_save_mode, 
        enformer_finetuned_weights_dir=args.enformer_finetuned_weights_dir
    )
