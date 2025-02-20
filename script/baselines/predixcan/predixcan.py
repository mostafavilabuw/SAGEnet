import numpy as np 
import pandas as pd
import os
import argparse
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import ElasticNetCV,ElasticNet
import warnings
from sklearn.exceptions import ConvergenceWarning
import SAGEnet.tools

def predixcan(results_save_dir, gene_list_path, sub_list_dir, maf_threshold, input_len, rosmap_vcf_path, hg38_file_path, rosmap_expr_data_path,tss_data_path, gtex_vcf_path):
    """
    Train prediXcan (elastic net) model on ROSMAP training individuals, select hyperparameters on ROSMAP validation individuals, evaluate on all ROSMAP and GTEx individuals.
    
    Parameters: 
    - results_save_dir: String path to directory in which to prediXcan results. 
    - gene_list_path: String path to csv containing list of genes to train a prediXcan model for. 
    sub_list_dir: String path to directory containing lists of individuals (ROSMAP train/validation/test, GTEx all individuals). 
    - maf_threshold: Float, MAF threshold. 
    - input_len: Integer, size of the genomic window model input. 
    - rosmap_vcf_file_path: String path to the VCF file with ROSMAP variant information.
    - hg38_file_path: String path to the human genome (hg38) reference file.
    - rosmap_expr_data_path: String path to DataFrame with ROSMAP expression data, indexed by gene names, with sample names as columns.
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - gtex_vcf_path: String path to the VCF file with GTEx variant information.
    
    Saves: Creates a folder for each gene. Within each gene folder, saves one DataFrame with predictions for ROSMAP individuals (rosmap_pred.csv), one DataFrame with predictions for GTEx individuals (gtex_pred.csv), one DataFrame with feature information (position, reference allele, alternatie allele, coefficient) (coef.csv), and selected alpha value (alpha.npy). 
    """
    
    results_save_dir=f'{results_save_dir}{input_len}/maf_filter_{maf_threshold}/'
    os.makedirs(results_save_dir, exist_ok=True)
    
    gene_list = np.loadtxt(gene_list_path,delimiter=',',dtype=str)
    print(f'len(gene_list):{len(gene_list)}')
    
    rosmap_train_subs = np.loadtxt(f'{sub_list_dir}ROSMAP/train_subs.csv',delimiter=',',dtype=str)
    rosmap_val_subs = np.loadtxt(f'{sub_list_dir}ROSMAP/val_subs.csv',delimiter=',',dtype=str)
    rosmap_test_subs = np.loadtxt(f'{sub_list_dir}ROSMAP/test_subs.csv',delimiter=',',dtype=str)
    rosmap_all_subs = np.concatenate((rosmap_train_subs,rosmap_val_subs,rosmap_test_subs))
    gtex_all_subs = np.loadtxt(f'{sub_list_dir}GTEx/all_subs.csv',delimiter=',',dtype=str)

    expr_data = pd.read_csv(rosmap_expr_data_path, index_col=0) # genes x individuals 
    tss_data = pd.read_csv(tss_data_path,sep='\t',index_col=1)
    
    for k in range(len(gene_list)): 
        ensg = gene_list[k]
        print(f'gene idx:{k}')
        print(f'gene:{ensg}') 
        
        if ensg not in os.listdir(results_save_dir): 
        
            curr_gene_results_save_dir=f'{results_save_dir}{ensg}/'
            os.makedirs(curr_gene_results_save_dir, exist_ok=True)
            
            y_data = expr_data.loc[ensg,rosmap_all_subs].values
            curr_chr = tss_data.loc[ensg]['chr']
            tss_pos = tss_data.loc[ensg]['tss']
            pos_start = tss_pos - input_len//2
            pos_end = tss_pos + input_len//2
            
            # get variant info for this gene 
            rosmap_all_features, rosmap_pos, rosmap_ref, rosmap_alt =  SAGEnet.tools.get_variant_info(curr_chr, pos_start,pos_end, rosmap_all_subs,contig_prefix='',vcf_file_path=rosmap_vcf_path,hg38_file_path=hg38_file_path, maf_threshold=maf_threshold, train_subs_vcf_path=rosmap_vcf_path,train_subs=rosmap_train_subs, train_subs_contig_prefix='') # rosmap_all_features is (num subs x num features) 
            
            if len(rosmap_all_features)==0: 
                print('no variants present, skipping gene')

            else: 
                print(f'variant df shape: {rosmap_all_features.shape}')
                rosmap_feature_info = pd.DataFrame()
                rosmap_feature_info['pos'] = rosmap_pos
                rosmap_feature_info['ref'] = rosmap_ref
                rosmap_feature_info['alt'] = rosmap_alt
                
                train_var_data = rosmap_all_features[:len(rosmap_train_subs),]
                test_fold = np.concatenate([np.full(len(rosmap_train_subs),-1), np.zeros(len(rosmap_val_subs))]) # choose hyperparameter based on val fold 
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    cv = PredefinedSplit(test_fold)
                    regr = ElasticNetCV(cv=cv, l1_ratio = .5)
                    regr.fit(rosmap_all_features, y_data) # use CV to find best hyperparameter
                    final_model = ElasticNet(alpha=regr.alpha_, l1_ratio=.5) # retrain with chosen hyperparameter on just train data 
                    final_model.fit(train_var_data, expr_data.loc[ensg,rosmap_train_subs].values)

                # save elastic net alpha and coeffs 
                np.save(curr_gene_results_save_dir + 'alpha', final_model.alpha)
                rosmap_feature_info['coef'] = final_model.coef_
                rosmap_feature_info.to_csv(curr_gene_results_save_dir + 'coef.csv')

                # pred w final model
                rosmap_pred = final_model.predict(rosmap_all_features)
                rosmap_pred_df = pd.DataFrame(index=rosmap_all_subs)
                rosmap_pred_df['pred']=rosmap_pred
                rosmap_pred_df.to_csv(f'{curr_gene_results_save_dir}rosmap_pred.csv')
            
                # eval on gtex 
                # don't have to include MAF bc we later intersect with ROSMAP variant set 
                gtex_all_features, gtex_pos, gtex_ref, gtex_alt =  SAGEnet.tools.get_variant_info(curr_chr, pos_start,pos_end, gtex_all_subs,contig_prefix='chr',vcf_file_path=gtex_vcf_path,hg38_file_path=hg38_file_path)
                # gtex_all_features is (num subs x num features) 

                if len(gtex_all_features)==0: 
                    print('no variants present, skipping gene eval (gtex)')

                else: 
                    gtex_feature_info = pd.DataFrame()
                    gtex_feature_info['pos'] = gtex_pos
                    gtex_feature_info['ref'] = gtex_ref
                    gtex_feature_info['alt'] = gtex_alt

                    # intersect dataframes to evaluate ROSMAP-trained model on GTEx
                    rosmap_feature_info = rosmap_feature_info.drop(columns=['coef'])
                    rosmap_feature_info['rosmap_indices'] = rosmap_feature_info.index
                    gtex_feature_info['gtex_indices'] = gtex_feature_info.index
                    merged_df = pd.merge(gtex_feature_info, rosmap_feature_info, how='inner', left_on=list(gtex_feature_info.columns)[:-1], right_on=list(rosmap_feature_info.columns)[:-1])
                    trans_gtex_all_features = np.zeros((gtex_all_features.shape[0], rosmap_all_features.shape[1]))
                    # samples x features 
                    corresponding_gtex_vals = gtex_all_features[:,merged_df['gtex_indices']]
                    rosmap_indices_for_gtex_vals = merged_df['rosmap_indices']
                    trans_gtex_all_features[:,rosmap_indices_for_gtex_vals] = corresponding_gtex_vals

                    gtex_pred = final_model.predict(trans_gtex_all_features)
                    gtex_pred_df = pd.DataFrame(index=gtex_all_subs)
                    gtex_pred_df['pred']=gtex_pred
                    gtex_pred_df.to_csv(f'{curr_gene_results_save_dir}gtex_pred.csv')

if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_save_dir")
    parser.add_argument("--maf_threshold",default=0.01,type=float)
    parser.add_argument("--input_len",default=40000,type=int)
    parser.add_argument("--gene_list_path",default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/protein_coding_genes.csv')
    parser.add_argument("--sub_list_dir",default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/individual_sets/')
    parser.add_argument("--rosmap_vcf_path",default='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz')
    parser.add_argument("--gtex_vcf_path",default='/data/tuxm/GTEX_v8/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz')
    parser.add_argument("--hg38_file_path",default='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa')
    parser.add_argument("--rosmap_expr_data_path",default='/data/mostafavilab/personal_genome_expr/data/rosmap/expressionData/vcf_match_covariate_adjusted_log_tpm.csv')
    parser.add_argument("--tss_data_path",default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/gene-ids-and-positions.tsv')
    args = parser.parse_args()
   
    predixcan(results_save_dir=args.results_save_dir, gene_list_path=args.gene_list_path, sub_list_dir=args.sub_list_dir, maf_threshold=args.maf_threshold, input_len=args.input_len, rosmap_vcf_path=args.rosmap_vcf_path, hg38_file_path=args.hg38_file_path, rosmap_expr_data_path=args.rosmap_expr_data_path,tss_data_path=args.tss_data_path, gtex_vcf_path=args.gtex_vcf_path)
