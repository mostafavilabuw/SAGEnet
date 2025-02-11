import numpy as np 
import pandas as pd
import argparse
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet

def finetune_enformer(enformer_res_path,expr_data_path,enformer_gene_assignments_path):
    """
    Save Enformer fine-tuning weights. 
    
    Paramters: 
    - enformer_res_path: String path to directory containing per-gene Enformer predictions (all tracks saved, shape (3, 5313))
    - expr_data_path: String path to expression Dataframe to use for finetuning. 
    - enformer_gene_assignments_path: String path to dataframe containing Enformer gene splits. 
    """
    gene_names = []
    gene_res = []
    single_gene_res_path=f'{enformer_res_path}gene_res/'
    for file in os.listdir(single_gene_res_path): 
        curr_gene = np.load(f'{single_gene_res_path}{file}')
        gene_names.append(file.split('.')[0])
        gene_res.append(np.log(curr_gene+1)[0,:,:].sum(axis=0)) # sum over three center bins 
    gene_res_df = pd.DataFrame(index=gene_names,data=np.array(gene_res))
        
    expr_data = pd.read_csv(expr_data_path, index_col=0)
    use_ensgs = [gene for gene in gene_res_df.index if gene in expr_data.index]
    y_data = expr_data.loc[use_ensgs].mean(axis=1)
    x_data = gene_res_df.loc[use_ensgs]
    
    assignments_df = pd.read_csv(enformer_gene_assignments_path,index_col=0)
    train_genes = assignments_df[assignments_df['enformer_set']=='train'].index
    val_genes = assignments_df[assignments_df['enformer_set']=='valid'].index
    train_and_val_genes = np.concatenate((train_genes,val_genes))
    
    fold_info = np.concatenate([np.full(len(train_genes),-1), np.zeros(len(val_genes))]) # choose hyperparameter based on val fold 
    cv = PredefinedSplit(fold_info)
    regr = ElasticNetCV(cv=cv, l1_ratio = .5)
    regr.fit(x_data.loc[train_and_val_genes].values, y_data.loc[train_and_val_genes].values) # use CV to find best hyperparameter
    final_model = ElasticNet(alpha=regr.alpha_, l1_ratio=.5) # retrain with chosen hyperparameter on just train data 
    final_model.fit(x_data.loc[train_genes].values, y_data.loc[train_genes].values)
    np.save(enformer_res_path + 'coef', final_model.coef_)
    np.save(enformer_res_path + 'intercept', final_model.intercept_)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enformer_res_path',default='/data/mostafavilab/personal_genome_expr/final_results/enformer/ref_seq_all_tracks/')
    parser.add_argument("--expr_data_path",default='data/tuxm/GTEX_v8/Brain-Cortex_covariate_adjusted_log_tpm.csv')
    parser.add_argument('--enformer_gene_assignments_path', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/enformer_gene_splits.csv')
    args = parser.parse_args()
    
    finetune_enformer(args.enformer_res_path,args.expr_data_path,args.enformer_gene_assignments_path)