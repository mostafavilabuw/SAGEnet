import numpy as np 
import pandas as pd
import pysam
import SAGEnet.plot
import os
import scipy.stats


def save_boxplots(paths,labels,obs_data,experiment_label,fig_save_dir,xlabel='Model',ylabel='Pearson R',fig_width=11,gene_set=None,non_dual_output=[],fontsize=14):
    """
    Given a list of paths to directories containing model evaluation results, plot the results in all paths as boxplots. Save seperate boxplots for three correlations: train gene per gene, test gene per gene, test gene across gene. 
    
    Parameters: 
    - paths: List of string paths to directories containing model evaluation results. 
    - labels: List of string labels associated with each path. 
    - obs_data: Dataframe of observed data to use to get correlations with predicted data. Indexed by gene names, with sample names as columns.
    - experiment_label: String label to create directory within fig_save_dir to save boxplots in. 
    - fig_save_dir: String path to directory in which to make a directory to save boxplots in. 
    - xlabel: String label for boxplot x-axis. 
    - ylabel: String label for boxplot y-axis. 
    - title: String boxplot title. 
    - fig_width: Float width of figure. 
    - gene_set: Numpy array of gene set (containing train, validation and test genes) to plot in boxplots. If None provided, gene_set is loaded from paths containing evaluation results. 
    """    
    save_dir = f'{fig_save_dir}{experiment_label}/'
        
    # save predictions and correlations (resave, even if these results exist already) 
    for path in paths:
        if path in non_dual_output: 
            save_corr_from_per_gene(path,obs_data,dual_output=False)
        else: 
            save_corr_from_per_gene(path,obs_data)

    # get gene sets 
    if gene_set is None: 
        gene_set = np.load(f'{paths[0]}gene_list.npy',allow_pickle=True)
    gene_set = [gene for gene in gene_set if gene in obs_data.index.values]
    train_genes, val_genes, test_genes = SAGEnet.tools.get_train_val_test_genes(gene_set)
    
    # plot per gene train gene 
    comb_res = pd.DataFrame(index=train_genes)
    for i in range(len(paths)):
        corr = pd.read_csv(f'{paths[i]}per_gene_pearson_corr.csv',index_col=0).loc[train_genes,'pearson']
        comb_res[labels[i]] = corr

    SAGEnet.plot.sb_plot(comb_res,xlabel=xlabel,ylabel=ylabel,save_dir=save_dir,save_name='per_gene_train_gene',title=f'Train gene (n={len(comb_res)}) per gene Pearson R',save_fig=True,dot_size=5,dot_alpha=.1,fig_width=fig_width,fontsize=fontsize)
    
    # plot per gene test gene 
    comb_res = pd.DataFrame(index=test_genes)
    for i in range(len(paths)):
        corr = pd.read_csv(f'{paths[i]}per_gene_pearson_corr.csv',index_col=0).loc[test_genes,'pearson']
        comb_res[labels[i]] = corr
    SAGEnet.plot.sb_plot(comb_res,xlabel=xlabel,ylabel=ylabel,save_dir=save_dir,save_name='per_gene_test_gene',title=f'Test gene (n={len(comb_res)}) per gene Pearson R',save_fig=True,dot_size=5,dot_alpha=.3,fig_width=fig_width,fontsize=fontsize)

   
    # plot across gene test gene 
    comb_res = pd.DataFrame(index=list(range(1)))
    for i in range(len(paths)): 
        if paths[i] in non_dual_output: 
            pred = pd.read_csv(f'{paths[i]}predictions.csv',index_col=0).loc[test_genes].mean(axis=1).values
        else: 
            pred = pd.read_csv(f'{paths[i]}mean_predictions.csv',index_col=0).loc[test_genes].mean(axis=1).values
        corr, pval = scipy.stats.pearsonr(pred,obs_data.loc[test_genes].mean(axis=1).values)
        comb_res[labels[i]]=corr
    comb_res = comb_res.fillna(0)
    SAGEnet.plot.sb_plot(comb_res,xlabel=xlabel,ylabel=ylabel,save_dir=save_dir,save_name='across_gene_test_gene',title=f'Test gene across gene Pearson R',save_fig=True,dot_size=10,dot_alpha=1,fig_width=fig_width,plot_type='na')


def save_corr_from_per_gene(save_dir,obs_data,dual_output=True):
    """
    Given a path to a directory containing model evaluation results (saved seperately per-gene) and expression data, calculate and save the per-gene correlation and predictions for evaluation results. 
    
    Parameters: 
    - save_dir: String path to directory containing model evaluation results.
    - obs_data: Dataframe of observed data to use to get correlations with predicted data. Indexed by gene names, with sample names as columns.
    - dual_output: Boolean indicating whether the model outputs mean and difference predictions seperately (for pSAGEnet, dual_output=True)
    """    
    sub_list = np.load(f'{save_dir}sub_list.npy')
    corrs = []
    genes = []
    mean_preds = []
    diff_preds = []
    preds = []
    for file in os.listdir(save_dir+'gene_res/'):
        gene = file.split('.')[0]
        if gene in obs_data.index: 
            genes.append(gene)
            pred = np.load(save_dir+'gene_res/'+file)
            obs = obs_data.loc[gene,sub_list].values
            if np.any(np.isnan(pred)):
                print(f'{file} contains nan, changing to 0')
                pred[np.isnan(pred)] = 0
            if dual_output: 
                corr, p = scipy.stats.pearsonr(obs,pred[:,1])
                mean_preds.append(pred[:,0])
                diff_preds.append(pred[:,1])
            else: 
                if len(pred.shape)>1: 
                    pred=pred[:,1] # for non contrastive, dual output 
                corr, p = scipy.stats.pearsonr(obs,pred)
                preds.append(pred)
            corrs.append(corr)
    res = pd.DataFrame(index=genes)
    res[f'pearson'] = corrs
    res = res.fillna(0)
    res.to_csv(f'{save_dir}per_gene_pearson_corr.csv')
    
    if dual_output: 
        mean_preds = np.array(mean_preds)
        diff_preds = np.array(diff_preds)
        mean_res_df = pd.DataFrame(mean_preds,index=genes,columns=sub_list)
        diff_res_df = pd.DataFrame(diff_preds,index=genes,columns=sub_list)
        mean_res_df.to_csv(f'{save_dir}mean_predictions.csv')
        diff_res_df.to_csv(f'{save_dir}diff_predictions.csv')
    else: 
        preds = np.array(preds) 
        res_df = pd.DataFrame(preds,index=genes,columns=sub_list)
        res_df.to_csv(f'{save_dir}predictions.csv')