import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import SAGEnet.tools

def filter_known_covariates(known_covariates, inferred_covariates, unadjusted_r2_cutoff=0.9):
    '''
    Based off of https://github.com/heatherjzhou/PCAForQTL/blob/master/R/22.01.04_main1.4_filterKnownCovariates.R
    Filter out known covariates that are captured well by inferred covariates. 
    
    Parameters: 
    - known_covariates: Dataframe of known covariates (samples x covariates)
    - inferred_covariates: DataFrame of inferred covariates from PCA (samples x PCs)
        Note: known_covariates and inferred_covariates must have the same index 
    - unadjusted_r2_cutoff: Float R2 cutoff 
    
    Returns: R2 value for each known covariate, filered Dataframe of known covariates 
    '''
    if not known_covariates.index.equals(inferred_covariates.index): 
        raise ValueError("known_covariates and inferred_covariates must have the same index")   
    
    # standardize 
    mean_vals = known_covariates.mean()
    std_vals = known_covariates.std()
    known_covariates_standardized = (known_covariates - mean_vals) / std_vals
    known_covariates_standardized = known_covariates_standardized.fillna(0) 
    R2s = []
    
    for i in range(known_covariates_standardized.shape[1]):
        model = sm.OLS(known_covariates_standardized.iloc[:, i], inferred_covariates) # fit linear regression
        results = model.fit()
        R2s.append(results.rsquared)
    indices_of_known_covariates_to_keep = np.where(np.array(R2s) < unadjusted_r2_cutoff)[0]
    indices_to_drop = np.where(np.array(R2s) >= unadjusted_r2_cutoff)[0]
    features_to_drop = known_covariates_standardized.columns[indices_to_drop]
    print('dropping:')
    print(list(features_to_drop))
    to_return = known_covariates.iloc[:, indices_of_known_covariates_to_keep]
    return R2s, to_return


def save_gene_set(save_dir, genocode_annotations_path): 
    '''
    Save list of unique protein coding gene ENSG IDs from gencode annotations dataframe into save_dir.  
    '''
    parsed = pd.read_csv(genocode_annotations_path,index_col=0)
    protein_coding_parsed = parsed[parsed['gene_type']=='protein_coding']
    protein_coding_parsed = protein_coding_parsed[protein_coding_parsed['feature']=='gene']
    acceptable_chrs = []
    acceptable_chrs.append('chrX')
    acceptable_chrs.append('chrY')
    for i in range(1,23):
        acceptable_chrs.append('chr'+str(i))
    sel_protein_coding_parsed = protein_coding_parsed[protein_coding_parsed['seqname'].isin(acceptable_chrs)]
    final_ensgs = []
    for item in list(sel_protein_coding_parsed['gene_id']):
        final_ensgs.append(item.split('.')[0]) # ENS(species)(object type)(identifier).(version), get rid of version 
    final_ensgs = list(set(final_ensgs)) 
    print('num protein coding genes')
    print(len(final_ensgs))
    np.savetxt(save_dir+ 'protein_coding_genes.csv',final_ensgs,delimiter=',',fmt='%s')
    
    
def save_rosmap_individual_sets(save_dir,vcf_path, raw_gene_expr_path, train_frac=.8,val_frac=.1, test_frac=.1,random_seed=17): 
    '''
    From ROSMAP VCF and expression data, find individuals with both data types and save in save_dir. 
    '''
    vcf_samples = SAGEnet.tools.get_sample_names(vcf_path)
    vcf_samples = np.array([item.split('_')[0][1:] for item in vcf_samples])
    
    tpm_data = pd.read_csv(raw_gene_expr_path,sep= ' ',index_col=0)
    tpm_samples = np.array(tpm_data.index)
  
    overlap_expr_and_vcf_subs = list(set(vcf_samples).intersection(set(tpm_samples)))
    overlap_expr_and_vcf_subs = np.array(overlap_expr_and_vcf_subs)
    
    print(f'num individuals with vcf and expr data: {len(overlap_expr_and_vcf_subs)}')
    
    # random split train/val/test 
    np.random.seed(random_seed) 
    shuffled_indices = np.random.permutation(len(overlap_expr_and_vcf_subs))
    shuffled_individs = overlap_expr_and_vcf_subs[shuffled_indices]
    
    val_size = int(len(overlap_expr_and_vcf_subs) * val_frac)
    test_size = int(len(overlap_expr_and_vcf_subs) * test_frac)
    train_size = len(overlap_expr_and_vcf_subs) - val_size - test_size
    
    train_set = shuffled_individs[:train_size]
    val_set = shuffled_individs[train_size:train_size + val_size]
    test_set = shuffled_individs[train_size + val_size:]

    np.savetxt(save_dir+'train_subs.csv',train_set,delimiter=',',fmt='%s')
    np.savetxt(save_dir+'val_subs.csv',val_set,delimiter=',',fmt='%s')
    np.savetxt(save_dir+'test_subs.csv',test_set,delimiter=',',fmt='%s')

    
def rosmap_adjust_for_covariates(save_dir, sub_data_dir, raw_gene_expr_path, qual_metrics_covariates_path, phen_covariates_path, genotype_pcs_path, num_expr_pcs=10, num_genotype_pcs=3): 
    '''
    Adjust ROSAMP expression data for known and unknown covariates and save final adjusted data in save_dir. 
    '''
    train_subs = np.loadtxt(sub_data_dir + 'train_subs.csv',delimiter=',',dtype=str)
    val_subs = np.loadtxt(sub_data_dir + 'val_subs.csv',delimiter=',',dtype=str)
    test_subs = np.loadtxt(sub_data_dir + 'test_subs.csv',delimiter=',',dtype=str)
    overlap_subs = np.concatenate((train_subs,val_subs,test_subs)) # all subs

    raw_tpm = pd.read_csv(raw_gene_expr_path,sep= ' ',index_col=0)
    raw_tpm = np.log2(raw_tpm.T+1) 
    print('raw tpm shape')
    print(raw_tpm.shape)
    
    # get top pcs from log2(tpm+1)
    top_expr_pcs = SAGEnet.tools.get_pcs(raw_tpm,num_expr_pcs)
    
    # load confounds covariate data 
    qual_metrics_covariates = pd.read_csv(qual_metrics_covariates_path,sep=' ')
    sample_ids = np.array(qual_metrics_covariates['Sample'].astype(str))
    sample_ids = np.char.zfill(sample_ids.astype(str), 8) # go from ints of varying lengths to strings of length 8 for consistency 
    qual_metrics_covariates['Sample']=sample_ids
    
    # load phenotype covariate data 
    phen_covariates =  pd.read_csv(phen_covariates_path,sep='\t')
    proj_ids = np.array(phen_covariates['projid'].astype(str))
    proj_ids = np.char.zfill(proj_ids.astype(str), 8) # go from ints of varying lengths to strings of length 8 for consitency 
    phen_covariates['projid'] = proj_ids

    known_covariates_from_qual_metrics= ['Batch','LOG_ESTIMATED_LIBRARY_SIZE','LOG_PF_READS_ALIGNED',
    'PCT_CODING_BASES','PCT_INTERGENIC_BASES','PCT_PF_READS_ALIGNED','PCT_RIBOSOMAL_BASES',
    'PCT_UTR_BASES','PERCENT_DUPLICATION','MEDIAN_3PRIME_BIAS','MEDIAN_5PRIME_TO_3PRIME_BIAS',
    'MEDIAN_CV_COVERAGE','RinScore']
    known_covariates_from_phen= ['study', 'pmi', 'age_death','msex']
    all_known_covars = known_covariates_from_qual_metrics+known_covariates_from_phen
    print('num known covariates')
    print(len(all_known_covars))
    
    merged_df = qual_metrics_covariates.merge(phen_covariates, left_on='Sample', right_on='projid', how='inner')
    merged_df.set_index('Sample', inplace=True)
    
    # adjust known covariates:

    # take log2 
    merged_df['LOG_ESTIMATED_LIBRARY_SIZE'] = np.log2(merged_df['ESTIMATED_LIBRARY_SIZE'])
    merged_df['LOG_PF_READS_ALIGNED'] = np.log2(merged_df['PF_READS_ALIGNED'])

    # deal with categorical vars 
    merged_df = pd.get_dummies(merged_df, columns=['Batch','study'], drop_first=True)

    # replace nan w mean 
    pmi_mean = merged_df['pmi'].mean(skipna=True)
    merged_df['pmi'] = merged_df['pmi'].fillna(pmi_mean)
    RinScore_mean = merged_df['RinScore'].mean(skipna=True)
    merged_df['RinScore'] = merged_df['RinScore'].fillna(RinScore_mean)
    
    # get df of just selected known covariates 
    selected_known_covars = pd.DataFrame()
    for item in all_known_covars: 
        selected_known_covars[item] = merged_df[item]
        
    # load in genotype PCs 
    ancestry_pcs = pd.read_csv(genotype_pcs_path,sep=' ',header=None)
    new_ids = []
    for item in ancestry_pcs[0]:
        new_ids.append(item[1:])
    ancestry_pcs.drop(0, axis=1, inplace=True) 
    ancestry_pcs.drop(1, axis=1, inplace=True) 
    ancestry_pcs.set_index(pd.Index(new_ids), inplace=True)
    
    # put indices in the same order 
    ancestry_pcs = ancestry_pcs[ancestry_pcs.index.isin(overlap_subs)]
    ancestry_pcs = ancestry_pcs.reindex(overlap_subs)
    selected_known_covars = selected_known_covars[selected_known_covars.index.isin(overlap_subs)]
    selected_known_covars = selected_known_covars.reindex(overlap_subs)
    top_expr_pcs = top_expr_pcs[top_expr_pcs.index.isin(overlap_subs)]
    top_expr_pcs = top_expr_pcs.reindex(overlap_subs)
    raw_tpm = raw_tpm[raw_tpm.index.isin(overlap_subs)]
    raw_tpm = raw_tpm.reindex(overlap_subs)

    # add in ancestry PCs as known covariates 
    for i in range(num_genotype_pcs): 
        selected_known_covars.loc[:,'ANC_PC_' + str(i)] = np.array(ancestry_pcs.iloc[:,i])
    R2s, final_covars = filter_known_covariates(selected_known_covars,top_expr_pcs)
    print(R2s)
    
    # add top pcs in 
    for i in range(num_expr_pcs):
        final_covars.loc[:,'PC_' + str(i)] = top_expr_pcs.iloc[:,i]
    
    # regress out final covariates, retain gene means 
    LR = LinearRegression()
    LR.fit(final_covars,raw_tpm)
    prediction = LR.predict(final_covars)
    adjusted_tpm = raw_tpm - prediction + raw_tpm.mean(axis=0)
    adjusted_tpm.index = adjusted_tpm.index.astype(str)
    adjusted_tpm.to_csv(save_dir + 'covariate_adjusted_log_tpm.csv')
    
    

def gtex_adjust_for_covariates(save_dir, tissue, metadata_path, phen_covariates_path, vcf_file, raw_gene_expr_path,sub_data_dir, genotype_pcs_path, num_expr_pcs=10, num_genotype_pcs=3): 
    '''
    Adjust GTEx expression data for known and unknown covariates and save final adjusted data in save_dir. 
    '''
    gtex_metadata = pd.read_csv(metadata_path,sep='\t',dtype=str)
    rel_gtex_metadata = gtex_metadata[gtex_metadata['SMTSD']==tissue]
    columns_to_drop = ['SMPTHNTS', 'SMTS', 'SMTSD', 'SMNABTCHD', 'SMGEBTCHD', 'SMNABTCH','SMGEBTCH']
    rel_gtex_metadata = rel_gtex_metadata.drop(columns=columns_to_drop)
    rel_gtex_metadata.set_index('SAMPID', inplace=True)
 
    phen_covariates = pd.read_csv(phen_covariates_path,sep='\t',index_col=0)
    use_subids = ['-'.join(item.split('-')[:2]) for item in rel_gtex_metadata.index]
    phen_covariates = phen_covariates.loc[use_subids] # put in same order
    rel_gtex_metadata['SEX'] = list(phen_covariates['SEX'])
    rel_gtex_metadata['AGE'] = [int(item.split('-')[0]) for item in phen_covariates['AGE']]
    rel_gtex_metadata['DTHHRDY'] = list(phen_covariates['DTHHRDY'])

    unique_counts = rel_gtex_metadata.nunique()
    columns_with_same_value = unique_counts[unique_counts == 1].index
    rel_gtex_metadata = rel_gtex_metadata.drop(columns=columns_with_same_value)
    nan_columns = rel_gtex_metadata.columns[rel_gtex_metadata.isna().all()]
    rel_gtex_metadata = rel_gtex_metadata.drop(columns=nan_columns)

    if tissue=='Brain - Frontal Cortex (BA9)':
        make_categorical = ['SMCENTER','SMGEBTCHT','SMAFRZE','DTHHRDY','SEX']

    else:
        make_categorical = ['SMCENTER','SMNABTCHT','SMGEBTCHT','SMAFRZE','DTHHRDY','SEX']

    take_log2 = ['SMCHMPRS', 'SMMPPD','SMRRNANM','SMRDTTL','SMVQCFL','SMTRSCPT','SMMPPDPR',
                'SMMPPDUN','SME2ANTI','SMALTALG','SME2SNSE','SME1ANTI','SMSPLTRD',
                'SME1SNSE']
    for col in rel_gtex_metadata.columns: # make into floats 
        if col not in make_categorical:
            rel_gtex_metadata[col] = rel_gtex_metadata[col].astype(float) 
    for col in take_log2: # take log2 of vars related to read counts 
        rel_gtex_metadata[col] = np.log2(rel_gtex_metadata[col])
    for col in make_categorical: # fill na in categorical vars with mode 
        if rel_gtex_metadata[col].isna().any(): 
            curr_mode = rel_gtex_metadata[col].value_counts().idxmax()
            rel_gtex_metadata[col] = rel_gtex_metadata[col].fillna(curr_mode)
    for col in rel_gtex_metadata.columns: # fill na in other vars with mean 
        if rel_gtex_metadata[col].isna().any(): 
            curr_mean = rel_gtex_metadata[col].mean(skipna=True)
            rel_gtex_metadata[col] = rel_gtex_metadata[col].fillna(curr_mean)
    rel_gtex_metadata = pd.get_dummies(rel_gtex_metadata, columns=make_categorical, drop_first=True)
  
    # get sample IDs with both WGS and expression data 
    final_sample_names = SAGEnet.tools.get_sample_names(vcf_file)
    expr_subs = pd.read_csv(raw_gene_expr_path,skiprows=2,sep='\t',nrows=1).columns[2:]
    overlap_samp_ids = []
    for item in expr_subs:
        if '-'.join(item.split('-')[:2]) in final_sample_names and item in rel_gtex_metadata.index:
            overlap_samp_ids.append(item)
    overlap_samp_ids=np.array(overlap_samp_ids)
    np.savetxt(f'{sub_data_dir}{tissue}{all_subs.csv}', overlap_samp_ids, delimiter=',', fmt='%s')
    
    rel_gtex_metadata=rel_gtex_metadata.loc[overlap_samp_ids]
    
    # load expr data 
    expr_cols_to_load=np.append(overlap_samp_ids,'Name')
    expr = pd.read_csv(gtex_raw_expr_data_path,skiprows=2,sep='\t',usecols=expr_cols_to_load)
    gene_names = [item.split('.')[0] for item in expr['Name']]
    expr.index=gene_names
    expr = expr.drop(columns=['Name'])
    raw_tpm=np.log2(expr+1)
    raw_tpm = raw_tpm.reset_index().drop_duplicates(subset='index').set_index('index')
    raw_tpm=raw_tpm.T

    # get top pcs from log2(tpm+1)
    top_expr_pcs = SAGEnet.tools.get_pcs(raw_tpm,num_expr_pcs)
        
    # load in genotype PCs 
    ancestry_pcs = pd.read_csv(genotype_pcs_path,sep=' ',header=None,index_col=0)
    ancestry_pcs.drop(1, axis=1, inplace=True) 
    vcf_lables= ['-'.join(item.split('-')[:2]) for item in overlap_samp_ids]
    ancestry_pcs=ancestry_pcs.loc[vcf_lables]
   
    # add in ancestry PCs as known covariates 
    for i in range(num_genotype_pcs): 
        rel_gtex_metadata.loc[:,'ANC_PC_' + str(i)] = np.array(ancestry_pcs.iloc[:,i])

    R2s, final_covars = filter_known_covariates(rel_gtex_metadata,top_expr_pcs)
    print(R2s)
    
    # add top pcs in 
    for i in range(num_expr_pcs):
        final_covars.loc[:,'PC_' + str(i)] = top_expr_pcs.iloc[:,i]

    tissue_save_name = ''.join(tissue.split(' '))
    
    # regress out final covariates, retain gene means 
    LR = LinearRegression()
    LR.fit(final_covars,raw_tpm)
    prediction = LR.predict(final_covars)
    adjusted_tpm = raw_tpm - prediction + raw_tpm.mean(axis=0)
    adjusted_tpm.index = adjusted_tpm.index.astype(str)
    adjusted_tpm.to_csv(save_dir + tissue_save_name + '_covariate_adjusted_log_tpm.csv')
    
    
if __name__ == '__main__':  

    # save list of protein coding genes 
    # annotations downloaded from https://www.gencodegenes.org/human/release_27.html, gencode v27
    gtf_dir = '/data/mostafavilab/personal_genome_expr/gene_info/'
    gene_list_save_dir = '/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/'
    gencode_annotation_path = f'{gtf_dir}gencode.v27.annotation.gtf'
    parsed = SAGEnet.tools.gtf2df(gencode_annotation_path)
    parsed.to_csv(f'{gtf_dir}gencode_v27_annotation.csv')
    save_gene_set(gene_list_save_dir, f'{gtf_dir}gencode_v27_annotation.csv')
    
    # save individual sets for ROSMAP
    rosmap_vcf_path='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz'
    rosmap_raw_gene_expr_path = '/data/mostafavilab/bng/rosmapAD/data/expressionData/DLPFC/20220207-bulk-RNAseq/raw_geneTpm.txt'
    rosmap_sub_data_dir = '/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/individual_sets/ROSMAP/'
    save_rosmap_individual_sets(rosmap_sub_data_dir,rosmap_vcf_path, rosmap_raw_gene_expr_path, train_frac=.8,val_frac=.1, test_frac=.1,random_seed=17): 

    # preprocess ROSMAP expression data 
    rosmap_expr_data_save_dir='/data/mostafavilab/personal_genome_expr/data/rosmap/expressionData/'
    rosmap_qual_metrics_covariates_path = '/data/mostafavilab/bng/rosmapAD/data/expressionData/DLPFC/20220207-bulk-RNAseq/qualityMetrics.txt'
    rosmap_phen_covariates_path = '/data/mostafavilab/bng/rosmapAD/data/phenotypes/basic_Spt2019.txt'
    rosmap_genotype_pcs_path = '/data/mostafavilab/personal_genome_expr/ancestry/ancestry.eigenvec'
    # these are saved with plink --bfile chrAll.phased --out ancestry --pca 1161
        
    rosmap_adjust_for_covariates(rosmap_expr_data_save_dir, rosmap_individual_sets_save_dir, rosmap_raw_gene_expr_path, rosmap_qual_metrics_covariates_path, rosmap_phen_covariates_path, rosmap_genotype_pcs_path, num_expr_pcs=10, num_genotype_pcs=3)
    
    # preprocess GTEx exprssion data 
    gtex_expr_save_dir='/data/tuxm/GTEX_v8/'
    gtex_sub_data_dir='homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/individual_sets/ROSMAP/'
    gtex_metadata_path='/data/mostafavilab/personal_genome_expr/data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
    gtex_phen_covariates_path = '/data/tuxm/GTEX_v8/gtex_public_phenotype_covars.txt'
    gtex_vcf_file='/data/tuxm/GTEX_v8/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz'
    gtex_raw_expr_data_path = '/data/tuxm/GTEX_v8/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'
    gtex_genotype_pcs_path = '/data/tuxm/GTEX_v8/ancestry.eigenvec'
    # these are saved with plink plink --bfile GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze --out ancestry --pca 838
    
    gtex_adjust_for_covariates(gtex_expr_save_dir, 'Brain - Cortex', gtex_metadata_path, gtex_phen_covariates_path, gtex_vcf_file, gtex_raw_expr_data_path,gtex_sub_data_dir, gtex_genotype_pcs_path, num_expr_pcs=10, num_genotype_pcs=3)



    

    
