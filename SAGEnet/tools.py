import numpy as np
import os
import pysam
import pandas as pd
#from SAGEnet.models import rSAGEnet

def get_pos_idx_in_seq(gene, pos,tss_data_path='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/gene-ids-and-positions.tsv',input_len=40000,allow_reverse_complement=True):
    """
    Get given a gene and a position, determine the index of the given position in the sequence from PersonalGenomeDataset, ReferenceDataset, or VariantDataset. 
    
    Parameters: 
    - gene: String gene ENSG id. 
    - pos: Integer position in the gene.
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
     - input_len: Integer, size of the genomic window model input. 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
    
    Returns: integer position index of position in sequence. 
    """
    gene_meta_info = pd.read_csv(tss_data_path, sep="\t")
    gene_info = gene_meta_info[gene_meta_info['gene_id']==gene].iloc[0]
    chr = gene_info["chr"]
    tss_pos = gene_info["tss"]
    pos_start = max(0, tss_pos - input_len // 2)
    pos_idx = pos-pos_start
    if pos_idx<0 or pos_idx>=input_len:
        raise ValueError('pos not in sequence range')
    rc=(gene_info['strand']!='+') # reverse complement genes on the negative strand 
    if rc and allow_reverse_complement: 
        pos_idx=input_len-pos_idx
    return pos_idx


def select_gene_set(predixcan_res_path, rand_genes, top_genes_to_consider=5000,seed=42, num_genes=1000,gene_idx_start=0): 
    """
    Select gene set, either based on prediXcan ranking or randomly. 
    - predixcan_res_path: String path to predixcan results path, to be used to construct ranked gene sets. 
    - rand_genes: Boolean indicating whether or not to randomly select genes (from top_genes_to_consider gene set) to use in model evaluation. If False, select gene set from top-prediXcan ranked genes. 
    - top_genes_to_consider: Integer, length of prediXcan-ranked top gene set to consider when randomly selecting genes (only relevant if rand_genes==True). 
    - seed: Integer seed to determine random shuffling of gene set. 
    - num_genes: Integer number of genes to select. 
    - gene_idx_start: Integer index in prediXcan-ranked gene list of first gene to use in model evaluation.
    
    Returns: List of genes (each gene is a string ENSG id). 
    """
    predixcan_res = pd.read_csv(predixcan_res_path)
    predixcan_res = predixcan_res.sort_values(by='val_pearson', ascending=False)

    if rand_genes: 
        print(f'randomly selecting from top {top_genes_to_consider} genes')
        ensg_options = predixcan_res['ensgs'][:top_genes_to_consider].values
        np.random.seed(seed)
        np.random.shuffle(ensg_options)
        gene_list = ensg_options[:num_genes]
    else: 
        print(f'selecting genes from {gene_idx_start} to {gene_idx_start+num_genes}')
        gene_list = predixcan_res['ensgs'][gene_idx_start:gene_idx_start+num_genes].values
        
    return gene_list


def get_sample_names(vcf_file_path):
    """
    Get sample names from header of a given VCF path. 
    """
    with pysam.VariantFile(vcf_file_path) as vcf:
        return np.array(list(vcf.header.samples))
    

def get_pcs(X,num_pcs):
    """
    Given input data (samples x features), return PCs of standardized data. 
    """
    mean_vals = X.mean()
    std_vals = X.std()
    X_standardized = (X - mean_vals) / std_vals
    X_standardized = X_standardized.fillna(0) # for not expressed genes where std is 0 
    pca = PCA(n_components=num_pcs)
    prcomp_result = pca.fit_transform(X_standardized)
    pca_df = pd.DataFrame(data=prcomp_result, index=X.index)
    return pca_df


def gtf2df(gtf: str) -> pd.DataFrame:
    """
    Process gencode gtf file into Dataframe.  
    """
    df = pd.read_csv(gtf, sep='\t', header=None, comment='#')
    df.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    fields = ['gene_id', 'transcript_id', 'gene_type', 'gene_name', 'level', 'transcript_name', 'tag']
    for field in fields:
        df[field] = df['attribute'].apply(lambda x: re.findall(rf'{field} "([^"]*)"', x)[0] if rf'{field} "' in x else '')
    df.replace('', np.nan, inplace=True)
    df.drop('attribute', axis=1, inplace=True)
    return df


def init_model_from_ref(model, weights_to_load_model_ckpt_path): 
    """
    Given a model (pSAGEnet or rSAGEnet), load in weights from the layers in conv0, convlayers, dilated_convlayers from the model at weights_to_load_model_ckpt_path. 
    
    Parameters:
    - model: Model of class pSAGEnet or rSAGEnet. 
    - weights_to_load_model_ckpt_path: String to path of model ckpt to use weights from (that model is of class pSAGEnet or rSAGEnet). 
    
    Returns: Model with weights loaded into layers in conv0, convlayers, dilated_convlayers. 
    """
    ref_model = rSAGEnet.load_from_checkpoint(ref_ckpt_path)
    model.conv0.load_state_dict(ref_model.conv0.state_dict())
    for i in range(len(model.convlayers)):
        model.convlayers[i].load_state_dict(ref_model.convlayers[i].state_dict())
    for i in range(len(model.dilated_convlayers)):
        model.dilated_convlayers[i].load_state_dict(ref_model.dilated_convlayers[i].state_dict())
    return model


def get_records_list(chr,pos_start,pos_end,contig_prefix,vcf_file_path='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz',hg38_file_path='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa'):
    """
    Fetches variant records from a VCF file within a specified genomic region.
    
    Parameters:
    - chr: String chromosome identifier (e.g., '1', 'X').
    - pos_start: Int genomic start position. 
    - pos_end: Int genomic end position. 
    - contig_prefix: Prefix for chromosome names in the VCF file (e.g., "chr" or "").
    - vcf_file_path : String Path to the VCF file. 
    - hg38_file_path : String path to the reference genome FASTA file.

    Returns: List of variant records from the VCF file within the specified region.
    """
    with pysam.VariantFile(vcf_file_path, mode="r") as vcf_data, \
         pysam.FastaFile(hg38_file_path) as genome:
        pos_start=max(0,pos_start)
        pos_end = min(genome.get_reference_length(f"chr{chr}"), pos_end)
        vcf_chr = "23" if chr in {'X', 'Y'} and contig_prefix == '' else str(chr)
        records_list = list(vcf_data.fetch(contig_prefix + str(vcf_chr), pos_start, pos_end))
        return records_list

    
def get_variant_info(chr, pos_start,pos_end, subs_list,contig_prefix='',vcf_file_path='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz',hg38_file_path='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa', maf_threshold=-1, train_subs_vcf_path=None,train_subs=None, train_subs_contig_prefix=''): 
    """
    Gets all variant information (filtered by MAF) within a specified genomic region.
    
    Parameters:
      - chr: String chromosome identifier (e.g., '1', 'X').
    - pos_start: Int genomic start position. 
    - pos_end: Int genomic end position. 
    - contig_prefix: Prefix for chromosome names in the VCF file (e.g., "chr" or "").
    - vcf_file_path : String Path to the VCF file. 
    - hg38_file_path : String path to the reference genome FASTA file.
    - maf_threshold: Float MAF threshold (only include variants with MAF>maf_threshold)
    - train_subs_vcf_path: String path to VCF containing genotype data for the list of individuals we are using to calculate MAF. If None, set to vcf_file_path. 
    - train_subs: List of strings giving individual sample IDs for individuals used to calculate MAF. If None, set to subs_list 
    - train_subs_contig_prefix: String before chromosome number in VCF file (for, example ROSMAP VCF uses 'chr1', etc.) in the set of individuals used to calculate MAF. 
    
    Returns: 
    all_features: Numpy array of variant data of shape (num subs, num features). Each entry is 0, 1, or 2. 
    pos: List of integer genomic positions of variants within region. 
    ref: List of string reference allele values of variants within region. 
    alt: List of string alt allele values of variants within region. 
    
    If no variants exist given the specifications, all_features, pos, ref, alt will all be empty lists. 
    """
    if train_subs_vcf_path is None: 
        train_subs_vcf_path=vcf_file_path
    if train_subs is None: 
        train_subs=subs_list
    
    records_list = get_records_list(chr=chr,pos_start=pos_start,pos_end=pos_end,contig_prefix=contig_prefix,vcf_file_path=vcf_file_path,hg38_file_path=hg38_file_path)
    pos, ref, alt, all_features = [], [], [], []
    for record in records_list: 
        features_per_pos = len(record.alts)
        for feature in range(features_per_pos): 
            feature_idx = feature+1
            maf = calc_maf(record,feature_idx, train_subs_vcf_path,train_subs, train_subs_contig_prefix)
            if maf>maf_threshold: 
                pos.append(record.pos)
                ref.append(record.ref)
                alt.append(record.alts[feature])
                feature_info = []
                for sample in subs_list: 
                    feature_info.append(record.samples[sample]["GT"].count(feature_idx))
                all_features.append(feature_info)
    if len(all_features)>0: 
        all_features = np.vstack(all_features).T # (num subs x num features)
    return all_features, pos, ref, alt


def calc_maf(record, record_allele_idx, train_subs_vcf_path, train_subs, train_subs_contig_prefix=''): 
    """
    Calculate the minor allele frequency (MAF) for a given variant and alternative allele index within a list of individuals. 

    Parameters: 
    - record: pysam.VariantRecord containing variant information.  
      From, for example: 
      ```
      records_list = vcf_data.fetch(contig_prefix + str(fasta_chr), pos_start, pos_end)
      record = records_list[0]
      ```
    - record_allele_idx: Integer index of the alternative allele in "GT" (1 or higher).  
    - train_subs_vcf_path: String path to VCF containing genotype data for the list of individuals we are using to calculate MAF. 
    - train_subs: List of strings giving individual sample IDs for individuals used to calculate MAF. 
    - train_subs_contig_prefix: String before chromosome number in VCF file (for, example ROSMAP VCF uses 'chr1', etc.) in the set of individuals used to calculate MAF. 

    Returns: Float, the minor allele frequency (MAF) of the specified allele in the provided individuals (returns 0 if the allele is not found in the reference population). 
    """
    train_subs_vcf = pysam.VariantFile(train_subs_vcf_path, mode='r')
    
    if 'chr' in record.chrom: # for ex, record from GTEx 
        if train_subs_contig_prefix=='': # for ex, train_subs are ROSMAP 
            train_subs_chrom = record.chrom.split('r')[1] # remove 'chr' for searching VCF
    else: 
        train_subs_chrom=record.chrom  
    train_subs_chrom = "23" if train_subs_chrom in {'X', 'Y'} and contig_prefix == '' else str(train_subs_chrom)

    train_subs_genotypes = []
    train_subs_records = [record for record in train_subs_vcf.fetch(train_subs_chrom, record.pos-1, record.pos)]    
    maf=0 # initialize
    if len(train_subs_records)>0:
        # check if the record we are looking for exists in this VCF, these individuals  
        for train_subs_record in train_subs_records: 
            if train_subs_record.pos==record.pos and train_subs_record.ref==record.ref and train_subs_record.alts[record_allele_idx-1]==record.alts[record_allele_idx-1]:
                matching_record = train_subs_record
                train_subs_genotypes = [
                    matching_record.samples[individual]["GT"]
                    for individual in train_subs
                    if matching_record.samples[individual]["GT"] is not None
                ]
                total_alleles = len(train_subs)*2 
                train_subs_genotypes=np.concatenate(train_subs_genotypes)                
                allele_freq = np.count_nonzero(train_subs_genotypes == record_allele_idx)/total_alleles
                allele_freqs = [allele_freq,1-allele_freq]
                maf = np.min(allele_freqs)
    return maf


def get_null_corr(obs,pred):
    """
    Computes null Pearson correlation values between rows of two provided Dataframes by shuffling the labels of one dataframe. 
    Parameters:
    - obs: pd.DataFrame containing observed values, with rows corresponding to different genes, columns corresponding to different samples.
    - pred: pd.DataFrame containing predicted values, with rows corresponding to different genes, columns corresponding to different samples.

    Returns: pd.DataFrame with the same rows as obs and pred, single column containing Pearson correlation coefficient between each row of `obs` and the shuffled `pred`.
    """
    shuffled_labels = np.random.permutation(pred.columns)
    pred.columns = shuffled_labels
    null_corr = pred.corrwith(obs, axis=1).values
    null_corr_res = pd.DataFrame(index=obs.index)
    null_corr_res['pearson']=null_corr
    return null_corr_res


def get_train_val_test_genes(gene_list,tss_data_path='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/gene-ids-and-positions.tsv', use_enformer_gene_assignments=False,enformer_gene_assignments_path=None):
    """
    Sort a given gene list into train, validaiton, and test based on either chromsome split or enformer gene assignments. 
    
    Parameters: 
    - gene_list: List of gene_ids 
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr' and 'gene'. 

    Returns: Tuple of numpy arrays for train, validation, and test genes 
    """
    if use_enformer_gene_assignments: 
        print('selecting train/val/test gene sets based on enformer gene sets')
        assignments_df = pd.read_csv(enformer_gene_assignments_path,index_col=0)
        train_genes = assignments_df[assignments_df['enformer_set']=='train'].index
        val_genes = assignments_df[assignments_df['enformer_set']=='valid'].index
        test_genes = assignments_df[assignments_df['enformer_set']=='valid'].index

    else: 
        print('selecting train/val/test gene sets based on chromosome split')
        train_chrs = np.array(list(range(1,17))).astype(str)
        val_chrs = np.array([17,18,21,22]).astype(str)
        test_chrs = np.array([19, 20]).astype(str)
        gene_meta_info = pd.read_csv(tss_data_path, sep="\t")
        sel_gene_meta_info = gene_meta_info[gene_meta_info['gene_id'].isin(gene_list)]
        train_genes = sel_gene_meta_info[sel_gene_meta_info['chr'].isin(train_chrs)]['gene_id'].values
        val_genes = sel_gene_meta_info[sel_gene_meta_info['chr'].isin(val_chrs)]['gene_id'].values
        test_genes = sel_gene_meta_info[sel_gene_meta_info['chr'].isin(test_chrs)]['gene_id'].values
    return train_genes, val_genes, test_genes


def select_ckpt_path(ckpt_dir,max_epochs=10,best_ckpt_metric='train_gene_gene'):
    """
    Identify "best" model ckpt in a directory based on a given metric. 
    
    Paramters: 
    ckpt_dir: String of directory containing model ckpts and csv files with model metrics. 
    max_epochs: Max epoch of model training to consider when selecting best ckpt. 
    best_ckpt_metric: Metric used to select best model. Can be one of {'train_gene_gene', 'train_gene_sample', 'val_gene_gene', 'val_gene_sample'}
    
    Returns: String of ckpt path of best model 
    """
    print(f'identifying best ckpt from dir {ckpt_dir} based on {best_ckpt_metric}')
    model_corrs = []
    for epoch in range(max_epochs): 
        corr_filename = f'epoch={epoch}_{best_ckpt_metric}_corrs.csv'
        if corr_filename in os.listdir(ckpt_dir): 
            epoch_corr_info = pd.read_csv(f'{ckpt_dir}{corr_filename}',index_col=0)
            model_corrs.append(epoch_corr_info['Correlation'].median())
    best_epoch = np.argmax(model_corrs)
    print(f'best epoch:{best_epoch}')
    ckpt_path = f'{ckpt_dir}epoch={best_epoch}.ckpt'
    return ckpt_path


def mean_center_attributions(attributions,axis=1):
    """
    Mean center attributions by subtracting the mean from each position. 
    
    Paramters: 
    - attributions: Numpy array with model attributions (one value per base pair per position, for ex from ISM or gradients). 
    - axis: Axis along which to take mean, should correspond to taking the mean across the 4 base pairs. 
    
    Returns: Numpy array of mean-centered attributions. 
    """
    print(f'attributions.shape:{attributions.shape}')
    mean = np.mean(attributions, axis=axis, keepdims=True)  
    print(f'mean.shape:{mean.shape}')
    centered = attributions - mean 
    print(f'centered.shape:{centered.shape}')
    return centered
    




