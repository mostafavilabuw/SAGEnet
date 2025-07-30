import numpy as np
import os
import pysam
import pandas as pd
from sklearn.decomposition import PCA
from SAGEnet.models import rSAGEnet

def get_pos_idx_in_seq(gene, pos,tss_data_path='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/gene-ids-and-positions.tsv',input_len=40000,allow_reverse_complement=True):
    """
    Given a gene and a position in that gene, determine the index of the given position in the sequence (from PersonalGenomeDataset, ReferenceDataset, or VariantDataset). 
    
    Parameters: 
    - gene: String gene ENSG id. 
    - pos: Integer position in the gene.
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - input_len: Integer, size of the genomic window for model input. 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
    
    Returns: integer position index of position in sequence. 
    """
    gene_meta_info = pd.read_csv(tss_data_path, sep='\t', index_col='region_id')
    gene_info = gene_meta_info.loc[gene]
    tss_pos = gene_info["tss"]
    pos_start = max(0, tss_pos - input_len // 2)
    pos_idx = pos-pos_start
    if pos_idx<0 or pos_idx>=input_len:
        raise ValueError('pos not in sequence range')
    rc=(gene_info['strand']!='+') # reverse complement genes on the negative strand 
    if rc and allow_reverse_complement: 
        pos_idx=input_len-pos_idx
    return pos_idx


def select_region_set(enet_path, rand_regions, top_regions_to_consider=5000,seed=42, num_regions=1000,region_idx_start=0): 
    """
    Select region set, either based on prediXcan ranking or randomly. 
    - predixcan_res_path: String path to prediXcan results path, to be used to construct ranked region sets. 
    - rand_regions: Boolean indicating whether or not to randomly select regions (from top_regions_to_consider region set) to use in model evaluation. If False, select region set from top-prediXcan ranked regions. 
    - top__regions_to_consider: Integer, length of prediXcan-ranked top region set to consider when randomly selecting regions (only relevant if rand_regions==True). 
    - seed: Integer seed to determine random shuffling of region set. 
    - num_regions: Integer number of genes to select. 
    - region_idx_start: Integer index in prediXcan-ranked region list of first region to use in model evaluation.
    
    Returns: List of regions (each region is a string).
    """
    enet_res = pd.read_csv(enet_path,index_col=0)
    enet_res = enet_res.sort_values(by='val_pearson', ascending=False)

    if rand_regions: 
        print(f'randomly selecting from top {top_regions_to_consider} regions')
        region_options = enet_res.index[:top_regions_to_consider].values
        np.random.seed(seed)
        np.random.shuffle(region_options)
        region_list = region_options[:num_regions]
    else: 
        print(f'selecting regions from {region_idx_start} to {region_idx_start+num_regions}')
        region_list = enet_res.index[region_idx_start:region_idx_start+num_regions].values
        
    return region_list


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
    X_standardized = X_standardized.fillna(0) # for genes where std is 0 
    pca = PCA(n_components=num_pcs)
    prcomp_result = pca.fit_transform(X_standardized)
    pca_df = pd.DataFrame(data=prcomp_result, index=X.index)
    return pca_df


def gtf2df(gtf: str) -> pd.DataFrame:
    """
    Given string path to gencode gtf file, return Dataframe containing infomation in the gtf file.  
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
    - weights_to_load_model_ckpt_path: String to path of rSAGEnet model ckpt to use weights from. 
    
    Returns: Model with weights loaded into layers in conv0, convlayers, dilated_convlayers. 
    """
    ref_model = rSAGEnet.load_from_checkpoint(weights_to_load_model_ckpt_path)
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
        vcf_chr = "23" if chr in {'X', 'Y'} and contig_prefix == '' else str(chr) # adjust chr label for ROSMAP VCF
        records_list = list(vcf_data.fetch(contig_prefix + str(vcf_chr), pos_start, pos_end))
        return records_list

    
def get_variant_info(chr, pos_start,pos_end, subs_list,vcf_file_path,hg38_file_path, contig_prefix='', maf_min=-1, maf_max=2,train_subs_vcf_path=None,train_subs=None, train_subs_contig_prefix='',record_from_train_subs_vcf=True,determine_minor_allele_by_reference=True): 
    """
    Gets all variant information (filtered by MAF) within a specified genomic region.
    
    Parameters:
    - chr: String chromosome identifier (e.g., '1', 'X').
    - pos_start: Int genomic start position. 
    - pos_end: Int genomic end position. 
    - subs_list: List of individuals to get variant info for. 
    - hg38_file_path: String path to the human genome (hg38) reference file (hg19 can also be used). 
    - contig_prefix: String before chromosome number in VCF file (for, example ROSMAP VCF uses '1' vs. GTEx uses 'chr1', 
        so for ROSMAP set contig_prefix='', for GTEx set contig_prefix='chr'. Only used if majority_seq=True. 
    - vcf_file_path : String Path to the VCF file. 
    - maf_threshold: Float MAF threshold (only include variants with MAF>maf_threshold)
    - maf_min: Float indicating the threshold MAF min. If less than zero, does not prevent any variants from being included. 
    - maf_max: Float indicating the threshold MAF max. If greater than 1, does not prevent any variants from being included. 
    - train_subs_vcf_path: String path to VCF containing genotype data for the list of individuals to be used to calculate MAF. If None, set to vcf_file_path. 
    - train_subs: List of (string) individual sample IDs for individuals used to calculate MAF. If None, set to subs_list 
    - train_subs_contig_prefix: String before chromosome number in VCF file (e.g., "chr" or "") in the set of individuals used to calculate MAF. By default, set to "" (ROSMAP contig prefix).
    - record_from_train_subs_vcf: Boolean indiciating whether or not the individuals used to calculate MAF (training individuals) are from the same VCF 
        as the individuals we are getting variant info for. True if they are the same, False is not. If True, this function runs faster. 
    - determine_minor_allele_by_reference: Boolean, whether or not to determine MAF based on "which allele is not in the referene allele" (True) or 
        "which allele is less common among the training individuals" (False).
    
    Returns: 
    all_features: DataFrame of variant data of shape (num_subs, num_features), indexed but individuals. Each entry is 0, 1, or 2. 
    pos: List of integer genomic positions of variants within region (length len(num_features)). 
    ref: List of string reference allele values of variants within region (length len(num_features)). 
    alt: List of string alt allele values of variants within region (length len(num_features)). 
    
    If no variants exist given the specifications, all_features, pos, ref, alt will all be empty lists. 
    """
    insert_all_variants = maf_min < 0 and maf_max > 1 # no MAF constraints 

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

            variant_passes_maf_filter=True
            if not insert_all_variants: # maf is supplied: 
                maf = calc_maf(record,feature_idx, train_subs_vcf_path,train_subs, train_subs_contig_prefix,record_from_train_subs_vcf=record_from_train_subs_vcf,determine_minor_allele_by_reference=determine_minor_allele_by_reference)
                if maf<=maf_min or maf>=maf_max:
                    variant_passes_maf_filter=False
            if variant_passes_maf_filter: # maf not supplied or maf calculated is > threshold 
                pos.append(record.pos)
                ref.append(record.ref)
                alt.append(record.alts[feature])
                feature_info = []
                for sample in subs_list: 
                    feature_info.append(record.samples[sample]["GT"].count(feature_idx))
                all_features.append(feature_info)
    if len(all_features)>0: 
        all_features = np.vstack(all_features).T # (num_subs x num_features)
        all_features=pd.DataFrame(index=subs_list,data=all_features)
    return all_features, pos, ref, alt


def calc_maf(record, record_allele_idx, train_subs_vcf_path, train_subs, train_subs_contig_prefix='',record_from_train_subs_vcf=True,determine_minor_allele_by_reference=True): 
    """
    Calculate the minor allele frequency (MAF) for a given variant and alternative allele index for a list of individuals. 

    Parameters: 
    - record: pysam.VariantRecord containing variant information.  
      From, for example: 
      ```
      records_list = vcf_data.fetch(contig_prefix + str(fasta_chr), pos_start, pos_end)
      record = records_list[0]
      ```
    - record_allele_idx: Integer index of the alternative allele in "GT" (1 or higher).  
    - train_subs_vcf_path: String path to VCF containing genotype data for the list of individuals we are using to calculate MAF. 
    - train_subs: List of (string) individual sample IDs for individuals used to calculate MAF.
    - train_subs_contig_prefix: String before chromosome number in VCF file (e.g., "chr" or "") in the set of individuals used to calculate MAF. By default, set to "" (ROSMAP contig prefix).
    - record_from_train_subs_vcf: Boolean indiciating whether or not the individuals used to calculate MAF (training individuals) are from the same VCF 
            as the individuals we are getting variant info for. True if they are the same, False is not. If True, this function runs faster. 
    - determine_minor_allele_by_reference: Boolean, whether or not to determine MAF based on "which allele is not in the referene allele" (True) or 
        "which allele is less common among the training individuals" (False).
    
    Returns: Float, the minor allele frequency (MAF) of the specified allele in the provided individuals (returns 0 if the allele is not found in the reference population). 
    """
    maf=0 
    matching_record=None
    if record_from_train_subs_vcf: # same VCF for record and train subs, so don't need to find the matching record  -- speed up 
        matching_record=record 
    else: 
        train_subs_vcf = pysam.VariantFile(train_subs_vcf_path, mode='r')
        
        if 'chr' in record.chrom: # record from GTEx 
            if train_subs_contig_prefix=='': # train_subs are ROSMAP 
                train_subs_chrom = record.chrom.split('r')[1] # remove 'chr' for searching VCF
            else: 
                train_subs_chrom=record.chrom  
        else: 
            train_subs_chrom=record.chrom  
        
        train_subs_chrom = "23" if train_subs_chrom in {'X', 'Y'} and train_subs_contig_prefix == '' else str(train_subs_chrom)
        train_subs_genotypes = []
        train_subs_records = [record for record in train_subs_vcf.fetch(train_subs_chrom, record.pos-1, record.pos)]    
        
        # check if the record we are looking for exists in this VCF, these individuals  
        if len(train_subs_records)>0:
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
    if determine_minor_allele_by_reference: 
        maf=allele_freq
    else: 
        maf=min(allele_freq,1-allele_freq)
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
    - use_enformer_gene_assignments: Boolean, whether to use gene splits from enformer_gene_assignments_path (or based on chromosome).
    - enformer_gene_assignments_path: String path to DataFrame containing Enformer gene split assignments. Only relevant if use_enformer_gene_assignments==True. 

    Returns: Tuple of numpy arrays containing train, validation, and test genes 
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
        gene_meta_info = pd.read_csv(tss_data_path, sep='\t', index_col='region_id')
        sel_gene_meta_info = gene_meta_info.loc[gene_list]
        train_genes = sel_gene_meta_info[sel_gene_meta_info['chr'].isin(train_chrs)].index.values
        val_genes = sel_gene_meta_info[sel_gene_meta_info['chr'].isin(val_chrs)].index.values
        test_genes = sel_gene_meta_info[sel_gene_meta_info['chr'].isin(test_chrs)].index.values
    return train_genes, val_genes, test_genes

def select_ckpt_path(ckpt_dir,max_epochs=5,best_ckpt_metric='train_region_region'):
    """
    Identify "best" model ckpt in a directory based on a given metric. 
    
    Paramters: 
    ckpt_dir: String of directory containing model ckpts and csv files with model metrics. 
    max_epochs: Max epoch of model training to consider when selecting best ckpt. 
    best_ckpt_metric: Metric used to select best model. Can be one of {'train_gene_gene', 'train_gene_sample', 'val_gene_gene', 'val_gene_sample'}. 
    
    Returns: String of ckpt path of best model 
    """
    if best_ckpt_metric=='last_epoch':
        print('selecting last model epoch ckpt')
        epochs = []
        for filename in os.listdir(ckpt_dir):
            if filename[-4:]=='ckpt' and filename[:5]=='epoch':
                epoch=int(filename[-6])
                epochs.append(epoch)
        best_epoch = np.max(epochs)
    else: 
        print(f'identifying best ckpt from dir {ckpt_dir} based on {best_ckpt_metric}')
        model_corrs = []
        for epoch in range(max_epochs): 
            corr_filename = f'epoch={epoch}_{best_ckpt_metric}_corrs.csv'
            if corr_filename in os.listdir(ckpt_dir): 
                epoch_corr_info = pd.read_csv(f'{ckpt_dir}{corr_filename}',index_col=0)
                median_corr=epoch_corr_info['Correlation'].fillna(0).median()
                if np.isnan(median_corr):
                    median_corr = 0
                model_corrs.append(median_corr)
            else: 
                print(f'{corr_filename} not found in {ckpt_dir}')
        best_epoch = np.argmax(model_corrs)
    print(f'best epoch:{best_epoch}')
    ckpt_path = f'{ckpt_dir}epoch={best_epoch}.ckpt'
    return ckpt_path


def zero_center_attributions(attributions,axis=1):
    """
    Zero-center attributions by subtracting the mean from each position. 
    
    Paramters: 
    - attributions: Numpy array with model attributions (one value per base pair per position, for ex from ISM or gradient). 
    - axis: Axis along which to take mean, should correspond to taking the mean across the 4 base pairs. 
    
    Returns: Numpy array of zero-centered attributions. 
    """
    if attributions.shape[axis]!=4: 
        raise ValueError(f'attributions.shape[axis]={attributions.shape[axis]} (must equal 4)')
    mean = np.mean(attributions, axis=axis, keepdims=True)  
    centered = attributions - mean 
    return centered
    
    




