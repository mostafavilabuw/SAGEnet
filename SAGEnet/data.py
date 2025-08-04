import torch
from torch.utils.data import Dataset
import pysam
import numpy as np
import SAGEnet.tools

def onehot_encoding(
    sequence: str,
    length: int,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value=0.25,
    dtype=np.float32,
    reverse_complement: bool = False,
    allow_reverse_complement: bool = True # if this is set to False, the sequence will not be reverse complemented, even if reverse_complement==True
) -> np.ndarray:
    """ One-hot encode sequence. """
    sequence=sequence.upper()
    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)
    if allow_reverse_complement and reverse_complement:
        # define complement mapping
        complement = str.maketrans("ACGT", "TGCA")
        # translate the sequence to its complement and then reverse it
        reversed_complement = sequence.translate(complement)[::-1]
        # assign the reversed complement to the sequence variable
        sequence = reversed_complement
    encoded_sequence = neutral_value * np.ones((length, len(alphabet)), dtype=dtype)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    encoded_sequence[: len(sequence)] = hash_table[to_uint8(sequence)]
    return encoded_sequence.T


def get_seq_start_and_end(chr, center_pos, input_len, hg38_file_path):
    """ Get the start and end of a genomic region given a center position (deal with cases where the window would extend beyond the start/end of the chromosome). """
    genome = pysam.FastaFile(hg38_file_path)
    pos_start = max(0, center_pos - input_len // 2)
    pos_end = min(genome.get_reference_length(f"chr{chr}"), center_pos + input_len // 2)
    return pos_start, pos_end


def get_reference_sequence(chr, pos_start, pos_end, majority_seq,hg38_file_path, vcf_file_path=None, subs_to_consider=None, contig_prefix=''):
    """ Get reference sequence -- either the majority sequence of a set of individuals, or just the reference sequence. """
    input_len=pos_end-pos_start
    genome = pysam.FastaFile(hg38_file_path)
    seq = genome.fetch(f"chr{chr}", pos_start, pos_end).upper()
    if majority_seq: 
        records_list = SAGEnet.tools.get_records_list(chr, pos_start, pos_end, contig_prefix=contig_prefix,vcf_file_path=vcf_file_path,hg38_file_path=hg38_file_path)
        passing_records =[]
        passing_records_alt_idxs = []
        for record in records_list: 
            features_per_pos = len(record.alts)
            for feature in range(features_per_pos): 
                feature_idx = feature+1
                maf = SAGEnet.tools.calc_maf(record,feature_idx, vcf_file_path,subs_to_consider, train_subs_contig_prefix=contig_prefix,record_from_train_subs_vcf=True,determine_minor_allele_by_reference=True)
                if maf>.5: 
                    passing_records.append(record)
                    passing_records_alt_idxs.append(feature)
        seq = modify_ref_seq_from_records_list(seq, passing_records, passing_records_alt_idxs, pos_start)[0:input_len]
    return seq 


def modify_ref_seq_from_records_list(seq, records_list, records_alt_idxs, seq_start_pos,verbose=True):
    """ 
    Modify reference sequence provided with records list provided (and the corresponding alt allele idxs)
    Parameters:
    - seq: String reference sequence to modify. 
    - records_list: List of variant records from VCF file to insert. 
    - records_alt_idxs: List of indices specifying which alternate allele to use for each record in records_list. Usually allele idxs are 0, but they can be greater 
        (if there is more than one possible alt allele at a given position).
    - seq_start_pos: Integer of genomic position that correpsonds to the start of the seq provided.     
    """
    shift=0
    for record_idx, record in enumerate(sorted(records_list, key=lambda r: r.pos)):

        position= record.pos - seq_start_pos + shift - 1
        ref_allele = record.ref.upper()
        allele_index=records_alt_idxs[record_idx] # usually 0 but can be greater if multiple alts at single pos 
        alt_allele=record.alts[allele_index].upper()

        seq_before_variant = seq[:position]
        seq_after_variant = seq[position + len(ref_allele) :]

        if seq[position : position + len(ref_allele)] != ref_allele:
            if verbose:
                print(f"warning: reference allele at position idx {position} (position {record.pos}) ({ref_allele}) does not match the sequence ({seq[position : position + len(ref_allele)]}) \n this is likely due to overlapping variant positions; you can use bcftools plugin remove-overlaps to remove these records from your VCF")

        seq = seq_before_variant + alt_allele + seq_after_variant
        shift += len(alt_allele) - len(ref_allele)
    return seq


def modify_personal_seqs_from_records_list(seq, records_list, sample_of_interest, seq_start_pos,only_snps, maf_min, maf_max,train_subs_vcf_file_path,train_subs,verbose=True):
    """
    Modifies maternal and paternal DNA sequences based on variant information. Returns tuple containing maternal sequence and paternal sequence. 
    """
    maternal_sequence = paternal_sequence = seq.upper()
    maternal_shift = paternal_shift = 0

    for record in sorted(records_list, key=lambda r: r.pos):
        GT = record.samples[sample_of_interest]["GT"]
        for idx, seq in enumerate([maternal_sequence, paternal_sequence]):
            shift = maternal_shift if idx == 0 else paternal_shift
            position = record.pos - seq_start_pos + shift - 1
            ref_allele = record.ref.upper()

            try:
                allele = GT[idx]
                if allele is None:
                    allele_index = 0
                else:
                    allele_index = int(allele)
            except IndexError:
                allele_index = 0  # fallback if GT has only one allele
                
            alt_allele = (
                ref_allele
                if allele_index == 0
                else record.alts[allele_index - 1].upper()
            )
            
            if allele_index!=0: # we do not need to do anything if the allele index indicates its ref seq
                # consider whether to insert 

                # based on only_snps and maf_threshold, decide whether or not to insert a given variant
                insert_variant=True
                
                if only_snps: 
                    if len(alt_allele)!=1 or len(ref_allele)!=1: 
                        insert_variant=False
                if maf_min>=0 or maf_max<=1: # maf is supplied 
                    curr_maf = SAGEnet.tools.calc_maf(record, allele_index, train_subs_vcf_file_path, train_subs)
                    if curr_maf <= maf_min or curr_maf >= maf_max:
                        insert_variant=False
                        
                if insert_variant: 
                    seq_before_variant = seq[:position]
                    seq_after_variant = seq[position + len(ref_allele) :]

                    if seq[position : position + len(ref_allele)] != ref_allele:
                        if verbose:
                            print(
                                f"warning: reference allele at position idx {position} (position {record.pos}) ({ref_allele}) does not match the sequence for {'maternal' if idx == 0 else 'paternal'} ({seq[position : position + len(ref_allele)]}) \n this is likely due to overlapping variant positions; you can use bcftools plugin remove-overlaps to remove these records from your VCF"
                            )
                        continue

                    seq = seq_before_variant + alt_allele + seq_after_variant
                    shift += len(alt_allele) - len(ref_allele)
                    
            if idx == 0:
                maternal_sequence = seq
                maternal_shift = shift
            else:
                paternal_sequence = seq
                paternal_shift = shift

    return maternal_sequence, paternal_sequence


def get_personal_tensor(sample_of_interest, region, metadata, input_len, hg38_file_path, contig_prefix, vcf_file_path, train_subs, train_subs_vcf_file_path, only_snps, maf_min,maf_max,allow_reverse_complement,verbose):
    '''
    Gets maternal and paternal DNA sequences based on variant information for a given individual. Returns these sequences as stacked one-hot encoded tensors.
    '''
    region_info = metadata.loc[region]
    chr = region_info["chr"]
    pos = region_info["pos"]
    pos_start, pos_end = get_seq_start_and_end(chr, pos, input_len, hg38_file_path=hg38_file_path)

    all_records_list = SAGEnet.tools.get_records_list(chr, pos_start, pos_end, contig_prefix=contig_prefix,vcf_file_path=vcf_file_path,hg38_file_path=hg38_file_path)
    records_list = [record for record in all_records_list if record.samples[sample_of_interest]["GT"] != (0, 0) and record.samples[sample_of_interest]["GT"] != (None, None)]                       
    ref_sequence_for_personal=get_reference_sequence(chr, pos_start, pos_end, majority_seq=False,subs_to_consider=train_subs,hg38_file_path=hg38_file_path,contig_prefix=contig_prefix,vcf_file_path=train_subs_vcf_file_path) 
    
    maternal_seq, paternal_seq = modify_personal_seqs_from_records_list(ref_sequence_for_personal, records_list, sample_of_interest, pos_start,only_snps=only_snps, maf_min=maf_min, maf_max=maf_max,train_subs_vcf_file_path=train_subs_vcf_file_path,train_subs=train_subs,verbose=verbose)
    maternal_seq = maternal_seq[0 : input_len]
    paternal_seq = paternal_seq[0 : input_len]

    # one-hot encode the sequences
    if 'strand' in  metadata.columns: 
        rc=(region_info['strand']!='+') # reverse complement regions on the negative strand 
    else: 
        rc=False

    maternal_encoded = onehot_encoding(maternal_seq, input_len, reverse_complement=rc,allow_reverse_complement=allow_reverse_complement)
    paternal_encoded = onehot_encoding(paternal_seq, input_len, reverse_complement=rc,allow_reverse_complement=allow_reverse_complement)
    
    personal_sequence_tensor = torch.from_numpy(
        np.concatenate((maternal_encoded, paternal_encoded), axis=0)
    )
    return personal_sequence_tensor


class ReferenceGenomeDataset(Dataset):
    def __init__(
        self,
        metadata,
        hg38_file_path,
        y_data=None,
        input_len=40000,
        allow_reverse_complement=True,
        single_seq=True,
        pos_start=None,
        pos_end=None,
        majority_seq=True,
        vcf_file_path=None,
        train_subs=None,
        contig_prefix='',
        median_or_mean_y_data='median',
    ):
        """
        Initialize the ReferenceGenomeDataset object. 

        Parameters:
        - hg38_file_path: String path to the human genome (hg38) reference file.
        - input_len: Integer, size of the genomic window for model input. 
        - allow_reverse_complement: Boolean, whether or not to reverse complement regions on the negative strand. 
        - single_seq: Boolean, if True, return output of the shape of a ReferenceGenomeDataset datapoint, if False, return output of the shape of a PersonalGenomeDataset datapoint
        - pos_start: Integer, genome position start. If None, set to max(0, pos - self.input_len // 2)
        - pos_end: Integer, genome position start. If None, set to min(self.genome.get_reference_length(f"chr{chr}"), pos + self.input_len // 2)
        """
        
        self.metadata = metadata
        self.input_len = input_len
        self.y_data = y_data
        self.allow_reverse_complement=allow_reverse_complement
        self.genome = pysam.FastaFile(hg38_file_path)
        self.hg38_file_path=hg38_file_path
        self.single_seq=single_seq
        self.pos_start=pos_start
        self.pos_end=pos_end

        # for majority seq 
        self.majority_seq=majority_seq
        self.vcf_file_path=vcf_file_path
        self.train_subs=train_subs
        self.contig_prefix=contig_prefix
        self.median_or_mean_y_data=median_or_mean_y_data

    def __len__(self):
        return len(self.metadata) 

    def __getitem__(self, idx):
        """
        Retrieve and process data for a specific region by index.
        
        Returns: 
        One-hot-encoded reference sequence, mean y_data value (or placeholder 0 if no y_data is provided)
        """        
        # retrieve region information and sample identifier
        region_info = self.metadata.iloc[idx]
        chr = region_info["chr"]
        pos = region_info["pos"]
        
        if self.pos_end is not None: 
            if self.pos_end>self.genome.get_reference_length(f"chr{chr}"): 
                raise ValueError(f"provided {pos_end} out of chromosome range")
        
        if self.pos_start is None or self.pos_end is None: 
            pos_start, pos_end = get_seq_start_and_end(chr, pos, self.input_len, hg38_file_path=self.hg38_file_path)

        ref_sequence=get_reference_sequence(chr, pos_start, pos_end, majority_seq=self.majority_seq,subs_to_consider=self.train_subs,hg38_file_path=self.hg38_file_path,contig_prefix=self.contig_prefix,vcf_file_path=self.vcf_file_path) 

        # one-hot encode the sequences
        if 'strand' in  self.metadata.columns: 
            rc=(region_info['strand']!='+') # reverse complement regions on the negative strand 
        else: 
            rc=False
        ref_encoded = onehot_encoding(ref_sequence, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)

        # get y_data data, if provided. if not, use 0 as placeholder (for example, in model evaluation) 
        if self.y_data is None: 
            avg_y_data=np.array(0)
        else: 
            if self.median_or_mean_y_data=='mean':
                avg_y_data = np.array(self.y_data.loc[region_info.name,self.train_subs].mean())
            elif self.median_or_mean_y_data=='median':
                avg_y_data = np.array(np.median(np.array(self.y_data.loc[region_info.name,self.train_subs])))

        if self.single_seq: 
            return torch.from_numpy(ref_encoded).float(), torch.from_numpy(avg_y_data).float()

        else: # shape to be input to pSAGEnet
            ref_sequence_tensor = torch.from_numpy(
                np.concatenate((ref_encoded, ref_encoded), axis=0)
            )
            personal_sequence_tensor = torch.from_numpy(
                np.concatenate((ref_encoded, ref_encoded), axis=0)
            )
            # stack personal and reference sequence tensors
            all_seq_tensor = torch.stack(
                (ref_sequence_tensor, personal_sequence_tensor), dim=0
            )
            # placeholder y_data tensor 
            y_data_tensor = torch.from_numpy(np.stack((avg_y_data, avg_y_data), axis=0))

            # include placeholder region_idx, sample_idx 
            return all_seq_tensor.float(), y_data_tensor.float(), 0, 0


class VariantDataset(Dataset):
    def __init__(
        self,
        metadata,
        hg38_file_path,
        variant_info,
        input_len=40000,
        allow_reverse_complement=True,
        insert_variants=True,
        single_seq=False,
        pos_start=None,
        pos_end=None
    ):
        """
        Initialize the VariantDataset object.
        Insert variant from variant_info into reference sequence (if insert_variants==True).

        Parameters:
        - metadata: DataFrame containing genome region-related information, specifically the columns 'chr', 'pos', and, optionally 'strand'. 'pos' should 
            be the center of the region -- i.e., TSS for gene expression. Index should be region ID (for example, gene ENSG).        
        - hg38_file_path: String path to the human genome (hg38) reference file (hg19 can also be used). 
        - variant_info: DataFrame containing variant information, specifially the columns 'region_id', 'chr', 'pos', 'ref', and 'alt'. 
        - input_len: Integer, size of the genomic window for model input. 
        - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
        - insert_variants: Boolean, if True, insert variant, if False, do not insert variant (predict from reference sequence). 
        - single_seq: Boolean, if True, return output of the shape of a ReferenceGenomeDataset datapoint, if False, return output of the shape of a PersonalGenomeDataset datapoint
        - pos_start: Integer, genome position start. If None, set to max(0, pos - self.input_len // 2)
        - pos_end: Integer, genome position start. If None, set to min(self.genome.get_reference_length(f"chr{chr}"), pos + self.input_len // 2)
        """
        
        self.hg38_file_path = hg38_file_path
        self.input_len = input_len
        self.allow_reverse_complement=allow_reverse_complement
        self.metadata =metadata.set_index('region_id', drop=False)
        self.insert_variants=insert_variants
        self.single_seq=single_seq
        self.variant_info=variant_info
        self.pos_start=pos_start
        self.pos_end=pos_end

        if self.insert_variants:
            print('inserting variants')
        else: 
            print('not inserting variants')
    
    def __len__(self):
        return len(self.variant_info) 

    def __getitem__(self, idx):
        """
        Retrieve and process data for a specific variant by index.
        
        Returns: 
        One-hot-encoded sequence, either in the shape of a ReferenceGenomeDataset datapoint (single_seq==True), or in the shape of a PersonalGenomeDataset datapoint (single_seq==False). For single_seq==False, the variant (if inserted) is inserted into both haplotypes. Y values are placeholders (0s). 
        """                
        self.genome = pysam.FastaFile(self.hg38_file_path) # important to initialize for each datapoint to avoid errors when num_workers>1
        
        # get variant info 
        variant_info = self.variant_info.iloc[idx]
        variant_region_id = variant_info['region_id']
        variant_pos = variant_info['pos']
        variant_ref = variant_info['ref']
        variant_alt = variant_info['alt']
        variant_chr = variant_info['chr']
        
        # use region_id in variant info to get metadata 
        region_info = self.metadata.loc[variant_region_id]
        chr = region_info["chr"]
        pos = region_info["pos"]
        
        if pos_end>self.genome.get_reference_length(f"chr{chr}"): 
            raise ValueError(f"provided {pos_end} out of chromosome range")
        
        if self.pos_start is None or self.pos_end is None: 
            pos_start, pos_end = get_seq_start_and_end(chr, pos, self.input_len, hg38_file_path=self.hg38_file_path)

        ref_sequence=get_reference_sequence(chr, pos_start, pos_end, majority_seq=False,hg38_file_path=self.hg38_file_path) 
        
        # check to make sure chromsome in variant info matches chromosome in region info 
        if str(chr)!=str(variant_chr): 
            raise ValueError(f"variant chromsome ({variant_chr}) != region chromosome ({chr})")
        if variant_pos not in range(pos_start,pos_end):
            raise ValueError(f"variant position {variant_pos} out of range({pos_start},{pos_end})")
        
        # get idx of variant position based on sequence start 
        adjusted_variant_pos = variant_pos-pos_start-1
        
        # check that the reference sequence in variant info matches what is found in the reference sequence 
        if (ref_sequence[adjusted_variant_pos:adjusted_variant_pos+len(variant_ref)]).upper()!=variant_ref.upper(): 
            raise ValueError(f"ref_sequence[adjusted_variant_pos:adjusted_variant_pos+len(variant_ref)] ({(ref_sequence[adjusted_variant_pos:adjusted_variant_pos+len(variant_ref)]).upper()}) != variant_ref ({variant_ref.upper()})")
        
        # insert variant 
        ref_with_variant_inserted = ref_sequence[:adjusted_variant_pos]+variant_alt+ref_sequence[adjusted_variant_pos+len(variant_alt):]
    
        # one-hot encode the sequences
        if 'strand' in  self.metadata.columns: 
            rc=(region_info['strand']!='+') # reverse complement regions on the negative strand 
        else: 
            rc=False
        ref_encoded = onehot_encoding(ref_sequence, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)
        ref_with_variant_inserted_encoded = onehot_encoding(ref_with_variant_inserted, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)

        if self.single_seq: 
            if self.insert_variants: 
                return torch.from_numpy(ref_with_variant_inserted_encoded), torch.from_numpy(np.array(0)).float() # use placeholder y tensor 
            else: 
                return torch.from_numpy(ref_encoded), torch.from_numpy(np.array(0)).float()

        else: 
            ref_sequence_tensor = torch.from_numpy(
                np.concatenate((ref_encoded, ref_encoded), axis=0)
            )
            if self.insert_variants: 
                personal_sequence_tensor = torch.from_numpy(
                    np.concatenate((ref_with_variant_inserted_encoded, ref_with_variant_inserted_encoded), axis=0)
                )
            else: 
                personal_sequence_tensor = torch.from_numpy(
                    np.concatenate((ref_encoded, ref_encoded), axis=0)
                )
            # stack personal and reference sequence tensors
            all_seq_tensor = torch.stack(
                (ref_sequence_tensor, personal_sequence_tensor), dim=0
            )
            # placeholder y tensor 
            y_tensor = torch.from_numpy(np.stack((0, 0), axis=0))

            # include placeholder region_idx, sample_idx 
            return all_seq_tensor.float(), y_tensor.float(), 0, 0
        
        
class PersonalGenomeDataset(Dataset):
    def __init__(
        self,
        metadata,
        vcf_file_path,
        hg38_file_path,
        sample_list,
        y_data=None,
        input_len=40000,
        contig_prefix="",
        split_y_data=True,
        y_data_zscore=None,
        train_subs=None,
        only_snps=0,
        maf_min=-1,
        maf_max=2,
        train_subs_vcf_file_path=None,
        train_subs_y_data=None,
        allow_reverse_complement=True,
        only_personal=False,
        median_or_mean_y_data='mean',
        majority_seq=False,
        verbose=False
    ):
        """
        Initialize the PersonalGenomeDataset object.

        Parameters:
        - metadata: DataFrame containing genome region-related information, specifically the columns 'chr', 'pos', and, optionally 'strand'. 'pos' should 
            be the center of the region -- i.e., TSS for gene expression. Index should be region ID (for example, gene ENSG). 
        - vcf_file_path: String path to the VCF file with variant information.
        - hg38_file_path: String path to the human genome (hg38) reference file (hg19 can also be used). 
        - sample_list: List of sample names (as they appear in VCF) to include in dataset. 
        - y_data: DataFrame with y data, indexed by region names, with sample names as columns.
                If y_data is not provided, 0 will be used as a placeholder y value. 
        - input_len: Integer, size of the genomic window for model input. 
        - contig_prefix: String before chromosome number in VCF file (for, example ROSMAP VCF uses 'chr1', etc.) 
        - split_y_data: Boolean indicating if y_data is to be decomposed into mean and difference from mean. If True, idx 1 of the y output represents difference from mean y (either z-score or not). If False, idx 1 of the y output represents personal y (used to train non-contrastive model). 
        - y_data_zscore: DataFrame in the same format as y_data (indexed by region names, sample names as columns) with pre calculated zscores. If provided, zscores are used for model output idx 1 instead of (personal y - mean y). 
        - train_subs: List of training individuals (to be used for calculating mean y and MAF). If not provided, train_subs is set to sample_list. 
        - only_snps: Boolean indicating if only SNPs should be inserted or if all variants (including indels) should be inserted. 
        - maf_min: Float indicating the threshold MAF min for a variant to be inserted. If less than zero, does not prevent any variants from being inserted. 
        - maf_max: Float indicating the threshold MAF max for a variant to be inserted. If greater than 1, does not prevent any variants from being inserted. 
        - train_subs_vcf_file_path: String path to the VCF file for train_subs. If not provided, train_subs_vcf_file_path is set to vcf_file_path. 
        - train_subs_y_data: DataFrame with y data for train_subs. If not provided, train_subs_y_data is set to y_data. 
        - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand. 
        - only_personal: Boolean, if True, only return personal sequence (no reference).
        - majority_seq: Boolean, if True, use the "majority sequence" of the training individuals as reference sequence. If false, just use the reference sequence. 
            "majority sequence" means that for each variant in the training individuals, if a majority of training individuals have the variant, it is inserted.
        - median_or_mean_y_data: String, whether to set the y value paired with the reference sequence to be the mean or the median of the y values provided. 
        """
     
        self.metadata = metadata
        self.vcf_file_path = vcf_file_path
        self.hg38_file_path = hg38_file_path
        self.sample_list = sample_list
        self.n_samples = len(self.sample_list)
        self.input_len = input_len
        self.y_data = y_data 
        self.y_data_zscore = y_data_zscore
        self.contig_prefix = contig_prefix
        self.split_y_data=split_y_data
        self.only_snps=only_snps
        self.maf_min=maf_min
        self.maf_max=maf_max
        self.allow_reverse_complement=allow_reverse_complement
        self.only_personal=only_personal
        self.median_or_mean_y_data=median_or_mean_y_data
        self.majority_seq=majority_seq
        self.verbose=verbose
        
        if train_subs_y_data is None: 
            self.train_subs_y_data=y_data
        else: 
            self.train_subs_y_data = train_subs_y_data
        
        if train_subs is None: 
            self.train_subs = sample_list
        else: 
            self.train_subs=train_subs
            
        if train_subs_vcf_file_path is None: 
            self.train_subs_vcf_file_path=vcf_file_path
        else: 
            self.train_subs_vcf_file_path=train_subs_vcf_file_path
        
        if self.y_data_zscore is not None: 
            print('using y_data zscores')
        if not self.split_y_data: 
            print('not splitting y_data')
        print(f'acceptable maf range: {maf_min}<maf<{maf_max}')
        print(f'avg is {median_or_mean_y_data}')

    def __len__(self):
        """
        Return the total count of region-sample pairs.
        """
        return len(self.metadata) * len(self.sample_list)

    def __getitem__(self, idx):
        """
        Retrieve and process data for a specific region-sample pair by index.
        Returns:
        Tuple containing (reference sequence,personal sequence), (mean y_data, difference from mean y_data), region_idx, sample_idx 
        """
        # calculate region and sample index

        #self.n_vars_inserted=0
        #self.unmatched_count=0

        region_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples
        region_info = self.metadata.iloc[region_idx]
        region=region_info.name
        sample_of_interest = self.sample_list[sample_idx]

        chr = region_info["chr"]
        pos = region_info["pos"]

        pos_start, pos_end = get_seq_start_and_end(chr, pos, self.input_len, hg38_file_path=self.hg38_file_path)
        ref_sequence=get_reference_sequence(chr, pos_start, pos_end, majority_seq=self.majority_seq,subs_to_consider=self.train_subs,hg38_file_path=self.hg38_file_path,contig_prefix=self.contig_prefix,vcf_file_path=self.train_subs_vcf_file_path) 
        if 'strand' in  self.metadata.columns: 
            rc=(region_info['strand']!='+') # reverse complement regions on the negative strand 
        else: 
            rc=False
        ref_encoded = onehot_encoding(ref_sequence, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)

        personal_sequence_tensor = get_personal_tensor(sample_of_interest, region,self.metadata, self.input_len, self.hg38_file_path,self.contig_prefix,self.vcf_file_path,self.train_subs,self.train_subs_vcf_file_path,self.only_snps,self.maf_min,self.maf_max,self.allow_reverse_complement,verbose=self.verbose)

        if self.y_data is not None: 
        
            # get and process y_data
            if self.median_or_mean_y_data=='mean':
                avg_y_data = np.array(self.train_subs_y_data.loc[region_info.name,self.train_subs].mean())
            elif self.median_or_mean_y_data=='median':
                avg_y_data = np.median(np.array(self.train_subs_y_data.loc[region_info.name,self.train_subs]))

            curr_y_data = np.array(
                self.y_data.loc[region_info.name][sample_of_interest])

            if self.y_data_zscore is None: 
                curr_y_data_diff = curr_y_data - avg_y_data
            else: 
                curr_y_data_diff = np.array(
                    self.y_data_zscore.loc[region_info.name][sample_of_interest]
                )
     
            if self.split_y_data: 
                y_data_tensor = torch.from_numpy(
                    np.stack((avg_y_data, curr_y_data_diff), axis=0)
                )
            else: 
                y_data_tensor = torch.from_numpy(
                    np.stack((avg_y_data, curr_y_data), axis=0)
                )
        else: # construct placeholder for y_data_tensor (used in model evaluation)             
            curr_y_data=np.array(0)
            y_data_tensor = torch.from_numpy(
                np.stack((curr_y_data, curr_y_data), axis=0)
            )
            
        ref_sequence_tensor = torch.from_numpy(
            np.concatenate((ref_encoded, ref_encoded), axis=0)
        )

        # stack personal and reference sequence tensors
        all_seq_tensor = torch.stack(
            (ref_sequence_tensor, personal_sequence_tensor), dim=0
        )

        if self.only_personal: 
            return personal_sequence_tensor, torch.from_numpy(curr_y_data).float(), region_idx, sample_idx

        return all_seq_tensor.float(), y_data_tensor.float(), region_idx, sample_idx