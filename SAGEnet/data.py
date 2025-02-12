import torch
from torch.utils.data import Dataset
import pysam
import numpy as np
import pandas as pd
import time
import SAGEnet.tools

def onehot_encoding(
    sequence: str,
    length: int,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value=0.25,
    dtype=np.float32,
    reverse_complement: bool = False,
    allow_reverse_complement: bool = True # if this is set to False, the sequence will not be reverse complemented, even if reverse_complement=True
) -> np.ndarray:
    """One-hot encode sequence."""
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

class ReferenceGenomeDataset(Dataset):
    def __init__(
        self,
        gene_metadata,
        hg38_file_path,
        expr_data=None,
        input_len=40000,
        allow_reverse_complement=True,
        single_seq=True,
        pos_start=None,
        pos_end=None
    ):
        """
        Initialize the ReferenceGenomeDataset object. 

        Parameters:
        - gene_metadata: DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
        - hg38_file_path: String path to the human genome (hg38) reference file.
        - expr_data: DataFrame with expression data, indexed by gene names, with sample names as columns.
            This should include all of the samples that we want to use in the mean expression calculation. 
            If not provided, 0 will be used for expression value. 
        - input_len: Integer, size of the genomic window model input. 
        - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand. 
        - single_seq: Boolean, if True, return output of the shape of a ReferenceGenomeDataset datapoint, if False, return output of the shape of a PersonalGenomeDataset datapoint
        - pos_start: Integer, genome position start. If None, set to max(0, tss_pos - self.input_len // 2)
        - pos_end: Integer, genome position start. If None, set to min(self.genome.get_reference_length(f"chr{chr}"), tss_pos + self.input_len // 2)
        """
        
        self.gene_metadata = gene_metadata
        self.hg38_file_path = hg38_file_path
        self.input_len = input_len
        self.expr_data = expr_data
        self.allow_reverse_complement=allow_reverse_complement
        self.genome = pysam.FastaFile(hg38_file_path)
        self.single_seq=single_seq
        self.pos_start=pos_start
        self.pos_end=pos_end

    def __len__(self):
        return len(self.gene_metadata) 

    def __getitem__(self, idx):
        """
        Retrieve and process data for a specific gene by index.
        
        Returns: 
        One-hot-encoded reference sequence, mean expression value (or placeholder 0 if no expr_data is provided)
        """        
        # retrieve gene information and sample identifier
        gene_info = self.gene_metadata.iloc[idx]
        chr = gene_info["chr"]
        tss_pos = gene_info["tss"]
        
        if self.pos_start is None: 
            pos_start = max(0, tss_pos - self.input_len // 2)  
        else:
            pos_start=self.pos_start
        if self.pos_end is None: 
            pos_end = min(self.genome.get_reference_length(f"chr{chr}"), tss_pos + self.input_len // 2)
        else: 
            pos_end=self.pos_end
            
        if pos_end>self.genome.get_reference_length(f"chr{chr}"): 
            raise ValueError(f"provided {pos_end} out of chromosome range")
        
        ref_sequence = self.genome.fetch(f"chr{chr}", pos_start, pos_end)

        # one-hot encode the sequences
        rc=(gene_info['strand']!='+') # reverse complement genes on the negative strand 
        ref_encoded = onehot_encoding(ref_sequence, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)

        # get expression data, if provided. if not, use 0 as placeholder (for example, for evaluation) 
        if self.expr_data is None: 
            mean_expr=np.array(0)
        else: 
            mean_expr = np.array(self.expr_data.loc[gene_info["ensg"]].mean())
            
        if self.single_seq: 
            return torch.from_numpy(ref_encoded).float(), torch.from_numpy(mean_expr).float()

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
            # placeholder expression tensor 
            express_tensor = torch.from_numpy(np.stack((mean_expr, 0), axis=0))

            # include placeholder gene_idx, sample_idx 
            #return all_seq_tensor.float(), express_tensor.float(), 0, 0
            return all_seq_tensor.float(), express_tensor.float()


class VariantDataset(Dataset):
    def __init__(
        self,
        gene_metadata,
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
        
        Insert variant from variant_info into reference sequence. Return input to p-SAGE-net: ref/ref, personal/personal, with the variant inserted into the first personal 

        Parameters:
        - gene_metadata: DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
        - hg38_file_path: String path to the human genome (hg38) reference file.
        - variant_info: DataFrame containing variant information, specifially the columns 'gene', 'chr', 'pos', and 'alt'
        - input_len: Integer, size of the genomic window model input. 
        - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
        - insert_variants: Boolean, if True, insert variant, if False, do not insert variant (predict on reference sequence) 
        - single_seq: Boolean, if True, return output of the shape of a ReferenceGenomeDataset datapoint, if False, return output of the shape of a PersonalGenomeDataset datapoint
        - pos_start: Integer, genome position start. If None, set to max(0, tss_pos - self.input_len // 2)
        - pos_end: Integer, genome position start. If None, set to min(self.genome.get_reference_length(f"chr{chr}"), tss_pos + self.input_len // 2)
        """
        
        self.hg38_file_path = hg38_file_path
        self.input_len = input_len
        self.allow_reverse_complement=allow_reverse_complement
        self.gene_metadata =gene_metadata.set_index('ensg', drop=False)
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
        One-hot-encoded sequence and expression, either in the shape of a ReferenceGenomeDataset datapoint (single_seq==True), or in the shape of a PersonalGenomeDataset datapoint (single_seq==False). For single_seq==False, the variant (if inserted) is inserted into both haplotypes. Expression values are placeholders (0s). 
        """                
        self.genome = pysam.FastaFile(self.hg38_file_path) # important to initialize for each datapoint to avoid errors when num_workers>1
        
        # get variant info 
        variant_info = self.variant_info.iloc[idx]
        variant_gene = variant_info['gene']
        variant_pos = variant_info['pos']
        variant_ref = variant_info['ref']
        variant_alt = variant_info['alt']
        variant_chr = variant_info['chr']
        
        # use gene in variant info to get gene info 
        gene_info = self.gene_metadata.loc[variant_gene]
        chr = gene_info["chr"]
        tss_pos = gene_info["tss"]
        if self.pos_start is None: 
            pos_start = max(0, tss_pos - self.input_len // 2)  
        else:
            pos_start=self.pos_start
        if self.pos_end is None: 
            pos_end = min(self.genome.get_reference_length(f"chr{chr}"), tss_pos + self.input_len // 2)
        else: 
            pos_end=self.pos_end
        if pos_end>self.genome.get_reference_length(f"chr{chr}"): 
            raise ValueError(f"provided {pos_end} out of chromosome range")
        ref_sequence = self.genome.fetch(f"chr{chr}", pos_start, pos_end)
        
        # check to make sure chromsome in variant info matches chromosome in gene info 
        if str(chr)!=str(variant_chr): 
            raise ValueError(f"variant chromsome ({variant_chr}) != gene chromosome ({chr})")
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
        rc=(gene_info['strand']!='+') # reverse complement genes on the negative strand 
        ref_encoded = onehot_encoding(ref_sequence, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)
        ref_with_variant_inserted_encoded = onehot_encoding(ref_with_variant_inserted, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)

        if self.single_seq: 
            if self.insert_variants: 
                return torch.from_numpy(ref_with_variant_inserted_encoded), torch.from_numpy(np.array(0)).float() # use placeholder expression tensor 
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
            # placeholder expression tensor 
            express_tensor = torch.from_numpy(np.stack((0, 0), axis=0))

            # include placeholder gene_idx, sample_idx 
            return all_seq_tensor.float(), express_tensor.float(), 0, 0
        

        
class PersonalGenomeDataset(Dataset):
    def __init__(
        self,
        gene_metadata,
        vcf_file_path,
        hg38_file_path,
        sample_list,
        expr_data=None,
        input_len=40000,
        verbose=False,
        contig_prefix="",
        split_expr=True,
        unmatched_threshold=10,
        expr_data_zscore=None,
        train_subs=None,
        only_snps=0,
        maf_threshold=-1,
        train_subs_vcf_file_path=None,
        train_subs_expr_data=None,
        allow_reverse_complement=True
    ):
        """
        Initialize the PersonalGenomeDataset object.

        Parameters:
        - gene_metadata: DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
        - hg38_file_path: String path to the human genome (hg38) reference file.
        - expr_data: DataFrame with expression data, indexed by gene names, with sample names as columns.
            This should include all of the samples that we want to use in the mean expression calculation. 
            If not provided, 0s will be returned for expression values. 
        - sample_list: List of sample names (as they appear in VCF) to include in dataset 
        - input_len: Integer, size of the genomic window model input. 
        - vcf_file_path: String path to the VCF file with variant information.
        - verbose: Boolean flag for verbose output (default: False).
        - contig_prefix: String before chromosome number in VCF file (for, example ROSMAP VCF uses 'chr1', etc.) 
        - split_expr: Boolean indicating if expression is to be decomposed into mean, difference from mean. If True, idx 1 of the expression output represents difference from mean expression (either z-score or not). If False, idx 1 of the expression output represents personal gene expression. Used to train non-contrastive model. 
        - unmatched_threshold: Integer number of mismatches between reference allele in VCF and reference in sequecne before RuntimeError.
        - expr_data_zscore: DataFrame in the same format as expr_data (indexed by gene names, sample names as columns) with pre calculated zscores. If provided, zscores are used for model output idx 1 instead of (personal expression - mean expression). 
        - train_subs: List of training individuals (to be used for calculating mean expression and MAF). If not provided, train_subs is set to sample_list. 
        - only_snps: Boolean indicating if only SNPs should be inserted or if all variants (including indels) should be inserted. 
        - maf_threshold: Float indicating the minimum MAF for a variant to be inserted. If less than zero, all variants will be inserted. 
        - train_subs_vcf_file_path: String path to the VCF file for train_subs. If not provided, train_subs_vcf_file_path is set to vcf_file_path. 
        - train_subs_expr_data: DataFrame with expression data for train_subs. If not provided, train_subs_expr_data is set to expr_data. 
        - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand 
        """
     
        self.gene_metadata = gene_metadata
        self.vcf_file_path = vcf_file_path
        self.hg38_file_path = hg38_file_path
        self.sample_list = sample_list
        self.n_samples = len(self.sample_list)
        self.input_len = input_len
        self.verbose = verbose
        self.expr_data = expr_data 
        self.expr_data_zscore = expr_data_zscore
        self.contig_prefix = contig_prefix
        self.split_expr=split_expr
        self.only_snps=only_snps
        self.maf_threshold=maf_threshold
        self.unmatched_threshold=unmatched_threshold
        self.allow_reverse_complement=allow_reverse_complement
        
        if train_subs_expr_data is None: 
            self.train_subs_expr_data=expr_data
        else: 
            self.train_subs_expr_data = train_subs_expr_data
        
        if train_subs is None: 
            self.train_subs = sample_list
        else: 
            self.train_subs=train_subs
            
        if train_subs_vcf_file_path is None: 
            self.train_subs_vcf_file_path=vcf_file_path
        else: 
            self.train_subs_vcf_file_path=train_subs_vcf_file_path
        
        if self.expr_data_zscore is not None: 
            print('using expr zscores')
        if not self.split_expr: 
            print('not splitting expression')
        if self.maf_threshold>=0: 
            print(f'maf_threshold:{maf_threshold}')

        # load reference genome
        self.genome = pysam.FastaFile(hg38_file_path)

    def __len__(self):
        """
        Return the total count of gene-sample pairs.
        """
        return len(self.gene_metadata) * len(self.sample_list)

    def __getitem__(self, idx):
        """
        Retrieve and process data for a specific gene-sample pair by index.
        Returns:
        - Tuple containing (reference sequence,personal sequence), (mean expression, difference from mean expression), gene_idx, sample_idx 
        """
        # calculate gene and sample index
        gene_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples
        gene_info = self.gene_metadata.iloc[gene_idx]
        sample_of_interest = self.sample_list[sample_idx]
        
        (maternal_seq, paternal_seq), ref_sequence = self.process_gene_variants(
            gene_info, sample_of_interest
        )

        # truncate the sequences to input_len base pairs
        maternal_seq = maternal_seq[0 : self.input_len]
        paternal_seq = paternal_seq[0 : self.input_len]
        ref_sequence = ref_sequence[0 : self.input_len]

        # one-hot encode the sequences
        rc=(gene_info['strand']!='+') # reverse complement genes on the negative strand 
        maternal_encoded = onehot_encoding(maternal_seq, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)
        paternal_encoded = onehot_encoding(paternal_seq, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)
        ref_encoded = onehot_encoding(ref_sequence, self.input_len, reverse_complement=rc,allow_reverse_complement=self.allow_reverse_complement)

        if self.expr_data is not None: 
        
            # get and process expression data
            mean_expr = np.array(self.train_subs_expr_data.loc[gene_info["ensg"],self.train_subs].mean())

            if self.expr_data_zscore is not None and not self.split_expr: 
                raise ValueError(f"zcore expression data is provided but self.split_expr=False. zscore data will not be used")

            curr_expr_data = np.array(
                    self.expr_data.loc[gene_info["ensg"]][sample_of_interest])

            if self.expr_data_zscore is None: 
                curr_expr_diff = curr_expr_data - mean_expr 
            else: 
                curr_expr_diff = np.array(
                    self.expr_data_zscore.loc[gene_info["ensg"]][sample_of_interest]
                )
     
            if self.split_expr: 
                express_tensor = torch.from_numpy(
                    np.stack((mean_expr, curr_expr_diff), axis=0)
                )
            else: 
                express_tensor = torch.from_numpy(
                    np.stack((mean_expr, curr_expr_data), axis=0)
                )
        else: # construct placeholder for express_tensor (used in model evaluation) 
            express_tensor = torch.from_numpy(
                np.stack((np.array(0), np.array(0)), axis=0)
            )
            
        # convert sequence data to tensors
        personal_sequence_tensor = torch.from_numpy(
            np.concatenate((maternal_encoded, paternal_encoded), axis=0)
        )
        ref_sequence_tensor = torch.from_numpy(
            np.concatenate((ref_encoded, ref_encoded), axis=0)
        )

        # stack personal and reference sequence tensors
        all_seq_tensor = torch.stack(
            (ref_sequence_tensor, personal_sequence_tensor), dim=0
        )
        return all_seq_tensor.float(), express_tensor.float(), gene_idx, sample_idx

        
    def process_gene_variants(self, gene_info, sample_of_interest):
        """
        Process gene variants using pre-loaded VCF and reference genome data.
        """        
        chr = gene_info["chr"]
        tss_pos = gene_info["tss"]
        pos_start = max(0, tss_pos - self.input_len // 2)
        pos_end = min(self.genome.get_reference_length(f"chr{chr}"), tss_pos + self.input_len // 2)
    
        records_list = SAGEnet.tools.get_records_list(chr, pos_start, pos_end, contig_prefix=self.contig_prefix,vcf_file_path=self.vcf_file_path,hg38_file_path=self.hg38_file_path)
    
        vcf_of_sample = [record for record in records_list if record.samples[sample_of_interest]["GT"] != (0, 0) and record.samples[sample_of_interest]["GT"] != (None, None)]             
        sequence = self.genome.fetch(f"chr{chr}", pos_start, pos_end)
        return self.modify_sequences_based_on_vcf(sequence, vcf_of_sample, sample_of_interest, pos_start), sequence

    
    def modify_sequences_based_on_vcf(
        self, sequence, vcf_of_sample, sample_of_interest, test_start):
        """
        Modifies maternal and paternal DNA sequences based on variant information.
        """
        maternal_sequence = paternal_sequence = sequence.upper()
        maternal_shift = paternal_shift = 0

        for record in sorted(vcf_of_sample, key=lambda r: r.pos):
            GT = record.samples[sample_of_interest]["GT"]
            unmatched_count=0
            for idx, seq in enumerate([maternal_sequence, paternal_sequence]):
                shift = maternal_shift if idx == 0 else paternal_shift
                position = record.pos - test_start + shift - 1
                ref_allele = record.ref.upper()

                if GT[idx]==None: # genotype is missing/unknown due to tech issues
                    allele_index = 0
                else:
                    allele_index = int(GT[idx])  # 0 for ref, 1 or greater for alt
                    
                alt_allele = (
                    ref_allele
                    if allele_index == 0
                    else record.alts[allele_index - 1].upper()
                )
                
                # based on only_snps and maf_threshold, decide whether or not to insert a given variant
                insert_variant=True
                
                if self.only_snps: 
                    if len(alt_allele)!=1 or len(ref_allele)!=1: 
                        insert_variant=False
                if self.maf_threshold>=0: # maf is supplied 
                    curr_maf = SAGEnet.tools.calc_maf(record, allele_index, self.train_subs_vcf_file_path, self.train_subs) 
                    if curr_maf<=self.maf_threshold: 
                        insert_variant=False
                        
                if insert_variant: 
                    seq_before_variant = seq[:position]
                    seq_after_variant = seq[position + len(ref_allele) :]

                    if seq[position : position + len(ref_allele)] != ref_allele:
                        unmatched_count+=1
                        if unmatched_count >= self.unmatched_threshold:
                            raise RuntimeError(f"Fatal error: unmatched_count has reached the threshold of {self.unmatched_threshold}.")

                        if self.verbose:
                            print(
                                f"Warning: Reference allele at position {record.pos} does not match the sequence for {'maternal' if idx == 0 else 'paternal'}."
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
    





