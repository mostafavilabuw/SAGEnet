from bisect import bisect_left, bisect_right
import numpy as np 
import pandas as pd 

input_len=40000
input_data_path = '/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/'

tss_data_path=f'{input_data_path}gene-ids-and-positions.tsv'
gene_list_path = f'{input_data_path}protein_coding_genes.csv'
gene_list = np.loadtxt(gene_list_path,delimiter=',',dtype=str)

gene_meta_info = pd.read_csv(tss_data_path, sep="\t")
gene_win_info=gene_meta_info.loc[gene_list]

# enformer_human_seqs from https://console.cloud.google.com/storage/browser/basenji_barnyard/data
content = []
with open('/data/aspiro17/enformer_res/enformer_paper_data/enformer_human_seqs.bed')as f:
    for line in f:
        content.append(line.strip().split())

valid_chrs = [str(i) for i in (range(1,23))]
valid_chrs.append('X')

chr_regions = {chrom: {'starts': [], 'ends': [], 'labels': []} for chrom in valid_chrs}

for item in content:
    current_chr = item[0].split('r')[1]
    curr_start = int(item[1])
    curr_end = int(item[2])
    curr_label = item[3]
    chr_regions[current_chr]['starts'].append(curr_start)
    chr_regions[current_chr]['ends'].append(curr_end)
    chr_regions[current_chr]['labels'].append(curr_label)

# sort regions for each chromosome by start positions
for chrom in chr_regions:
    sorted_indices = sorted(range(len(chr_regions[chrom]['starts'])), key=lambda i: chr_regions[chrom]['starts'][i])
    chr_regions[chrom]['starts'] = [chr_regions[chrom]['starts'][i] for i in sorted_indices]
    chr_regions[chrom]['ends'] = [chr_regions[chrom]['ends'][i] for i in sorted_indices]
    chr_regions[chrom]['labels'] = [chr_regions[chrom]['labels'][i] for i in sorted_indices]

# assign regions using binary search
sel_starts = []
sel_ends = []
assignments = []
tss = []
chroms = []

for gene in gene_win_info.index:
    curr_chr = str(gene_win_info.loc[gene]['chr'])
    curr_tss = int(gene_win_info.loc[gene]['tss'])
    tss.append(curr_tss)
    chroms.append(curr_chr)

    if curr_chr not in chr_regions:
        assignments.append('na')
        sel_starts.append('na')
        sel_ends.append('na')
        continue

    # find candidate regions using binary search
    rel_regions = chr_regions[curr_chr]
    left_bound = curr_tss - (input_len//2)
    right_bound = curr_tss + (input_len//2)

    # get indices of candidate regions that might overlap
    start_idx = bisect_right(rel_regions['ends'], left_bound)
    end_idx = bisect_left(rel_regions['starts'], right_bound)

    # check for overlaps in candidate regions
    assign = 'na'
    for i in range(start_idx, end_idx):
        if rel_regions['starts'][i] <= left_bound and rel_regions['ends'][i] >= right_bound:
            assign = rel_regions['labels'][i]
            sel_starts.append(rel_regions['starts'][i])
            sel_ends.append(rel_regions['ends'][i])
            break
    if assign=='na': 
        sel_starts.append('na')
        sel_ends.append('na')
    assignments.append(assign)
    
assignments_df = pd.DataFrame(index=gene_win_info.index)
assignments_df['enformer_set'] = assignments
assignments_df['sel_start'] = sel_starts
assignments_df['sel_end'] = sel_ends
assignments_df['tss'] = tss
assignments_df['chrom'] = chroms

assignments_df.to_csv(f'{input_data_path}enformer_gene_splits.csv')