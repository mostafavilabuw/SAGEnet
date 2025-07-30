import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from SAGEnet.data import PersonalGenomeDataset
from SAGEnet.models import pSAGEnet
import SAGEnet.tools
import glob
import os

# add your WANDB key 
#os.environ["WANDB_API_KEY"] = 'your_key' 

def train_on_personal(model_save_dir, tss_data_path,expr_data_path,sub_data_dir,hg38_file_path,vcf_file_path,wandb_project,start_from_ref,ref_model_ckpt_path,num_nodes,input_len,batch_size,num_workers,max_epochs,wandb_job_name,device,lam_diff,lam_ref,zscore,predixcan_res_path,num_top_train_genes,num_top_val_genes,only_snps,split_expr,num_training_subs,save_all_epochs,seed,rand_genes,top_genes_to_consider,maf_threshold,gene_idx_start,allow_reverse_complement,block_type,first_layer_kernel_number,int_layers_kernel_number,hidden_size,h_layers): 
    """
    Train pSAGEnet model on paired WGS and personal gene expression data. 
    
    Parameters: 
    - model_save_dir: String path to directory in which to save model ckpts and metrics. 
    - tss_data_path: String path to DataFrame containing gene-related information, specifically the columns 'chr', 'tss', and 'strand'. 
    - expr_data_path: String path to DataFrame with expression data, indexed by gene names, with sample names as columns.
    - sub_data_dir: String path to directory containing lists of individual train/validation/test splits. 
    - hg38_file_path: String path to the human genome (hg38) reference file.
    - vcf_file_path: String path to the VCF file with variant information.
    - wandb_project: String WANDB project name. 
    - start_from_ref: Boolean indicating whether or not to initialize pSAGEnet model by loading in rSAGEnet weights. 
    - ref_model_ckpt_path: String path to rSAGEnet model to use to initialize pSAGEnet model. 
    - num_nodes: Integer number of nodes to use in model training. 
    - input_len: Integer, size of the genomic window model input. 
    - batch_size: Integer, batch size to use in model training. 
    - num_workers: Integer, number of workers to use in model training. 
    - max_epochs: Integer, maximum number of training epochs. 
    - wandb_job_name: String, WANDB job name. 
    - device: Integer, GPU index. 
    - lam_diff: Float, weight on "difference" component of loss function (idx 1).  
    - lam_ref: Float, weight on "mean" component of loss function (idx 0). 
    - zscore: Boolean, whether to use zscore for model idx 1 output (instead of difference from mean). 
    - predixcan_res_path: String path to predixcan results path, to be used to construct ranked gene sets. 
    - num_top_train_genes: Integer gene set size from which to select train genes. 
    - num_top_val_genes: Integer gene set size from which to select validation genes. 
    - only_snps: Boolean indicating if only SNPs should be inserted or if all variants (including indels) should be inserted. 
    - split_expr: Boolean indicating if expression is to be decomposed into mean, difference from mean. If True, idx 1 of the expression output represents difference from mean expression (either z-score or not). If False, idx 1 of the expression output represents personal gene expression. Used to train non-contrastive model. 
    - num_training_subs: Integer number of training individuals. 
    - save_all_epochs: Boolean indicating whether to save ckpt for each model training epoch. If False, only save "best" model epochs, as determined by validation metrics. 
    - seed: Integer seed to determine random shuffling of gene set. 
    - rand_genes: Boolean indicating whether or not to randomly select genes (from top_genes_to_consider gene set) to use in model training. If False, select gene set from top-prediXcan ranked genes. 
    - top_genes_to_consider: Integer, length of prediXcan-ranked top gene set to consider when randomly selecting genes (only relevant if rand_genes==True). 
    - maf_threshold: Float, MAF threshold to use in training. 
    - gene_idx_start: Integer index in prediXcan-ranked gene list of first gene to use in model training. 
    - allow_reverse_complement: Boolean, whether or not to reverse complement genes on the negative strand.  
    - block_type: String block type for the lowest resolution block ("mamba", "transformer", or "conv"). 
    - first_layer_kernel_number: Integer n input channels for the first convolutional layer. 
    - int_layers_kernel_number: Integer n input & output channels for all convolutional layers after the first. 
    - hidden_size: Integer number of nodes in fully connected layers.  
    - h_layers: Integer n hidden layers.  
    """
    model_save_dir=f'{model_save_dir}{wandb_job_name}'
    print(f'creating dir {model_save_dir}')
    os.makedirs(model_save_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=wandb_project, name=wandb_job_name, id=wandb_job_name, resume="allow")

    train_subs = np.loadtxt(f'{sub_data_dir}ROSMAP/train_subs.csv',delimiter=',',dtype=str)
    val_subs = np.loadtxt(f'{sub_data_dir}ROSMAP/val_subs.csv',delimiter=',',dtype=str)

    sel_train_subs=train_subs[:num_training_subs]
    print(f'num train subs={len(sel_train_subs)}')
    print(f'num val subs={len(val_subs)}')
    expr_data = pd.read_csv(expr_data_path, index_col=0)
    
    # get gene lists 
    train_gene_list = SAGEnet.tools.select_region_set(enet_path=predixcan_res_path, rand_regions=rand_genes, top_regions_to_consider=top_genes_to_consider,seed=seed, num_regions=num_top_train_genes,region_idx_start=gene_idx_start) 
    val_gene_list = SAGEnet.tools.select_region_set(enet_path=predixcan_res_path, rand_regions=rand_genes, top_regions_to_consider=top_genes_to_consider,seed=seed, num_regions=num_top_val_genes,region_idx_start=gene_idx_start) 

    # only include genes that are present in the expression data 
    train_gene_list = np.array([gene for gene in train_gene_list if gene in expr_data.index])
    val_gene_list = np.array([gene for gene in val_gene_list if gene in expr_data.index])
    
    # split train and validation gene lists by chromosome 
    train_gene_list, _ , _ = SAGEnet.tools.get_train_val_test_genes(train_gene_list, tss_data_path=tss_data_path)
    _, val_gene_list, _ = SAGEnet.tools.get_train_val_test_genes(val_gene_list, tss_data_path=tss_data_path)

    gene_meta_info = pd.read_csv(tss_data_path, sep="\t",index_col='region_id')
    train_gene_meta=gene_meta_info.loc[train_gene_list]
    val_gene_meta=gene_meta_info.loc[val_gene_list]

    print(f'n train genes = {len(train_gene_meta)}')
    print(f'n val genes = {len(val_gene_meta)}')
    np.save(f'{model_save_dir}/train_genes',train_gene_meta.index.values)
    np.save(f'{model_save_dir}/val_genes',val_gene_meta.index.values)
    
    if zscore: 
        expr_dir = os.path.dirname(expr_data_path)
        if 'zscore_expr_from_train_subs.csv' not in os.listdir(expr_dir): 
            print('getting expr zscores')
            zscores = expr_data.apply(lambda x: (x - x[train_subs].mean()) / x[train_subs].std(), axis=1)
            zscores.to_csv(f'{expr_dir}/zscore_expr_from_train_subs.csv')
        else: 
            zscores = pd.read_csv(f'{expr_dir}/zscore_expr_from_train_subs.csv', index_col=0)
    else: 
        zscores = None

    train_dataset = PersonalGenomeDataset(metadata=train_gene_meta, vcf_file_path=vcf_file_path, hg38_file_path=hg38_file_path, y_data=expr_data, sample_list=sel_train_subs,y_data_zscore=zscores,train_subs=train_subs, input_len=input_len,only_snps=only_snps,split_y_data=split_expr,maf_min=maf_threshold,train_subs_vcf_file_path=vcf_file_path,train_subs_y_data=expr_data,allow_reverse_complement=allow_reverse_complement)
    val_subs_dataset = PersonalGenomeDataset(metadata=train_gene_meta, vcf_file_path=vcf_file_path, hg38_file_path=hg38_file_path, y_data=expr_data, sample_list=val_subs,y_data_zscore=zscores,train_subs=train_subs, input_len=input_len,only_snps=only_snps,split_y_data=split_expr,maf_min=maf_threshold,train_subs_vcf_file_path=vcf_file_path,train_subs_y_data=expr_data,allow_reverse_complement=allow_reverse_complement)
    val_genes_dataset = PersonalGenomeDataset(metadata=val_gene_meta, vcf_file_path=vcf_file_path, hg38_file_path=hg38_file_path, y_data=expr_data, sample_list=val_subs, y_data_zscore=zscores,train_subs=train_subs, input_len=input_len,only_snps=only_snps,split_y_data=split_expr,maf_min=maf_threshold,train_subs_vcf_file_path=vcf_file_path,train_subs_y_data=expr_data,allow_reverse_complement=allow_reverse_complement)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_subs_dataloader = DataLoader(val_subs_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_genes_dataloader = DataLoader(val_genes_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
          
    val_dataloaders=[val_subs_dataloader,val_genes_dataloader]

    es = EarlyStopping(monitor="train_gene_val_sub_diff_loss/dataloader_idx_0", patience=5,mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # used to save every model epoch 
    all_epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,  
        filename="{epoch}",  
        save_top_k=-1,  
        every_n_epochs=1,
        save_last=False
    )
    
    # save last ckpt to be able to resume model training if job is killed 
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        filename="last",     
        save_top_k=0,        
        every_n_train_steps=300,  
        save_last=True      
    )

    # if not saving model ckpt every epoch, save based on validation metrics 
    train_gene_val_sub_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="train_gene_val_sub_loss/dataloader_idx_0", save_top_k=1, mode="min", save_last=False, every_n_epochs=1,filename="{epoch}-{step}-{train_gene_val_sub_loss/dataloader_idx_0:.4f}")
    train_gene_val_sub_diff_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="train_gene_val_sub_diff_loss/dataloader_idx_0", save_top_k=1, mode="min", save_last=False, every_n_epochs=1,filename="{epoch}-{step}-{train_gene_val_sub_diff_loss/dataloader_idx_0:.4f}")
    train_gene_val_sub_ref_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="train_gene_val_sub_ref_loss/dataloader_idx_0", save_top_k=1, mode="min", save_last=True, every_n_train_steps=300,filename="{epoch}-{step}-{train_gene_val_sub_ref_loss/dataloader_idx_0:.4f}")
    val_gene_val_sub_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="val_gene_val_sub_loss/dataloader_idx_1", save_top_k=1, mode="min", save_last=False, every_n_epochs=1,filename="{epoch}-{step}-{val_gene_val_sub_loss/dataloader_idx_1:.4f}")
    val_gene_val_sub_diff_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="val_gene_val_sub_diff_loss/dataloader_idx_1", save_top_k=1, mode="min", save_last=False, every_n_epochs=1,filename="{epoch}-{step}-{val_gene_val_sub_diff_loss/dataloader_idx_1:.4f}")
    val_gene_val_sub_ref_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="val_gene_val_sub_ref_loss/dataloader_idx_1", save_top_k=1, mode="min", save_last=False, every_n_epochs=1,filename="{epoch}-{step}-{val_gene_val_sub_ref_loss/dataloader_idx_1:.4f}")
    
    if save_all_epochs: 
        print('saving ckpt each epoch')
        ckpt_list = [all_epoch_checkpoint_callback,last_checkpoint_callback]
    else: 
        print('saving ckpts based on val metrics')
        ckpt_list = [train_gene_val_sub_loss_checkpoint_callback,train_gene_val_sub_diff_loss_checkpoint_callback,train_gene_val_sub_ref_loss_checkpoint_callback,val_gene_val_sub_loss_checkpoint_callback,val_gene_val_sub_diff_loss_checkpoint_callback,val_gene_val_sub_ref_loss_checkpoint_callback]
    ckpt_list.append(es)
    ckpt_list.append(lr_monitor)
    
    if glob.glob(os.path.join(model_save_dir, "*.ckpt"))!=[]:
        last_checkpoint = model_save_dir + "/last.ckpt"
    else:
        last_checkpoint = None
        
    trainer = pl.Trainer(
    accelerator="gpu", 
    devices=[int(device)] if device else 1, 
    num_nodes=num_nodes, 
    strategy="ddp" if not device else 'auto', 
    callbacks=ckpt_list, 
    max_epochs=max_epochs, 
    benchmark=False, 
    profiler='simple', 
    gradient_clip_val=1, 
    logger=wandb_logger, 
    log_every_n_steps=10)
    my_model = pSAGEnet(lam_diff=lam_diff,lam_ref=lam_ref,start_from_ref=start_from_ref,num_train_regions=num_top_train_genes,num_val_regions=num_top_val_genes,num_training_subs=num_training_subs,model_save_dir=model_save_dir,h_layers=h_layers,int_layers_kernel_number=int_layers_kernel_number,hidden_size=hidden_size,first_layer_kernel_number=first_layer_kernel_number,block_type=block_type,split_y_data=split_expr)
        
    if start_from_ref: 
        print(f'loading rSAGEnet weights from {ref_model_ckpt_path} into pSAGEnet model')
        my_model = SAGEnet.tools.init_model_from_ref(my_model, ref_model_ckpt_path)
          
    wandb_logger.watch(my_model)
    if last_checkpoint is None:
        print('fitting model')
        trainer.fit(my_model, train_dataloader, val_dataloaders=val_dataloaders)
    else:
        print(f'fitting model from ckpt={last_checkpoint}')
        trainer.fit(my_model, train_dataloader, val_dataloaders=val_dataloaders, ckpt_path=last_checkpoint)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--job_id', type=str, required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_len', type=int, default=40000)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_job_name', type=str, default='job0')
    parser.add_argument('--start_from_ref', type=int,default=1)
    parser.add_argument('--model_save_dir', required=True)
    parser.add_argument('--device', default='')
    parser.add_argument('--lam_diff', default=1,type=float)
    parser.add_argument('--lam_ref', default=1,type=float)
    parser.add_argument('--zscore', default=1,type=int)
    parser.add_argument('--only_snps', default=0,type=int)
    parser.add_argument('--maf_threshold', default=-1,type=float)
    parser.add_argument('--tss_data_path', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/gene-ids-and-positions.tsv')
    parser.add_argument('--expr_data_path', default='/data/mostafavilab/personal_genome_expr/data/rosmap/expressionData/vcf_match_covariate_adjusted_log_tpm.csv')
    parser.add_argument('--sub_data_dir', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/individual_sets/')
    parser.add_argument('--predixcan_res_path', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/results_data/predixcan/rosmap_pearson_corr.csv')
    parser.add_argument('--hg38_file_path', default='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa')
    parser.add_argument('--vcf_file_path', default='/data/mostafavilab/bng/rosmapAD/data/wholeGenomeSeq/chrAll.phased.vcf.gz')
    parser.add_argument('--ref_model_ckpt_path', default='/data/mostafavilab/personal_genome_expr/ref_cnn_models/rc_neg_strand/job0/epoch=12-step=32045.ckpt')
    parser.add_argument("--num_top_train_genes",type=int,default=1000)
    parser.add_argument("--num_top_val_genes",type=int,default=1000)
    parser.add_argument("--split_expr",type=int,default=1)
    parser.add_argument("--num_training_subs",type=int,default=689)
    parser.add_argument("--save_all_epochs",type=int,default=0)
    parser.add_argument("--rand_genes",type=int,default=0)
    parser.add_argument("--top_genes_to_consider",type=int,default=5000)
    parser.add_argument("--gene_idx_start",type=int,default=0)
    parser.add_argument("--allow_reverse_complement",type=int,default=1)
    parser.add_argument("--block_type",default='conv')    
    parser.add_argument("--first_layer_kernel_number",type=int,default=900)
    parser.add_argument("--int_layers_kernel_number",type=int,default=256)
    parser.add_argument("--hidden_size",type=int,default=256)
    parser.add_argument("--h_layers",type=int,default=1)    
    args = parser.parse_args()
    
    pl.seed_everything(args.seed) 
    
    train_on_personal(
        model_save_dir=args.model_save_dir, 
        tss_data_path=args.tss_data_path, 
        expr_data_path=args.expr_data_path, 
        sub_data_dir=args.sub_data_dir, 
        hg38_file_path=args.hg38_file_path, 
        vcf_file_path=args.vcf_file_path, 
        wandb_project=args.wandb_project, 
        start_from_ref=args.start_from_ref, 
        ref_model_ckpt_path=args.ref_model_ckpt_path, 
        num_nodes=args.num_nodes, 
        input_len=args.input_len, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        max_epochs=args.max_epochs, 
        wandb_job_name=args.wandb_job_name, 
        device=args.device, 
        lam_diff=args.lam_diff, 
        lam_ref=args.lam_ref, 
        zscore=args.zscore, 
        predixcan_res_path=args.predixcan_res_path, 
        num_top_train_genes=args.num_top_train_genes, 
        num_top_val_genes=args.num_top_val_genes, 
        only_snps=args.only_snps, 
        split_expr=args.split_expr, 
        num_training_subs=args.num_training_subs, 
        save_all_epochs=args.save_all_epochs, 
        seed=args.seed, 
        rand_genes=args.rand_genes, 
        top_genes_to_consider=args.top_genes_to_consider, 
        maf_threshold=args.maf_threshold, 
        gene_idx_start=args.gene_idx_start, 
        allow_reverse_complement=args.allow_reverse_complement, 
        block_type=args.block_type, 
        first_layer_kernel_number=args.first_layer_kernel_number, 
        int_layers_kernel_number=args.int_layers_kernel_number, 
        hidden_size=args.hidden_size, 
        h_layers=args.h_layers
    )