import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from SAGEnet.data import ReferenceGenomeDataset
from SAGEnet.models import rSAGEnet
import SAGEnet.tools
import glob
import os

# add your WANDB key 
#os.environ["WANDB_API_KEY"] = 'your_key' 

def train_ref(batch_size, num_workers, max_epochs, model_save_dir, num_nodes, h_layers, n_conv_blocks, batch_norm, dropout, int_layers_kernel_number, first_layer_kernel_size, int_layers_kernel_size, hidden_size, pooling_size, pooling_type, input_len,learning_rate,first_layer_kernel_number,n_dilated_conv_blocks,increasing_dilation,wandb_project,wandb_job_name,tss_data_path,expr_data_path,hg38_file_path,enformer_gene_assignments_path,use_enformer_gene_assignments,allow_reverse_complement, gene_list_path,device,sub_data_dir):

    model_save_dir=f'{model_save_dir}/{wandb_job_name}'
    print(f'creating dir {model_save_dir}')
    os.makedirs(model_save_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=wandb_project, name=wandb_job_name, id=wandb_job_name, resume="allow")

    gene_meta_info = pd.read_csv(tss_data_path, sep="\t")
    expr_data = pd.read_csv(expr_data_path, index_col=0)
    gene_list = np.loadtxt(gene_list_path,delimiter=',',dtype=str)
    train_subs = np.loadtxt(f'{sub_data_dir}ROSMAP/train_subs.csv',delimiter=',',dtype=str)
    expr_data=expr_data.loc[gene_list,train_subs]

    train_genes, val_genes, test_genes = SAGEnet.tools.get_train_val_test_genes(gene_list,tss_data_path=tss_data_path, use_enformer_gene_assignments=use_enformer_gene_assignments,enformer_gene_assignments_path=enformer_gene_assignments_path)
    train_genes_meta = gene_meta_info[gene_meta_info['gene_id'].isin(train_genes)]
    val_genes_meta = gene_meta_info[gene_meta_info['gene_id'].isin(val_genes)]
    
    print(f'n train genes: {len(train_genes_meta)}')
    print(f'n val genes: {len(val_genes_meta)}')
    
    train_dataset = ReferenceGenomeDataset(metadata=train_genes_meta, hg38_file_path=hg38_file_path, y_data=expr_data, input_len=input_len,allow_reverse_complement=allow_reverse_complement)
    val_dataset = ReferenceGenomeDataset(metadata=val_genes_meta, hg38_file_path=hg38_file_path, y_data=expr_data, input_len=input_len,allow_reverse_complement=allow_reverse_complement)
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    es = EarlyStopping(monitor="val_pearson", patience=10,mode='max')
    checkpoint_callback = ModelCheckpoint(dirpath=model_save_dir, monitor="val_pearson", save_top_k=1, mode="max", save_last=True, every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks=[es,checkpoint_callback,lr_monitor]

    if glob.glob(os.path.join(model_save_dir, "*.ckpt"))!=[]:
        last_checkpoint = model_save_dir + "/last.ckpt"
    else:
        last_checkpoint = None
        
    trainer = pl.Trainer(
    accelerator="gpu", 
    devices=[int(device)] if device else 1, 
    num_nodes=num_nodes, 
    strategy="ddp" if not device else None, 
    callbacks=callbacks, 
    max_epochs=max_epochs, 
    benchmark=False, 
    profiler='simple', 
    gradient_clip_val=1, 
    logger=wandb_logger, 
    log_every_n_steps=10)
   
    my_model = rSAGEnet(input_length=input_len, int_layers_kernel_number=int_layers_kernel_number, first_layer_kernel_size=first_layer_kernel_size, int_layers_kernel_size=int_layers_kernel_size, hidden_size=hidden_size, pooling_size=pooling_size, pooling_type=pooling_type, h_layers=h_layers, n_conv_blocks=n_conv_blocks, batch_norm=batch_norm, dropout=dropout, learning_rate=learning_rate,first_layer_kernel_number=first_layer_kernel_number,n_dilated_conv_blocks=n_dilated_conv_blocks,increasing_dilation=increasing_dilation)    

    wandb_logger.watch(my_model)
    if last_checkpoint is None:
        print('fitting model')
        trainer.fit(my_model, train_dataloader, val_dataloader)
    else:
        print(f'fitting model from ckpt={last_checkpoint}')
        trainer.fit(my_model, train_dataloader, val_dataloader, ckpt_path=last_checkpoint)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--h_layers', type=int, default=1)
    parser.add_argument('--n_conv_blocks', type=int, default=5)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--int_layers_kernel_number', type=int, default=256)
    parser.add_argument('--first_layer_kernel_size', type=int, default=10)
    parser.add_argument('--int_layers_kernel_size', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--pooling_size', type=int, default=10)
    parser.add_argument('--pooling_type', type=str, default="avg")
    parser.add_argument('--input_len', type=int, default=40000)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--first_layer_kernel_number', type=int, default=900)
    parser.add_argument('--n_dilated_conv_blocks', type=int, default=0)
    parser.add_argument('--increasing_dilation', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--wandb_job_name', type=str, default='job0')
    parser.add_argument('--tss_data_path', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/data/ROSMAP/expressionData/gene-ids-and-positions.tsv')
    parser.add_argument('--expr_data_path', default='/data/mostafavilab/personal_genome_expr/data/rosmap/expressionData/vcf_match_covariate_adjusted_log_tpm.csv')
    parser.add_argument('--hg38_file_path', default='/data/tuxm/project/Decipher-multi-modality/data/genome/hg38.fa')
    parser.add_argument('--enformer_gene_assignments_path', default='/data/mostafavilab/personal_genome_expr/final_results/enformer/enformer_gene_splits.csv')
    parser.add_argument('--use_enformer_gene_assignments', type=int, default=0)
    parser.add_argument("--allow_reverse_complement",type=int,default=1)
    parser.add_argument("--gene_list_path",default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/protein_coding_genes.csv')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sub_data_dir', default='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/data/ROSMAP/sub_lists/')
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)  
    
    train_ref(
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    max_epochs=args.max_epochs,
    model_save_dir=args.model_save_dir,
    num_nodes=args.num_nodes,
    h_layers=args.h_layers,
    n_conv_blocks=args.n_conv_blocks,
    batch_norm=args.batch_norm,
    dropout=args.dropout,
    int_layers_kernel_number=args.int_layers_kernel_number,
    first_layer_kernel_size=args.first_layer_kernel_size,
    int_layers_kernel_size=args.int_layers_kernel_size,
    hidden_size=args.hidden_size,
    pooling_size=args.pooling_size,
    pooling_type=args.pooling_type,
    input_len=args.input_len,
    learning_rate=args.learning_rate,
    first_layer_kernel_number=args.first_layer_kernel_number,
    n_dilated_conv_blocks=args.n_dilated_conv_blocks,
    increasing_dilation=args.increasing_dilation,
    wandb_project=args.wandb_project,
    wandb_job_name=args.wandb_job_name,
    tss_data_path=args.tss_data_path,
    expr_data_path=args.expr_data_path,
    hg38_file_path=args.hg38_file_path,
    enformer_gene_assignments_path=args.enformer_gene_assignments_path,
    use_enformer_gene_assignments=args.use_enformer_gene_assignments,
    allow_reverse_complement=args.allow_reverse_complement,
    gene_list_path=args.gene_list_path,
    device=args.device,
    sub_data_dir=args.sub_data_dir)