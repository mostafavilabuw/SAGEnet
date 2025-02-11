from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import scipy.stats
from SAGEnet.nn import ConvBlock, MambaBlock, TransformerBlock, Residual, CrossAttention
from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl



def calculate_correlations(self, outputs):
    """
    Computes Pearson correlation coefficients between predicted and actual values across genes and samples. This function computes correlation coefficients separately for each gene (across individuals) and for each sample (across genes). It returns two DataFrames containing correlation results.

    Parameters:
    - outputs : List of tuples where each element is a tuple of:`preds` (torch.Tensor, model predictions), `actuals` (torch.Tensor, ground truth values), `gene_idx` (torch.Tensor, gene indices corresponding to each prediction), `sample_idx` (torch.Tensor, sample indices corresponding to each prediction) 

    Returns:
    gene_corrs_df : pd.DataFrame with columns:
        - 'GeneID': Unique gene identifiers.
        - 'Correlation': Pearson correlation coefficient between predictions 
          and actual values for that gene across all samples.

    sample_corrs_df : pd.DataFrame with columns:
        - 'SampleID': Unique sample identifiers.
        - 'Correlation': Pearson correlation coefficient between predictions 
          and actual values for that sample across all genes.
    """
    
    # Lists for temporary storage of data to be concatenated
    temp_preds = []
    temp_actuals = []
    temp_genes = []
    temp_samples = []

    # Loop over outputs to append the data
    for (preds, actuals, gene_idx, sample_idx) in outputs:

        # Convert BFloat16 tensors to Float32 before converting to NumPy arrays
        if preds.dtype == torch.bfloat16:
            preds = preds.to(torch.float32)

        temp_preds.append(preds.cpu().numpy())  # Assuming first column is the prediction
        temp_actuals.append(actuals.cpu().numpy())  # Assuming second column is the actual value
        temp_genes.append(gene_idx.cpu().numpy())
        temp_samples.append(sample_idx.cpu().numpy())

    # Concatenate lists of arrays into single arrays

    all_preds = np.concatenate(temp_preds)
    all_actuals = np.concatenate(temp_actuals)
    all_genes = np.concatenate(temp_genes)
    all_samples = np.concatenate(temp_samples)

    # Prepare to collect correlations
    gene_correlations = []
    sample_correlations = []

    # Unique gene and sample identifiers
    unique_genes = np.unique(all_genes)
    unique_samples = np.unique(all_samples)

    # Calculate correlations across individuals for the same gene
    for gene in unique_genes:
        gene_mask = all_genes == gene
        if np.std(all_actuals[gene_mask]) == 0 or np.std(all_preds[gene_mask]) == 0:
            # Adding NaN if variance is zero
            gene_correlations.append((gene, np.nan))
            continue

        corr = np.corrcoef(all_actuals[gene_mask], all_preds[gene_mask])[0, 1]
        gene_correlations.append((gene, corr))

    # Calculate correlations across genes for the same individual
    for sample in unique_samples:
        sample_mask = all_samples == sample
        if np.std(all_actuals[sample_mask]) == 0 or np.std(all_preds[sample_mask]) == 0:
            # Adding NaN if variance is zero
            sample_correlations.append((sample, np.nan))
            continue
        corr = np.corrcoef(all_actuals[sample_mask], all_preds[sample_mask])[0, 1]
        sample_correlations.append((sample, corr))

    # Convert to DataFrame for easier handling
    gene_corrs_df = pd.DataFrame(gene_correlations, columns=['GeneID', 'Correlation'])
    sample_corrs_df = pd.DataFrame(sample_correlations, columns=['SampleID', 'Correlation'])

    return gene_corrs_df, sample_corrs_df


def reshape_collected_data(data):
    """
    Reshapes collected data by flattening the first two dimensions.

    Used to process tensors, dictionaries, lists, or tuples that store collected data from distributed or batched computations. Flattens the first two dimensions (typically `world_size` and `batch`) for tensors while preserving the structure of nested dictionaries, lists, or tuples.

    Parameters:
    - data : torch.Tensor | dict | list | tuple
        If `data` is a `torch.Tensor`, the function flattens its first two dimensions.
        If `data` is a `dict`, it applies the transformation  to all values.
        If `data` is a `list` or `tuple`, it applies the transformation to all elements while maintaining the original type.
    
    Returns: A reshaped tensor with the first two dimensions flattened, or a dictionary, list, or tuple with the same structure, but with reshaped elements.
    """
    
    if torch.is_tensor(data):
        # assuming the first dimension is world_size and the second is batch,
        # flatten these two dimensions.
        return data.view(-1, *data.shape[2:])
    elif isinstance(data, dict):
        return {k: reshape_collected_data(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(reshape_collected_data(v) for v in data)
    else:
        raise TypeError("Unsupported type for reshaping")


class Base(pl.LightningModule):
    """
    Base model class used in pSAGEnet and rSAGEnet
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    def test_step(self, batch, batch_idx):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5
        )

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=5,
                ),
                "monitor": "val_loss",
            }
        elif self.hparams.scheduler == "cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.hparams.learning_rate / 2,
                    max_lr=self.hparams.learning_rate * 2,
                    cycle_momentum=False,
                ),
                "interval": "step",
            }
        else:
            print("No scheduler is used")
        
        return [optimizer], [lr_scheduler]

    
class rSAGEnet(Base):
    def __init__(
        self,
        input_length=40000,
        first_layer_kernel_number=900,
        int_layers_kernel_number=256,
        first_layer_kernel_size=10,
        int_layers_kernel_size=5,
        hidden_size=256,
        learning_rate=5e-4,
        n_conv_blocks=5,
        n_dilated_conv_blocks=0,
        h_layers=1,
        pooling_size=10,
        pooling_type="avg",
        batch_norm=True,
        padding="same",
        scheduler="cycle",
        dropout=0,
        block_type='conv',
        increasing_dilation=False,
        predict_from_personal=False,
        using_personal_dataset=False
    ):
        """
        Initialize rSAGEnet

        Parameters:
        - input_length: Integer input window size 
        - first_layer_kernel_number: Integer n input channels for the first convolutional layer 
        - int_layers_kernel_number: Integer n input & output channels for all convolutional layers after the first 
        - first_layer_kernel_size: Integer kernel size for the first convolutional layer 
        - int_layers_kernel_size: Integer kernel size for all convolutional layers after the first 
        - hidden_size: Integer number of nodes in fully connected layers 
        - learning_rate: Float
        - n_conv_blocks: Integer n convolutional blocks (residual (convolutional layer, activation) and pooling layers with dilation=1 
        - n_dilated_conv_blocks: Integer n convolutional blocks and pooling layers with dilation increasing exponentially  
        - h_layers: Integer n hidden layers 
        - pooling_size: Integer pooling kernel size
        - pooling_type: String pooling type ("max" or "avg")
        - batch_norm: Boolean, whether to add batch normalization at the beginning of each convolutional block 
        - padding: String, padding type in convoluational layers 
        - scheduler: String learning rate scheduler 
        - dropout: Float, dropout in fully connected layers 
        - block_type: String block type for the lowest resolution block ("mamba", "transformer", or "conv")
        - increasing_dilation: Boolean, whether or not to exponentially increase dilation in dilated_conv_layers (or keep at 2) 
        - predict_from_personal: If True, model predicts from the "personal" sequence component of the PersonalGenomeDataset, if False, model predicts from the "reference" sequence component. Only relevant if using_personal_dataset==True. 
        - using_personal_dataset: If True, model expects PersonalGenomeDataset, if False, model expects ReferenceGenomeDataset
        """
        
        super().__init__()
        
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.val_pearsons = []
        self.predict_from_personal = predict_from_personal
        self.using_personal_dataset = using_personal_dataset
            
        self.conv0 = ConvBlock(
            4, first_layer_kernel_number, first_layer_kernel_size, padding=padding, batch_norm=batch_norm
        )
        
        self.convlayers = nn.ModuleList() 
        fc_dim = input_length
        
        self.convlayers.append(
            ConvBlock(
                first_layer_kernel_number,
                int_layers_kernel_number,
                int_layers_kernel_size,
                padding=padding,
                batch_norm=batch_norm,
            )
        )
        if pooling_type == "max":
            self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
        elif pooling_type == "avg":
            self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
        else:
            raise ValueError("pooling type must be either max or avg")
        fc_dim = ceil(fc_dim / pooling_size)

        for i in range(n_conv_blocks-1):
            if i == n_conv_blocks-2 and block_type=='mamba': 
                self.convlayers.append(
                    Residual(
                        MambaBlock(
                            n_filters=int_layers_kernel_number,
                            d_state=16,
                            d_conv=int_layers_kernel_size,
                            expand=2,
                        )
                    )
                )
            elif i == n_conv_blocks-2 and block_type=='transformer':
                self.convlayers.append(
                    Residual(
                        TransformerBlock(
                            n_filters=int_layers_kernel_number,
                            nhead=int_layers_kernel_size,
                            expand=2,
                            n_layers=1,
                        )
                    )
                )
            else: 
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            int_layers_kernel_number,
                            int_layers_kernel_number,
                            int_layers_kernel_size,
                            padding=padding,
                            batch_norm=batch_norm,
                        )
                    )
                )

            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            else:
                raise ValueError("Pooling type must be either max or avg")

            fc_dim = ceil(fc_dim / pooling_size)

        self.dilated_convlayers = nn.ModuleList()
        for layer in range(n_dilated_conv_blocks):
            if increasing_dilation: 
                d =  2**(layer+1)
            else: 
                d = 2 
            self.dilated_convlayers.append(
                Residual(
                    ConvBlock(
                        int_layers_kernel_number,
                        int_layers_kernel_number,
                        int_layers_kernel_size,
                        dilation=d,
                        padding=padding,
                        batch_norm=batch_norm
                    )
                )
            )        
        
        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * int_layers_kernel_number, hidden_size), nn.ReLU()
        )

        self.fclayers = nn.ModuleList()
        for i in range(h_layers):
            self.fclayers.append(nn.Linear(hidden_size, hidden_size))
            self.fclayers.append(nn.ReLU())
            self.fclayers.append(nn.Dropout(dropout))

        self.ref_out = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        ref_x = self.conv0(x)

        for layer in self.convlayers:
            ref_x = layer(ref_x)
            
        if len(self.dilated_convlayers)>0: 
            for layer in self.dilated_convlayers:
                ref_x = layer(ref_x)
            
        ref_x = ref_x.flatten(1)
        ref_x = self.fc0(ref_x)

        for layer in self.fclayers:
            ref_x = layer(ref_x)

        ref_x = self.ref_out(ref_x)
        return ref_x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat[:,0], y) # make y_hat 1D
        loss=loss.to(self.device)
        self.log("train_loss", loss,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = None):
        if self.using_personal_dataset: 
            x, y, gene_idx, sample_idx = batch

            if self.predict_from_personal: 
                y=y[:,1]
                x_mat = x[:, 1, 0:4, :]
                x_pat = x[:, 1, 4:, :]
                y_hat_mat = self(x_mat)
                y_hat_pat = self(x_pat)
                y_hat = (y_hat_mat+y_hat_pat)/2
            else: # predicting from ref 
                y=y[:,0]
                x_ref = x[:, 0, 0:4, :]
                y_hat = self(x_ref)
            self.validation_step_outputs.append((y_hat,y, gene_idx, sample_idx))
        else: 
            x, y = batch
            y_hat = self(x)[:,0]
        
        # keep track of validation outputs for metric logging 
        curr_result = torch.stack([y_hat.cpu().detach(), y.cpu().detach()], axis=1)
        self.val_result = torch.cat([self.val_result,curr_result], dim=0)
        loss = F.mse_loss(y_hat, y) 
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        if self.using_personal_dataset or self.predict_from_personal: 
            x, y, gene_idx, sample_idx = batch
            if self.predict_from_personal: 
                y=y[:,1]
                x_mat = x[:, 1, 0:4, :]
                x_pat = x[:, 1, 4:, :]
                y_hat_mat = self(x_mat)
                y_hat_pat = self(x_pat)
                y_hat = (y_hat_mat+y_hat_pat)/2
            else: # predicting from ref 
                y=y[:,0]
                x_ref = x[:, 0, 0:4, :]
                y_hat = self(x_ref)
        else: 
            x, y = batch
            y_hat = self(x)[:,0]
        return y_hat
    
    def on_validation_epoch_start(self): 
        self.val_result = torch.Tensor([])

    def on_validation_epoch_end(self) -> None:
        if self.using_personal_dataset: 
            self.validation_step_outputs = self.all_gather(self.validation_step_outputs)
            if self.trainer.global_rank == 0:
                # reshape all elements in self.validation_step_outputs the (world_size, batch,..)  to (world_size*batch, ...)
                self.validation_step_outputs = reshape_collected_data(self.validation_step_outputs)
                gene_corrs_df, sample_corrs_df = calculate_correlations(self.validation_step_outputs)
                self.log("val_median_gene_corr", gene_corrs_df['Correlation'].median(), sync_dist=True)
                self.log("val_median_sample_corr", sample_corrs_df['Correlation'].median(), sync_dist=True)
                validation_step_outputs = self.validation_step_outputs
            self.validation_step_outputs.clear()
        else: 
            y_hat = self.val_result[:,0].numpy()
            y = self.val_result[:,1].numpy()
            val_pearson = scipy.stats.pearsonr(y_hat, y)[0] 
            self.log('val_pearson', val_pearson, sync_dist=True)
            self.val_pearsons.append(val_pearson)
            self.log('best_val_pearson', np.max(self.val_pearsons), sync_dist=True)
       

class pSAGEnet(Base):
    def __init__(
        self,
        input_length=40000,
        first_layer_kernel_number=900,
        int_layers_kernel_number=256,
        first_layer_kernel_size=10,
        int_layers_kernel_size=5,
        hidden_size=256,
        learning_rate=5e-4,
        n_conv_blocks=5,
        n_dilated_conv_blocks=0,
        h_layers=1,
        pooling_size=10,
        pooling_type="avg",
        batch_norm=True,
        padding="same",
        scheduler="cycle",
        dropout=0,
        block_type='conv',
        increasing_dilation=False,
        lam_diff=1,
        lam_ref=1,
        start_from_ref=False,
        num_top_train_genes=1000,
        num_top_val_genes=1000,
        num_training_subs=0,
        model_save_dir='',
        split_expr=True
    ):
        """
        Initialize pSAGEnet

        Parameters:
        - input_length: Integer input window size 
        - first_layer_kernel_number: Integer n input channels for the first convolutional layer 
        - int_layers_kernel_number: Integer n input & output channels for all convolutional layers after the first 
        - first_layer_kernel_size: Integer kernel size for the first convolutional layer 
        - int_layers_kernel_size: Integer kernel size for all convolutional layers after the first 
        - hidden_size: Integer number of nodes in fully connected layers 
        - learning_rate: Float
        - n_conv_blocks: Integer n convolutional blocks (residual (convolutional layer, activation) and pooling layers with dilation=1 
        - n_dilated_conv_blocks: Integer n convolutional blocks and pooling layers with dilation increasing exponentially  
        - h_layers: Integer n hidden layers 
        - pooling_size: Integer pooling kernel size
        - pooling_type: String pooling type ("max" or "avg")
        - batch_norm: Boolean, whether to add batch normalization at the beginning of each convolutional block 
        - padding: String, padding type in convoluational layers 
        - scheduler: String learning rate scheduler 
        - dropout: Float, dropout in fully connected layers 
        - block_type: String block type for the lowest resolution block ("mamba", "transformer", or "conv")
        - increasing_dilation: Boolean, whether or not to exponentially increase dilation in dilated_conv_layers (or keep at 2) 
        - lam_dff: Float, weight on "difference" component of loss function (idx 1) 
        - lam_ref: Float, weight on "mean" component of loss function (idx 0) 
        - split_expr: Boolean, if True, model "difference" output (idx 1) is predicted straight from personal sequence (no intermediate subtraction with reference) 

        - start_from_ref: Boolean, whether model was initialized with weights from r-SAGE-net (for tracking model runs with wandb). 
        - num_top_train_genes: Integer gene set size from which to select train genes (for tracking model runs with wandb). 
        - num_top_val_genes: Integer gene set size from which to select validation genes (for tracking model runs with wandb). 
        - num_training_subs: Integer number of training individuals (for tracking model runs with wandb). 
        - model_save_dir: String, directory where model is saved  (for tracking model runs with wandb). 
        """
        super().__init__()
        
        self.split_expr=split_expr
        self.save_hyperparameters()
        
        # initialize training/validation metrics 
        self.train_genes_val_step_outputs = []
        self.val_genes_val_step_outputs = []
        self.train_genes_val_step_outputs_mean = []
        self.val_genes_val_step_outputs_mean = []
        self.train_genes_val_step_outputs_diff = []
        self.val_genes_val_step_outputs_diff = []

        self.conv0 = ConvBlock(
            4, first_layer_kernel_number, first_layer_kernel_size, padding=padding, batch_norm=batch_norm
        )
        
        self.convlayers = nn.ModuleList() 
        fc_dim = input_length
        
        self.convlayers.append(
            ConvBlock(
                first_layer_kernel_number,
                int_layers_kernel_number,
                int_layers_kernel_size,
                padding=padding,
                batch_norm=batch_norm,
            )
        )
        if pooling_type == "max":
            self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
        elif pooling_type == "avg":
            self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
        else:
            raise ValueError("pooling type must be either max or avg")
        fc_dim = ceil(fc_dim / pooling_size)

        for i in range(n_conv_blocks-1):
            if i == n_conv_blocks-2 and block_type=='mamba': 
                self.convlayers.append(
                    Residual(
                        MambaBlock(
                            n_filters=int_layers_kernel_number,
                            d_state=16,
                            d_conv=int_layers_kernel_size,
                            expand=2,
                        )
                    )
                )
            elif i == n_conv_blocks-2 and block_type=='transformer':
                self.convlayers.append(
                    Residual(
                        TransformerBlock(
                            n_filters=int_layers_kernel_number,
                            nhead=int_layers_kernel_size,
                            expand=2,
                            n_layers=1,
                        )
                    )
                )
            else: 
                self.convlayers.append(
                    Residual(
                        ConvBlock(
                            int_layers_kernel_number,
                            int_layers_kernel_number,
                            int_layers_kernel_size,
                            padding=padding,
                            batch_norm=batch_norm,
                        )
                    )
                )

            if pooling_type == "max":
                self.convlayers.append(nn.MaxPool1d(pooling_size, ceil_mode=True))
            elif pooling_type == "avg":
                self.convlayers.append(nn.AvgPool1d(pooling_size, ceil_mode=True))
            else:
                raise ValueError("Pooling type must be either max or avg")

            fc_dim = ceil(fc_dim / pooling_size)

        self.dilated_convlayers = nn.ModuleList()
        for layer in range(n_dilated_conv_blocks):
            if increasing_dilation: 
                d =  2**(layer+1)
            else: 
                d = 2 
            self.dilated_convlayers.append(
                Residual(
                    ConvBlock(
                        int_layers_kernel_number,
                        int_layers_kernel_number,
                        int_layers_kernel_size,
                        dilation=d,
                        padding=padding,
                        batch_norm=batch_norm
                    )
                )
            )   
        
        self.fc0 = nn.Sequential(
            nn.Linear(fc_dim * int_layers_kernel_number, hidden_size), nn.ReLU()
        )
        
        self.fclayers = nn.ModuleList()  
        for i in range(h_layers):
            self.fclayers.append(nn.Linear(hidden_size, hidden_size))
            self.fclayers.append(nn.ReLU())
            self.fclayers.append(nn.Dropout(dropout))

        self.diff_fclayers = nn.ModuleList()  
        for i in range(h_layers):
            self.diff_fclayers.append(nn.Linear(hidden_size, hidden_size))
            self.diff_fclayers.append(nn.ReLU())
            self.diff_fclayers.append(nn.Dropout(dropout))

        self.diff_out = nn.Sequential(nn.Linear(hidden_size, 1))
        self.ref_out = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        ref_x = x[:, 0, :, :]
        personal_x = x[:, 1, :, :]
        ref_x = ref_x[:, 0:4, :] # 4:8 is identical  
        personal_x_maternal = personal_x[:, 0:4, :]
        personal_x_paternal = personal_x[:, 4:8, :]
        
        ref_x = self.conv0(ref_x)
        personal_x_maternal = self.conv0(personal_x_maternal)
        personal_x_paternal = self.conv0(personal_x_paternal)

        for layer in self.convlayers:
            ref_x = layer(ref_x)
            personal_x_maternal = layer(personal_x_maternal)
            personal_x_paternal = layer(personal_x_paternal)
            
        if len(self.dilated_convlayers)>0: 
            for layer in self.dilated_convlayers:
                ref_x = layer(ref_x)
                personal_x_maternal = layer(personal_x_maternal)
                personal_x_paternal = layer(personal_x_paternal)

        ref_x = ref_x.flatten(1)
        personal_x_maternal = personal_x_maternal.flatten(1)
        personal_x_paternal = personal_x_paternal.flatten(1)

        personal_x = (personal_x_maternal + personal_x_paternal) / 2 # average 

        ref_x = self.fc0(ref_x)
        personal_x = self.fc0(personal_x)

        
        if self.split_expr: 
            diff_x = ref_x - personal_x
        else: 
            diff_x = personal_x

        for layer in self.fclayers:
            ref_x = layer(ref_x)

        for layer in self.diff_fclayers:
            diff_x = layer(diff_x)

        ref_x = self.ref_out(ref_x)
        diff_x = self.diff_out(diff_x)
        return torch.cat((ref_x, diff_x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y, gene_idx, sample_idx = batch
        y_hat = self(x)

        ref_loss = F.mse_loss(y_hat[:, 0], y[:, 0])
        diff_loss = F.mse_loss(
            y_hat[:, 1], y[:, 1]
        )  
        loss = self.hparams.lam_ref * ref_loss + self.hparams.lam_diff * diff_loss

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_ref_loss", ref_loss, sync_dist=True)
        self.log("train_diff_loss", diff_loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, gene_idx, sample_idx = batch
        y_hat = self(x)
        
        if dataloader_idx==0: 
            ref_loss = F.mse_loss(y_hat[:, 0], y[:, 0])
            diff_loss = F.mse_loss(y_hat[:, 1], y[:, 1])
            loss = self.hparams.lam_ref * ref_loss + self.hparams.lam_diff * diff_loss
            pred_labels = y_hat[:, 0] + y_hat[:, 1]
            actual_labels = y[:, 0] + y[:, 1]
            self.log("train_gene_val_sub_loss", loss, sync_dist=True)
            self.log("train_gene_val_sub_ref_loss", ref_loss, sync_dist=True)
            self.log("train_gene_val_sub_diff_loss", diff_loss, sync_dist=True)
            self.train_genes_val_step_outputs_mean.append((y_hat[:, 0],y[:, 0], gene_idx, sample_idx))
            self.train_genes_val_step_outputs_diff.append((y_hat[:, 1],y[:, 1], gene_idx, sample_idx))

        elif dataloader_idx==1: 
            ref_loss = F.mse_loss(y_hat[:, 0], y[:, 0])
            diff_loss = F.mse_loss(y_hat[:, 1], y[:, 1])
            loss = self.hparams.lam_ref * ref_loss + self.hparams.lam_diff * diff_loss
            pred_labels = y_hat[:, 0] + y_hat[:, 1]
            actual_labels = y[:, 0] + y[:, 1]
            self.log("val_gene_val_sub_loss", loss, sync_dist=True)
            self.log("val_gene_val_sub_ref_loss", ref_loss, sync_dist=True)
            self.log("val_gene_val_sub_diff_loss", diff_loss, sync_dist=True)
            self.val_genes_val_step_outputs_mean.append((y_hat[:, 0],y[:, 0], gene_idx, sample_idx))
            self.val_genes_val_step_outputs_diff.append((y_hat[:, 1],y[:, 1], gene_idx, sample_idx))
        return loss

    
    def on_validation_epoch_end(self):        
        self.train_genes_val_step_outputs_mean = self.all_gather(self.train_genes_val_step_outputs_mean)
        self.val_genes_val_step_outputs_mean = self.all_gather(self.val_genes_val_step_outputs_mean)
        self.train_genes_val_step_outputs_diff = self.all_gather(self.train_genes_val_step_outputs_diff)
        self.val_genes_val_step_outputs_diff = self.all_gather(self.val_genes_val_step_outputs_diff)
        
        if self.trainer.global_rank == 0:            
            # train genes, val subs 
            # reshape all elements in self.validation_step_outputs the (world_size, batch,..)  to (world_size*batch, ...)
            self.train_genes_val_step_outputs_diff = reshape_collected_data(self.train_genes_val_step_outputs_diff)
            self.train_genes_val_step_outputs_mean = reshape_collected_data(self.train_genes_val_step_outputs_mean)

            gene_corrs_df, _ = calculate_correlations(self.train_genes_val_step_outputs_diff)
            _, sample_corrs_df = calculate_correlations(self.train_genes_val_step_outputs_mean)
            self.log("train_gene_val_sub_median_gene_corr", gene_corrs_df['Correlation'].median(), sync_dist=False)
            self.log("train_gene_val_sub_median_sample_corr", sample_corrs_df['Correlation'].median(), sync_dist=False)
            gene_corrs_df.to_csv(f'{self.model_save_dir}/epoch={self.trainer.current_epoch}_train_gene_gene_corrs.csv')
            sample_corrs_df.to_csv(f'{self.model_save_dir}/epoch={self.trainer.current_epoch}_train_gene_sample_corrs_df.csv')

            # val genes, val subs 
            # reshape all elements in self.validation_step_outputs the (world_size, batch,..)  to (world_size*batch, ...)
            self.val_genes_val_step_outputs_diff = reshape_collected_data(self.val_genes_val_step_outputs_diff)
            self.val_genes_val_step_outputs_mean = reshape_collected_data(self.val_genes_val_step_outputs_mean)

            gene_corrs_df, _ = calculate_correlations(self.val_genes_val_step_outputs_diff)
            _, sample_corrs_df = calculate_correlations(self.val_genes_val_step_outputs_mean)
            self.log("val_gene_val_sub_median_gene_corr", gene_corrs_df['Correlation'].median(), sync_dist=False)
            self.log("val_gene_val_sub_median_sample_corr", sample_corrs_df['Correlation'].median(), sync_dist=False)
            gene_corrs_df.to_csv(f'{self.model_save_dir}/epoch={self.trainer.current_epoch}_val_gene_gene_corrs.csv')
            sample_corrs_df.to_csv(f'{self.model_save_dir}/epoch={self.trainer.current_epoch}_val_gene_sample_corrs_df.csv')
           
        self.train_genes_val_step_outputs.clear()
        self.val_genes_val_step_outputs.clear()
        self.train_genes_val_step_outputs_mean.clear()
        self.train_genes_val_step_outputs_diff.clear()
        self.val_genes_val_step_outputs_mean.clear()
        self.val_genes_val_step_outputs_diff.clear()
      
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, y, gene_idx, sample_idx = batch
        y_hat = self(x)
        return y_hat
    
    