#!/bin/bash

# Define arrays of hyperparameter values
first_layer_kernel_numbers=(256,900)
int_layers_kernel_numbers=(256,512,900)
first_layer_kernel_sizes=(25,10)
int_layers_kernel_sizes=(5)
n_conv_block_options=(5,8)
pooling_sizes=(25,10,5)
pooling_types=("max","avg")
n_dilated_conv_block_options=(0,3,5)
dropouts=(0,.2)
h_layer_options=(2,1)
increasing_dilations=(0,1)
batch_norms=(0,1)
hidden_sizes=(256,512,900)
learning_rates=(5e-4, 1e-3)
n_models_per_hyp=1

for first_layer_kernel_number in "${first_layer_kernel_numbers[@]}"; do
    for int_layers_kernel_number in "${int_layers_kernel_numbers[@]}"; do
        for first_layer_kernel_size in "${first_layer_kernel_sizes[@]}"; do
            for int_layers_kernel_size in "${int_layers_kernel_sizes[@]}"; do
                for n_conv_blocks in "${n_conv_blocks_options[@]}"; do
                    for pooling_size in "${pooling_sizes[@]}"; do
                        for pooling_type in "${pooling_types[@]}"; do
                            for n_dilated_conv_blocks in "${n_dilated_conv_block_options[@]}"; do
                                for dropout in "${dropouts[@]}"; do
                                    for h_layers in "${h_layer_options[@]}"; do
                                        for increasing_dilation in "${increasing_dilations[@]}"; do
                                            for batch_norm in "${batch_norms[@]}"; do
                                                for hidden_size in "${hidden_sizes[@]}"; do
                                                    for learning_rate in "${learning_rates[@]}"; do
                                                        for ((i=1; i<=n_models_per_hyp; i++)); do
                                                            sbatch --nodes=1  --export=ALL run_train_sagenet.sh
                                                        done 
                                                    done
                                                done
                                            done
                                        done
                                    done 
                                done 
                            done 
                        done 
                    done 
                done 
            done 
        done 
    done
done 


