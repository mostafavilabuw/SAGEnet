## preprocess_data.py 
- Save gene set of all protein-coding genes to use in the analysis
- Randomly split ROSMAP individuals into train/validation/test
- Preprocess ROSMAP expression data (take log(TPM+1) and regress out known covariates and top expression PCs)
- Preprocess GTEx expression data (take log(TPM+1) and regress out known covariates and top expression PCs)
