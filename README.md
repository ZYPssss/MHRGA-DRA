# 
=======
# 



## Overview 


## Data
'data/DRUG.xlsx : Drug information files, including drug names and SMILES strings.

'data/DRUG-RNA.xlsx': A file containing association information between drugs and RNAs, where each line in the file indicates an association between a drug and an RNA.


'data/DRUG_RNA_MATRIX.csv ': Drug-RNA association matrix file; 1 indicates an association, 0 indicates unknown.


'data/DRUG_similarity.csv ': The drug similarity matrix is obtained by summing and averaging the results of drug structural similarity and Gaussian kernel similarity.


'data/RNA.xlsx ': RNA information files, including RNA names and category.


'data/RNA_similarity.csv ': RNA similarity matrix, obtained by summing and averaging RNA sequence similarity and Gaussian kernel similarity.



## Environment
`You can create a conda environment for xxxxxx  by ‘conda env create -f environment.yml‘.`


## Train and test the model
- ### step 1
  - #### Pre-trained Multi-view Molecular Feature Encoder
        'python pre_train_Multi_View_E_model.py'
- ### step 2
  - #### Training a machine-based autocomplete model 
        'python pre_train_Machine_E_model.py'
- ### step 3 
  - #### Obtaining corresponding molecular features using a multi-view molecular encoder 
        'python get_pre_train_data.py'
- ### step 4 
  - #### Retrieve the RNA-drug association matrix with weights completed by the machine learning enhancement module
        'python get_add_adj.py'
- ### step 5 
  - #### Run the main code
        'python main.py'
 
