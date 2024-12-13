import os
import numpy
import torch
"""
configuration file includes all related datasets 
"""
CUDA_ID = 'cuda:2' if torch.cuda.is_available() else 'cpu'

root_data_folder = 'data/'
raw_data_folder = os.path.join(root_data_folder, 'raw_dat')
preprocessed_data_folder = os.path.join(root_data_folder, 'preprocessed')
gene_feature_file = os.path.join(preprocessed_data_folder, 'CosmicHGNC_list.tsv')

#TCGA_datasets
tcga_multi_label_file = os.path.join(root_data_folder, 'TCGA_labels.csv')
tcga_folder = os.path.join(raw_data_folder, 'TCGA')
tcga_sample_file = os.path.join(tcga_folder, 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz')

#CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'zscore.csv')
ccle_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'CCLE_expression.csv')
ccle_sample_file = os.path.join(ccle_folder, 'sample_info.csv')

#gex features
gex_feature_file = os.path.join(preprocessed_data_folder, '1000gene_features.csv')

#GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_raw_target_file = os.path.join(gdsc_folder, 'gdsc_ic50flag.csv')
gdsc_sample_file = os.path.join(gdsc_folder, 'gdsc_cell_line_annotation.csv')
gdsc_preprocessed_target_file = os.path.join(preprocessed_data_folder, 'gdsc_ic50flag.csv')
gdsc_drugs = os.path.join(gdsc_folder, 'drug_smiles.csv')

label_graph = numpy.array(object=object)
label_graph_norm = numpy.array(object=object)

tissue_map = numpy.array(object=object)
drug_feat = numpy.array(object=object)