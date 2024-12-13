import gzip
import random
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, AllChem
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader
from operator import itemgetter
import config
from copy import deepcopy

pd.set_option('future.no_silent_downcasting', True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def str_collate(batch):
    return np.array(batch)

def get_unlabeled_dataloaders(gex_features_df, seed, test_ratio, batch_size):
    """CCLE as source domain, TCGA as target domain"""
    set_seed(seed)
    ccle_sample_info_df = pd.read_csv(config.ccle_sample_file, index_col=0)
    with gzip.open(config.tcga_sample_file) as f:
        tcga_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
    tcga_samples = tcga_sample_info_df.index.intersection(gex_features_df.index)
    ccle_samples = gex_features_df.index.difference(tcga_samples)
    tcga_sample_info_df = tcga_sample_info_df.loc[tcga_samples]
    ccle_sample_info_df = ccle_sample_info_df.loc[ccle_samples.intersection(ccle_sample_info_df.index)]
    tcga_df = gex_features_df.loc[tcga_samples]
    ccle_df = gex_features_df.loc[ccle_samples]
    
    #ensuring that samples can be 'train-test' separately according to the tissue type
    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.OncotreePrimaryDisease.value_counts()[
        ccle_sample_info_df.OncotreePrimaryDisease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.OncotreePrimaryDisease.isin(excluded_ccle_diseases)].index)

    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]
    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=test_ratio,
                                                   stratify=ccle_sample_info_df.loc[to_split_ccle_df.index].OncotreePrimaryDisease, random_state=seed)
    ccle_df = pd.concat([train_ccle_df, ccle_df.loc[excluded_ccle_samples]])
    
    train_tcga_df, test_tcga_df = train_test_split(tcga_df, test_size=test_ratio,
                                                   stratify=tcga_sample_info_df['_primary_disease'], random_state=seed)
    #Build a mapping dictionary for all tissue types
    ccle_tissue_type = list(ccle_df.Tissue.value_counts().index)
    tcga_tissue_type = list(tcga_df.Tissue.value_counts().index)
    tissue_type = set(ccle_tissue_type + tcga_tissue_type)
    config.tissue_map = dict(zip(tissue_type, list(range(0,len(tissue_type)))))

    #Construct the tensor dataset
    train_tcga_dataset = TensorDataset(torch.from_numpy(tcga_df.drop(columns='Tissue').values.astype('float32')), 
                                       torch.from_numpy(np.array(itemgetter(*tcga_df.Tissue)(config.tissue_map))))
    train_ccle_dataset = TensorDataset(torch.from_numpy(ccle_df.drop(columns='Tissue').values.astype('float32')), 
                                       torch.from_numpy(np.array(itemgetter(*ccle_df.Tissue)(config.tissue_map))))
    test_tcga_dateset = TensorDataset(torch.from_numpy(test_tcga_df.drop(columns='Tissue').values.astype('float32')),
                                      torch.from_numpy(np.array(itemgetter(*test_tcga_df.Tissue)(config.tissue_map))))
    test_ccle_dateset = TensorDataset(torch.from_numpy(test_ccle_df.drop(columns='Tissue').values.astype('float32')),
                                      torch.from_numpy(np.array(itemgetter(*test_ccle_df.Tissue)(config.tissue_map))))

    train_tcga_dataloader = DataLoader(train_tcga_dataset, batch_size=batch_size, shuffle=False)
    test_tcga_dataloader = DataLoader(test_tcga_dateset, batch_size=batch_size, shuffle=False)
    train_ccle_dataloader = DataLoader(train_ccle_dataset, batch_size=batch_size, shuffle=False)
    test_ccle_dataloader = DataLoader(test_ccle_dateset, batch_size=batch_size, shuffle=False)

    return (train_ccle_dataloader, test_ccle_dataloader), (train_tcga_dataloader, test_tcga_dataloader)


def get_ccle_multi_labeled_dataloader(gex_features_df, batch_size, drug, seed, measurement,
                                      threshold_gdsc, threshold_label, n_splits=5):
    drugs_to_keep = [item.lower() for item in drug]
    gex_features_df = gex_features_df.drop(columns='Tissue')
    # read file
    gdsc1_response = pd.read_csv(config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(config.gdsc_target_file2)

    # filter data columns, only get three columns COSMIC_ID, DRUG_NAME and measurement(IC50,AUC,RMSE,Z_SCORE)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    # filter drugs
    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]

    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()

    # Delete the duplicates in gdsc1_target_df and gdsc2_target_df
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    # Take the "COSMIC_ID" as the new row index and take the "DRUG_NAME" as the column index
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')

    # read ccle sample info, get the "COSMICID" column
    ccle_sample_info = pd.read_csv(config.ccle_sample_file, index_col=1)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    # read gdsc sample info, get the "COSMIC identifier" column
    gdsc_sample_info = pd.read_csv(config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')

    # inner join with index
    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, 
                                                 how='inner')[['ModelID']]
    # the index is the COSMIC_ID and the value is the DepMap_ID
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['ModelID']
    
    # change the data index, the final index is the DepMap_ID. Values with no mapping relationship are set to NaN.
    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]
    ccle_target_df = target_df[drugs_to_keep]
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)
    
    # Get the tissue type for each sample
    tmp = ccle_sample_info.set_index("ModelID")
    samples = tmp.loc[ccle_labeled_samples]
    tissues = np.array(itemgetter(*samples.Tissue)(config.tissue_map))
    
    # According to the set of threshold, label the data
    ccle_labels = ccle_target_df.loc[ccle_labeled_samples]
    masked_labels = (~np.array(ccle_labels.isnull())).astype(int)
    if threshold_gdsc is None:
        threshold_gdsc = list(ccle_labels.mean(axis=0))
    
    for i in range(len(ccle_labels)):
        for j in range(len(drugs_to_keep)):
            if np.isnan(ccle_labels.iloc[i, j]):
                ccle_labels.iloc[i, j] = 0
            else:                     
                # ccle_labels.iloc[i, j] = (ccle_labels.iloc[i, j] < threshold_gdsc[j]).astype(int) 
                ccle_labels.iloc[i, j] = (ccle_labels.iloc[i, j] < threshold_gdsc).astype(int)  
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]  
    
    '''Init label graph (sample occurence smiliarty)''' 
    label_graph = np.eye(len(drugs_to_keep), dtype=float)
    drug_id_map = dict()
    for idx, item in enumerate(drugs_to_keep):
        drug_id_map[item] = idx
    
    for idx, item in enumerate(drugs_to_keep):
        occurence_dict = dict()
        for i in range(len(ccle_labels)):
            if ccle_labels.iloc[i,idx] == 1:
                for j in range(len(ccle_labels.iloc[i])):
                    if j == idx:
                        continue
                    if ccle_labels.iloc[i,j] == 1:
                        if drugs_to_keep[j] in occurence_dict.keys():
                            occurence_dict[drugs_to_keep[j]] = occurence_dict[drugs_to_keep[j]] + 1
                        else:
                            occurence_dict[drugs_to_keep[j]] = 1
        occurence_rank = sorted(occurence_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(occurence_rank)):
            label_graph[idx][drug_id_map[occurence_rank[i][0]]] = occurence_rank[i][1]
            
    row, col = np.diag_indices_from(label_graph)
    label_graph[row, col] = np.array(ccle_labels.sum().values.tolist())
    for idx_col in range(label_graph.shape[1]):
        normalizer = np.sum(label_graph[:, idx_col])
        label_graph[:, idx_col] = label_graph[:, idx_col] * 1.0 / normalizer
    print("Normalizered Label Graph============================================================")
    config.label_graph = label_graph - np.diag(np.diag(label_graph))
    config.label_graph = (config.label_graph >= threshold_label).astype(int)
    print(config.label_graph) 

    # Node feature setting of label graph
    SMILES = pd.read_csv(config.gdsc_drugs, header=0, index_col=0)
    SMILES.index = SMILES.index.map(str.lower)
    mol_list = [Chem.MolFromSmiles(x) for x in list(SMILES.loc[drugs_to_keep]['Isosmiles'])]
    #fp_list = np.array([MACCSkeys.GenMACCSKeys(x) for x in mol_list])
    fp_list = np.array([Chem.RDKFingerprint(x, fpSize=64) for x in mol_list])
    config.drug_feat = fp_list

    # check the processed data
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    kfold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_index, test_index in kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df, = ccle_labeled_feature_df.values[train_index], \
                                                       ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]
        train_ccle_mask, test_ccle_mask = masked_labels[train_index], masked_labels[test_index]
        train_ccle_tissue, test_ccle_tissue = tissues[train_index], tissues[test_index]

        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(train_labeled_ccle_df.astype('float32')),
            torch.from_numpy(train_ccle_labels.astype('float32')),
            torch.from_numpy(train_ccle_mask.astype('float32')),
            torch.from_numpy(train_ccle_tissue))
            
        test_labeled_ccle_df = TensorDataset(
            torch.from_numpy(test_labeled_ccle_df.astype('float32')),
            torch.from_numpy(test_ccle_labels.astype('float32')),
            torch.from_numpy(test_ccle_mask.astype('float32')),
            torch.from_numpy(test_ccle_tissue))

        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset, batch_size=batch_size, shuffle=False)
        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df, batch_size=batch_size, shuffle=False)

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader

def get_tcga_multi_labeled_dataloaders(gex_features_df, drug, batch_size):
    # Filter by beginning string "TCGA"
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df = tcga_gex_feature_df.drop(columns='Tissue')
    # Take the first 12 characters of the original string as the new data id.
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    # Group by the new id and get the average of each column as the features.
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()
    # TCGA label preprocessing
    tcga_labeled_df = pd.read_csv(config.tcga_multi_label_file, header=0, index_col=0)
    intersection_index = tcga_gex_feature_df.index.intersection(tcga_labeled_df.index)
    sample_tissue = np.array(itemgetter(*tcga_labeled_df.Cancer)(config.tissue_map))
    
    tcga_labeled_df = tcga_labeled_df.drop(columns='Cancer').loc[intersection_index] 
    tcga_labeled_df = tcga_labeled_df[drug]
    tcga_labeled_gex_feature_df = tcga_gex_feature_df.loc[intersection_index]
    masked_labels = (~np.array(tcga_labeled_df.isnull())).astype(int)
    print("number of non-NA labels: {}".format(np.sum(masked_labels)))
    # Binarize labels  
    Binarize={'Complete Response':1.0, 'Partial Response':0.0, 'Clinical Progressive Disease':0.0,
              'Stable Disease':0.0, np.nan:-1.0}
    
    tcga_labeled_df = tcga_labeled_df.replace(Binarize, regex=False)
    # Dataloading
    tcga_data = TensorDataset(torch.from_numpy(tcga_labeled_gex_feature_df.values.astype('float32')), 
                  torch.from_numpy(tcga_labeled_df.values.astype('float32')),
                  torch.from_numpy(masked_labels.astype('float32')),
                  torch.from_numpy(sample_tissue))
    tcga_dataloader = DataLoader(tcga_data, batch_size=batch_size, shuffle=False)
    
    return tcga_dataloader


def get_multi_labeled_dataloader(gex_features_df, drug, seed, batch_size, ccle_measurement,
                                 threshold_gdsc, threshold_label, n_splits=5):
    tcga_labeled_dataloader = get_tcga_multi_labeled_dataloaders(gex_features_df=gex_features_df,                                                   
                                                                 drug=drug, batch_size=batch_size)
    
    ccle_labeled_dataloader = get_ccle_multi_labeled_dataloader(gex_features_df=gex_features_df,
                                                                batch_size=batch_size,
                                                                drug=drug, seed=seed,
                                                                measurement=ccle_measurement,
                                                                threshold_gdsc=threshold_gdsc,
                                                                threshold_label=threshold_label,
                                                                n_splits=n_splits)
    
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, tcga_labeled_dataloader