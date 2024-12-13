import sys  
import pandas as pd
import torch
import json
import os
import argparse
import itertools
import dataload
import config
import pretraining, classifier, finetuning
from copy import deepcopy
from utility import set_seed_all
import time
import numpy as np
import warnings
from torch_geometric.nn.inits import reset, uniform

def wrap_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])
    return aux_dict

def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])

def main(args, drug, params_dict):
    start_time = time.time()
    device = config.CUDA_ID
    set_seed_all(2024)
    # Load mix gene expressions for both ~9000 tcga and ~1000 cell line
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    # Load traning params
    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)
    training_params['unlabeled'].update(params_dict)
    training_params['labeled'].update(params_dict)
    param_str = dict_to_str(params_dict)
    model_save_folder = os.path.join('model_save')

    training_params.update({'device': device, 
                            'input_dim': gex_features_df.shape[1]-1, 
                            'model_save_folder': os.path.join(model_save_folder, param_str),
                            'retrain_flag': args.retrain_flag, 
                            'norm_flag': args.norm_flag,
                            'alph': args.alph,
                            'beta': args.beta})
    
    make_dir(training_params['model_save_folder'])
    
    # Data construction for the pre-training
    s_dataloaders, t_dataloaders = dataload.get_unlabeled_dataloaders(gex_features_df=gex_features_df,                                            
                                                                    seed=2024,
                                                                    test_ratio=0.25,
                                                                    batch_size=training_params['unlabeled']['batch_size'])

    # Start pretraining, obtain shared encoder
    encoder = pretraining.training(s_dataloaders=s_dataloaders,
                                   t_dataloaders=t_dataloaders,
                                   **wrap_params(training_params, type='unlabeled'))
    
    type_prototypes = pretraining.get_prototype(s_dataloaders=s_dataloaders,
                                   t_dataloaders=t_dataloaders,
                                   encoder=encoder,
                                   device=training_params['device'])
    
    # Data construction for the fine-tuning
    labeled_dataloader = dataload.get_multi_labeled_dataloader(gex_features_df=gex_features_df,
                                                            seed=2024,
                                                            batch_size=training_params['labeled']['batch_size'],
                                                            drug=drug,
                                                            ccle_measurement=args.measurement,
                                                            threshold_gdsc=args.thres_g,
                                                            threshold_label=args.thres_s,
                                                            n_splits=args.n) 
    # start finetuning, obtain UDA predictor
    fold = 0
    all_results = []
    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:
        print('\n################ Dataset Fold: {} ################'.format(fold))
        ft_encoder = deepcopy(encoder)
        print('================Drugs:================\n', drug)
        # Initial training of the multi-label (drug) predictor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = classifier.multi_training(encoder=ft_encoder,
                                                train_dataloader=train_labeled_ccle,
                                                val_dataloader=test_labeled_ccle,
                                                drug=drug,
                                                **wrap_params(training_params, type='labeled'))
            # Store the best GDSC drug response predictor
            predictor.load_state_dict(torch.load(os.path.join(model_save_folder, param_str, 'predictor.pt')))
            
            # Domain adversarial training for the TCGA dataset
            network = finetuning.training(encoder=ft_encoder,
                                            classifier=predictor,
                                            s_dataloader=train_labeled_ccle,
                                            t_dataloader=labeled_tcga,
                                            drug=drug,
                                            prototype=type_prototypes, 
                                            params_str=param_str,
                                            **wrap_params(training_params, type='labeled'))
            
            # Transfer drug response prediction for TCGA dataset   
            print("\n================Transfer testing for TCGA data================")   
            test_loss, results, y_true, y_pred, y_mask = finetuning.testing(model=network,
                                            t_dataloader=labeled_tcga,
                                            drug=drug,
                                            device=training_params['device'])
            print('testing loss : {:.4f}'.format(test_loss))
            print('test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(np.around(results.mean(axis=1), 4)))   
            # sample_save_folder = os.path.join('results', 'fold-'+str(fold), 'samples', param_str)
            # make_dir(sample_save_folder) 
            # np.save(os.path.join(sample_save_folder, 'y_true.npy'), y_true)
            # np.save(os.path.join(sample_save_folder, 'y_pred.npy'), y_pred)
            # np.save(os.path.join(sample_save_folder, 'y_mask.npy'), y_mask) 
        """
        from utility import classification_metric, edge_extract
        device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        node_x = torch.from_numpy(config.drug_feat.astype('float32'))
        node_x = node_x.to(device)
        edge_index = edge_extract(config.label_graph)
        edge_index = torch.from_numpy(edge_index.astype('int'))
        edge_index = edge_index.to(device)
        s_all, t_all = [], []
        s_types, t_types = [], []
        for i, batch in enumerate(train_labeled_ccle):
            s_ty = batch[3].to(device)
            s_x = batch[0].to(device)
            _, _, s_feat = network(s_x, 0, node_x, edge_index)
            s_all.append(s_feat)
            s_types.append(s_ty)
        for i, batch in enumerate(test_labeled_ccle):
            s_ty = batch[3].to(device)
            s_x = batch[0].to(device)
            _, _, s_feat = network(s_x, 0, node_x, edge_index)
            s_all.append(s_feat)
            s_types.append(s_ty)
        for i, batch in enumerate(labeled_tcga):
            t_ty = batch[3].to(device)
            t_x = batch[0].to(device)
            _, _, t_feat = network(t_x, 0, node_x, edge_index)
            t_all.append(t_feat)
            t_types.append(t_ty)
        s_all=torch.cat(s_all, dim=0); s_types=torch.cat(s_types, dim=0) 
        t_all=torch.cat(t_all, dim=0); t_types=torch.cat(t_types, dim=0)
        np.save('s_all.npy',s_all.cpu().detach().numpy())
        np.save('t_all.npy',t_all.cpu().detach().numpy())
        np.save('s_types.npy',s_types.cpu().detach().numpy())
        np.save('t_types.npy',t_types.cpu().detach().numpy())
        """

        results = pd.DataFrame(results)
        results.columns = drug; results.index = ['auc','aupr','f1','acc']
        task_save_folder = os.path.join('results', 'fold-'+str(fold))
        make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str)
        with open(f'{file_name}.csv', 'w') as f:
            results.to_csv(f)      
        fold = fold+1
        all_results.append(results)
        
    # Calculate the average result of 5-CV    
    avg_result = np.mean(np.array(all_results), 0)  
    avg_result = pd.DataFrame(avg_result)
    avg_result.columns = drug; avg_result.index = ['auc','aupr','f1','acc']
    file_name = os.path.join('results', param_str)
    with open(f'{file_name}.csv', 'w') as f:
        avg_result.to_csv(f)      
    
    # Computer the model running time   
    elapsed = time.time() - start_time
    print('9-drug elapsed time: ', round(elapsed, 4)/5)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining and Fine_tuning')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE', choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--alph', dest='alph', nargs='?', type=float, default=0.2, help='Coefficient of transfer loss')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.3, help='Coefficient of contrastive loss')

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)
    
    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    params_grid = {
        "pretrain_num_epochs": [100, 200, 300],
        "uda_num_epochs": [300, 400, 500]
    }

    keys, values = zip(*params_grid.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param in params_list:      
        main(args=args, params_dict=param,
             drug=["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", 
                   "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"])