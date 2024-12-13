import os
import torch
import numpy as np
from models import ConnectNetwork, GraphMLP
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, auc, precision_recall_curve
import config
from utility import classification_metric, edge_extract

def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def eval_epoch(model, node_x, edge_index, loader, loss_fn, device):
    y_true, y_pred, y_mask = [], [], []
    auc_list, aupr_list, f1_list, acc_list = [], [], [], []
    avg_loss = 0
    edge_index = edge_index.to(device)
    node_x = node_x.to(device)
    for x, y, mask, _ in loader:
        x = x.to(device)
        y = y.to(device)
        mask_tmp = (mask > 0).to(device)
        with torch.no_grad():
            yp = model(x, node_x, edge_index)
            yp = yp.squeeze(dim=1)
            loss_mat = loss_fn(yp, y.double())
            loss_mat = torch.where(mask_tmp, loss_mat,
                                   torch.zeros(loss_mat.shape).to(device))
            loss = torch.sum(loss_mat) / torch.sum(mask_tmp)
            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()
            avg_loss += loss / x.size(0)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)
    # results = classification_metric(y_true, y_pred)
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            auc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            aupr_list.append(auprc(y_true[is_valid, i], y_pred[is_valid, i]))
            f1_list.append(f1_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
            acc_list.append(accuracy_score(y_true[is_valid, i], (y_pred[is_valid, i]>=0.5).astype('int')))
        else:
            print('{} is invalid'.format(i))
    all_results = [auc_list, aupr_list, acc_list, f1_list]
    
    return np.array(all_results), avg_loss.cpu().detach().item()

def multi_classifier_train_step(model, node_x, edge_index, batch, device, optimizer, loss_fn, scheduler=None):
    x = batch[0].to(device)
    y = batch[1].to(device)
    mask = batch[2].to(device)
    mask = (mask > 0)
    edge_index = edge_index.to(device)
    node_x = node_x.to(device)

    yp = model(x, node_x, edge_index)
    loss_mat = loss_fn(yp, y.double())
    loss_mat = torch.where(mask, loss_mat,
                           torch.zeros(loss_mat.shape).to(device))
    loss = torch.sum(loss_mat) / torch.sum(mask)
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()
    return loss / x.size(0)

def multi_training(encoder, train_dataloader, val_dataloader, drug, **kwargs):
    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    classifier = GraphMLP(input_dim=kwargs['latent_dim']+node_x.size(1), output_dim=1,
                              hidden_dims=kwargs['classifier_hidden_dims'],
                              drug_num=len(drug),
                              drop=kwargs['drop']).to(kwargs['device'])
    predictor = ConnectNetwork(encoder, classifier, noise_flag=False).to(kwargs['device'])
    
    cl_optimizer = torch.optim.AdamW(predictor.parameters(), lr=kwargs['lr'])
    classifier_loss = nn.BCEWithLogitsLoss(reduction='none') #Filter itself
    print("\n================Pre-training for drug response classifier================")
    best_auc = 0
    for epoch in range(int(kwargs['train_num_epochs'])):
        train_loss = 0
        # Training    
        predictor.train()
        cl_optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            loss = multi_classifier_train_step(model=predictor,
                                        node_x=node_x,
                                        edge_index=edge_index,
                                        batch=batch,
                                        device=kwargs['device'],
                                        optimizer=cl_optimizer,
                                        loss_fn=classifier_loss)
            train_loss += loss.cpu().detach().item()
        if (epoch+1) % 50 == 0:
            print('classification training epoch = {}'.format(epoch+1))
            print('training loss : {:.4f}'.format(train_loss))
        # Evaluating
        predictor.eval()
        metrics, valid_loss = eval_epoch(model=predictor,
                                    node_x=node_x,
                                    edge_index=edge_index,
                                    loader=val_dataloader,
                                    loss_fn=classifier_loss,
                                    device=kwargs['device'])
        if epoch % 50 == 0:
            print('validating loss : {:.4f}'.format(valid_loss))
            print('validation avg (GDSC) metrics (auc,aupr,f1,acc): {}'.format(np.around(metrics.mean(axis=1), 4)))
            
        if metrics[0].mean() > best_auc:
            torch.save(predictor.decoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'predictor.pt'))
            best_auc = metrics[0].mean()
    
    return predictor.decoder
