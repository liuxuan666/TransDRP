import os
import torch
import numpy as np
from models import EncoderDecoder, FeatMLP
import config
from itertools import chain
from utility import PrototypeData
from torch.utils.data import DataLoader
from itertools import cycle

def eval_epoch(model, loader, device):
    avg_loss = 0
    for x_batch in loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            inputs, recon, z = model(x_batch)
            loss = model.loss_function(inputs, recon, z)
            avg_loss += loss.cpu().detach().item() / x_batch.size(0)
            
    return avg_loss

def ae_train_step(model_s, model_t, s_batch, t_batch, device, optimizer):
    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)
    input_s, recon_s, z_s = model_s(s_x)
    input_t, recon_t, z_t = model_t(t_x)
    s_loss = model_s.loss_function(input_s, recon_s, z_s)
    t_loss = model_t.loss_function(input_t, recon_t, z_t)
    loss = s_loss + t_loss
    loss.backward()
    optimizer.step()
    
    return loss.cpu().detach().item() / s_x.size(0)

def training(s_dataloaders, t_dataloaders, **kwargs):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]
    
    shared_encoder = FeatMLP(input_dim = kwargs['input_dim'],
                         output_dim = kwargs['latent_dim'],
                         hidden_dims = kwargs['encoder_hidden_dims'],
                         drop = kwargs['drop']).to(kwargs['device'])   
    shared_decoder = FeatMLP(input_dim = 2 * kwargs['latent_dim'],
                         output_dim = kwargs['input_dim'],
                         hidden_dims = kwargs['decoder_hidden_dims'],
                         drop = kwargs['drop']).to(kwargs['device'])
    s_AE = EncoderDecoder(encoder = shared_encoder,
                    decoder = shared_decoder,
                    input_dim = kwargs['input_dim'],
                    output_dim = kwargs['latent_dim'],
                    hidden_dims = kwargs['encoder_hidden_dims'],
                    drop = kwargs['drop'],
                    norm_flag = kwargs['norm_flag'],
                    noise_flag = True).to(kwargs['device'])
    t_AE = EncoderDecoder(encoder = shared_encoder,
                    decoder = shared_decoder,
                    input_dim = kwargs['input_dim'],
                    output_dim = kwargs['latent_dim'],
                    hidden_dims = kwargs['encoder_hidden_dims'],
                    drop = kwargs['drop'],
                    norm_flag = kwargs['norm_flag'],
                    noise_flag = True).to(kwargs['device'])
    AE_params = [s_AE.private_encoder.parameters(),
                 t_AE.private_encoder.parameters(),
                 shared_encoder.parameters(),
                 shared_decoder.parameters()]

    #AE = EncoderDecoder(shared_encoder, shared_decoder, noise_flag=True)
    ae_optimizer = torch.optim.AdamW(chain(*AE_params), lr=kwargs['lr'])
    best_threshold = np.inf
    print("================Pre-training for feature extraction================")
    if kwargs['retrain_flag']:
        # starting pre-training for domain feature 
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            t_AE.train(); s_AE.train()
            ae_optimizer.zero_grad()
            for i, batch in enumerate(zip(t_train, cycle(s_train))):
                train_loss = ae_train_step(model_s=s_AE, model_t=t_AE,
                                           s_batch=batch[1], t_batch=batch[0],
                                           device=kwargs['device'], optimizer=ae_optimizer)
                train_loss_all += train_loss
            
            t_AE.eval(); s_AE.eval()
            s_val_loss = eval_epoch(model=s_AE, loader=s_test, device=kwargs['device'])
            t_val_loss = eval_epoch(model=t_AE, loader=t_test, device=kwargs['device'])
            
            val_loss_all = s_val_loss + t_val_loss
            if (epoch+1) % 50 == 0:
                print('AE training epoch = {}, loss = {:.4f}'.format(epoch+1, val_loss_all))
            if (val_loss_all < best_threshold):
                torch.save(s_AE.state_dict(), os.path.join(kwargs['model_save_folder'], 's_AE.pt'))
                torch.save(t_AE.state_dict(), os.path.join(kwargs['model_save_folder'], 't_AE.pt'))
                best_threshold = val_loss_all
    else:
        try:
            s_AE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 's_AE.pt')))
            t_AE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_AE.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")
    t_AE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_AE.pt')))    
    
    return t_AE.shared_encoder


def get_prototype(s_dataloaders, t_dataloaders, encoder, device):
    s_train = s_dataloaders[0]
    t_train = t_dataloaders[0]
    prototype_s = {}; prototype_t = {}
    #Get tisssue prototype for source domain
    all_feat_s = []; all_type_s =[]
    encoder.eval()
    with torch.no_grad():
        for step, s_batch in enumerate(s_train):
            s_x = s_batch[0].to(device) #sample feature
            s_l = s_batch[1].to(device) #sample tissue type
            s_f = encoder(s_x)
            all_feat_s.append(s_f); all_type_s.append(s_l)
        all_feat_s = torch.cat(all_feat_s, dim = 0); all_type_s = torch.cat(all_type_s, dim = 0)
        for i in range(len(config.tissue_map)):
            idx = torch.where(all_type_s==i)
            if idx[0].size(0) != 0:
                prototype_s[i] = all_feat_s[idx[0],:].mean(dim=0).cpu().detach().numpy()
                
        #Get tisssue prototype for target domain
        all_feat_t = []; all_type_t =[]
        for step, t_batch in enumerate(t_train):
            t_x = t_batch[0].to(device) #sample feature
            t_l = t_batch[1].to(device) #sample tissue type
            t_f = encoder(t_x)
            all_feat_t.append(t_f); all_type_t.append(t_l)
        all_feat_t = torch.cat(all_feat_t, dim = 0); all_type_t = torch.cat(all_type_t, dim = 0)
        for i in range(len(config.tissue_map)):
            idx = torch.where(all_type_t==i)
            if idx[0].size(0) != 0:
                prototype_t[i] = all_feat_t[idx[0],:].mean(dim=0).cpu().detach().numpy()   
    
    # Convert the dict format to the tensor format
    s_type_id = np.array(list(prototype_s.keys()))
    t_type_id = np.array(list(prototype_t.keys()))
    s_type_prototype = np.array(list(prototype_s.values()))
    t_type_prototype = np.array(list(prototype_t.values()))
    
    source = PrototypeData(torch.from_numpy(s_type_id.astype('int')), 
                           torch.from_numpy(s_type_prototype.astype('float32')))
    target = PrototypeData(torch.from_numpy(t_type_id.astype('int')), 
                           torch.from_numpy(t_type_prototype.astype('float32')))
    
    prototype_source = DataLoader(source, batch_size=s_type_prototype.shape[0], shuffle=False)
    prototype_target = DataLoader(target, batch_size=t_type_prototype.shape[0], shuffle=False)
    
    return [prototype_source, prototype_target]
