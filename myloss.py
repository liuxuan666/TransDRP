# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from utility import uniform
import config
device = config.CUDA_ID

def Adversarial_loss(domain_s, domain_t, BCEloss):
    # Initialize the pseudo-labels of the domain confrontation
    domain_labels = torch.cat((domain_s, domain_t), dim=0) 
    bs_s = domain_s.size(0)
    bs_t = domain_t.size(0)
    target_labels = torch.from_numpy(np.array([[0]]*bs_s + [[1]]*bs_t).astype('float32')).to(device)
    domain_loss = BCEloss(domain_labels, target_labels) 
    
    return domain_loss


class InfoMax_loss(nn.Module):
    def __init__(self, hidden_dim, EPS = 1e-8):
        super(InfoMax_loss,self).__init__()
        self.EPS = EPS
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(self.hidden_dim, self.weight)
    
    def discriminate(self, z, summary, sigmoid: bool = True):
        """computes the probability scores assigned to this patch(z)-summary pair.
        Args:
            z (torch.Tensor): The latent space os samples.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
        """
        #summary = summary.t() if summary.dim() > 1 else summary
        z = torch.unsqueeze(z, dim = 1)
        summary = torch.unsqueeze(summary, dim = -1)
        #value = torch.matmul(z, torch.matmul(self.weight.to(device), summary))
        value = torch.matmul(z, summary)
        value = value.squeeze()
        mask = torch.where(value != 0.0)[0]

        return torch.sigmoid(value[mask]) if sigmoid else value[mask]
    
    def process(self, data_type, p_key, p_value):
        summary = []
        for i in data_type:
            idx = torch.where(p_key == i)
            if (len(idx[0]) != 0):
                summary.append(p_value[idx[0]])
            else:
                summary.append(torch.zeros(1, self.hidden_dim).to(device))
                
        return torch.cat(summary, dim = 0)
    
    def forward(self, s_feat, t_feat, s_type, t_type, prototype):
        """Computes the mutual information maximization objective"""
        batch_s = next(iter(prototype[0]))
        batch_t = next(iter(prototype[1]))
        # Extract the anchor (source) prototypes 
        pos_summary_t = self.process(t_type, batch_s[0].to(device), batch_s[1].to(device))
        neg_summary_t = self.process(t_type, batch_t[0].to(device), batch_t[1].to(device))
        # Extract the anchor (target) prototypes
        # pos_summary_s = self.process(s_type, batch_t[0].to(device), batch_t[1].to(device))
        # neg_summary_s = self.process(s_type, batch_s[0].to(device), batch_s[1].to(device))
        # Compute the contrastive loss of InfoMax
        pos_loss_t = -torch.log(self.discriminate(t_feat, pos_summary_t, sigmoid=True) + self.EPS).mean()
        neg_loss_t = -torch.log(1-self.discriminate(t_feat, neg_summary_t, sigmoid=True) + self.EPS).mean()

        # pos_loss_s = -torch.log(self.discriminate(s_feat, pos_summary_s, sigmoid=True) + self.EPS).mean()
        # neg_loss_s = -torch.log(1-self.discriminate(s_feat, neg_summary_s, sigmoid=True) + self.EPS).mean()

        return (pos_loss_t + neg_loss_t)/2 #+ (pos_loss_s + neg_loss_s)/2