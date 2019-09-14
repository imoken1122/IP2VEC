import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
import numpy as np
import random

class Skipgram(nn.Module):
    def __init__(self,vocab_size,emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.u_embedding = nn.Embedding(vocab_size,emb_dim)
        self.v_embedding = nn.Embedding(vocab_size,emb_dim)
        self.log_sigmoid = nn.LogSigmoid()
        
        init_range= 0.5/emb_dim
        self.u_embedding.weight.data.uniform_(-init_range,init_range)
        self.v_embedding.weight.data.uniform_(-0,0)
        
    def forward(self, target, context,neg):
        v_embedd = self.u_embedding(target)
        u_embedd = self.v_embedding(context)
        positive = self.log_sigmoid(th.sum(u_embedd * v_embedd, dim =1)).squeeze()
        
        u_hat = self.v_embedding(neg)
        #negative_ = th.bmm(u_hat, v_embedd.unsqueeze(2)).squeeze(2)
        negative_ = (v_embedd.unsqueeze(1) * u_hat).sum(2)
        negative = self.log_sigmoid(-th.sum(negative_,dim=1)).squeeze()
        
        loss = positive + negative
        return -loss.mean()

