import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
from tqdm import tqdm_notebook as tqdm
import numpy as np
import random
from model import Skipgram
class Trainer:
    def __init__(self,w2v,v2w,freq,emb_dim):
        self.v2w = v2w
        self.w2v = w2v
        self.unigram_table = self.noise(w2v,freq)
        self.vocab_size = len(w2v)
        self.model = Skipgram(self.vocab_size,emb_dim).to(device)
        self.optim = optim.Adam(self.model.parameters())
        
    def noise(self,w2v, freq):
        unigram_table = []
        total_word = sum([c for c in freq.values()])
        for w,v in w2v.items():
            unigram_table.extend([v]*int(((freq[w]/total_word)**0.75)/0.001))
        return unigram_table
    
    def negative_sampling(self,batch_size,neg_num,batch_target):
        neg = np.zeros((neg_num))
        for i in range(batch_size):
            sample = random.sample(self.unigram_table, neg_num)
            while batch_target[i] in sample:
                sample = random.sample(self.unigram_table, neg_num)
            neg = np.vstack([neg,sample])
        return neg[1:batch_size+1]
    
    def fit(self,data,max_epoch,batch_size,neg_num):
        run_losses = []
        for epoch in range(max_epoch):
            run_loss = 0
            
            for batch in tqdm(data):
                
                context,target = batch[:,1],batch[:,0]
                self.optim.zero_grad()
                batch_neg = self.negative_sampling(batch_size,neg_num,target)
                context = V(th.LongTensor(context)).to(device)
                target = V(th.LongTensor(target)).to(device)
                batch_neg = V(th.LongTensor(batch_neg.astype(int))).to(device)
                
                loss = self.model(target, context, batch_neg) 
                loss.backward()
                self.optim.step()
#                run_loss += loss.cpu().item()
 #           run_losses.append(run_loss/len(data)) 
            print(epoch,run_loss)
        return run_losses
    def most_similar(self,word,top):
        W = self.model.state_dict()["u_embedding.weight"]
        idx = w2v[word]
        similar_score = {}
        for i,vec in enumerate(W):
            if i != idx:
                d = vec.dot(W[idx])
                similar_score[self.v2w[i]] = d
        similar_score = sorted(similar_score.items(), key=lambda x: -x[1])[:top]
        for k,v in similar_score:
            print(k,":",round(v.item(),2))

