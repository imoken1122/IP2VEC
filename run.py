import numpy as np
import padnas as pd
import preprocess as p
import trainer as t

batch_size = 1024
path = ""
X = pd.read_csv(path)
d = X.to_numpy()
w2v,v2w = p._w2v(d)
corpus = pd.DataFrame(p._corpus(d, w2v)).to_numpy()
freq  = p._frequency(d)
train = p._data_loader(corpus, batch_size)


model = t.Trainer(w2v,v2w,freq,emb_dim=32)
model.fit(data = train,max_epoch=50,batch_size=256,neg_num=10
th.save(trainer.model.state_dict(),'ip2vec.pth')
