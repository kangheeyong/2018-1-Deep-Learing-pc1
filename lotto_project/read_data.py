import numpy as np
import pickle
import gzip


data = np.genfromtxt('lotto_1_787.csv',delimiter=',')
data[(0,0)] = 5

one_hot = np.eye(45)




a = one_hot[data[:,0].astype(np.int32)-1]
for i in range(1,6) :
    a = a + one_hot[data[:,i].astype(np.int32)-1]
b = one_hot[data[:,6].astype(np.int32)-1]

a = np.reshape(a,(-1,1,45,1))
b = np.reshape(b,(-1,1,45,1))

c = np.concatenate([a,b],3)


with gzip.open('lotto_1_787.pickle.gzip','wb') as f :
    pickle.dump(c,f)


with gzip.open('lotto_1_787.pickle.gzip','rb') as f :
    d = pickle.load(f)


print(d.shape)
