#!/usr/bin/env python
# coding: utf-8

# In[7]:


# references
# https://las.inf.ethz.ch/files/bachem18scalable.pdf
#


# In[8]:


# get_ipython().run_line_magic('pdb', '')


# In[9]:


# from google.colab import drive
# drive.mount('/content/gdrive')
# loc = "/content/gdrive/My\ Drive/IITGn\ Files/Semester\ 7/IDS/Project/ploting_files"


# In[51]:


import numpy as np
np.seterr(all='warn')
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import normalize


# In[52]:


class Levarage:
    def __init__(self, data, mode='reduced'):
        '''Assuming the datapoints are in rows'''
        data
        self.len = data.shape[0] # number 
        self.pdf = self._find_multinomial_distrib(data, mode) # saving pdf for sampling
    
    def sample_ix(self, percent, seed):
        np.random.seed(seed)
        '''samples percent points and returns thier indexes'''
        num_of_sampes = int(self.len * percent / 100)
        return np.random.choice(range(self.len), num_of_sampes, p=self.pdf)
    
    def _find_multinomial_distrib(self, data, mode):
        q, r = np.linalg.qr(data, mode=mode)
        q_row_norm2 = np.linalg.norm(q, ord=2, axis=1)**2 # assumed the rows are dpts
        q_fnorm = np.linalg.norm(q, ord='fro')**2
        return q_row_norm2 / q_fnorm


# In[53]:


class Sampling:
  '''def __init__(self): return'''

  def new_objective(self, data, new_centres):
    obj = 0
    for point in data:
      if point not in new_centres:
        distance = [sum((point-i)**2) for i in new_centres]
        obj += min(distance)
    return obj

  def sampling(self, data, percent = 4, seed=5):    #n x d 
    raise NotImplemented

  def plot_objective(self, data):
    return


# In[54]:


class Lev(Sampling):
  def sampling(self, data, percent=1, seed=5):
    lev_sampler = Levarage(data)
    ix = lev_sampler.sample_ix(percent=percent, seed=seed)
    return data[ix]

class Corsets(Sampling):
  def sampling(self, data, percent, seed):
    '''m is number of samples'''
    np.random.seed(seed)
    mean = np.mean(data, axis=0)
    assert(mean.shape[0] == data.shape[1])

    n = data.shape[0]
    q = np.zeros(n)
    cons = 1/(2*n)
    mult = 1/2
    deno = np.sum(np.linalg.norm(data - mean, axis=1)**2)
    num = np.linalg.norm(data - mean, axis=1)**2
    q = cons + mult*num/deno

    num_of_samples = int(n * percent / 100)
    ix = np.random.choice(np.arange(n), num_of_samples, p=q)

    return data[ix]

class Rand(Sampling):
  def sampling(self, data, percent = 1, seed=5):    #n x d 
    np.random.seed(seed)
    sampling_idx = np.random.randint(0,len(data),size=int(percent*len(data)/100))
    return data[sampling_idx]
  
class Volumetric(Sampling):
  def sampling(self, data, percent=4, seed=3):
    '''n x d matrix : data input'''
    np.random.seed(seed)
    X = np.transpose(data)
    X = normalize(X)
    Z = np.linalg.inv(np.dot(X,X.T))
    Z = normalize(Z)
    prob = np.zeros(X.shape[1])
    n = X.shape[1]
    number = percent/100*n
    for i in range(X.shape[1]):
       prob[i] = 1 - X[:,i].dot(Z).dot(X[:,i])
    S = [i for i in range(n)]
    prob=prob/prob.sum()
    while(len(S)>number):
      i = np.random.choice(np.arange(len(S)),p=prob)
      S.remove(S[i])
      v = Z.dot(X[:,i])/np.sqrt(prob[i])
      prob = np.delete(prob,i)
      for j in range(len(S)):
         temp = prob[j] - (X[:,j].dot(v))**2
      prob = prob/prob.sum()
      Z = Z + v.dot(np.transpose(v))
      Z = normalize(Z)
    return data[S]


# In[55]:


kdd_data_b =  pd.read_csv("./Dataset/bio_train.dat",delimiter='\t',header=None)
kdd_data_b = np.array(kdd_data_b)
chota = 8000 # full ke liye -1
kdd_data = kdd_data_b[:chota,3:]
print(kdd_data.shape)


# In[56]:


def get_costs(sampler,
              name,
              kdd_data,
              n_cluster,
              max_iters,
              tolerance,
              seeds,
              percents):
  states = {}
  for percent in percents:
    states['percent'+str(percent)] = {}
    for max_itern in max_iters:
      states['percent'+str(percent)]['max_iter'+str(max_itern)] = {}
      for seed in seeds:
        km = KMeans(n_clusters = n_cluster,
                    max_iter = max_itern,
                    tol = tolerance,
                    random_state = seed,
                    init='random',
                    n_jobs=-1)
        sampled_data = sampler.sampling(kdd_data,
                                        percent = percent,
                                        seed = seed)
        km.fit(sampled_data)
        new_centres = km.cluster_centers_
        post_sampling_cost = sampler.new_objective(kdd_data, new_centres)
        states['percent'+str(percent)]['max_iter'+str(max_itern)]['seed'+str(seed)] = post_sampling_cost
  with open('./pkls/costInd'+name+".pkl", 'wb') as f:
    pickle.dump(states, f)
  return states


# In[57]:


# params = {
#   'kdd_data': kdd_data,
#   'n_cluster' : 15,
#   'max_itern' : tuple(i for i in range(1, 10)),
#   'tolerance' : 1e-20,
#   'seeds' : (2, 4, 63),
#   'percents' : (2, 5, 8, 10),
# }


# In[58]:


kdd_data = kdd_data
n_cluster = 15
max_itern =  tuple(i for i in range(1, 10))
tolerance = 1e-20
seeds = (2, 4, 63)
percents = (2, 5, 8, 10)


# In[49]:


from multiprocessing import Process

if __name__ == '__main__':
    p1 = Process(target=get_costs, args=(Rand(), 'random', kdd_data,
                                                            n_cluster,
                                                            max_itern,
                                                            tolerance,
                                                            seeds,
                                                            percents,))
    p2 = Process(target=get_costs, args=(Lev(), 'leverage', kdd_data,
                                                            n_cluster,
                                                            max_itern,
                                                            tolerance,
                                                            seeds,
                                                            percents,))
    p3 = Process(target=get_costs, args=(Corsets(), 'coreset', kdd_data,
                                                            n_cluster,
                                                            max_itern,
                                                            tolerance,
                                                            seeds,
                                                            percents,))
    p4 = Process(target=get_costs, args=(Volumetric(), 'volumetric', kdd_data,
                                                            n_cluster,
                                                            max_itern,
                                                            tolerance,
                                                            seeds,
                                                            percents,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
