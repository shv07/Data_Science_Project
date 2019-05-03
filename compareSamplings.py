# references
# https://las.inf.ethz.ch/files/bachem18scalable.pdf

import numpy as np
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.preprocessing import normalize


# In[50]:


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


# In[51]:


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


# In[52]:


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


# In[53]:


kdd_data_b =  pd.read_csv("./Dataset/bio_train.dat",delimiter='\t',header=None)
kdd_data_b = np.array(kdd_data_b)
chota = 8000 # full ke liye -1
kdd_data = kdd_data_b[:chota,3:]
print(kdd_data.shape)


# In[54]:


def get_costs(sampler,
              name,
              kdd_data,
              n_cluster,
              max_itern,
              tolerance,
              seeds,
              percents):
  
  states = {}
  for percent in percents:
#     print (percents)
    states['percent'+str(percent)] = {}
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
      states['percent'+str(percent)]['seed'+str(seed)] = post_sampling_cost
  with open('./pkls/'+name+".pkl", 'wb') as f:
        pickle.dump(states, f)
  return states


# In[55]:


params = {
  'kdd_data': kdd_data,
  'n_cluster' : 15,
  'max_itern' : 20,
  'tolerance' : 1e-20,
  'seeds' : (2, 4, 5),
  'percents' : tuple(i for i in range(1, 11)),
}


# In[56]:


kdd_data = kdd_data
n_cluster = 15
max_itern = 20
tolerance = 1e-20
seeds = (2, 4, 5)
percents = tuple(i for i in range(1, 11))


# In[ ]:


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


# def get_mean_std(new_dict):
#   df = pd.DataFrame(new_dict)
#   mean = df.mean(axis=0).values
#   std = df.std(axis=0).values
#   xdata = list(range(len(new_dict)))

# std_flag = False
# plt.figure(figsize=(20,10))
# for i, datas in zip(['red', 'grey', 'blue', 'green'], [rands, levs, cors, vols]):
#   xdata, ydata, dydata= get_mean_std(datas)
#   if not std_flag:
#     plt.plot(xdata, ydata, color=i)
#   else: 
#     plt.errorbar(xdata, ydata, yerr = dydata, marker = '.', color = i)
#     plt.fill_between(xdata, ydata - dydata, ydata + dydata,
#                    color=i, alpha=0.2)
# 
# plt.legend(('Random', 'Leverage', 'Corsets', 'Volumetric'))
# plt.xlabel('Sample %')
# plt.ylabel('Cost Value')
# plt.show()

# # Lightweight Coreset Sampling
# n_cluster = 5
# max_itern = 400
# tolerance = 1e-6
# sample_size = 0.6
# 
# #km = KMeans(n_clusters = n_cluster, max_iter = max_itern, tol = tolerance)
# #km.fit(kdd_data)
# #pre_sampling_cost = km.inertia_
# 
# LWC_sampled_data = sampler.light_weight_coresets(kdd_data, sample_size)
# km.fit(LWC_sampled_data)
# new_centres = km.cluster_centers_
# post_sampling_cost = sampler.new_objective(kdd_data, new_centres)

# pre_sampling_cost*10**(-11), post_sampling_cost*10**(-11)

# np.abs(np.array([1,2,2])-np.array([4,5,6]))

# In[ ]:




