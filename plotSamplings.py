import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def get_mean_std(new_dict):
  df = pd.DataFrame(new_dict)
  mean = df.mean(axis=0).values
  std = df.std(axis=0).values
  xdata = list(range(len(new_dict)))
  return xdata, mean, std

vals = []
fnames = ['random.pkl', 'leverage.pkl','coreset.pkl','volumetric.pkl']
for file in fnames:
    with open('./pkls/'+file, 'rb') as f:
        vals.append(pickle.load(f))

std_flag = False
plt.figure(figsize=(20,10))
for i, datas in zip(['red', 'grey', 'blue', 'green'], vals):
  xdata, ydata, dydata= get_mean_std(datas)
  if not std_flag:
    plt.plot(xdata, ydata, color=i)
  else: 
    plt.errorbar(xdata, ydata, yerr = dydata, marker = '.', color = i)
    plt.fill_between(xdata, ydata - dydata, ydata + dydata,
                   color=i, alpha=0.2)

plt.legend(('Uniform', 'Leverage', 'Corsets', 'Volumetric'))
plt.xlabel('Sample %')
plt.title('Comparison between different Sampling Methods')
plt.ylabel('Cost Value')
plt.savefig("./plots/compareCosts"+str(int(std_flag))+".pdf", dpi=300)
plt.show()

std_flag = True
plt.figure(figsize=(20,10))
for i, datas in zip(['red', 'grey', 'blue', 'green'], vals):
  xdata, ydata, dydata= get_mean_std(datas)
  if not std_flag:
    plt.plot(xdata, ydata, color=i)
  else: 
    plt.errorbar(xdata, ydata, yerr = dydata, marker = '.', color = i)
    plt.fill_between(xdata, ydata - dydata, ydata + dydata,
                   color=i, alpha=0.2)

plt.legend(('Uniform', 'Leverage', 'Corsets', 'Volumetric'))
plt.title('Comparison between different Sampling Methods with std_dev')
plt.xlabel('Sample %')
plt.ylabel('Cost Value')
plt.savefig("./plots/compareCosts"+str(int(std_flag))+".pdf", dpi=300)
plt.show()
