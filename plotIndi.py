import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def get_mean_std2(new_dict):
  '''for a perticular percent you get max_iters and seeds'''
  df = pd.DataFrame(new_dict)
  mean = df.mean(axis=0).values
  std = df.std(axis=0).values
  xdata = tuple(i for i in range(1, 10)) # iters
  return xdata, mean, std


fnames = ['random', 'leverage','coreset','volumetric']
for file in fnames:
    pers = ('2%', '5%', '8%', '10%')
    with open('./pkls/costInd'+file+'.pkl', 'rb') as f:
        vals = pickle.load(f)

    std_flag = False
    plt.figure(figsize=(20,10))
    for i, datas in zip(['red', 'grey', 'blue', 'green'], vals.values()): # for diff percents
      xdata, ydata, dydata= get_mean_std2(datas)
      plt.plot(xdata, ydata, color=i)
      if std_flag:
#         plt.errorbar(xdata, ydata, yerr = dydata, marker = '.', color = i)
        plt.fill_between(xdata, ydata - dydata, ydata + dydata,
                       color=i, alpha=0.2)

    plt.legend(pers)
    plt.xlabel('# of iterations')
    plt.ylabel('Cost Value')
    plt.title('Cost vs # of iters: ' + file)
    plt.savefig("./plots/costsampleInd"+str(int(std_flag))+file+".pdf", dpi=300)
    plt.show()

for file in fnames:
    pers = ('2%', '5%', '8%', '10%')
    with open('./pkls/costInd'+file+'.pkl', 'rb') as f:
        vals = pickle.load(f)

    std_flag = True
    plt.figure(figsize=(20,10))
    for i, datas in zip(['red', 'grey', 'blue', 'green'], vals.values()): # for diff percents
      xdata, ydata, dydata= get_mean_std2(datas)
      plt.plot(xdata, ydata, color=i)
      if std_flag:
#         plt.errorbar(xdata, ydata, yerr = dydata, marker = '.', color = i)
        plt.fill_between(xdata, ydata - dydata, ydata + dydata,
                       color=i, alpha=0.2)

    plt.legend(pers)
    plt.title('Comparison between different Sampling Methods')
    plt.xlabel('# of iterations')
    plt.ylabel('Cost Value')
    plt.savefig("./plots/costsampleInd"+str(int(std_flag))+file+".jpg", dpi=300)
    plt.show()

