import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def get_mean_std2(new_dict):
  df = pd.DataFrame(new_dict)
  mean = df.mean(axis=0).values
  std = df.std(axis=0).values
  xdata = list((5, 10, 20))
  return xdata, mean, std

''' Changes with sample size ... 
'''


fnames = ['Indrandom.pkl', 'Indleverage.pkl','Indcoreset.pkl','Indvolumetric.pkl']
for file in fnames:
    pers = ('2%', '5%', '8%', '10%')
    with open('./pkls/cost'+file, 'rb') as f:
        vals = pickle.load(f)

    std_flag = False
    plt.figure(figsize=(20,10))
    for i, datas in zip(['red', 'grey', 'blue', 'green'], vals.values()):
      xdata, ydata, dydata= get_mean_std2(datas)
      plt.plot(xdata, ydata, color=i)
      if std_flag:
#         plt.errorbar(xdata, ydata, yerr = dydata, marker = '.', color = i)
        plt.fill_between(xdata, ydata - dydata, ydata + dydata,
                       color=i, alpha=0.2)

    plt.legend(pers)
    plt.xlabel('# of iterations')
    plt.ylabel('Cost Value')
    plt.savefig("costsample"+str(std_flag)+file+".pdf", dpi=300)
    plt.show()

