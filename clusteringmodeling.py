# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:13:13 2021

@author: user
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns
import numpy as np
pd.options.display.float_format = '{:.5f}'.format
#pd.reset_option('display.float_format')
np.set_printoptions(precision=6, suppress=True)
import torch
import numpy as np
from sklearn.cluster import DBSCAN



april = pd.read_csv('C:/Users/user\Desktop/wemakeprice_sample_data/April/Product_Korean/Product_Korean2.csv')
april2=april[['mid','lcate']]
april2=pd.get_dummies(april2, columns = ['lcate'])
april3=april2.groupby(by=['mid'], as_index=False).sum()
april3.set_index(['mid'], inplace = True) #열을 index로 지정
april4=april3.iloc[0:1000]

km=KMeans(n_clusters=10, algorithm='auto', init='k-means++', n_init=10, max_iter=300)
km.fit(april4)
predict=pd.DataFrame(km.predict(april4))

for i in range(0,11):
    locals()["y_clust_"+str(i)] = np.where(predict == i)
    locals()['p'+str(i)]=[]

#for i in range(len(locals()['y_clyst_'+str(i)][0]))
    for x in range(len(locals()['y_clust_'+str(i)][0])):
        locals()['p'+str(i)].append(april4.iloc[locals()['y_clust_'+str(i)][0][x]])
        
p0 = []
fig, ax = plt.subplots(1, sharex=True, sharey=True)
plt.grid()
fig.autofmt_xdate()
#ax.xaxis.set_ticks(np.arange(0, 96, 10))   ## x축 step(간격) : 5
plt.xlabel('L_cate', fontsize=18)
plt.xticks(size=10); plt.yticks(size=10)
plt.ylabel('구매횟수', size=20)
#plt.ylim(ylim_low,ylim_top)
for i in range(len(y_clust_0[0])):
    plt.plot(april4.iloc[y_clust_0[0][i]], 'b', alpha = 0.3)
    plt.xticks(rotation=90)
    p0.append(april4.iloc[y_clust_0[0][i]]) 
    
"""#########################################################"""

model = DBSCAN(min_samples=5)
predict = pd.DataFrame(model.fit_predict(april4))
r = pd.concat([feature,predict],axis=1)
ct = pd.crosstab(data['labels'],r['predict'])
print (ct)

for i in range(0,11):
    locals()["y_clust_"+str(i)] = np.where(predict == i)
    locals()['p'+str(i)]=[]