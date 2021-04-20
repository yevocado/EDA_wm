# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:13:13 2021

@author: user
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:\\windows\\fonts\\malgun.ttf").get_name()
import numpy as np
pd.options.display.float_format = '{:.5f}'.format
#pd.reset_option('display.float_format')
np.set_printoptions(precision=6, suppress=True)
import numpy as np
from sklearn.cluster import DBSCAN
import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib

april = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/April/Product_Korean/Product_Korean2.csv')
april2=april[['mid','lcate']]
april2=pd.get_dummies(april2, columns = ['lcate'])
april3=april2.groupby(by=['mid'], as_index=False).sum()
april3.set_index(['mid'], inplace = True) #열을 index로 지정
april4=april3.iloc[0:1000]
 
input_dim=april4.shape[1] #원래시퀀스수(101?)
encoding_dim=10
input_layer=Input(shape=(input_dim, ))
encoder_layer_1=Dense(64, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder_layer_2=Dense(32, activation='tanh')(encoder_layer_1)
encoder_layer_3=Dense(16, activation='tanh')(encoder_layer_2)
encoder_layer_4=Dense(encoding_dim, activation='tanh')(encoder_layer_3)
#차원축소

scaler=MinMaxScaler()
data_scaled=scaler.fit_transform(april4)
encoder=Model(inputs=input_layer, outputs=encoder_layer_4)
encoded_data=pd.DataFrame(encoder.predict(data_scaled))
encoded_data.columns=['factor_1','factor_2','factor_3','factor_4','factor_5','factor_6','factor_7','factor_8','factor_9','factor_10']

#april = pd.read_csv('C:/Users/user/Desktop/wemakeprice_sample_data/encoded_data.csv')
april=april.drop(columns="Unnamed: 0")
km=KMeans(n_clusters=8, algorithm='auto', init='k-means++', n_init=10, max_iter=300, random_state=1)
km.fit(encoded_data)
predict=pd.DataFrame(km.predict(encoded_data))
 
for i in range(0,8):
    locals()["y_clust_"+str(i)] = np.where(predict == i)
    locals()['p'+str(i)]=[]
#for i in range(len(locals()['y_clyst_'+str(i)][0]))
for x in range(len(locals()['y_clust_'+str(i)][0])):
    locals()['p'+str(i)].append(encoded_data.iloc[locals()['y_clust_'+str(i)][0][x]])

#april2=april[['mid','lcate']]
#april2=pd.get_dummies(april2, columns = ['lcate'])
#april3=april2.groupby(by=['mid'], as_index=False).sum()
#april3.set_index(['mid'], inplace = True) #열을 index로 지정
#april4=april3.iloc[0:1000]
#april_real=april3.iloc[0:1000]

#for i in range(0,8):
 #   locals()["y_clust_"+str(i)] = np.where(predict == i)
  #  locals()['p'+str(i)]=[]

#for x in range(len(locals()['y_clust_'+str(i)][0])):
 #   locals()['p'+str(i)].append(april.iloc[locals()['y_clust_'+str(i)][0][x]])
       
#fig, ax = plt.subplots(1, sharex=True, sharey=True)
#plt.grid()
#fig.autofmt_xdate()
#ax.xaxis.set_ticks(np.arange(0, 96, 10))   ## x축 step(간격) : 5
#plt.xlabel('', fontsize=18)
#plt.xticks(size=10); plt.yticks(size=10)
#plt.ylabel('', size=20)
#plt.ylim(ylim_low,ylim_top)
#for i in range(len(y_clust_3[0])):
 #   plt.plot(april.iloc[y_clust_7[0][i]], 'b', alpha = 0.3)
  #  plt.xticks(rotation=90)
   # p7.append(april.iloc[y_clust_7[0][i]]) 
   
#fig, ax = plt.subplots(1, sharex=True, sharey=True)
#plt.grid()
#fig.autofmt_xdate()
#ax.xaxis.set_ticks(np.arange(0, 96, 10))   ## x축 step(간격) : 5
#plt.xlabel('', fontsize=18)
#plt.xticks(size=10); plt.yticks(size=10)
#plt.ylabel('', size=20)
for x in range(0,8):
    for i in range(len(locals()['y_clust_'+str(x)][0])):
        locals()['p'+str(x)].append(encoded_data.iloc[locals()['y_clust_'+str(x)][0][i]]) 

for x in range(0,8):
    for i in range(len(locals()['y_clust_'+str(x)][0])):
        locals()['y_clust_' + str(x)[0] + 'plot'] = plt.plot(encoded_data.iloc[locals()['y_clust_'+str(x)][0][i]], 'b', alpha = 0.3)
        plt.xticks(rotation=90)
    plt.show()
        #plt.plot(april.iloc[locals()['y_clust_'+str(x)][0][i]], 'b', alpha = 0.3)
        #plt.xticks(rotation=90)


"""#########################################################"""
test=april4
test.reset_index(level=0, inplace=True)
april_fin=pd.concat([test,predict], axis=1)
april_group = april_fin.groupby(april_fin[0]).mean()
april_group=april_group.drop('mid', axis=1)

april_fin.to_csv('C:/Users/user/Desktop/april_fin.csv', index=False, encoding='cp949') 

april_group=april_group.transpose()

for i in range(0, 8):
    plt.figure(figsize=(23, 5))
    plt.plot(april_group.index, april_group[i])
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

joblib.dump(km, 'DEC_model.pkl') 

md_test = km
