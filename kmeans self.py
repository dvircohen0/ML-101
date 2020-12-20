# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:07:19 2020

@author: דביר
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


df=pd.read_csv(r"C:\Users\דביר\Downloads\iris.data",header=None)
df=df.iloc[:,:-1]
df=df.values

plt.scatter(df[:,0], df[:,1], s=20)
plt.show()

K=3
##choosing K random points to be centroids for start: 
centroids_i=np.random.choice(len(df),K,replace=False)
centroids_val=df[centroids_i]
 

def cluster_data(data,centroids):
    clustered_data={}
    dist=[]
    for i in range(len(data)):
        for j in range(K):
            dist.append(np.linalg.norm(data[i]-centroids[j]))
        clustered_data.setdefault(dist.index(min(dist)), []).append(data[i])
        dist=[]
    return clustered_data

#yy=cluster_data(df,centroids) 
 
def new_centroids(clustered_data):
    new_centers=[]
    for i in range(len(clustered_data)):
        new_centers.append(sum(clustered_data.get(i))/len(clustered_data.get(i)))
    return new_centers
#zz= new_centroids(yy)       
centers=centroids_val            
for i in range(70):
    data_class=cluster_data(df,centers)
    centers=new_centroids(data_class)
    cent=np.asarray(centers)
    plt.scatter(cent[:,0],cent[:,1],c='g',s=100)
    for j in range(K):
        dataplot=np.asarray(data_class.get(j))
        dataplot=dataplot[:,:2]
        plt.scatter(dataplot[:,0],dataplot[:,1], s=30)
    plt.show()
    

    
    
    
    
    
    
    