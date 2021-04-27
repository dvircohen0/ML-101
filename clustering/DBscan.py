
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



df=pd.read_csv(r"C:\Users\דביר\Downloads\iris.data",header=None)
df=df.iloc[:,:-1]
df=df.values
dim=(len(df),len(df))
dis=np.zeros(dim)
for i in range(len(df)):
    for j in range(len(df)):
        dis[i,j]=np.linalg.norm(df[i,:]-df[j,:])

def MyDBSCAN(Data, eps, MinPts):
    labels = [0]*len(Data)
    C = 0
    for i in range(len(Data)):
        if (labels[i] == 0):
            Neighbors_i = my_neighbors(Data, i, eps)
            if len(Neighbors_i) < MinPts:
               labels[i] = -1
            else: 
               C += 1
               create_cluster(Data, labels, i, Neighbors_i, C, eps, MinPts)
    return labels



def create_cluster(data,labels,P,neighbors_of_P,Cluster_N,eps,MinPts):
    labels[P]=Cluster_N
    i=0
    while i < len(neighbors_of_P):
        if (labels[neighbors_of_P[i]] == -1 or labels[neighbors_of_P[i]] == 0):
           labels[neighbors_of_P[i]] = Cluster_N
           PnNeighborPts = my_neighbors(data, neighbors_of_P[i], eps)
           if len(PnNeighborPts) >= MinPts:
              neighbors_of_P=neighbors_of_P + PnNeighborPts
        i+=1
                 

def my_neighbors(data,P,eps):
    neighbors=[]
    for i in range(len(data)):
        if dis[P,i] < eps:
            neighbors.append(i)
    return neighbors


my_labels = MyDBSCAN(df, eps=0.8, MinPts=19)

colors=['b','r','y']
for i in range(len(my_labels)):
    for j in range(len(np.unique(my_labels))):
        if my_labels[i]==np.unique(my_labels)[j]:
           plt.scatter(df[i,2],df[i,3],c=colors[j], s=30)
plt.show()
        
my_labels2 = DBSCAN(eps=0.8, min_samples=19).fit_predict(df)

for i in range(len(my_labels2)):
    for j in range(len(np.unique(my_labels2))):
        if my_labels2[i]==np.unique(my_labels2)[j]:
           plt.scatter(df[i,2],df[i,3],c=colors[j], s=30)
plt.show()
