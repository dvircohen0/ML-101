
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df=pd.read_csv(r"C:\Users\דביר\Downloads\iris.data",header=None)
df=df.iloc[:,:-1]
df=df.values
n, d = df.shape
k=3
mean = df[np.random.choice(n, k)]
Sigma= [np.eye(d)] * k
w = [1/k] * k
R = np.zeros((n, k))


#
def prob (sigma,mean): 
    Data = df - mean
    prob = np.sum(np.dot(Data, np.linalg.inv(sigma))*Data, 1)
    prob = np.exp(-0.5*prob)/np.sqrt((np.power((2*np.pi), d))*np.absolute(np.linalg.det(sigma)))      
    return prob 

def E_Step(R):
    for i in range(k):
        R[:, i] = w[i] * prob (Sigma[i],mean[i])
    R = (R.T / np.sum(R, axis = 1)).T
    weight_sum = np.sum(R, axis = 0)
    return  weight_sum ,R
#
def M_step(weight_sum ):
    for i in range(k):
        mean[i] = 1/weight_sum [i] * np.sum(R[:, i] * df.T, axis = 1).T
        x_mu = np.matrix(df - mean[i])
        Sigma[i] = np.array(1 / weight_sum [i] * np.dot(np.multiply(x_mu.T,  R[:, i]), x_mu))
        w[i] = 1/n * weight_sum [i]
#

max_iters = 1000
for i in range(max_iters):
    N_ks, R = E_Step(R)
    M_step(N_ks)

   
idx=np.argmax(R, axis=1)
classes=np.unique(idx)   
clusters={}

for i in range(len(idx)):
    for j in range(len(classes)):
        if idx[i]==classes[j]:
           clusters.setdefault(classes[j], []).append(df[i])

colors=['b','r','y'] 
for j in range(len(classes)):
    dataplot=np.asarray(clusters.get(j))
    plt.scatter(dataplot[:,1],dataplot[:,2],c=colors[j], s=30)
plt.show()       
