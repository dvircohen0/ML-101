
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

var=1
LR=200
itter=500
momentom=0.99
PERPLEXITY=5
data=pd.read_csv(r"C:\Users\דביר\Downloads\iris.data",header=None)
data=data.iloc[:,:-1]
data=data.values
#data=np.random.rand(10,4)
#data=np.tile(data,(2,1))
#data[:10]*=0.1
n=[len(data),len(data)]
dis=np.zeros(n)
dis2=np.zeros(n)
p_val=np.zeros(n)
N=len(data)
var=1
data_2d=data.dot(np.random.rand(data.shape[1],2))
data_2d-=np.mean(data_2d)
data_2d/=np.std(data_2d)


def k_neighbours(data,x1_index,p_or_q='p'):
    x1=data[x1_index]
    list_k_neighbours=[]
    for i in range(N):
        if i!=x1_index:
            xi=data[i]
            if p_or_q=='p':
                distance=np.exp(-np.linalg.norm(x1-xi)**2/(2*var**2))
            else:
                distance=(1+np.linalg.norm(x1-xi)**2)**-1
            list_k_neighbours.append(distance)
    
    list_k_neighbours=sorted(list_k_neighbours)
    return list_k_neighbours[-PERPLEXITY:]

for i in range(N):
    for j in range(N):
        dis[i,j]=np.exp(-np.linalg.norm(data[i]-data[j])**2/2*var**2)

for i in range(N):
    for j in range(N):
        dis2[i,j]=(1+np.linalg.norm(data_2d[i]-data_2d[j])**2)**-1


def Pji(x1_index,x2_index):
    pij=dis[x1_index,x2_index]/sum(k_neighbours(data,x1_index,'p'))
    pji=dis[x2_index,x1_index]/sum(k_neighbours(data,x2_index,'p'))
    return (pji+pij)/(2*N)

def Qji(y1_index,y2_index):
    qij=dis2[y1_index,y2_index]/sum(k_neighbours(data,y1_index,'q'))
    qji=dis2[y2_index,y1_index]/sum(k_neighbours(data,y2_index,'q')) 
    return (qji+qij)/(2*N)

def Q_table(data_2d):
    q_val=np.zeros(n)
    for i in range(N):
        for j in range(N):
            if i!=j:
               q_val[i,j]=Qji(i,j) 
    return q_val
    
for i in range(N):
    for j in range(N):
        if i!=j:
           p_val[i,j]=Pji(i,j)
           
q_val=Q_table(data_2d)

def gradient(data_2d):
    dim=data_2d.shape
    history=np.zeros(dim)
    history2=np.zeros(dim)
    global q_val
    
    for k in range(itter):
        for i in range(N):
            count=0
            for j in range(N):
                count+=((p_val[j,i]-q_val[j,i])*(data_2d[i]-data_2d[j])*(1+np.linalg.norm(data_2d[i]-data_2d[j]**2))**-1)
            data_2d[i]-= 4*LR*count + momentom*(history[i]-history2[i])
            history2[i]=history[i]
            history[i]=data_2d[i]
    q_val=Q_table(data_2d)
    data_2d-=np.mean(data_2d)
    data_2d/=np.std(data_2d)   
    return data_2d
##            
#
color=['blue']*50+['red']*50+['green']*50
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2],color=color)
plt.show()
#

plotty=gradient(data_2d)
##
#
plt.scatter(plotty[:,0],plotty[:,1],color=color)
plt.show()



