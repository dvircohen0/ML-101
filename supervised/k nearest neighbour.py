import numpy as np
from sklearn.datasets import load_iris
from math import sqrt

dataset=load_iris()
dataset=np.column_stack((dataset.data, dataset.target))
np.random.shuffle(dataset)

def train_test_split(data,split):
    train_size = int(np.round(split*np.size(data,0)))
    return data[:train_size,:-1], data[train_size:,:-1], data[:train_size,-1], data[train_size:,-1]
#, y_train, y_test

def distance(x,y):
    dis=0.0
    for i in range(len(x)-1):
        dis+=(x[i]-y[i])**2
    return sqrt(dis)  
x_train, x_test, y_train, y_test=train_test_split(dataset,0.6) 


def nearest_neighbours(row,dataset,k):
    dis=[]
    index=[]
    for i in range(len(dataset)-1):
        dis.append(distance(row,dataset[i]))
    index=np.argsort(dis)  
    dis=np.sort(dis) 
    return dis[:k],index[:k]
#        
zz,yy=nearest_neighbours(x_test[1],x_test,3)

def voting_function(labels,index):
    value=[]
    for i in range(len(index)):
        value.append(labels[index[i]])
    return max(set(value), key = value.count)

def accuracy(predict,actual):
    count=0
    for i in range(len(predict)):
        if predict[i]==actual[i]:
            count=count+1
    return(count/len(predict))

dis,index,predict=[],[],[]
for i in range(len(x_test)):
    dis,index=nearest_neighbours(x_test[i],x_train,5)
    predict.append(voting_function(y_train,index))
    
   

print('accuracy percentage {:.2%}'.format(accuracy(predict,y_test)))    
 
    
    
#        
#    
#    

