# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 22:36:38 2020

@author: דביר
"""

import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\דביר\Downloads\iris.data",header=None)
  
labels=[]
for i in range(len(df)):
    if df.iloc[i,4]=='Iris-setosa':
       labels.append(1)
    else:
        labels.append(0)

df_train,df_test,label_train,label_test = train_test_split(df,labels) 
X=df_train.iloc[:,:2]  

colors = {0:'r', 1:'b'}        
fig, ax = plt.subplots()
##
for i in range(len(X)):
    ax.scatter(X.iloc[i,0], X.iloc[i,1],color=colors[label_train[i]])
ax.set_title('Iris Dataset')
#
label_train=pd.DataFrame(label_train)
#
def logit(z):
    return 1/(1+np.exp(-z))

def hx(w,X):
    z=list()
    y=0
    for i in range(len(X)):
        y=w[0]+w[1]*X.iloc[i,0]+w[2]*X.iloc[i,1]
        z.append(logit(y))
    return pd.DataFrame(z)

def cost(w,X,y):
    h=hx(w,X)
    return(-1*(y*np.log(h)+(1-y))*np.log(1-h)).mean()

    
def grand(w, X, Y):
    total1,total2,total3=0,0,0
    h = hx(w,X)
    for i in range(1,len(X)):
        total1 += h.iloc[i]-Y.iloc[i]
        total2 += (h.iloc[i]-Y.iloc[i])*X.iloc[i,0]
        total3 += (h.iloc[i]-Y.iloc[i])*X.iloc[i,1]
    return total1 / len(X), total2 / len(X), total3 / len(X)

def accuracy(predict,actual):
    count=0
    for i in range(len(predict)):
        if predict[i]==actual[i]:
            count=count+1
    return(count/len(predict))

w=[1,1,1]
learn=0.01
a,b,c=[0,0,0]
n=100

for i in range(n):
    s1, s2, s3  = grand(w, X, label_train)
    a = a - learn * s1
    b = b - learn * s2
    c = c - learn * s3
    w=[a,b,c]
#
#predict
result=list() 
p=list()
   
for i in range(len(df_test)):
    p=w[0]+w[1]*df_test.iloc[i,0]+w[2]*df_test.iloc[i,1]
    p=logit(p)
    print(p)
    if p.values >=0.32:
        result.append(1)
    else: result.append(0)
    
print('accuracy percentage {:.2%}'.format(accuracy(result,label_test))) 
    