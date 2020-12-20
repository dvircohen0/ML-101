# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:52:05 2020

@author: דביר
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df=pd.read_csv(r"C:\Users\דביר\Downloads\pima-indians-diabetes.csv",header=None).values
y=df[:,-1]
y=(y*2)-1    
X=df[:,:-1]
x_train,x_test,y_train,y_test = train_test_split(X,y)

def hinge_loss(x,y,C,w,b):
    part1 = 1-np.dot(np.dot(x, w) - b, y)
    part2= C * (np.linalg.norm(w)**2)
    return np.maximum(0, part1 ) + part2
  
def svm_SGD(x,y,a,C):
    ww = np.ones(len(x[0]))       # initilizing weight
    bb = 1          # initilizing bias
    iterr = 10000                
    for e in range(iterr):
        for i in range(len(x)):
            val1= np.dot(x[i], ww) - bb
            if (y[i]*val1 < 1):
                ww -= a * ( -y[i]*x[i] + 2*C*ww )
                bb -= a * (y[i])
            else:
                ww -= a * 2*C*ww
                bb -= a * 0
    return ww, bb

def accuracy(predict,actual):
    count=0
    for i in range(len(predict)):
        if predict[i]==actual[i]:
            count+=1
    return(count/len(predict))

start_W= np.ones(len(x_train[0]))  
start_B= 1
C = 0.01
loss_val=hinge_loss(x_train, y_train, C, start_W, start_B)
W, B = svm_SGD(x_train, y_train, 0.001,C)
   

result= []
for i  in range(len(x_test)):
    if np.dot(x_test[i], W) - B > 1:
       result.append(1)
    else:
        result.append(-1)

print('accuracy percentage {:.2%}'.format(accuracy(result,y_test)))     
count, count1=0,0
for i in range(len(y_test)):
    if y_test[i]==1:
        count+=1
        if result[i]==1:
            count1+=1

print('accuracy percentage {:.2%}'.format(count1/count)) 
  
    