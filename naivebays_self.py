# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:42:51 2020

@author: דביר
"""

import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
import math


#import dataset
dataset=pd.read_csv(r"C:\Users\דביר\Downloads\pima-indians-diabetes.csv",header=None)
#split into test and train
train,test = train_test_split(dataset)   
#split into 2 groups of labels
group1 = train[train[8] == 1]
group2 = train[train[8] == 0]
#calculate p for each group
group1_p=len(group1)/len(dataset)
group2_p=len(group2)/len(dataset)

#func calculte the mean and std
def mean_std_vactor(x):
    mean0=list()
    std0=list()
    for i in range(len(x.columns)-1):
        mean0.append(x.iloc[:,i].mean())
        std0.append(x.iloc[:,i].std())
    return mean0,std0
#func calculte the gausseian p given x mean and std
def gausse(x,mean,std):
    gausse=(1/(np.sqrt(2*math.pi)*std))*np.exp(-(x-mean)**2/(2*std**2))
    return gausse
#func calculte the accuracy
def accuracy(predict,actual):
    count=0
    for i in range(len(predict)):
        if predict[i]==actual[i]:
            count=count+1
    return(count/len(predict))
#calculate mean and std for every feature in both groups    
mean1,std1=mean_std_vactor(group1)
mean2,std2=mean_std_vactor(group2)
#create temp veraible
p1,p2,result=list(),list(),list()

for j in range(len(test)):
    for i in range(len(mean1)):
        #calculate sum of product of fetures p gausse for each group
        p1.append(gausse(test.iloc[j,i],mean1[i],std1[i]))
        p2.append(gausse(test.iloc[j,i],mean2[i],std2[i]))
        #check which group is more likely
    if np.prod(p1)*group1_p>=np.prod(p2)*group2_p:
       result.append(1)
    else:
       result.append(0)
    p1,p2=list(),list()
    
print('accuracy percentage {:.2%}'.format(accuracy(result,test.iloc[:,-1].tolist())))   
    
    
     
     