# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:46:26 2020

@author: דביר
"""

import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r"C:\Users\דביר\Downloads\wdbc.data",header=None)
X_train,X_test= train_test_split(dataset)  

def split(data,i,value):
    left = data[data[i] >= value]
    right = data[data[i] < value]
    return left,right

def check_purity(X):
    label_column = X.iloc[:,1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False
    
def classify_data(data):
    label_column = data.iloc[:,1].tolist()
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification    
    
def gini(Y1,Y2):
    size_L=len(Y1)
    size_R=len(Y2)
    M_L=Y1.count('M')
    B_L=size_L-M_L
    M_R=Y2.count('M')
    B_R=size_R-M_R
    P_l=1-1*(M_L/size_L)**2 -1*(B_L/size_L)**2 
    P_r=1-1*(M_R/size_R)**2 -1*(B_R/size_R)**2 
    gini_value=(P_l*size_L/(size_L+size_R))+(P_r*size_R/(size_L+size_R))
    return gini_value


def find_best_gini(X):
    mean1,gini1=list(),list()
    for i in range(2,len(X.columns)):
        mean1.append(X[i].mean())
        L,R=split(X,i,X[i].mean())
        gini_val=gini(L.iloc[:,1].tolist(),R.iloc[:,1].tolist())
        gini1.append(gini_val)
    return gini1.index(min(gini1))+2,mean1[gini1.index(min(gini1))]

def accuracy(predict,actual):
    count=0
    for i in range(len(predict)):
        if predict[i]==actual[i]:
            count+=1
    return(count/len(predict))

  
def decision_tree_algorithm(data, counter=0, min_samples=2, max_depth=4):
   
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification
    else:    
        counter += 1
        index,value=find_best_gini(data)
        L,R = split(data, index, value)
        question = "{} <= {}".format(index, value)
        sub_tree = {question: []}
        yes_answer = decision_tree_algorithm(L, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(R, counter, min_samples, max_depth)
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        return sub_tree

def classify(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    feature_name=int(feature_name)
    if example[feature_name] <= float(value):
        answer = tree[question][1]
    else:
        answer = tree[question][0]
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify(example, residual_tree)    
    
tree=decision_tree_algorithm(X_train, counter=0, min_samples=2, max_depth=5)
result=list()
for i in range(len(X_test)):
    result.append(classify(X_test.iloc[i,:].tolist(),tree))
result=pd.DataFrame(result)  
print('accuracy percentage {:.2%}'.format(accuracy(result.iloc[:,0],X_test.iloc[:,1].tolist())))    