# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:02:31 2020

@author: דביר
"""
import numpy as np

pi=np.array([0.6,0.4])
A=np.array([[0.69,0.3,0.01],[0.4,0.59,0.01]])
B=np.array([[0.5,0.4,0.1],[0.1,0.3,0.6]])
y=np.array([0,1,2])
S=[1,2]


def forward_backward(observations, states, start_prob, trans_prob,emm_prob):
    fw=np.zeros(([len(states),len(observations)]))
    bw=np.zeros(([len(states),len(observations)]))
    posterior=np.zeros(([len(states),len(observations)]))
    fw[:,0]=start_prob
    for obs in range(len(observations)):
        for st in range(len(states)):
            for k in range(len(states)):
                if obs==0:
                    fw[st,obs]=start_prob[st]*emm_prob[st,observations[obs]]
                else:
                    fw[st,obs]+=fw[k,obs-1]*trans_prob[k,st]*emm_prob[st,observations[obs]]
                    
    for obs in range(len(observations)):
        for st in range(len(states)):
            for k in range(len(states)): 
                if obs==0:
                    bw[st,len(states)-obs]=trans_prob[st,-1]
                else:
                    bw[st,len(states)-obs]+=bw[k,len(states)-obs+1]*trans_prob[st,k]*emm_prob[k,observations[len(states)-obs]+1]

    p_fw=sum(fw[:,-1]*trans_prob[:,-1])
    p_bw=sum(bw[:,0]*start_prob*emm_prob[:,observations[0]])
    for i in range(len(observations)):
        for j in range(len(states)):
            posterior[j,i]=fw[j,i]*bw[j,i]/p_fw
     
    return fw,bw,posterior
    
fw,bw,posterior=forward_backward(y,S,pi,A,B)
print('Forward values')
print('Healthy:',fw[0,:])
print('Fever:',fw[1,:])
print('Backward values')
print('Healthy:',bw[0,:])
print('Fever:',bw[1,:])
print('Combined values')
print('Healthy:',posterior[0,:])
print('Fever:',posterior[1,:])
    
    
