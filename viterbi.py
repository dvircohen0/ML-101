
import numpy as np


pi = np.array([[0.04, 0.02, 0.06, 0.04, 0.11, 0.11, 0.01, 0.09, 0.03, 0.05, 0.06, 0.11, 0.05, 0.11, 0.03, 0.08]]).T
A = np.array([ \
    [0.08, 0.02, 0.10, 0.05, 0.07, 0.08, 0.07, 0.04, 0.08, 0.10, 0.07, 0.02, 0.01, 0.10, 0.09, 0.01], \
    [0.06, 0.10, 0.11, 0.01, 0.04, 0.11, 0.04, 0.07, 0.08, 0.10, 0.08, 0.02, 0.09, 0.05, 0.02, 0.02], \
    [0.08, 0.07, 0.08, 0.07, 0.01, 0.03, 0.10, 0.02, 0.07, 0.03, 0.06, 0.08, 0.03, 0.10, 0.10, 0.08], \
    [0.08, 0.04, 0.04, 0.05, 0.07, 0.08, 0.01, 0.08, 0.10, 0.07, 0.11, 0.01, 0.05, 0.04, 0.11, 0.06], \
    [0.03, 0.03, 0.08, 0.10, 0.11, 0.04, 0.06, 0.03, 0.03, 0.08, 0.03, 0.07, 0.10, 0.11, 0.07, 0.03], \
    [0.02, 0.05, 0.01, 0.09, 0.05, 0.09, 0.05, 0.12, 0.09, 0.07, 0.01, 0.07, 0.05, 0.05, 0.11, 0.06], \
    [0.11, 0.05, 0.10, 0.07, 0.01, 0.08, 0.05, 0.03, 0.03, 0.10, 0.01, 0.10, 0.08, 0.09, 0.07, 0.02], \
    [0.03, 0.02, 0.16, 0.01, 0.05, 0.01, 0.14, 0.14, 0.02, 0.05, 0.01, 0.09, 0.07, 0.14, 0.03, 0.01], \
    [0.01, 0.09, 0.13, 0.01, 0.02, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.06, 0.11, 0.06, 0.03, 0.14], \
    [0.09, 0.03, 0.04, 0.05, 0.04, 0.03, 0.12, 0.04, 0.07, 0.02, 0.07, 0.10, 0.11, 0.03, 0.06, 0.09], \
    [0.09, 0.04, 0.06, 0.06, 0.05, 0.07, 0.05, 0.01, 0.05, 0.10, 0.04, 0.08, 0.05, 0.08, 0.08, 0.10], \
    [0.07, 0.06, 0.01, 0.07, 0.06, 0.09, 0.01, 0.06, 0.07, 0.07, 0.08, 0.06, 0.01, 0.11, 0.09, 0.05], \
    [0.03, 0.04, 0.06, 0.06, 0.06, 0.05, 0.02, 0.10, 0.11, 0.07, 0.09, 0.05, 0.05, 0.05, 0.11, 0.08], \
    [0.04, 0.03, 0.04, 0.09, 0.10, 0.09, 0.08, 0.06, 0.04, 0.07, 0.09, 0.02, 0.05, 0.08, 0.04, 0.09], \
    [0.05, 0.07, 0.02, 0.08, 0.06, 0.08, 0.05, 0.05, 0.07, 0.06, 0.10, 0.07, 0.03, 0.05, 0.06, 0.10], \
    [0.11, 0.03, 0.02, 0.11, 0.11, 0.01, 0.02, 0.08, 0.05, 0.08, 0.11, 0.03, 0.02, 0.10, 0.01, 0.11]])
B = np.array([[0.01,0.99], \
                [0.58,0.42], \
                [0.48,0.52], \
                [0.58,0.42], \
                [0.37,0.63], \
                [0.33,0.67], \
                [0.51,0.49], \
                [0.28,0.72], \
                [0.35,0.65], \
                [0.61,0.39], \
                [0.97,0.03], \
                [0.87,0.13], \
                [0.46,0.54], \
                [0.55,0.45], \
                [0.23,0.77], \
                [0.76,0.24]])
y = [[0, 0, 0, 0, 0, 0, 1, 0, 1, 1], [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]]


def predict_S(B,y,A,pi):
    LP=len(pi)
    LY=len(y)
    T1=np.zeros((LP,LY))
    for i in range(LP):
        T1[i,0]=pi[i]*B[i,1] if y[0]==1 else pi[i]*B[i,0]
    T2=np.zeros((LP,LY))
    temp=np.zeros(LP)
    for k in range(1,LY): 
        for i in range(LP):
            P=B[i,1] if y[k]==1 else B[i,0]
            for j in range(LP):
                temp[j]=T1[j,k-1]*P*A[j,i]
            T1[i,k]=max(temp)
            T2[i,k]=np.argmax(temp) 
    s=np.zeros(LY)
    s[0]=np.argmax(T1[:,LY-1])
    for i in range(1,LY):
        s[i]=T2[int(s[i-1]),LY-i]
    s=np.flip(s)    
    return s

for i in range(np.shape(y)[0]) :
    print(predict_S(B,y[i],A,pi))
    
    
    

    
