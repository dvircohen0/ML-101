import numpy as np
import matplotlib.pyplot as plt
import os
import imageio as io
from skimage import color
#import cv2 

path = r'C:\faces94\faces94'
h = 200
w = 180
N=50

def read_images(path):
    X = []
    index = 0
    for root, dirs, files in os.walk(path,topdown=True):
        for name in files:
            full_file = os.path.join(root, name)
            if full_file.endswith("1.jpg") and not (full_file.endswith("11.jpg")):
                X.append(color.rgb2gray(io.imread(full_file)).reshape([1,w*h]))
                index +=1
    X = np.array(X)
    X = X.squeeze().T
    i=np.random.randint(0,np.shape(X)[1])
    testimage=X[:,i]
    X=X.T
    X=np.delete(X,i,axis=0)
    X=X.T
    return X,testimage

def find_eigenvectors(data):
    mean_img=np.mean(data,axis=1)
    av_data=(data.T-mean_img).T
    c=np.dot(av_data.T,av_data)
    eig_val, eig_vec = np.linalg.eigh(c)
    eig_vec=np.dot(av_data,eig_vec)#/eig_val**0.5
    eig_vec /= np.linalg.norm(eig_vec, axis=0)
    eig_vec=eig_vec[:,-N:]
    return eig_vec,mean_img

def create_face(X_data,eig_vec,mean_img):
    weight=np.dot(eig_vec.T,(X_data.T-mean_img).T)
    i=np.random.randint(0,len(weight)-1)
    print('Original train image: ')
    plt.imshow(X_data[:,i].reshape((h,w)),cmap=plt.cm.gray)
    plt.show()
    new_img=np.dot(weight[:,i],eig_vec.T)+mean_img
    print('Restored train image: ')
    plt.imshow(new_img.reshape((h,w)),cmap=plt.cm.gray)
    plt.show()
    return new_img,weight

def test_image(image,eig_vec,mean_img):
    print('Original test image: ')
    plt.imshow(image.reshape((h,w)),cmap=plt.cm.gray)
    plt.show()
    weights=np.dot(eig_vec.T,(image-mean_img).T)
    image=np.dot(weights,eig_vec.T)+mean_img
    print('Restored test image: ')
    plt.imshow(image.reshape((h,w)),cmap=plt.cm.gray)
    plt.show()
    return image

X_data,testimage = read_images(path)
eig_vec,mean_img=find_eigenvectors(X_data)
create_face(X_data,eig_vec,mean_img)
test_image(testimage,eig_vec,mean_img)




