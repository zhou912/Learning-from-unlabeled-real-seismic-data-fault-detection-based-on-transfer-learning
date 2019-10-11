#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:54:12 2019

@author: zrs
"""

from keras.models import load_model
from unet3 import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target
from utils import *
from matplotlib.colors import Normalize
from PIL import Image

np.set_printoptions(threshold = np.inf)
# load json and create model 
json_file = open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model3.h5")
print("Loaded model from disk")

gx,m1,m2,m3 = np.fromfile("data/validation/seis/7.dat",dtype=np.single),128,128,128
gx = gx-np.min(gx)
gx = gx/np.max(gx)
k = 50
x = np.reshape(gx,(1,128,128,128,1))
Y = loaded_model.predict(x,verbose=1)
print(Y.shape)

# Y1 = Y[0]
# Y2 = Y[1]
# Y3 = Y[2]
# Y4 = Y[3]
# Y5 = Y[4]
#Y6 = Y[5]
fig = plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
imgplot1 = plt.imshow(np.transpose(x[0,k,:,:,0]),cmap=plt.cm.bone,interpolation='nearest',aspect=1)
'''
plt.subplot(1, 2, 2)
imgplot2 = plt.imshow(np.transpose(Y[0,k,:,:,0]),cmap=plt.cm.bone,interpolation='nearest',aspect=1)
'''
# training image dimensions
n1, n2, n3 = 128, 128, 128


'''
train_target_x = []   
new_raw_data = np.load("new_raw_data.npy").astype(np.float32)
#print (new_raw_data)
m1,m2,m3 = 401,1301,321
gs = np.zeros((n1,n2,n3,1),dtype=np.single)
for k1 in range(3):
    for k2 in range(10):
        for k3 in range(2):
            gs[:,:,:,0]=new_raw_data[128 * k1:128 * (k1+1),128 * k2:128 *(k2+1),0:128]
            Y = loaded_model.predict(x,verbose=1)
            
            #print(gs)
            train_target_x.append(gs)
train_target_x = np.array(train_target_x)
pre_x =train_target_x[0]
x_2 = np.reshape(pre_x,(1,128,128,128,1))
Y_2 = loaded_model.predict(x_2,verbose=1)
plt.subplot(1, 2, 2)
imgplot2 = plt.imshow(np.transpose(Y_2[0,k,:,:,0]),cmap=plt.cm.bone,interpolation='nearest',aspect=1)
'''

# set gaussian weights in the overlap bounaries
def getMask(os):
    sc = np.zeros((n1,n2,n3),dtype=np.single)
    sc = sc+1
    sp = np.zeros((os),dtype=np.single)
    sig = os/4
    sig = 0.5/(sig*sig)
    for ks in range(os):
        ds = ks-os+1
        sp[ks] = np.exp(-ds*ds*sig)
    for k1 in range(os):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k1]
                sc[n1-k1-1][k2][k3]=sp[k1]
    for k1 in range(n1):
        for k2 in range(os):
            for k3 in range(n3):
                sc[k1][k2][k3]=sp[k2]
                sc[k1][n3-k2-1][k3]=sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(os):
                sc[k1][k2][k3]=sp[k3]
                sc[k1][k2][n3-k3-1]=sp[k3]
    return sc


gx,m1,m2,m3 =np.load("new_raw_data.npy").astype(np.float32),401,1301,321
os = 12 #overlap width
c1 = np.round((m1+os)/(n1-os)+0.5)
c2 = np.round((m2+os)/(n2-os)+0.5)
c3 = np.round((m3+os)/(n3-os)+0.5)
c1 = int(c1)
c2 = int(c2)
c3 = int(c3)
p1 = (n1-os)*c1+os
p2 = (n2-os)*c2+os
p3 = (n3-os)*c3+os
gx = np.reshape(gx,(m1,m2,m3))
gp = np.zeros((p1,p2,p3),dtype=np.single)
gy = np.zeros((p1,p2,p3),dtype=np.single)
mk = np.zeros((p1,p2,p3),dtype=np.single)
gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
gp[0:m1,0:m2,0:m3]=gx
sc = getMask(os)
print (c1)
print (c2)
print (c3)

for k1 in range(c1):
    for k2 in range(c2):
        for k3 in range(c3):
            b1 = k1*n1-k1*os
            e1 = b1+n1
            b2 = k2*n2-k2*os
            e2 = b2+n2
            b3 = k3*n3-k3*os
            e3 = b3+n3
            gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
            gs = gs-np.min(gs)
            gs = gs/np.max(gs)
            Y = loaded_model.predict(gs,verbose=1)
            Y = np.array(Y)
            gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
            mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
gy = gy/mk
gy = gy[0:m1,0:m2,0:m3]
gy.tofile("data/prediction/f3d/"+"fp.dat",format="%4")


gx = np.reshape(gx,(m1,m2,m3))
gy = np.reshape(gy,(m1,m2,m3))

k1,k2,k3 = 29,29,99
gx1 = np.transpose(gx[k1,:,:])
gy1 = np.transpose(gy[k1,:,:])
gx2 = np.transpose(gx[:,k2,:])
gy2 = np.transpose(gy[:,k2,:])
gx3 = np.transpose(gx[:,:,k3])
gy3 = np.transpose(gy[:,:,k3])

#xline slice
fig = plt.figure(figsize=(9,9))
p1 = plt.subplot(1, 2, 1)
p1.imshow(gx1,aspect=1.5,cmap=plt.cm.gray)
p2 = plt.subplot(1,2,2)
p2.imshow(gy1,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)

#inline slice
fig = plt.figure(figsize=(12,12))
p1 = plt.subplot(1, 2, 1)
p1.imshow(gx2,aspect=1.5,cmap=plt.cm.gray)
p2 = plt.subplot(1,2,2)
p2.imshow(gy2,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)

#time slice
fig = plt.figure(figsize=(12,12))
p1 = plt.subplot(1, 2, 1)
p1.imshow(gx3,cmap=plt.cm.gray)
p2 = plt.subplot(1,2,2)
p2.imshow(gy3,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
