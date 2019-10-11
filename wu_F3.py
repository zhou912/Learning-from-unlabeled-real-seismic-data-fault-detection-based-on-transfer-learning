#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:43:47 2019

@author: zrs
"""

from faultSeg_classes import DataGenerator
from unet3 import *
from keras import callbacks
from keras.utils import to_categorical
from Gradient_Reverse_Layer import GradientReversal
import copy
from utils import *
from keras.models import load_model
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold = np.inf)
#这个是用train batch 训练的，应该改成fit_generator 来完成
#先按照自己熟悉的方式来
dim = (128,128,128)
batch_size = 1
epouch_num = 30000
n_channels = 1
shuffle = True
tdpath = 'data/train/seis/'
tfpath = 'data/train/fault/'
vdpath = 'data/validation/seis/'
vfpath = 'data/validation/fault/'
K.set_image_data_format('channels_last')
#读入源域的训练数据以及标签
#train_x的shape为2097152
total_train_source_x = []
total_train_source_y = []
for i in range(200):
    train_source_x  = np.fromfile(tdpath + str(i)+'.dat',dtype=np.single)
    train_source_x = np.reshape(train_source_x,dim)
    #数据预处理
    train_source_x = (train_source_x - np.min(train_source_x)) / (np.max(train_source_x) - np.min(train_source_x))
    train_source_y = np.fromfile(tfpath + str(i)+'.dat',dtype=np.single)
    train_source_y = np.reshape(train_source_y,dim)
    #(128*128*128*1)后面的1并不是输入的batch_size大小，而是数据的通道,在原始方法中存在数据增强
    train_source_x = np.reshape(train_source_x, ( 128, 128, 128, 1))
    train_source_y = np.reshape(train_source_y, ( 128, 128, 128, 1))
    
    for j in range(4):
        total_train_source_x.append(np.reshape(np.rot90(train_source_x,j,(0,1)), ( 128, 128, 128, 1)))
        total_train_source_y.append(np.reshape(np.rot90(train_source_y,j, (0,1)), (128, 128, 128, 1)))
   
total_train_source_x = np.array(total_train_source_x)
total_train_source_y = np.array(total_train_source_y)
#print (total_train_source_x)
#print(total_train_source_y[1])
#print(total_train_source_x.shape)
#print (total_train_source_y.shape)
#目标域数据读入
n1, n2, n3 = 128, 128, 128
# set gaussian weights in the overlap bounaries

source_batch = batch_generator([total_train_source_x, total_train_source_y], batch_size)

#定义损失函数
def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits,
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    #K为keras.optimizer里面的模糊因子epsilon
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)
    

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

#构建网络
input_img = Input(shape=(128, 128, 128,1))
conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(input_img)
conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(pool1)
conv2 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool2)
conv3 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

conv4 = Conv3D(512, (3,3,3), activation='relu', padding='same')(pool3)
conv4 = Conv3D(512, (3,3,3), activation='relu', padding='same')(conv4)

up5 = concatenate([UpSampling3D(size=(2,2,2))(conv4), conv3], axis=4)
conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up5)
conv5 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv5)

up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv2], axis=4)
conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(up6)
conv6 = Conv3D(32, (3,3,3), activation='relu', padding='same')(conv6)

up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv1], axis=4)
conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(up7)
conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)

conv8 = Conv3D(1, (1,1,1), activation='sigmoid')(conv7)

model = Model(inputs=[input_img], outputs=[conv8])
model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])


for i in range (epouch_num):
    X0, y0 = source_batch.__next__()
    cost = model.train_on_batch(X0,y0)
    if ((i + 1) % 100 == 0):
        print (model.metrics_names)
        print (i,'epouch_num')
        print (cost,'loss')
    if ((i + 1) % 5000 == 0):
        model_json = model.to_json()
        json_name = "model+" + str(i) + ".json"
        print (json_name)
        model_name =str(i) +  "model3.h5"
        with open(json_name,"w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_name)
        print("Save model to disk")

#model_json = source_classification_model.to_json()
#with open("model3.json","w") as json_file:
#    json_file.write(model_json)
#source_classification_model.save_weights("model3.h5")
#print("Save model to disk")


#os = 12 #overlap width
#c1 = np.round((m1+os)/(n1-os)+0.5)
#c2 = np.round((m2+os)/(n2-os)+0.5)
#c3 = np.round((m3+os)/(n3-os)+0.5)
#c1 = int(c1)
#c2 = int(c2)
#c3 = int(c3)
#p1 = (n1-os)*c1+os
#p2 = (n2-os)*c2+os
#p3 = (n3-os)*c3+os
#gp = np.zeros((p1,p2,p3),dtype=np.single)
#gy = np.zeros((p1,p2,p3),dtype=np.single)
#mk = np.zeros((p1,p2,p3),dtype=np.single)
#gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
#gp[0:m1,0:m2,0:m3]=gx
#sc = getMask(os)
#
#for k1 in range(c1):
#    for k2 in range(c2):
#        for k3 in range(c3):
#            b1 = k1*n1-k1*os
#            e1 = b1+n1
#            b2 = k2*n2-k2*os
#            e2 = b2+n2
#            b3 = k3*n3-k3*os
#            e3 = b3+n3
#            gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
#            gs = gs-np.min(gs)
#            gs = gs/np.max(gs)
#            Y = source_classification_model.predict(gs,verbose=1)
#            Y = np.array(Y)
#            gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
#            mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
#gy = gy/mk
#gy = gy[0:m1,0:m2,0:m3]
#gy.tofile("data/prediction/f3d/"+"fp.dat",format="%4")
#
#
#gx = np.reshape(gx,(m1,m2,m3))
#gy = np.reshape(gy,(m1,m2,m3))
#
#k1,k2,k3 = 29,29,99
#gx1 = np.transpose(gx[k1,:,:])
#gy1 = np.transpose(gy[k1,:,:])
#gx2 = np.transpose(gx[:,k2,:])
#gy2 = np.transpose(gy[:,k2,:])
#gx3 = np.transpose(gx[:,:,k3])
#gy3 = np.transpose(gy[:,:,k3])
#
##xline slice
#fig = plt.figure(figsize=(9,9))
#p1 = plt.subplot(1, 2, 1)
#p1.imshow(gx1,aspect=1.5,cmap=plt.cm.gray)
#p2 = plt.subplot(1,2,2)
#p2.imshow(gy1,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
#
##inline slice
#fig = plt.figure(figsize=(12,12))
#p1 = plt.subplot(1, 2, 1)
#p1.imshow(gx2,aspect=1.5,cmap=plt.cm.gray)
#p2 = plt.subplot(1,2,2)
#p2.imshow(gy2,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
#
##time slice
#fig = plt.figure(figsize=(12,12))
#p1 = plt.subplot(1, 2, 1)
#p1.imshow(gx3,cmap=plt.cm.gray)
#p2 = plt.subplot(1,2,2)
#p2.imshow(gy3,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
