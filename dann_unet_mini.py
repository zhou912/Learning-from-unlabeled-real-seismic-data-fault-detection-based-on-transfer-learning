#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:10:01 2019
unet_dann_mini
increase the speed 
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
batch_size = 8
epouch_num = 20010
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
    #print (train_source_x.shape)
    mini_train_source_x = train_source_x[0:64,0:64,0:64,:]
    mini_train_source_y = train_source_y[0:64,0:64,0:64,:]
    #print (mini_train_source_x.shape)
    total_train_source_x.append(mini_train_source_x)
    total_train_source_y.append(mini_train_source_y)
   
total_train_source_x = np.array(total_train_source_x)
total_train_source_y = np.array(total_train_source_y)
print (total_train_source_x.shape,'total_train_soure_x')
#目标域数据读入
n1, n2, n3 = 64, 64, 64
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
      
train_target_x = []   
gx,m1,m2,m3 = np.fromfile("data/prediction/f3d/gxl.dat",dtype=np.single),512,384,128
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
            #print (gs)
            #print (gs.shape)
            gs = np.reshape(gs,(n1,n2,n3,1)) 
            #print (gs.shape)
            train_target_x.append(gs)
            gs = np.zeros((1,n1,n2,n3,1),dtype=np.single)
train_target_x = np.array(train_target_x)  
train_target_x = train_target_x[0:200]
#print (train_source_x[1])
#print (train_target_x.shape)
#类别batch生成
source_batch = batch_generator([total_train_source_x, total_train_source_y], batch_size)
target_batch = batch_generator([train_target_x, total_train_source_y], batch_size)
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
input_img = Input(shape=(64, 64, 64,1))
conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
"""
#构建自编码器
de1 = UpSampling3D(size=(2, 2, 2))(conv4)
de2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(de1)
de2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(de2)
de3 = UpSampling3D(size=(2, 2, 2))(de2)
de4 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(de3)
de4 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(de4)
de5 = UpSampling3D(size=(2, 2, 2))(de4)
decoder_output = Conv3D(1, (3, 3, 3), activation='relu', padding='same',name = 'decoder_output')(de5)
"""
#构建断层类别鉴别器
up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=4)
conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)
conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv2], axis=4)
conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)
conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv1], axis=4)
conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same',name = "class_1")(up7)
conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same',name = "class_2")(conv7)
class_output  = Conv3D(1, (1, 1, 1), activation='sigmoid',name="class_output")(conv8)

#构建领域鉴别器
conv_d = Conv3D(1, (1, 1, 1), activation='relu', padding='same',name = 'dis_0')(conv6)
discriminator_feature = Flatten(name = 'dis_3')(conv_d)
#domain classifier
#grl_layer = GradientReversal(-1.0)
#dann classifier
grl_layer = GradientReversal(1.0)
discriminator_feature = grl_layer(discriminator_feature)
domain_1 = Dense(256,activation='relu',name = 'dis_1')(discriminator_feature)
domain_2 = Dropout(0.5,name = 'dis_2')(domain_1)
discriminator_output = Dense(2,activation="softmax",name="discriminator_output")(domain_2)
discriminator_model = Model(inputs=[input_img], outputs=[discriminator_output])
discriminator_model.compile(optimizer=Adam(lr = 1e-5),
                                    loss='categorical_crossentropy', metrics=['accuracy'], )

model = Model(inputs = [input_img],outputs = [class_output,discriminator_output])
#这里需要改动，现在这里已经是sigmoid单输出了，另外还要加上权重
model.compile(optimizer= "Adam",loss={'class_output':cross_entropy_balanced,
                                     'discriminator_output':'categorical_crossentropy'},
              loss_weights = {'class_output':1,'discriminator_output':0 },
              metrics=['accuracy'])

source_classification_model = Model(inputs=[input_img], outputs=[class_output])
source_classification_model.compile(optimizer=Adam(lr = 1e-4),
                                    loss=cross_entropy_balanced, metrics=['accuracy'], )


#领域训练权重以及标签
y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))
y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))
sample_weights_adversarial = np.ones((batch_size * 2,))
sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
train_mode = 'dann'
#保存训练过程中的模型
#checkpointer = ModelCheckpoint(filepath='F:/123/UNET/model/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',verbose=1)
for i in range (epouch_num):
#    print (i,'epouch_num')
    if train_mode == 'dann':
        X0, y0 = source_batch.__next__()
        X1, y1 = target_batch.__next__()
        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, y0])
        #save class_weight and train domain_dann to get common feature
        class_weights = []
        for layer in model.layers:
            if (layer.name.startswith("class")):
                class_weights.append(layer.get_weights())
        
        cost_domain = discriminator_model.train_on_batch(X_adv,y_adversarial_1)
        #update weight except class_weight
        k = 0
        for layer in model.layers:
            if(layer.name.startswith("class")):
                layer.set_weights(class_weights[k])
                k += 1
        
        #update class_weights
        adv_weights = []
        for layer in model.layers:
            if(layer.name.startswith("dis")):
                adv_weights.append(layer.get_weights())
        for z in range(2):
            cost_class = source_classification_model.train_on_batch(X0,y0)
        k = 0
        for layer in model.layers:
            if(layer.name.startswith("dis")):
                layer.set_weights(adv_weights[k])
                k += 1
        if ((i + 1) % 100 == 0):
            print(i, 'epouch_num')
            print (discriminator_model.metrics_names)
            print (cost_domain,'domain_loss')
            print (source_classification_model.metrics_names)
            print (cost_class,'class_loss')
        
        
#        cost = model.train_on_batch(X_adv, [y_class, y_adversarial_1],
#               sample_weight=[sample_weights_class, sample_weights_adversarial])
#        if ((i + 1) % 100 == 0):
#            print(i, 'epouch_num')
#            print (model.metrics_names)
#            #print (cost_1,'loss_1')
#            print (cost,'loss')
        '''
        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, y0])
        adv_weights = []
        for layer in model.layers:
            #检测指定命名开头的神经网络层
            if (layer.name.startswith("di")):
                adv_weights.append(layer.get_weights())
        cost = model.train_on_batch(X_adv, [y_class, y_adversarial_2],
                sample_weight=[sample_weights_class, sample_weights_adversarial])
                #将domain鉴别器的权重设置为上一次迭代的值
        k = 0
        for layer in model.layers:
            if (layer.name.startswith("di")):
                layer.set_weights(adv_weights[k])
                k += 1
        class_weights = []
        #存储鉴别器以及公共特征的权重
        for layer in model.layers:
            if (not layer.name.startswith("di")):
                class_weights.append(layer.get_weights())

        cost_2 = discriminator_model.train_on_batch(X_adv,y_adversarial_1)
        #训练领域鉴别器后，保留训练前的特征以及类别鉴别器的权重
        k = 0
        for layer in model.layers:
            if (not layer.name.startswith("di")):
                layer.set_weights(class_weights[k])
                k += 1
        '''
        #y_class = np.concatenate([y0, np.zeros_like(y0)])
        #cost = model.train_on_batch(X_adv,[y_class,y_adversarial_1],sample_weight=[sample_weights_class,sample_weights_adversarial])

#        if ((i + 1) % 500 == 0):
#            print(i, 'epouch_num')
#            print (model.metrics_names)
#            #print (cost_1,'loss_1')
#            print (cost,'loss')
#            '''
#            y_test_hat_t = source_classification_model.predict(validation_x)
#            y_output_mid = np.reshape(y_test_hat_t,128*128*128)
#            y_output_mid = np.array(y_output_mid).astype(float)
#            y_output = np.zeros((len(y_output_mid), 2))
#            for k in range(len(y_output_mid)):
#                if y_output_mid[k] > 0.5:
#                    y_output[k][1] = 1 
#                else:
#                    y_output[k][0] = 1
#            #print (type_of_target(y_output))
#            print("Iteration %d, target accuracy = %.3f" % (i,accuracy_score(validation_y, y_output)))
#            '''
    if train_mode == 'class':
        X0, y0 = source_batch.__next__()
        X1, y1 = target_batch.__next__()
        #X1 = np.zeros_like(X1)
        X_adv = np.concatenate([X0, X0])
        #print (y0)
        #print (X_adv.shape)
        #test = X_adv[0:16,13:14,13:14,13:14,:] 
        #print (test)
        y_class = np.concatenate([y0, y0])
        #print (X_adv.shape)
        #print (y_class.shape)
        #print (sample_weights_class)
        cost = source_classification_model.train_on_batch(X_adv,y_class,sample_weight=sample_weights_class)
        #cost = source_classification_model.train_on_batch(X0,y0)
        if ((i + 1) % 100 == 0):
            print (source_classification_model.metrics_names)
            print (i,'epouch_num')
            print (cost,'loss')
    if train_mode == 'domain':
        X0, y0 = source_batch.__next__()
        X1, y1 = target_batch.__next__()
        X_adv = np.concatenate([X0, X1])
        cost = discriminator_model.train_on_batch(X_adv,y_adversarial_1)
        if ((i + 1) % 100 == 0):
            print (discriminator_model.metrics_names)
            print (i,'epouch_num')
            print (cost,'loss')

model_json = source_classification_model.to_json()
json_name = "model3" + str(epouch_num) + ".json"
print (json_name)
model_name =str(epouch_num) +  "model3.h5"
with open(json_name,"w") as json_file:
    json_file.write(model_json)
source_classification_model.save_weights(model_name)
print("Save model to disk")



'''
prediction test on a field seismic image extracted from
the Netherlands off-shore F3 block seismic data
'''
#a 3d array of gx[m1][m2][m3], please make sure the dimensions are correct!!!
#gx,m1,m2,m3 = np.fromfile("data/prediction/f3d/gxl.dat",dtype=np.single),321,401,1001
gx,m1,m2,m3 = np.fromfile("data/prediction/f3d/gxl.dat",dtype=np.single),512,384,128
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
#            gs = gs*255
            Y = source_classification_model.predict(gs,verbose=1)
            Y = np.array(Y)
            gy[b1:e1,b2:e2,b3:e3]= gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
            mk[b1:e1,b2:e2,b3:e3]= mk[b1:e1,b2:e2,b3:e3]+sc
gy = gy/mk
gy = gy[0:m1,0:m2,0:m3]
gy.tofile("data/prediction/f3d/"+"fp.dat",format="%4")



from matplotlib.colors import Normalize
from PIL import Image

#gx,m1,m2,m3 = np.fromfile("data/prediction/f3d/gxl.dat",dtype=np.single),321,401,1001
gx,m1,m2,m3 = np.fromfile("data/prediction/f3d/gxl.dat",dtype=np.single),512,384,128
gy.tofile("data/prediction/f3d/"+"fp.dat",format="%4")
gx = np.reshape(gx,(m1,m2,m3))
gy = np.reshape(gy,(m1,m2,m3))

k1,k2,k3 = 29,29,99
name = "1_"
#k1,k2,k3 = 59,59,59
#name = "2_"
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
#p2.imshow(gy1,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
#p2.imshow(gy1,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap='gray_r')
#gy1 = MaxMinNormalization(gy1,np.max(gy1),np.min(gy1))
p2.imshow(gy1,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=0.7,cmap='gray_r')
#imgplot2 = plt.imshow(np.transpose(Y[0,:,:,k3,0]),cmap=plt.cm.bone,interpolation='nearest',aspect=1)

plt.savefig(name+"predict_xline.png")

#inline slice
fig = plt.figure(figsize=(12,12))
p1 = plt.subplot(1, 2, 1)
p1.imshow(gx2,aspect=1.5,cmap=plt.cm.gray)
p2 = plt.subplot(1,2,2)
#p2.imshow(gy2,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
#gy2 = MaxMinNormalization(gy2,np.max(gy2),np.min(gy2))
p2.imshow(gy2,aspect=1.5,interpolation="bilinear",vmin=0.4,vmax=0.7,cmap='gray_r')
plt.savefig(name+"predict_inline.png")

#time slice
fig = plt.figure(figsize=(12,12))
p1 = plt.subplot(1, 2, 1)
p1.imshow(gx3,cmap=plt.cm.gray)
p2 = plt.subplot(1,2,2)
#p2.imshow(gy3,interpolation="bilinear",vmin=0.4,vmax=1.0,cmap=plt.cm.gray)
#gy3 = MaxMinNormalization(gy3,np.max(gy3),np.min(gy3))
p2.imshow(gy3,interpolation="bilinear",vmin=0.4,vmax=0.7,cmap='gray_r')
plt.savefig(name+"predict_time.png")

