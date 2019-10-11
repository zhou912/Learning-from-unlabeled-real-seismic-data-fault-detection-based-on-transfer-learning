#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:26:12 2019
npy translate to raw
@author: zrs
"""
import numpy as np
import scipy.io as io

'''
#guanxiedata
guanxie = np.load("guanxie_unet.npy").astype(np.float)
print (guanxie.shape)
guanxie_test = np.zeros(shape = (321,401,1001))
for i in range(321):
    for j in range (401):
        for k in range(1001):
            if guanxie[i][j][k] > 0.5:
                guanxie_test[i][j][k] = guanxie[i][j][k]
            else:
                guanxie_test[i][j][k] = 0
guanxie_test = guanxie_test.astype(np.float32)
guanxie_test.tofile("guanxie_unet.raw")
print("finish")
'''

#F3
guanxie = np.load("f3_pre_unet.npy").astype(np.float)
#print (guanxie)
guanxie_test = np.zeros(shape = (512,384,128))
for i in range(512):
    for j in range (384):
        for k in range(128):
            if guanxie[i][j][k] > 0.5:
                guanxie_test[i][j][k] = guanxie[i][j][k]
            else:
                guanxie_test[i][j][k] = 0
guanxie_test = guanxie_test.astype(np.float32)
guanxie_test.tofile("f3_unet.raw")
print("finish")

