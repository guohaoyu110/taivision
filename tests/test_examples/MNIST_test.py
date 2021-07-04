#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MNIST_test.py
@Time    :   2021/04/29 21:28:58
@Author  :   Haoyu Guo 
@Version :   1.0
@Contact :   haoyuguo@usc.edu
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
import taivision
data_path = '../../data'
print('MNIST example:')
mnist = taivision.data.MNIST(data_path,download=False)
print('MNIST have {} classes:{}'.format(len(mnist.classes),mnist.classes))
for img,label in mnist:
    print('image size is {},mode is {},one-hot label is {}'.format(img.size,img.mode,label))
    break

print('FashionMNIST example:')
fashion = taivision.data.FashionMNIST(data_path,download=True)
print('FashionMNIST have {} classes:{}'.format(len(fashion.classes),fashion.classes))
for img,label in fashion:
    print('image size is {},mode is {},one-hot label is {}'.format(img.size,img.mode,label))
    break

print('QMNIST example:')
qmnist = taivision.data.QMNIST(data_path,download=True)
print('QMNIST have {} classes:{}'.format(len(qmnist.classes),qmnist.classes))
for img,label in qmnist:
    print('image size is {},mode is {},one-hot label is {}'.format(img.size,img.mode,label))
    break


print('KMNIST example:')
kmnist = taivision.data.KMNIST(data_path,download=True)
print('KMNIST have {} classes:{}'.format(len(kmnist.classes),kmnist.classes))
for img,label in kmnist:
    print('image size is {},mode is {},one-hot label is {}'.format(img.size,img.mode,label))
    break

print('EMNIST example:')
emnist = taivision.data.EMNIST(data_path,split = 'mnist',download=True)
print('EMNIST have {} classes:{}'.format(len(emnist.classes),emnist.classes))
for img,label in emnist:
    print('image size is {},mode is {},one-hot label is {}'.format(img.size,img.mode,label))
    break
