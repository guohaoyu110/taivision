#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:06:29 2021

@author: xi
"""
import torch 
import numpy as np 
import matplotlib.pyplot as plt  
import torch.nn as nn 
import os
import taivision

# load mnist dataset
use_cuda = torch.cuda.is_available()

root = '../data'
if not os.path.exists(root):
    os.mkdir(root)

fashion = taivision.data.MNIST('../data',download=True)

print(fashion.classes)
for img,classes in fashion:
    break