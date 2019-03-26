#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:26:21 2019

@author: ahmaa
"""

import os
import csv 
import numpy as np
import matplotlib.pyplot as plt
import re
with open('./log.txt', 'r') as file:
    
    lines = file.read().splitlines()
    listt=[]
    for j in range(40):
        
        for i in range(len(lines)):
        
#        listt.append(lines[i])
            token=re.findall(r'\S+', lines[i])
            
            train_ppls= token[4]
            valid_ppls= token[10]
            times_clock= token[-1]
            
            
        
   


#
#with open('./log.txt') as TextFile:
#    ReaderText = csv.reader(TextFile)
#    ID = []
#    treated = []
#    Non_treated= []
#    
#    for row in csvReader:
#        ID.append(row[0])
#        Non_treated.append(row[1])
#        treated.append(row[2])

#    TextFile.close()


#x = np.load("learning_curves.npy")[()]
#train_ppls = x['train_ppls']
#val_ppls = x['val_ppls']
#train_losses = x['train_losses']
#val_losses = x['val_losses']
#
#epoch = len(train_ppls)
#
#plt.plot(range(1,1+epoch), train_ppls, label="Train")
#plt.plot(range(1,1+epoch), val_ppls, label="Validation")
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#       ncol=2, mode="expand", borderaxespad=0.)
#
#plt.xlabel("Epoch")
#plt.ylabel("PPL")
#plt.show()