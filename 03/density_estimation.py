#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from samplers import distribution1, distribution2, distribution3, distribution4

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
batch_size = 512
n_epochs = 500
lr = 0.0001



d_input_size = 1
d_hidden_size = 512  
d_output_size = 1
  
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.map1(x))
        x = self.relu(self.map2(x))
        return self.sigmoid(self.map3(x))
    


cuda_available = True if torch.cuda.is_available() else False
#cuda_available = False
if cuda_available:
    D.cuda()
Tensor = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor 

dist_p = iter(distribution3(batch_size))
dist_q = iter(distribution4(batch_size))


D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)

for epoch in range(n_epochs):
    
    samples_p = next(dist_p)
    samples_q = next(dist_q)
    
    unknown_dist = Tensor(samples_q)
    std_dist = Tensor(samples_p)

    optimizer_D.zero_grad()

    std_decision = D(std_dist)
    # Fake images
    unknown_decision = D(unknown_dist)
    
    d_loss = -torch.mean(torch.log(unknown_decision)) - torch.mean(torch.log(1 - std_decision))
    d_loss.backward()
    optimizer_D.step()
        
        

f0 = lambda x: (1/np.sqrt(2*math.pi))*np.exp(-np.power(x, 2)/2)

x = Tensor(np.linspace(-5,5,1000)).view(1000,1)
D_star = D(x)/(1 - D(x))
D_star = D_star.view(D_star.size(0))
x = x.view(1000)
f1 = torch.mul(f0(x), D_star)




############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density


x = torch.Tensor(xx).view(1000,1)
r = D(x) # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r.detach().numpy())
plt.title(r'$D(x)$')

estimate = f1.detach().numpy() # estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0.2" with your estimate

#print(np.ones_like(xx))
#print(estimate)                                
                                                       
plt.subplot(1,2,2)
#plt.ylim(0,1)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')











