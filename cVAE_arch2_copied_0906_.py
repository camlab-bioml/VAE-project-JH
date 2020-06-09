# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:02:09 2020


architecture copied and edited from:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py


@author: Jack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import umap
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


debug = False


class Display(nn.Module):
    def forward(self, input):
        if debug == True:
            print(input.shape)
        return input


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dims:list=[32, 64, 128, 256, 512],
                 z_dim=12):
        
        super(VAE, self).__init__()
        self.z_dim = z_dim
        modules = []
        h_dims = [32, 64, 128, 256, 512]
        self.h_dims = h_dims
        # Build Encoder
        for h_dim in h_dims:
            modules.append(
                nn.Sequential(
                    Display(),
                    nn.Conv2d(image_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    Display(),
                    nn.BatchNorm2d(h_dim),
                    Display(),
                    nn.LeakyReLU(),
                    Display())
            )
            image_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(h_dims[-1], z_dim)
        self.fc_var = nn.Linear(h_dims[-1], z_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(z_dim, h_dims[-1])

        h_dims.reverse()

        for i in range(len(h_dims) - 1):
            pad = 1
            if i == 2:
                pad = 0
            modules.append(
                nn.Sequential(
                    Display(),
                    nn.ConvTranspose2d(h_dims[i],
                                       h_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=pad),
                    Display(),
                    nn.BatchNorm2d(h_dims[i + 1]),
                    Display(),
                    nn.LeakyReLU(),
                    Display())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            Display(),
                            nn.ConvTranspose2d(32,
                                               1,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            Display(),
                            nn.BatchNorm2d(1),
                            Display(),
                            nn.ReLU(),
                            Display())
                            #nn.Conv2d(h_dims[-1], out_channels=1,
                            #        kernel_size= 3, padding= 1),
                            #Display(),
                            #nn.ReLU(), #was tanh
                            #Display())
        
        
    def encode(self, data):
        result = self.encoder(data)
        if debug == True:
            print(f"Enc output = {result.shape}")
        result = torch.flatten(result, start_dim=1)
        if debug == True:
            print(f"flattened: {result.shape}")
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        if debug == True:
            print(f"fc_mu: {mu.shape}")
        return mu, log_var


    def decode(self, z):
        
        result = self.decoder_input(z)
        if debug == True:
            print(f"decoderinput layer = {result.shape}")
        result = result.view(-1, self.h_dims[0], 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
        
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
        
    def representation(self, x):
        x = x.type(torch.float)
        mu, logvar = self.encode(x)
        return mu, logvar


    def forward(self, x):
        x = x.type(torch.float)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        if debug == True:
            print(output.shape)
        return output, mu, logvar
    
    
    def sample(self, num=1):
        z = torch.randn(num, self.z_dim)
        output = self.decode(z)
        return output


if __name__ == '__main__':
    x = VAE()

    