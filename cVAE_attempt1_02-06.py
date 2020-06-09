# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:31:03 2020

Shamelessly lifted from:
    https://github.com/sksq96/pytorch-vae/blob/master/vae.py
    
    
Taken the kernel sizes from:
    https://github.com/coolvision/vae_conv/blob/master/vae_conv_model_mnist.py


@"author": Jack
"""


#Load in every day:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import umap

import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import datetime as dt
from time import perf_counter
import copy as cp
import numpy as np
import pandas as pd

batches = 50
lr = 1e-2
epochs = 1



trans = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True, download=True, 
                          transform=trans)

test_ds = datasets.MNIST('./data', train=False, download=True,
        transform=trans
    )

train_it = torch.utils.data.DataLoader(train_ds, batch_size=batches, 
                                       shuffle=True, drop_last=True)
test_it = torch.utils.data.DataLoader(test_ds, batch_size=batches, 
                                       shuffle=False, drop_last=True)

''' For making a separate tensor of the scaled data
b = perf_counter()
test_sc = torch.Tensor()
for i in range(len(test_ds)):
    test_sc = torch.cat([test_sc, test_ds[i][0]], dim=0)
    print(i)
e = perf_counter()
print(f"Time taken: {e-b}s")
torch.save(train_sc, "saved_models/scaled_train_ds.pt")
torch.save(test_sc, "saved_models/scaled_test_ds.pt")
'''

#train_sc = torch.load("saved_models/scaled_train_ds.pt")
#test_sc = torch.load("saved_models/scaled_test_ds.pt")


fixed_x, _ = next(iter(train_it))

n_f = 32
z_dim = 32

class Flatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=-1):
        #print(input.shape)
        return input.view(input.size(0), size, 1, 1)
    
class Display(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=256, z_dim=12, kernel=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, n_f, kernel_size=kernel, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_f, n_f*2, kernel_size=kernel, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_f*2, n_f*4, kernel_size=kernel-1, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_f*4, n_f*8, kernel_size=kernel, stride=1, padding=0),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, n_f*4, kernel_size=kernel, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(n_f*4, n_f*2, kernel_size=kernel-1, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_f*2, n_f, kernel_size=kernel, stride=2, padding=1),
            nn.ReLU(),
            #Display(),
            nn.ConvTranspose2d(n_f, image_channels, kernel_size=kernel, stride=2, padding=1),
            nn.ReLU(),  #NEED TO CHANGE 
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        x = x.type(torch.float)
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        x = x.type(torch.float)
        h = self.encoder(x)
        #print(f"encoder output shape: {h.shape}")
        z, self.mu, self.logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), self.mu, self.logvar
    
    def sample(self, num=1):
        z = torch.randn(num, z_dim)
        hidden_out = self.fc3(z)
        output = self.decoder(hidden_out)
        return output
        
    
def KL_div(mu, logvar):
    mu_sq = pow(mu,2)  
    sig_sq = torch.exp(logvar)     #doesnt need /2 
    loss = 0.5 * torch.sum(mu_sq - logvar - 1 + sig_sq ) 
    return loss
    
def get_loss(input_x, output_x, mu, logvar):
    recon_loss = F.mse_loss(output_x, input_x, reduction='sum')
    kl_loss = KL_div(mu, logvar)
    #print(recon_loss, kl_loss)
    return recon_loss + kl_loss
    

def train(mod, epochs):
    
    mod.train()
    train_loss = 0
    loss_list = []
    average_loss = 999
    #while average_loss > 40:
    
    for j in range(epochs):
        percent = 0
        train_loss = 0
        for i, (data, target) in enumerate(train_it):
            output, mu, logvar = mod(data)
            #print(output.shape)
            loss = get_loss(data, output, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loss_list.append(loss.item())
            percent = i*batches*100/len(train_it.dataset)
            #print(loss.item())
            if percent % 20 == 0:
                print(f"{percent}% , {loss.item()}")
            #if i*batches % len(train_it.dataset) == 0:
                #print(f"{100*i/len(train_it.dataset)}% - Loss = {loss}")
        average_loss = train_loss/len(train_it.dataset)
        print(f"====> Epoch {j} - Average train loss = {average_loss}")
       
    return loss_list
   
    
def test(mod, epoch):
    
    mod.eval()
    test_loss = 0
    mu_list = torch.Tensor()
    logvar_list = torch.Tensor()
    output_list = torch.Tensor()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_it):
            
            test_output, mu, logvar = mod(data)
            loss = get_loss(data, test_output, mu, logvar)
            test_loss += loss.item()
            percent = i*batches*100/len(test_it.dataset)
            if percent % 50 == 0:
                print(f"{percent}% - test")
                
            mu_list = torch.cat([mu_list, mu], dim=0)
            logvar_list = torch.cat([logvar_list, logvar], dim=0)
            output_list = torch.cat([output_list, test_output], dim=0)
                
        print(f"====> Average test loss = {test_loss/len(test_it.dataset)}")
    return test_loss, output_list, mu_list, logvar_list

#%%


channels = fixed_x.size(1)
num_tests = 2

test_output_scores = []
test_latent_scores = []
loss_lists = []


fit_sizes = [15,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,
             800,900,1000,2000,4000,6000,8000,8250,8500,8600,8700,8800,8900,9000]


for i in range(num_tests):
    
    #Make new models each time
    model = VAE(image_channels=channels, kernel=4, z_dim=z_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_l = train(model, 1)

    #plt.scatter([i for i in range(1200)], train_l)
    loss_lists.append(train_l)
    #Produce outputs from model
    test_loss, outputs, mus, logvars = test(model, 0)

    
    num=15  #No. of neighbours to consider

    output_scores = []
    latent_scores = []
    test_size = 1000

    for size in fit_sizes: 

        knn_output = KNeighborsClassifier(n_neighbors=num)
        knn_output.fit(outputs[0:size].detach().view(-1,28*28), test_ds.targets[0:size])
    
        #predictions = knn_outputs.predict(outputs[size:size+test_size].detach().view(-1,28*28))
        score = knn_output.score(outputs[size:size+test_size].detach().view(-1,28*28),
                              test_ds.targets[size:size+test_size])
        output_scores.append(score)
        
        knn_latent = KNeighborsClassifier(n_neighbors=num)
        knn_latent.fit(mus[0:size].detach(), test_ds.targets[0:size])
        #latent_pred = knn_latent.predict(mus[size:size+test_size].detach())
        score = knn_latent.score(mus[size:size+test_size].detach(),
                         test_ds.targets[size:size+test_size])
        latent_scores.append(score)
        #For tracking progress:
        if size in (10,100,600,4000,9000):  
            print(size)
 
    test_output_scores.append(output_scores)
    test_latent_scores.append(latent_scores)
    print(f"Model {i} done")

test_output_scores = pd.DataFrame(test_output_scores)
test_latent_scores = pd.DataFrame(test_latent_scores)
loss_lists = pd.DataFrame(loss_lists)

mean_outputs = test_output_scores.mean(axis=0)
mean_latent = test_latent_scores.mean(axis=0)
std_output = test_output_scores.std(axis=0)
std_latent = test_latent_scores.std(axis=0)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar(fit_sizes, mean_outputs, yerr=std_output, s=10, 
             c='b', label='Output space')
ax1.errorbar(fit_sizes, mean_latent, yerr=std_latent, s=10,
             c='r', label='Latent space')
plt.xlabel("Number of supervised examples")
plt.ylabel("Mean accuracy")
plt.legend(loc='upper right');
#plt.savefig("saved_plots/MNIST_cVAE_AccuracyVSFitsize_zdim32_04-06.png")
plt.show()

'''
#Generate sample:
output = model.sample(num=1)
new_img = output
img = new_img.view(28,28).data
plt.imshow(img, cmap='gray')
plt.show()
'''

#%%     

'''

#Use embeddings to plot latent mus of test sample:

targets = test_ds.targets


fit = umap.UMAP(n_neighbors=50)
emb = fit.fit_transform(save_mus.detach())

fig,ax = plt.subplots(figsize=(10,8))
cmap = plt.get_cmap('Spectral', 10)
cax = ax.scatter(emb[:,0], emb[:,1], s=1, c=targets, cmap=cmap, 
                 vmin=0, vmax=train_ds.targets.max())
fig.colorbar(cax)
plt.title('Latent mean-vector space of the MNIST dataset', fontsize=14)
plt.show()
#plt.savefig("saved_plots/CAE_MNIST_latentspace_seed1_3e_02-06.png")


#Test loading in model if don't want to retrain:
loadname = f"saved_models/cVAE_trainedmodel_02_06-18_59.pt"
#model2 = VAE(image_channels=channels, kernel=4, z_dim=z_dim)
#model2.load_state_dict(torch.load(loadname))

----------------------------------------------------------------------------
For a k-nn classifier
I want to train on a dataset of a certain size, and learn a latent space
ie. use supervised data for those bits.

Then I want to test on the data and see whether the output of the image and the
latent space give accurate predictions for the class

We want two classifiers, one that uses the 28,28 outputs and the other the latent 
space

Plot of no. of labelled examples vs accuracy

- VAE seems to perform very badly over repeat tests




'''
