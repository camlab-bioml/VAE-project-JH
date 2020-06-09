# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:39:27 2020

Implementing a simple VAE.
Following the tutorial at: https://graviraja.github.io/vanillavae/#

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

import itertools

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transforms = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('./data', train=True, download=True, 
                          transform=transforms)

test_ds = datasets.MNIST('./data', train=False, download=True,
        transform=transforms
    )

''' TODO: remove - the DataLoader does scaling 
scaler = MinMaxScaler()
scaler.fit(train_ds.data.view(-1,28*28))
scaled = scaler.transform(train_ds.data.view(-1,28*28))
train_ds_sc = torch.Tensor(scaled).view(-1,28,28)

test_ds_sc = torch.Tensor(scaler.transform(test_ds.data.view(-1,28*28)))
test_ds_sc = test_ds_sc.view(-1,28,28)
'''

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim):
        
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.mu = nn.Linear(hidden_dim3, latent_dim)
        self.var = nn.Linear(hidden_dim3, latent_dim)
        
    def forward(self, data):
        
        hidden1 = F.relu(self.linear1(data))  #Not sure about using ReLu
        hidden2 = F.relu(self.linear2(hidden1))
        hidden3 = F.relu(self.linear3(hidden2))
        self.z_mu = self.mu(hidden3)
        self.z_var = self.var(hidden3)
        
        return self.z_mu, self.z_var
    
    
class Decoder(nn.Module):

    def __init__(self, latent_dim, hidden_dim3, hidden_dim2, hidden_dim1, out_dim):
        
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim3)
        self.linear2 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.out = nn.Linear(hidden_dim1, out_dim)
        
    def forward(self, data):
        
        hidden1 = F.relu(self.linear1(data))
        hidden2 = F.relu(self.linear2(hidden1))
        hidden3 = F.relu(self.linear3(hidden2))
        pred = F.relu(self.out(hidden3))  #Not sure about sigmoid output
        
        return pred
    
    
class VAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, data):
        
        self.z_mu, self.z_var = self.encoder(data)
        
        #Reparameterize:
        std = torch.exp(self.z_var/2)    
    
        e = torch.randn_like(std)   #Samples a N(0,1) in the shape of std
        latent_sample = e.mul(std).add_(self.z_mu)   #Does elementwise multiplication and addition
        
       
        prediction = self.decoder(latent_sample)
        
        return prediction, self.z_mu, self.z_var
    
    def sample(self, num=1):
        z = torch.randn(num, z_dim)
        output = self.decoder(z)
        return output
        
    
in_dim = 28*28
hidden_dim1 = 256
hidden_dim2 = 128
z_dim = 12
lr = 0.001
batches = 50


train_it = DataLoader(train_ds, batch_size=batches, shuffle=True, drop_last=True)
test_it = DataLoader(test_ds, batch_size=batches, drop_last=True)




#KL divergence between N(mu,var) and N(0,1)
def KL_div(mu, var):
    mu_sq = pow(mu,2)  
    sig_sq = torch.exp(var) #doesnt need /2 
    loss = 0.5 * torch.sum(mu_sq - var - 1 + sig_sq ) 
    return loss


def get_loss(input_x, output_x, mu, logvar):
    recon_loss = F.mse_loss(output_x, input_x, reduction='sum')
    kl_loss = KL_div(mu, logvar)
    return recon_loss + kl_loss
    

def train(mod, e):
    
    mod.train()
    train_loss = 0
    loss_list = []
    average_loss = 999
    #while average_loss > 40:
    percent = 0
    for i, (data, target) in enumerate(train_it):
        data = data.view(-1,28*28)
        optimizer.zero_grad()
        
        output, mu, logvar = mod(data)
        loss = get_loss(data, output, mu, logvar)
        
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
    print(f"====> Epoch {e} - Average train loss = {average_loss}")
       
    return train_loss, loss_list
   

def test(mod):
    
    mod.eval()
    
    test_loss = 0
    mu_list = torch.Tensor()
    logvar_list = torch.Tensor()
    output_list = torch.Tensor()
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_it):
            
            data = data.view(-1, 28*28)
            data = data.to(device)
            
            
            sample, z_mu, z_var = mod(data)
            mu_list = torch.cat([mu_list, z_mu], dim=0)
            logvar_list = torch.cat([logvar_list, z_var], dim=0)
            output_list = torch.cat([output_list, sample], dim=0)
            loss = get_loss(data, sample, z_mu, z_var)
            test_loss += loss.item()
            
    return test_loss, output_list, mu_list, logvar_list
        
num_tests = 5
epochs = 2

test_output_scores = []
test_latent_scores = []
loss_lists = []
fit_sizes = [15,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,
             800,900,1000,2000,4000,6000,8000,8250,8500,8600,8700,8800,8900,9000]


for t in range(num_tests):
    
    best_loss = 1e10
    wins = 0
        
    all_losses = []
    
    dim_list = [28*28, 256, 128, 64, 12]
    enc = Encoder(*dim_list)
    dim_list.reverse()
    dec = Decoder(*dim_list)
    model = VAE(enc, dec)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        
        train_loss, loss_list = train(model, e)
        all_losses.append(loss_list)
        test_loss, outputs, z_mus, z_vars = test(model)
        
        train_loss /= len(train_ds)
        test_loss /= len(test_ds)
        print(f"Epoch {e} - Train loss = {round(train_loss,2)}, Test loss = {round(test_loss,2)}")
        
        if test_loss < best_loss:
            best_loss = test_loss
        else:
            wins += 1
        
        if wins > 3:
            print(f"{e} epochs completed.")
            break
        
    loss_lists.append(all_losses)
       
    num = 15
    #size = 9000
    test_size = 1000
    test_loss, outputs, z_mus, z_vars = test(model)
    
    score_list = []
    latent_score_list = []
    
    for size in fit_sizes:
    
        knn_output = KNeighborsClassifier(n_neighbors=num)
        knn_output.fit(outputs[0:size].detach().view(-1,28*28), test_ds.targets[0:size])
        score = knn_output.score(outputs[size:size+test_size].detach().view(-1,28*28),
                                 test_ds.targets[size:size+test_size]) 
        
        knn_latent = KNeighborsClassifier(n_neighbors=num)
        knn_latent.fit(z_mus[0:size].detach(), test_ds.targets[0:size])
        latent_score = knn_latent.score(z_mus[size:size+test_size].detach(),
                                        test_ds.targets[size:size+test_size])
        
        score_list.append(score)
        latent_score_list.append(latent_score)
    
    test_output_scores.append(score_list)
    test_latent_scores.append(latent_score_list)
    print(f"Test {t} done.")

test_output_scores = pd.DataFrame(test_output_scores)
test_latent_scores = pd.DataFrame(test_latent_scores)
loss_lists = pd.DataFrame(loss_lists)

mean_outputs = test_output_scores.mean(axis=0)
mean_latent = test_latent_scores.mean(axis=0)
std_output = test_output_scores.std(axis=0)
std_latent = test_latent_scores.std(axis=0)

x = loss_lists.loc[0][0]
x += loss_lists.loc[0][1]

y = loss_lists.loc[1][0]
y += loss_lists.loc[1][1]

plt.scatter([i for i in range(len(x))], x)
plt.scatter([i for i in range(len(y))], y)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar(fit_sizes, mean_outputs, yerr=std_output,
             c='b', label='Output space')
ax1.errorbar(fit_sizes, mean_latent, yerr=std_latent,
             c='r', label='Latent space')
plt.xlabel("Number of supervised examples")
plt.ylabel("Mean accuracy")
plt.title("Effect of supervision on Architecture 1")
plt.legend(loc='middle right');
#plt.savefig("saved_plots/VAE_MNIST_arch1_AccuracyVSnum_08-06.png")

test_loss, outputs, z_mus, z_vars = test(model)

knn_latent = KNeighborsClassifier(n_neighbors=num)
knn_latent.fit(z_mus[0:size].detach(), test_ds.targets[0:size])
predictions = knn_latent.predict(z_mus[size:size+test_size].detach())

conf_mat = confusion_matrix(test_ds.targets[size:size+test_size], predictions)
plt.imshow(conf_mat, cmap=plt.cm.Blues)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues, save_name=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(9,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if not normalize:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_name != None:
        plt.savefig(f"{save_name}")
        print("Saving...")


plot_confusion_matrix(conf_mat, classes=range(10), normalize=True,
                      save_name="saved_plots/VAE_MNIST_confusion-matrix-normalised_9kSupervisedKNN_08-06.png")


'''
sample = model.sample(1)
img = sample.view(28,28).data
plt.imshow(img, cmap='gray')

losses = []
for i in range(len(all_losses)):
    for j in range(len(all_losses[i])):
        losses.append(all_losses[i][j])
    
    
plt.scatter([i for i in range(len(losses))], losses)

losses, outputs, mus, logvars = test(model)
plt.imshow(outputs[0].view(28,28), cmap='gray')
plt.imshow(test_ds.data[0].view(28,28), cmap='gray')
'''

#%%     

#Attempts at UMAP, with trained model:


'''
#Plot the separations in the data
fit = umap.UMAP(random_state=42)
u = fit.fit_transform(train_ds.data.view(-1,28*28))
fig,ax = plt.subplots(figsize=(10,8))
cmap = plt.get_cmap('gist_rainbow', 10)
cax = ax.scatter(u[:,0], u[:,1], s=5, c=train_ds.targets, cmap=cmap, 
                 vmin=0, vmax=train_ds.targets.max())
fig.colorbar(cax)#, extend='min')
#plt.savefig(fname="saved_plots/MNIST_UMAP.png", quality=100)
'''

'''
#test_data = model.z_mu.detach().numpy() Needed since z_mu requires grad
fit2 = umap.UMAP(n_neighbors=50)
emb = fit2.fit_transform(z_mu_tensor)

fig,ax = plt.subplots(figsize=(10,8))
cmap = plt.get_cmap('Spectral', 10)
cax = ax.scatter(emb[:,0], emb[:,1], s=1, c=targets, cmap=cmap, 
                 vmin=0, vmax=train_ds.targets.max())
fig.colorbar(cax)
plt.title('Latent mean-vector space of the MNIST dataset', fontsize=14)
plt.savefig(fname=f'saved_plots/latent_dim/MNIST_scaled_{z_dim}.png', quality=100)
'''


