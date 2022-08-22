"""
@author: Vlad Popa
"""
#import libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams
rcParams['figure.dpi'] = 600
torch.set_default_dtype(torch.float)
torch.manual_seed(5113628)

#use GPU if available, to speed up computation time 
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parameters to be changed in ANN
neurons_per_layer=[1,50,50,1]
activations = [nn.Tanh,nn.Tanh,'off']
epochs = 50000

m = 2
mu_true = 0.5
k = 2
A = 1
phase = 0

#different loss terms
labels_list=[] #MSE loss training samples
physics_list = [] #MSE PDE residue loss
mu_list = [] #mu

#analytic solution
def oscillator(t):
    delta = mu_true / (2*m)
    w0 = np.sqrt(k/m)
    w = np.sqrt(w0**2-delta**2)
    return A*torch.exp(-delta*t)*torch.cos(w*t+phase)

'''
* Create dense artificial neural network with specified neurons per layer and activation functions
*
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param activatornames: ndarray(#layers); activation functions per layer
*
* @return: model for a dense neural network
'''
def ANN(neurons_per_layer,activations):
    layers = []
    for i in range(len(neurons_per_layer)-1):
        layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i+1]))
        if activations[i] == 'off':
            pass
        else:
            layers.append(activations[i]())
    return nn.Sequential(*layers)

#training samples and collocation points
t_physics = torch.linspace(0,30,1000).view(-1,1).float().to(mydevice)
t_physics.requires_grad_(True)

mu0 = 0.01 #initial guess mu
mu = torch.tensor([mu0]).float().to(mydevice)
mu.requires_grad_(True)
mu = nn.Parameter(mu)

model = ANN(neurons_per_layer,activations)
model.register_parameter('mu', mu) #register mu as model parameter
model.to(mydevice)

utrue = oscillator(t_physics) #true labels
err = np.random.uniform(low=0.9, high=1.1, size=(len(utrue),1)) #noise
err = torch.from_numpy(err).float().to(mydevice)

#implementation using Adam or L-BFGS
def adam():
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6000], gamma=0.1) #adaptive learning rate
    for i in range(epochs):
        #Pytorch accumulates gradients during training, so set to zero before optimization step
        optimizer.zero_grad()
        mse = nn.MSELoss()
        
        uerr = err*oscillator(t_physics) #noisy data
        up = model(t_physics) #prediction
    
        updt  = torch.autograd.grad(up, t_physics, torch.ones_like(t_physics), create_graph=True)[0] #du/dt
        updt2 = torch.autograd.grad(updt,  t_physics, torch.ones_like(t_physics),  create_graph=True)[0] #d^2u/dt^2
        
        physics = m*updt2 + mu*updt + k*up #residue ODE with model parameter mu
        physics_loss = torch.mean(physics**2)
        label = mse(up,uerr)

        loss = label + physics_loss
        loss.backward() #backward propagation
        optimizer.step() #perform one optimization step
        scheduler.step() #update learning rate as specified by condition set in scheduler
        
        physics_list.append(physics_loss.detach().cpu().numpy())
        labels_list.append(label.detach().cpu().numpy())
        mu_list.append(mu[0].detach().cpu().numpy())

        if i %1000 ==0:
            print(i,'/',epochs)
            
def lbfgs(): 
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2, 
                              max_iter = epochs, 
                              max_eval = None, 
                              tolerance_grad = 1e-05, 
                              tolerance_change = 1e-09, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        mse = nn.MSELoss()
        optimizer.zero_grad()
        
        uerr = err*oscillator(t_physics)

        
        up = model(t_physics)
        label = mse(up,uerr)
        updt  = torch.autograd.grad(up, t_physics, torch.ones_like(t_physics), create_graph=True)[0]
        updt2 = torch.autograd.grad(updt,  t_physics, torch.ones_like(t_physics),  create_graph=True)[0]
        physics = m*updt2 + mu*updt + k*up
        physics_loss = torch.mean(physics**2)
        
        
        loss = label + physics_loss
        loss.backward()
        
        physics_list.append(physics_loss.detach().cpu().numpy())
        labels_list.append(label.detach().cpu().numpy())
        mu_list.append(mu[0].detach().cpu().numpy())

        return loss
    optimizer.step(closure)

        
start = time.time()
adam() #choose optimization algorithm
end = time.time()
print('elapsed time:',end - start,'s')

#prediction data
tpredict = torch.linspace(0,30,3000).float().to(mydevice)
upredict = model(tpredict.view(-1,1)).flatten().detach().cpu()
tpredict = tpredict.detach().cpu()

#plot prediction and solution
plt.figure()
plt.plot(tpredict,oscillator(tpredict),label='u')
plt.plot(tpredict,upredict,'--',label=r'$\hat{u}$')
plt.ylabel(r'$u$',fontsize=12,labelpad=10,rotation='horizontal')
plt.xlabel(r'$t$',fontsize=12)
plt.legend()

#evolution of mu during training
plt.figure()
i = np.arange(1,len(mu_list)+1)
plt.plot(i,mu_list)
plt.plot(i,mu_true*np.ones(len(mu_list)),'--',c='red')
plt.xlabel(r'$\it{epochs}$',fontsize=12)
plt.ylabel(r'$\mu$',fontsize=12,labelpad=10,rotation='horizontal')
plt.show()
mse = nn.MSELoss()
print('Prediction MSE:',mse(upredict,oscillator(tpredict)).item())
print('Training MSE:',labels_list[-1])
print('ODE MSE training loss:',physics_list[-1])
print('Estimated mu:',mu_list[-1])