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
epochs = 5000

m = 2
mu = 0.5
k = 2
A = 1
phase = 0

#different loss terms
interior = [] #MSE loss in collocation points on interior
IC_loss = [] #MSE loss at initial condition
total_loss = [] #interior+IC_loss+losslabels

#analytic solution
def oscillator(t):
    delta = mu / (2*m)
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


h=1e-2 #step in numeric differentiation

#central difference first derivative of second order accuracy
def diff1(x):
    dummy1 = x.clone()
    dummy2 = x.clone()
    dummy1 += h
    dummy2 -= h
    return (0.5*model(dummy1)-0.5*model(dummy2))/h

#central difference second derivative of second order accuracy
def diff2(x):
    dummy1 = x.clone()
    dummy2 = x.clone()
    dummy3 = x.clone()
    dummy1 += h
    dummy3 -= h
    return (model(dummy1)+model(dummy3)-2*model(dummy2))/h**2

    
t_physics = torch.linspace(0,30,100).view(-1,1).float().to(mydevice) #collocation points
t_labels = torch.linspace(0,15,10).view(-1,1).float().to(mydevice) #training samples
utrue = oscillator(t_labels)

model = ANN(neurons_per_layer,activations)
model.to(mydevice)

#implementation using Adam or L-BFGS
def adam():
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.5*epochs), gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100)
    for i in range(epochs):
        #Pytorch accumulates gradients during training, so set to zero before optimization step
        optimizer.zero_grad()
        mse = nn.MSELoss()
        
        up = model(t_physics) #prediction
        u_label = model(t_labels) #labels training samples
        
        physics = m*diff2(t_physics) + mu*diff1(t_physics) + k*up# computes the residue of the 1D harmonic oscillator ODE
        physics_loss = torch.mean(physics**2)
        lossIC = (up[0]-1)**2+((model(t_physics[0]+h)-model(t_physics[0]))/h)**2
        losslabel = mse(u_label,utrue)
        loss = physics_loss+lossIC+losslabel #total physics loss and loss of labels

        total_loss.append(loss.detach().cpu().numpy())
        interior.append(physics_loss.detach().cpu().numpy())
        IC_loss.append(lossIC.detach().cpu().numpy())
        
        loss.backward() #backpropagation
        optimizer.step() #perform one optimization step
        scheduler.step() #update learning rate as specified by condition set in scheduler
        if i %1000 ==0:
            print(i,'/',epochs)

def lbfgs(): 
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-2, 
                              max_iter = epochs, 
                              max_eval = epochs, 
                              tolerance_grad = 1e-05, 
                              tolerance_change = 1e-09, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        mse = nn.MSELoss()
        
        up = model(t_physics)
        u_label = model(t_labels)
        
        physics = m*diff2(t_physics) + mu*diff1(t_physics) + k*up
        physics_loss = torch.mean(physics**2)
        lossIC = (up[0]-1)**2+((model(t_physics)[0]-model(t_physics+h)[0])/h)**2
        losslabel = mse(u_label,utrue)
        loss = physics_loss+lossIC+losslabel
        
        total_loss.append(loss.detach().cpu().numpy())
        interior.append(physics_loss.detach().cpu().numpy())
        IC_loss.append(lossIC.detach().cpu().numpy())

        loss.backward()
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

#prediction and solution
plt.figure()
plt.plot(tpredict,oscillator(tpredict),label='u')
plt.plot(tpredict,upredict,'--',label=r'$\hat{u}$')
plt.ylabel(r'$u$',fontsize=12,labelpad=10,rotation='horizontal')
plt.xlabel(r'$t$',fontsize=12)
plt.legend()

#training loss
plt.figure()
i = np.arange(1,len(total_loss)+1)
plt.plot(i,total_loss,label = 'Total loss')
plt.xlabel(r'$\it{epochs}$',fontsize=12)
plt.ylabel(r'$\mathcal{L}$',fontsize=12,labelpad=10,rotation='horizontal')
plt.yscale("log")
plt.show()

MSE = nn.MSELoss()
print('Prediction MSE: ',MSE(upredict,oscillator(tpredict)).item())
print('Total training MSE loss: ',total_loss[-1][0])
print('Interior training MSE loss: ',interior[-1])
print('IC training MSE loss:',IC_loss[-1][0])