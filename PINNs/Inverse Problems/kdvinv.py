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

#bounds
x_min, x_max, t_min, t_max = 0, 20, 0, 5
ext = [t_min, t_max,x_min, x_max]

#soliton speeds
c1 = 6
c2 = 2

#different loss terms
labels_list=[] #MSE loss training samples
physics_list = [] #MSE PDE residue loss
beta_list = [] #mu

#2-soliton solution to the KdV equation
def kdv2(c1,c2,t,x):
    xi1 = x - c1*t+2
    xi2 = x - c2*t-6
    return (2*(c1-c2)*(c1*torch.cosh(0.5*xi2*(c2**0.5))**2+c2*torch.sinh(0.5*xi1*(c1**0.5))**2))/torch.square(((c1**0.5)-(c2**0.5))*torch.cosh(0.5*(xi1*(c1**0.5)+xi2*(c2**0.5)))+((c1**0.5)+(c2**0.5))*torch.cosh(0.5*(xi1*(c1**0.5)-xi2*(c2**0.5))))

#parameters to be changed in ANN
neurons_per_layer=[2,50,50,50,50,50,50,50,50,1]
activations = [nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,'off']
epochs = 20000

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

#amount of training samples and collocation points
nx = 100
nt = 100

#mesh
x=torch.linspace(x_min,x_max,nx).view(-1,1)
t=torch.linspace(t_min,t_max,nt).view(-1,1)
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))

u = kdv2(c1,c2,T,X).float().to(mydevice) #solution
err = np.random.uniform(low=0.9, high=1.1, size=(nx*nt,1)) #noise
err = torch.from_numpy(err).float().to(mydevice) 

#create grid
t_physics = torch.linspace(t_min,t_max,nt).repeat_interleave(nx).view(-1,1).to(mydevice)
x_physics = torch.linspace(x_min,x_max,nx).repeat(nt,1).view(-1,1).to(mydevice)
x_physics.requires_grad_(True)
t_physics.requires_grad_(True)
Xtrain = torch.cat([x_physics,t_physics],axis=1).to(mydevice)

beta0 = 1 #initial guess beta
beta = torch.tensor([beta0]).float().to(mydevice)
beta.requires_grad_(True)
beta = nn.Parameter(beta)

model = ANN(neurons_per_layer,activations)
model.register_parameter('beta', beta) #register beta as model parameter 
model.to(mydevice)

#implementation using Adam or L-BFGS
def adam():
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7000], gamma=0.1) #adaptive learning rate
    for i in range(epochs):
        #Pytorch accumulates gradients during training, so set to zero before optimization step
        optimizer.zero_grad()
        mse = nn.MSELoss()
        
        up = model(Xtrain) #prediction
        uerr = err*kdv2(c1,c2,t_physics,x_physics) #noisy data

        u_t = torch.autograd.grad(up,t_physics,torch.ones_like(t_physics), create_graph=True)[0] #du/dt
        u_x = torch.autograd.grad(up,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #du/dx
        u_xx = torch.autograd.grad(u_x,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #du^2/dx^2
        u_xxx = torch.autograd.grad(u_xx,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #du^3/dx^3
        
        physics = u_t + beta*up*u_x + u_xxx #residue KdV equation with model parameter beta
        zeros = torch.zeros(Xtrain.shape[0],1).to(mydevice)
        physics_loss = mse(physics,zeros)
        labels = mse(up,uerr)
        
        total_loss = physics_loss + labels

        physics_list.append(physics_loss.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        beta_list.append(beta[0].detach().cpu().numpy())
        
        total_loss.backward() #backward propagation
        torch.nn.utils.clip_grad_norm_(model.parameters(),1) #enable clipping
        optimizer.step() #perform one optimization step
        scheduler.step() #update learning rate as specified by condition set in scheduler
        if i % 1000 ==0:
            print(i,'/',epochs)
            
def bfgs(): 
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
        up = model(Xtrain)
        
        uerr = err*kdv2(c1,c2,t_physics,x_physics)

        u_t = torch.autograd.grad(up,t_physics,torch.ones_like(t_physics), create_graph=True)[0]
        u_x = torch.autograd.grad(up,x_physics,torch.ones_like(x_physics), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x_physics,torch.ones_like(x_physics), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx,x_physics,torch.ones_like(x_physics), create_graph=True)[0]
        
        physics = u_t + beta*up*u_x + u_xxx
        zeros = torch.zeros(Xtrain.shape[0],1).to(mydevice)
        physics_loss = mse(physics,zeros)
        labels = mse(up,uerr)
        total_loss = physics_loss + labels

        physics_list.append(physics_loss.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        beta_list.append(beta[0].detach().cpu().numpy())
        total_loss.backward()

        return total_loss

    optimizer.step(closure)


start = time.time()
adam() #choose optimization algorithm
end = time.time()
print('elapsed time:',end - start,'s')

#amount of prediction data
ntpredict=501
nxpredict=501

#solution
x=torch.linspace(x_min,x_max,nxpredict).view(-1,1)
t=torch.linspace(t_min,t_max,ntpredict).view(-1,1)
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
u = kdv2(c1,c2,T,X)

#predict
t_predict = torch.linspace(t_min,t_max,ntpredict).repeat_interleave(nxpredict).view(-1,1)
x_predict = torch.linspace(x_min,x_max,nxpredict).repeat(ntpredict,1).view(-1,1)
Xpredict = torch.cat((x_predict,t_predict),1).float().to(mydevice)

upredict = model(Xpredict).detach().cpu()
upredict = torch.transpose(upredict.reshape(ntpredict,nxpredict),0,1)

#common margins for colorbar for better comparison
umax=max(torch.max(u),torch.max(upredict))
umin=min(torch.min(u),torch.min(upredict))


#prediction
plt.figure()
plt.imshow(upredict,extent=ext,origin='lower',vmin=umin,vmax=umax,aspect=1/3)
plt.xlabel(r"$t$",fontsize=12)
plt.ylabel(r"$x$",fontsize=12,labelpad=10,rotation='horizontal')
plt.colorbar()

#solution
plt.figure()
plt.imshow(u,extent=ext,origin='lower',vmin=umin,vmax=umax,aspect=1/3)
plt.xlabel(r"$t$",fontsize=12)
plt.ylabel(r"$x$",fontsize=12,labelpad=10,rotation='horizontal')
plt.colorbar()

#absolute error
plt.figure()
plt.imshow(abs(upredict-u),extent=ext,origin='lower',aspect=1/3)
plt.xlabel(r"$t$",fontsize=12)
plt.ylabel(r"$x$",fontsize=12,labelpad=10,rotation='horizontal')
plt.colorbar()

#evolution of beta during training
plt.figure()
i = np.arange(1,len(beta_list)+1)
plt.plot(i,beta_list)
plt.plot(i,6*np.ones(len(beta_list)),'--',c='red')
plt.xlabel(r'$\it{epochs}$',fontsize=12)
plt.ylabel(r'$\beta$',fontsize=12,labelpad=10,rotation='horizontal')
plt.show()

mse = nn.MSELoss()
print('MSE:',mse(upredict,u).item())
print('MSE labels:',labels_list[-1])
print('physics loss:',physics_list[-1])
print('estimated beta:',beta_list[-1])