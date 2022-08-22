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

#use GPU if available, to speed up computation time 
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(5113628)

#bounds
x_min, x_max, t_min, t_max = -3, 3, 0, 2
ext = [t_min, t_max,x_min, x_max]

#soliton speed
c = 2


#loss terms
BC_loss_list = [] #MSE loss boundary conditions
IC_loss_list = [] #MSE loss initial conditions
interior_list = [] #MSE loss on interior at collocation points
total_loss_list=[] #BC_loss_list+IC_loss_list+interior_list

#1-soliton solution to KdV equation
def kdv1(c,t,x):
    return 0.5*c/torch.square((torch.cosh(0.5*np.sqrt(c)*(x-c*t+1))))

#parameters to be changed in ANN
neurons_per_layer=[2,50,50,1]
activations = [nn.Tanh,nn.Tanh,'off']
epochs = 10000


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

#amount of collocation points
nx = 201
nt = 201

h=1e-2 #step in numeric differentiation

#central difference first derivative of second order accuracy
def diff1(x,axis):
    dummy1 = x.clone()
    dummy2 = x.clone()
    dummy1[:,axis] += h
    dummy2[:,axis] -= h
    return (0.5*model(dummy1)-0.5*model(dummy2))/h

#central difference second derivative of second order accuracy
def diff2(x,axis):
    dummy1 = x.clone()
    dummy2 = x.clone()
    dummy3 = x.clone()
    dummy1[:,axis] += h
    dummy3[:,axis] -= h
    return (model(dummy1)+model(dummy3)-2*model(dummy2))/h**2

#central difference third derivative of second order accuracy
def diff3(x,axis):
    dummy1 = x.clone()
    dummy2 = x.clone()
    dummy3 = x.clone()
    dummy4 = x.clone()
    dummy1[:,axis] += 2*h
    dummy2[:,axis] += h
    dummy3[:,axis] -= h
    dummy4[:,axis] -= 2*h
    return (0.5*model(dummy1)-model(dummy2)+model(dummy3)-0.5*model(dummy4))/(h**3)
 
   
#create mesh
x=torch.linspace(x_min,x_max,nx).view(-1,1)
t=torch.linspace(t_min,t_max,nt).view(-1,1)
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
u = kdv1(c,T,X).float().to(mydevice)

#create grid of collocation points in correct shape
t_physics = torch.linspace(t_min,t_max,nt).repeat_interleave(nx).view(-1,1)
x_physics = torch.linspace(x_min,x_max,nx).repeat(nt,1).view(-1,1)
Xphysics = torch.cat((x_physics,t_physics),1)

#remove BC and IC to obtain an interior
t0=0 #initial time for IC
Xphysics = torch.cat((x_physics,t_physics),1)
Xphysics = Xphysics[Xphysics[:,1] != t0] #exclude IC
Xphysics = Xphysics[Xphysics[:,0] != x_min] #exclude BCs
Xphysics = Xphysics[Xphysics[:,0] != x_max]
x_physics = Xphysics[:,[0]].to(mydevice)
t_physics = Xphysics[:,[1]].to(mydevice)

x_physics.requires_grad_(True)
t_physics.requires_grad_(True)
Xtrain = torch.cat([x_physics,t_physics],axis=1).to(mydevice)

#IC
t_0 = t0*torch.ones(nt).view(-1,1)
x_0 = torch.linspace(x_min,x_max,nx).view(-1,1)
X_IC = torch.cat([x_0,t_0],axis=1).to(mydevice)
u0 = kdv1(c, t_0, x_0).float().to(mydevice) 

#BCs
t_bot = torch.linspace(t_min,t_max,nt).view(-1,1)
x_bot = x_min*torch.ones(nx).view(-1,1)
X_bot = torch.cat([x_bot,t_bot],axis=1).to(mydevice)
bottom_u = kdv1(c,t_bot,x_bot).float().to(mydevice)

t_top = torch.linspace(t_min,t_max,nt).view(-1,1)
x_top = x_max*torch.ones(nx).view(-1,1)
X_top = torch.cat([x_top,t_top],axis=1).to(mydevice)
top_u = kdv1(c,t_top,x_top).float().to(mydevice) 


model = ANN(neurons_per_layer,activations)
model.to(mydevice)

def adam():
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/5), gamma=0.1) #adaptive learning rate
    for i in range(epochs):
        #Pytorch accumulates gradients during training, so set to zero before optimization step
        optimizer.zero_grad()

        up = model(Xtrain)
        loss = nn.MSELoss()
        
        physics = diff1(Xtrain,1) + 6*up*diff1(Xtrain,0) + diff3(Xtrain,0) #residue KdV equation

        physics_loss = loss(physics,torch.zeros(Xtrain.shape[0],1).to(mydevice))
        IC_loss = loss(model(X_IC),u0)
        BC1_loss = loss(model(X_bot),bottom_u)
        BC2_loss = loss(model(X_top),top_u)
        
        total_loss = physics_loss + IC_loss + BC1_loss + BC2_loss

        total_loss_list.append(total_loss.detach().cpu().numpy())
        IC_loss_list.append(IC_loss.detach().cpu().numpy())
        BC_loss_list.append((BC1_loss+BC2_loss).detach().cpu().numpy())
        interior_list.append(physics_loss.detach().cpu().numpy())
        
        total_loss.backward() #backward propagation
        torch.nn.utils.clip_grad_norm_(model.parameters(),1) #enable clipping
        optimizer.step() #perform one optimization step
        scheduler.step() #update learning rate as specified by condition set in scheduler
        if i % 1000 ==0:
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
        
        up = model(Xtrain)
        loss = nn.MSELoss()
        
        physics = diff1(Xtrain,1) + 6*up*diff1(Xtrain,0) + diff3(Xtrain,0)

        physics_loss = loss(physics,torch.zeros(Xtrain.shape[0],1).to(mydevice))
        IC_loss = loss(model(X_IC),u0)
        BC1_loss = loss(model(X_bot),bottom_u)
        BC2_loss = loss(model(X_top),top_u)
        
        total_loss = physics_loss + IC_loss + BC1_loss + BC2_loss
  
        total_loss_list.append(total_loss.detach().cpu().numpy())
        IC_loss_list.append(IC_loss.detach().cpu().numpy())
        BC_loss_list.append((BC1_loss+BC2_loss).detach().cpu().numpy())
        interior_list.append(physics_loss.detach().cpu().numpy())
        
        total_loss.backward()
            
        return total_loss

    optimizer.step(closure)

        

start = time.time()
adam() #chosen optimization algorithm
end = time.time()
print('elapsed time:',end - start,'s')

#amount of grid points used to predict
ntpredict=501
nxpredict=501

#solution
x=torch.linspace(x_min,x_max,nxpredict).view(-1,1)
t=torch.linspace(t_min,t_max,ntpredict).view(-1,1)
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
u = kdv1(c,T,X)

#create grid for prediction data
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

#evolution of training loss
plt.figure()
i = np.arange(1,len(total_loss_list)+1)
plt.plot(i,total_loss_list)
plt.xlabel(r'$\it{epochs}$',fontsize=12)
plt.ylabel(r'$\mathcal{L}$',fontsize=12,labelpad=10,rotation='horizontal')
plt.yscale("log")

#prediction and solution at various times
plt.figure()
plt.title('t = 0.5')
t=0.5
uplot=kdv1(c,t*torch.ones(ntpredict),x.flatten()).detach().cpu()
plt.plot(x,uplot,label=r'$u$')
Xplot = Xpredict[Xpredict[:,1] == t]
umodelplot = model(Xplot).detach().cpu()
plt.plot(x,umodelplot,'--',label=r'$\hat{u}$')
plt.legend()
plt.xlabel(r'$x$',fontsize=12)
plt.ylabel(r'$u$',fontsize=12,labelpad=10,rotation='horizontal')
plt.show()
MSE = nn.MSELoss()
print('Prediction MSE: ',MSE(upredict,u).item())
print('Total training loss: ',total_loss_list[-1])
print('Interior training MSE loss: ',interior_list[-1])
print('IC training MSE loss: ',IC_loss_list[-1])
print('BC training MSE loss: ',BC_loss_list[-1])