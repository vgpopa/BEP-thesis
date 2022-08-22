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
BC_loss_list = [] #MSE at boundary condition
IC_loss_list = [] #MSE at initial condition
interior_list = [] #MSE at collocation points on interior
total_loss_list=[] #BC_loss_list + IC_loss_list + interior_list

#analytic 2-soliton solution to KdV equation
def kdv2(c1,c2,t,x):
    xi1 = x - c1*t-2
    xi2 = x - c2*t-6
    return (2*(c1-c2)*(c1*torch.cosh(0.5*xi2*(c2**0.5))**2+c2*torch.sinh(0.5*xi1*(c1**0.5))**2))/torch.square(((c1**0.5)-(c2**0.5))*torch.cosh(0.5*(xi1*(c1**0.5)+xi2*(c2**0.5)))+((c1**0.5)+(c2**0.5))*torch.cosh(0.5*(xi1*(c1**0.5)-xi2*(c2**0.5))))

#parameters to be changed in ANN
neurons_per_layer=[2,50,50,50,50,50,50,50,50,1]
activations = [nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,'off']
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

#create the mesh 
x=torch.linspace(x_min,x_max,nx).view(-1,1)
t=torch.linspace(t_min,t_max,nt).view(-1,1)
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
u = kdv2(c1,c2,T,X).float().to(mydevice)

#create grid of collocation points in correct shape
t_physics = torch.linspace(t_min,t_max,nt).repeat_interleave(nx).view(-1,1)
x_physics = torch.linspace(x_min,x_max,nx).repeat(nt,1).view(-1,1)#.to(mydevice)

#remove BC and IC to obtain an interior
t0=0 #initial time for IC
Xphysics = torch.cat((x_physics,t_physics),1)
Xphysics = Xphysics[Xphysics[:,1] != t0]
Xphysics = Xphysics[Xphysics[:,0] != x_min]
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
u0 = kdv2(c1,c2, t_0, x_0).float().to(mydevice)

#BCs
t_bot = torch.linspace(t_min,t_max,nt).view(-1,1)
x_bot = x_min*torch.ones(nx).view(-1,1)
X_bot = torch.cat([x_bot,t_bot],axis=1).to(mydevice)
bottom_u = kdv2(c1,c2,t_bot,x_bot).float().to(mydevice)

t_top = torch.linspace(t_min,t_max,nt).view(-1,1)
x_top = x_max*torch.ones(nx).view(-1,1)
X_top = torch.cat([x_top,t_top],axis=1).to(mydevice)
top_u = kdv2(c1,c2,t_top,x_top).float().to(mydevice)


model = ANN(neurons_per_layer,activations)
model.to(mydevice)

#implementation using Adam or L-BFGS
def adam():
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000,4000], gamma=0.1) #adaptive learning rate
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/5), gamma=0.1)
    for i in range(epochs):
        #Pytorch accumulates gradients during training, so set to zero before optimization step
        optimizer.zero_grad()

        up = model(Xtrain) #prediction
        mse = nn.MSELoss()

        u_t = torch.autograd.grad(up,t_physics,torch.ones_like(t_physics), create_graph=True)[0] #du/dt
        u_x = torch.autograd.grad(up,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #du/dx
        u_xx = torch.autograd.grad(u_x,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #d^2u/dx^2
        u_xxx = torch.autograd.grad(u_xx,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #d^3u/dx^3
        physics = u_t + 6*up*u_x + u_xxx #residue KdV equation
        zeros = torch.zeros(Xtrain.shape[0],1).to(mydevice)
        physics_loss = mse(physics,zeros)
        
        model0 = model(X_IC)
        IC_loss = mse(model0,u0)
        
        modelbot = model(X_bot)
        BC1_loss = mse(modelbot,bottom_u)
        
        modeltop = model(X_top)
        BC2_loss = mse(modeltop,top_u)
        
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
        mse = nn.MSELoss()

        u_t = torch.autograd.grad(up,t_physics,torch.ones_like(t_physics), create_graph=True)[0]
        u_x = torch.autograd.grad(up,x_physics,torch.ones_like(x_physics), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,x_physics,torch.ones_like(x_physics), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx,x_physics,torch.ones_like(x_physics), create_graph=True)[0]
        physics = u_t + 6*up*u_x + u_xxx
        zeros = torch.zeros(Xtrain.shape[0],1).to(mydevice)
        physics_loss = mse(physics,zeros)
        
        model0 = model(X_IC)
        IC_loss = mse(model0,u0)

        modelbot = model(X_bot)
        BC1_loss = mse(modelbot,bottom_u)
        
        modeltop = model(X_top)
        BC2_loss = mse(modeltop,top_u)
        
        total_loss = physics_loss + IC_loss + BC1_loss + BC2_loss
        total_loss.backward()
        
        total_loss_list.append(total_loss.detach().cpu().numpy())
        IC_loss_list.append(IC_loss.detach().cpu().numpy())
        BC_loss_list.append((BC1_loss+BC2_loss).detach().cpu().numpy())
        interior_list.append(physics_loss.detach().cpu().numpy())
      
        return total_loss
    optimizer.step(closure)


start = time.time()
adam() #chosen optimization algorithm
end = time.time()
print('elapsed time:',end - start,'s')

#amount of grid points used to predict
ntpredict=501
nxpredict=501

#true values
x=torch.linspace(x_min,x_max,nxpredict).view(-1,1)
t=torch.linspace(t_min,t_max,ntpredict).view(-1,1)
X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))
u = kdv2(c1,c2,T,X)

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
uplot=kdv2(c1,c2,t*torch.ones(ntpredict),x.flatten()).detach().cpu()
plt.plot(x,uplot,label=r'$u$')
Xplot = Xpredict[Xpredict[:,1] == t]
umodelplot = model(Xplot).detach().cpu()
plt.plot(x,umodelplot,'--',label=r'$\hat{u}$')
plt.legend()
plt.xlabel(r'$x$',fontsize=12)
plt.ylabel(r'$u$',fontsize=12,labelpad=10,rotation='horizontal')

plt.figure()
plt.title('t = 1')
t=1
uplot=kdv2(c1,c2,t*torch.ones(ntpredict),x.flatten()).detach().cpu()
plt.plot(x,uplot,label=r'$u$')
Xplot = Xpredict[Xpredict[:,1] == t]
umodelplot = model(Xplot).detach().cpu()
plt.plot(x,umodelplot,'--',label=r'$\hat{u}$')
plt.legend()
plt.xlabel(r'$x$',fontsize=12)
plt.ylabel(r'$u$',fontsize=12,labelpad=10,rotation='horizontal')

plt.figure()
plt.title('t = 2')
t=2
uplot=kdv2(c1,c2,t*torch.ones(ntpredict),x.flatten()).detach().cpu()
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

#animation
t=torch.linspace(t_min,t_max,ntpredict)
fig = plt.figure()
ax1 = plt.subplot(111,xlim=(x_min,x_max),ylim=(0,2))
line1, = ax1.plot(x,u[:,0],label=r'$u$')
line2, = ax1.plot(x,upredict[:,0],'--',label=r'$\hat{u}$')
plt.xlabel('$x$',fontsize=12)
plt.ylabel('$u$',fontsize=12,rotation='horizontal',labelpad=15)
plt.legend()

#to create animation, uncomment all lines below
time_text1 = ax1.text(0.4, 0.9, '', fontsize=12,transform=ax1.transAxes)

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=20)

def animate(i):
    line1.set_ydata(u[:,i])
    line2.set_ydata(upredict[:,i])
    time_text1.set_text('t = %.2f' %t[i])
    return line1,line2, time_text1

#anim = animation.FuncAnimation(fig, animate, interval=400,frames=ntpredict-200)
#anim.save('KdV2_vid.mp4', dpi = 300, writer=writer)