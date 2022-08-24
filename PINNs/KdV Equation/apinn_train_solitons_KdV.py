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

#Runge-Kutta 4th order
def RK4(odefunc,Fu):
    w = Fu
    for i in range(len(tlist)):
        w = RK4Step(odefunc, tlist[i], w, h)
        vlist.append(w)

def RK4Step(odefunc,t,w,h):
    k1 = odefunc(t,w)
    k2 = odefunc(t+0.5*h,w+0.5*k1*h)
    k3 = odefunc(t+0.5*h,w+0.5*k2*h)
    k4 = odefunc(t+h,w+k3*h)
    return w+(k1+2*k2+2*k3+k4)*(h/6)


#change of variables
#notation: Fu is Fourier transformation of u
def Fu_to_Fv(t,Fu):
    return np.exp(-1j*(kx**3)*delta2*t)*Fu

def Fv_to_Fu(t,Fv):
    return np.exp(1j*(kx**3)* delta2*t)*Fv

#time derivative Fv
def dFv(t, Fv):
    u = np.fft.ifft(Fv_to_Fu(t,Fv))
    return -0.5j*kx*Fu_to_Fv(t, np.fft.fft(u**2))


#bounds
x0=-1
xn=1
L=xn-x0

#grid
nx  = 200
x   = np.linspace(x0,xn, nx+1)
x   = x[:nx]

#k-space
kx1 = np.linspace(0,int(nx/2-1),int(nx/2))
kx2 = np.linspace(1,int(nx/2),  int(nx/2))
kx2 = -1*kx2[::-1]
kx  = (2*np.pi/L)*np.concatenate((kx1,kx2))

#delta^2 in KdV equation
delta2 = 0.001

#IC
u0      = np.cos(np.pi*x)
Fu0   = np.fft.fft(u0)

start = time.time()

#solve pde
t0 = 0 #initial time
tn  = 1 #final time
nt = 6000
h = (tn-t0)/nt
tlist=np.arange(t0,tn+h,h)
Fv0 = Fu_to_Fv(t0,Fu0)
vlist=[]

#Perform RK4
RK4(dFv,Fv0)


vlist = np.array(vlist)
ulist = []

for i in range(len(tlist)):
    ulist.append(np.fft.ifft(Fv_to_Fu(tlist[i],vlist[i])))

ulist = np.array(ulist)
ulist = np.real(np.transpose(ulist))

#add second IC
tIC2 = 1
tindex = np.where(tlist==tIC2)[0][0]
u1 = ulist[:,tindex]


#bounds for plot
t_min, t_max,x_min, x_max = t0,tn,x0,xn
ext = [t_min, t_max,x_min, x_max]

#different loss terms
IC1_loss_list = [] #MSE initial condition 1 
IC2_loss_list = [] #MSE initial condition 2
interior_list = [] #MSE interior of residue KdV
labels_list = [] #MSE training labels
total_loss_list=[] #IC1_loss_list+IC2_loss_list+interior_list+labels_list

#parameters to be changed in ANN
neurons_per_layer=[2,50,50,50,50,50,50,50,50,50,1]
activations = [nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,nn.Tanh,'off']
epochs = 50000

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
nx = 200
nt = 200

#grid
t_physics = torch.linspace(t_min,t_max,nt).repeat_interleave(nx).view(-1,1)
x_physics = torch.linspace(x_min,x_max,nx).repeat(nt,1).view(-1,1)#.to(mydevice)

#create grid in correct shape and exclude ICs
t0=0 #IC1
t1=tIC2 #IC2, final time
Xphysics = torch.cat((x_physics,t_physics),1)
Xphysics = Xphysics[Xphysics[:,1] != t0]
Xphysics = Xphysics[Xphysics[:,1] != t1]

x_physics = Xphysics[:,[0]].to(mydevice)
t_physics = Xphysics[:,[1]].to(mydevice)
x_physics.requires_grad_(True)
t_physics.requires_grad_(True)

#solution at given ICs
t_0 = t0*torch.ones(nt).view(-1,1)
t_1 = t1*torch.ones(nt).view(-1,1)
x_0 = torch.linspace(x_min,x_max,nx).view(-1,1)
u0 = torch.cos(np.pi*x_0).float().to(mydevice)
u1 = torch.from_numpy(u1).view(-1,1).float().to(mydevice)

X_IC1 = torch.cat([x_0,t_0],axis=1).to(mydevice)
X_IC2 = torch.cat([x_0,t_1],axis=1).to(mydevice)

model = ANN(neurons_per_layer,activations)
model.to(mydevice)

#add training samples on interior (optional)
nsamples = 500
tindex,xindex = np.random.randint(0,nt+1,nsamples),np.random.randint(0,nx,nsamples)
tnew=[]
xnew=[]
unew=[]
for i in range(nsamples):
    tnew.append(tlist[tindex[i]])
    xnew.append(x[xindex[i]])
    unew.append(ulist[xindex[i],tindex[i]])
t_labels = torch.FloatTensor(tnew).view(-1,1)
x_labels = torch.FloatTensor(xnew).view(-1,1)
utrue = torch.FloatTensor(unew).view(-1,1).to(mydevice)

Xtrain = torch.cat([x_physics,t_physics],axis=1).to(mydevice)
X_labels = torch.cat((t_labels,x_labels),1).to(mydevice)

#implementation using Adam        
def adam():
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,15000], gamma=0.1) #adaptive learning rate
    for i in range(epochs):
        #Pytorch accumulates gradients during training, so set to zero before optimization step
        optimizer.zero_grad()
     
        up = model(Xtrain) #prediciton at collocation points
        ut = model(X_labels) #prediction at training samples on interior
        mse = nn.MSELoss()

        u_t = torch.autograd.grad(up,t_physics,torch.ones_like(t_physics), create_graph=True)[0] #du/dt
        u_x = torch.autograd.grad(up,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #du/dx
        u_xx = torch.autograd.grad(u_x,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #d^2u/dx^2
        u_xxx = torch.autograd.grad(u_xx,x_physics,torch.ones_like(x_physics), create_graph=True)[0] #d^3u/dx^3

        physics = u_t + up*u_x + delta2*u_xxx #residue KdV equation
        zeros = torch.zeros(Xtrain.shape[0],1).to(mydevice)
        physics_loss = mse(physics,zeros)
        
        model0 = model(X_IC1) #prediction first IC
        model1 = model(X_IC2) #prediction second IC
        IC1_loss = mse(model0,u0)
        IC2_loss = mse(model1,u1)
        losslabels = mse(ut,utrue)
        

        #total loss, comment losslabels to exclude training samples interior
        total_loss = physics_loss + IC1_loss + IC2_loss+losslabels

        total_loss_list.append(total_loss.detach().cpu().numpy())
        IC1_loss_list.append(IC1_loss.detach().cpu().numpy())
        IC2_loss_list.append(IC2_loss.detach().cpu().numpy())        
        interior_list.append(physics_loss.detach().cpu().numpy())
        
        total_loss.backward() #backward propagation
        torch.nn.utils.clip_grad_norm_(model.parameters(),1) #enable clipping
        optimizer.step() #perform one optimization step
        scheduler.step() #update learning rate as specified by condition set in scheduler
        if i % 1000 ==0:
            print(i,'/',epochs)

start = time.time()
adam()
end = time.time()
print('elapsed time:',end - start,'s')

#prediction gridsize
ntpredict=6001
nxpredict=200

#prediction data in correct shape
t_predict = torch.linspace(t_min,t_max,ntpredict).repeat_interleave(nxpredict).view(-1,1)
x_predict = torch.linspace(x_min,x_max,nxpredict).repeat(ntpredict,1).view(-1,1)
Xpredict = torch.cat((x_predict,t_predict),1).float().cpu()

#move everything onto the same device: cpu
u1 = u1.cpu() 
model.to('cpu')

#prediction
upredict = model(Xpredict).detach().to('cpu')
upredict = torch.transpose(upredict.reshape(ntpredict,nxpredict),0,1)

#common margins for colorbar for better comparison
umax=max(np.max(ulist),torch.max(upredict))
umin=min(np.min(ulist),torch.min(upredict))

#prediction
plt.figure()
plt.imshow(upredict,extent=ext,origin='lower',vmin=umin,vmax=umax)
plt.xlabel(r"$t$",fontsize=12)
plt.ylabel(r"$x$",fontsize=12,rotation='horizontal',labelpad=15)
plt.colorbar()

#solution from numerical simulations
plt.figure()
plt.imshow(ulist,extent=ext,origin='lower',vmin=umin,vmax=umax)
plt.xlabel(r"$t$",fontsize=12)
plt.ylabel(r"$x$",fontsize=12,rotation='horizontal',labelpad=15)
plt.colorbar()

#training loss over time
plt.figure()
i = np.arange(1,epochs+1)
plt.plot(i,total_loss_list)
plt.xlabel(r'$\it{epochs}$',fontsize=12)
plt.ylabel(r'$\mathcal{L}$',fontsize=12,rotation='horizontal',labelpad=15)
plt.yscale("log")

#plot prediction and solution at final time
plt.figure()
t=tn
plt.title('t = %.2f' %t)
plt.plot(x,u1,label=r'$u$')
Xplot = Xpredict[Xpredict[:,1] == tn]
umodelplot = model(Xplot).detach().cpu()
plt.plot(x,umodelplot,'--',label=r'$\hat{u}$')
plt.legend()
plt.xlabel(r'$x$',fontsize=12)
plt.ylabel(r'$u$',fontsize=12,rotation='horizontal',labelpad=15)
plt.show()

MSE = nn.MSELoss()
ulist = torch.from_numpy(ulist)
print('Prediction MSE:',MSE(upredict,ulist).item())
print('Total training loss:',total_loss_list[-1])
print('Interior MSE training loss:',interior_list[-1])
print('IC1 MSE loss:',IC1_loss_list[-1])
print('IC2 MSE loss:',IC2_loss_list[-1])