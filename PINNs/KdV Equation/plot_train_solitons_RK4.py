"""
@author: Vlad Popa
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.dpi'] = 600
import time

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
x0=-2
xn=2
L=xn-x0

#grid
nx  = 512
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

#solve ode
t0 = 0 #initial time
tn  = 4 #final time
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

end = time.time()
print('elapsed time:',end-start,'s')

#bounds for plot
t_min, t_max,x_min, x_max = t0,tn,x0,xn
ext = [t_min, t_max,x_min, x_max]

#plot solution
plt.figure()
plt.imshow(ulist,extent=ext,origin='lower',vmin=np.min(ulist),vmax=np.max(ulist))
plt.xlabel(r"$t$",fontsize=12)
plt.ylabel(r"$x$",fontsize=12,rotation='horizontal',labelpad=10)
plt.colorbar()
plt.show()

#plot solution at certain time
t = 0.5
tindex = np.where(tlist==t)[0][0]
u = ulist[:,tindex]
plt.figure()
plt.plot(x,u)
plt.xlabel(r"$x$",fontsize=12)
plt.ylabel(r"$u$",fontsize=12,rotation='horizontal',labelpad=10)
plt.show()

#3d plot
plt.figure()
X,T= np.meshgrid(x,tlist)
ax = plt.axes(projection='3d')
ax.plot_surface(T, X, np.transpose(ulist),cmap="rainbow")
ax.set_xlabel(r'$t$',fontsize=12)
ax.set_ylabel(r'$x$',fontsize=12)
ax.set_zlabel(r'$u$',fontsize=12)
plt.show()