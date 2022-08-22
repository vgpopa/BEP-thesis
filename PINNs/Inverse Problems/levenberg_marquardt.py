"""
@author: Vlad Popa
"""
#import libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit 
import time
plt.rcParams['figure.dpi']= 600
np.random.seed(5113628)

m = 2
mu = 0.5
k = 2
A = 1
phase = 0


def oscillator(t,mu):
    delta = mu / (2*m)
    w0 = np.sqrt(k/m)
    w = np.sqrt(w0**2-delta**2)
    return A*np.exp(-delta*t)*np.cos(w*t+phase)


t_osc_physics = np.linspace(0,30,1000) #training samples
y_osc_true = oscillator(t_osc_physics,mu) #true labels
err_osc = np.random.uniform(low=0.99, high=1.01, size=(len(t_osc_physics))) #generate noise
y_osc = err_osc*y_osc_true #noisy data

#Levenberg–Marquardt algorithm
start = time.clock()
mupredict, covariance = curve_fit(oscillator,t_osc_physics,y_osc)
end = time.clock()

print('Elapsed time:',end-start,'s')
print("Predicted mu", mupredict[0], "with standard deviation: ", np.sqrt(covariance[0,0]))