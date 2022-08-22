'''
@author: Vlad Popa
'''

#import libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#from scipy.optimize import Bounds
from functools import partial
from matplotlib import rcParams
rcParams['figure.dpi'] = 600
np.random.seed(5113628)

#parameters in analytic solution to harmonic oscillator
m = 2
mu = 0.5
k = 2
A = 1
phase = 0

#analytic solution to damped harmonic oscillator
def oscillator(t):
    delta = mu / (2*m)
    w0 = np.sqrt(k/m)
    w = np.sqrt(w0**2-delta**2)
    return A*np.exp(-delta*t)*np.cos(w*t+phase)


#cost function
def loss(y_true, y_pred):
  return 0.5*np.sum(((y_true - y_pred) ** 2))

#list of some well known activation functions
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def dsigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(np.zeros(np.shape(x)),x)

def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def x(x):
    return x

def dx(x):
    return np.ones(np.shape(x))

#initialise A and Z list of matrices with correct dimensions
def init(x,neurons_per_layer):
    Z = []
    layers = len(neurons_per_layer)
    for i in range(layers-1):
        Z.append(np.random.random((len(x), neurons_per_layer[i+1])))
    A = Z.copy()
    return A,Z

'''
* Initialise W matrices and b vectors
* In scipy.optimize, all model parameters need to be concatenated into one list
* Each weight matrix is flattened and concatenated into one list
* Each bias vector is concatenated into one list
* Each list is concatenated, to obtain one list of all model parameters
* 
* @param neurons_per_layer: list; specify amount of neurons in each layer
*
* @return: 
* W0: ndarray; list weight matrices initialised
* b0: ndarray; list of bias vectors initialised
* end: float; index where all bias terms follow in the list of all model parameters
'''
def initWb(neurons_per_layer):
    end = 0
    W0 = [] 
    b0 = [] 
    layers = len(neurons_per_layer)
    for i in range(layers-1):
        W0.append(np.random.random((neurons_per_layer[i+1], neurons_per_layer[i])))
        b0.append(np.random.random((neurons_per_layer[i+1], 1)))
        end += neurons_per_layer[i]*neurons_per_layer[i+1]
    return W0,b0,end

#convert list of matrices into one list, where each matrix is flattened
def reshape_list(x):
    return np.concatenate([x[j].flatten() for j in range(len(x))])

'''
* Reshape list of weights into list of weight matrices per layer
* This is because in scipy.optimize, all model parameters need to be concatenated into one list
* 
* @param w: list; list of all weight terms after being updated by scipy.optimize
* @param neurons_per_layer: list; specify amount of neurons in each layer
*
* @return: 
* Wnew: list(ndarrays); list of nested matrices with different shapes, depending on the architecture
            of the neural network specified, of individual weights
'''
def reshapeW(w,neurons_per_layer):
    layers = len(neurons_per_layer)
    Wnew = []
    count = 0
    for i in range(layers-1):
        end = neurons_per_layer[i]*neurons_per_layer[i+1]
        Wnew.append(np.reshape(w[count:count+end],(neurons_per_layer[i+1], neurons_per_layer[i])))
        count += end
    return Wnew

'''
* Reshape list of bias terms into list of bias vectors per layer
* This is because in scipy.optimize, all model parameters need to be concatenated into one list
* 
* @w: list; list of all bias terms after being updated by scipy.optimize
* @param neurons_per_layer: list; specify amount of neurons in each layer
*
* @return: 
* @bnew: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms
'''
def reshapeB(w,neurons_per_layer):
    layers = len(neurons_per_layer)
    bnew = []
    count = 0
    for i in range(layers-1):
        end = neurons_per_layer[i+1]
        bnew.append(np.reshape(w[count:count+end],(neurons_per_layer[i+1], 1)))
        count += end
    return bnew
    

'''
* Wrapper of cost function needed in scipy.optimize
* 
* @param w: list; list of all model parameters returned by scipy.optimize
* @param x: input matrix with N samples
* @param y: ndarray(nL,N); matrix with true labels, needed for training
* @param Z: list of preactivation matrices per layer initialised
* @param A: list of activation matrices per layer initialised
*           identical format like Z, but included for better understanding
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param end: float; index where all bias terms follow in the list of all model parameters
* @param activatornames: ndarray(#layers); activation functions per layer
* @param dactivatornames: ndarray(#layers); derivative of activation functions per layer
*
* @return: 
  cost: list; sum of squares loss
'''
def wrapper1(W,x,y,A,Z,neurons_per_layer,end,activatornames,dactivatornames):
    w = W[0:end]
    b = W[end:]

    wmatrix = reshapeW(w,neurons_per_layer)
    bmatrix = reshapeB(b,neurons_per_layer)

    yhat = forward(x,wmatrix,bmatrix,A,Z,neurons_per_layer,activatornames)
    cost = loss(y,yhat)  
    return cost

'''
* Wrapper of gradients needed in scipy.optimize
* 
* @param w: list; list of all model parameters returned by scipy.optimize
* @param x: input matrix with N samples
* @param y: ndarray(nL,N); matrix with true labels, needed for training
* @param Z: list of preactivation matrices per layer initialised
* @param A: list of activation matrices per layer initialised
*           identical format like Z, but included for better understanding
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param end: float; index where all bias terms follow in the list of all model parameters
* @param activatornames: ndarray(#layers); activation functions per layer
* @param dactivatornames: ndarray(#layers); derivative of activation functions per layer
*
* @return: 
  gradients: list; gradients of all model parameters in each layer concatenated for scipy.optimize
'''
def wrapper2(W,x,y,A,Z,neurons_per_layer,end,activatornames,dactivatornames):
    w = W[0:end]
    b = W[end:]

    wmatrix = reshapeW(w,neurons_per_layer)
    bmatrix = reshapeB(b,neurons_per_layer)

    dw,db = calc_gradients(wmatrix,bmatrix,x,y,A,Z,neurons_per_layer,activatornames,dactivatornames)
    dwnew = reshape_list(dw)
    dbnew = reshape_list(db)
    gradients = np.concatenate((dwnew,dbnew))
    
    return gradients

'''
* callback in scipy.optimize to keep track of loss during training
* 
* @param w: list; list of all model parameters returned by scipy.optimize
* @param x: ndarray(n0,N); input matrix with N samples
* @param y: ndarray(nL,N); matrix with true labels, needed for training
* @param end: float; index where all bias terms follow in the list of all model parameters
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param MSE: list; MSE of training samples for certain epoch
* @param A: list of activation matrices per layer initialised
*           identical format like Z, but included for better understanding
* @param Z: list of preactivation matrices per layer initialised
* @param functions: ndarray(#layers); activation functions per layer
'''
def callbackfunction(w,x,y,end,neurons_per_layer,MSE,A,Z,functions):
    A,Z = init(x,neurons_per_layer)
    W = reshapeW(w[0:end],neurons_per_layer)
    b = reshapeB(w[end:],neurons_per_layer)
    yhat = forward(x,W,b,A,Z,neurons_per_layer,functions)

    MSE.append(loss(y,yhat)/len(y))

#set activator function and check if activator exists
'''
* Gives a method found in a select set of Activator functions
*
* @param key: string; name of the required activator function
*
* @return: callable; method
'''
def activator_finder(key):
    try:
        return {"sigmoid": sigmoid,"dsigmoid": dsigmoid,"tanh": tanh, "dtanh": dtanh,"x": x,"dx":dx,"relu": relu, "drelu": drelu}.get(key)
    except:
        raise ValueError("given Activator function does not exist")
        
#forward propagation algorithm
'''
* Performs one forward propagation through the dense neural network
*
* @param x: ndarray(n0,N); input matrix with N samples
* @param W: list(ndarrays); list of nested matrices with different shapes, depending on the architecture
            of the neural network specified, of individual weights to be optimized
* @param b: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms to be optimized
* @param A: list of activation matrices per layer initialised
*           identical format like Z, but included for better understanding
* @param Z: list of preactivation matrices per layer initialised
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param activatornames: ndarray(#layers); activation functions per layer
*
* @return: ndarray; prediction
'''
def forward(x,W,b,A,Z,neurons_per_layer,activatornames):
    activatorlist = [activator_finder(name) for name in activatornames]
    layers = len(neurons_per_layer)
    n = len(x)
    bdummy = b.copy()
    
    for i in range(0,layers-1):
        #change to correct shape s.t. bias term applied to each input in the same neuron
        bdummy[i] = np.tile(b[i],(1,n))

    Z[0] = np.dot(W[0],x)+bdummy[0] #preactivation
    A[0] = activatorlist[0](Z[0]) #activation
    
    for i in range(1,layers-1):
        Z[i] = np.dot(W[i],A[i-1])+bdummy[i]
        A[i] = activatorlist[i](Z[i])
    return A[-1]

#calculate gradients in each layer 
'''
* Performs one forward propagation through the dense neural network
*
* @param W: list(ndarrays); list of nested matrices with different shapes, depending on the architecture
            of the neural network specified, of individual weights to be optimized
* @param b: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms to be optimized
* @param x: ndarray(n0,N); input matrix with N samples
* @param y: ndarray(nL,N); matrix with true labels, needed for training
* @param A: list of activation matrices per layer initialised
*           identical format like Z, but included for better understanding
* @param Z: list of preactivation matrices per layer initialised
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param activatornames: ndarray(#layers); activation functions per layer
* @param dactivatornames: ndarray(#layers); derivative of activation functions per layer
*
* @return: tuple; list of matrices, which contrain gradients in each layer
'''
def calc_gradients(W,b,x,y,A,Z,neurons_per_layer,activatornames,dactivatornames):
    dactivatorlist = [activator_finder(name) for name in dactivatornames]
    delta = Z.copy() #initialise error matrices
    dW,db = W.copy(),b.copy()
    yhat = forward(x,W,b,A,Z,neurons_per_layer,activatornames)
    
    #backpropagation equations
    delta[-1] = np.multiply(-(y-yhat),dactivatorlist[-1](Z[-1]))
    layers = len(neurons_per_layer)
    for j in range(layers-2,0,-1):
        dW[j] = np.matmul(delta[j],A[j-1].T)
        delta[j-1] = np.dot(W[j].T,delta[j])*dactivatorlist[j-1](Z[j-1])
    dW[0] = np.dot(delta[0],x.T)
    for k in range(layers-1):
        db[k] = np.sum(delta[k],axis=1,keepdims=True)
    return dW,db
    
#training algorithm, using scipy.optimize
'''
* Performs one forward propagation through the dense neural network
*
* @param W: list(ndarrays); list of nested matrices with different shapes, depending on the architecture
            of the neural network specified, of individual weights to be optimized
* @param b: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms to be optimized
* @param x: ndarray(n0,N); input matrix with N samples
* @param y: ndarray(nL,N); matrix with true labels, needed for training
* @param neurons_per_layer: list; specify amount of neurons in each layer
* @param end: float; index where all bias terms follow in the list of all model parameters
* @param activatornames: ndarray(#layers); activation functions per layer
* @param dactivatornames: ndarray(#layers); derivative of activation functions per layer
* @param rep: float; epochs
* @param Optimization: string; specify which optimization algorithm to use 
*
* @return: 
* @Wnew: ndarray; new weigths after training
* @bnew: ndarray; new bias terms after training
* @MSE: list; list of MSE suring training
* @new_params['nit']: float; amount of iterations performed by scipy, may differ from
                             epochs due to specified tol or tol of algorithm in source code 
'''
def train(W,b,x,y,neurons_per_layer,end,activatornames,dactivatornames,rep,Optimization):
    A,Z = init(x,neurons_per_layer)
    MSE = []
    nnparams = np.concatenate((reshape_list(W),reshape_list(b)))
 
    new_params = minimize(wrapper1, nnparams, args=(x,y,A,Z,neurons_per_layer,end,activatornames,dactivatornames), method=Optimization, options={'maxiter':rep,'disp':False},\
            jac=wrapper2,callback=partial(callbackfunction,x=x,y=y,end=end,neurons_per_layer=neurons_per_layer,MSE=MSE,A=A,Z=Z,functions=activatornames),tol=1e-12)

    Wnew = reshapeW(new_params.x[0:end],neurons_per_layer)
    bnew = reshapeB(new_params.x[end:],neurons_per_layer)

    return Wnew,bnew,MSE,new_params['nit']

#one forward propagation
def predict(x,W,b,neurons_per_layer,functions):
    A,Z = init(x,neurons_per_layer)
    return forward(x,W,b,A,Z,neurons_per_layer,functions)

################################
#for comparison, a DNN implementation in Pytorch with Adam
import torch.nn as nn
import torch
torch.manual_seed(5113628)

def oscillator_torch(t):
    delta = mu / (2*m)
    w0 = np.sqrt(k/m)
    w = np.sqrt(w0**2-delta**2)
    return A*torch.exp(-delta*t)*torch.cos(w*t+phase)

torch.set_default_dtype(torch.float)
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ANN(neurons_per_layer,activations):
    layers = []
    for i in range(len(neurons_per_layer)-1):
        layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i+1]))
        if activations[i] == 'off':
            pass
        else:
            layers.append(activations[i]())
    return nn.Sequential(*layers)
################################

if __name__ == "__main__":
    #parameters to be changed
    neurons_per_layer = np.array([1,8,8,1])
    epochs = 500
    optimization='BFGS'
    functions = np.array(["tanh","tanh","x"])
    dfunctions = np.array(list(map(lambda x: 'd' + x,functions)))
        
    #training data
    ntraining = 500
    xtest = np.linspace(0, 15, ntraining).reshape(neurons_per_layer[0],ntraining)
    ytest = oscillator(xtest)[0]

    #prediction data
    npredict = 3000
    xpredict = np.linspace(0, 30, npredict).reshape(neurons_per_layer[0],npredict)
    
    #labels from prediction data
    ytrue = oscillator(xpredict)[0]

#############
    #create plot for comparison between different optimization algorithms
    def plots():
        optimization=['L-BFGS-B','BFGS','CG','Powell']
        nlist=[]
        loss=[ [] for _ in range(len(optimization)) ]

        #Adam
        model = ANN(neurons_per_layer,[nn.Tanh,nn.Tanh,'off']).to(mydevice)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
        loss_adam=[]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.5*epochs), gamma=0.1)
        
        t_test = torch.linspace(0,15,500).view(-1,1).float().to(mydevice)
        u_test = oscillator_torch(t_test)
        
        for i in range(epochs):
            optimizer.zero_grad()
            
            u_adam = model(t_test)
            model_loss = nn.MSELoss()
            Loss = model_loss(u_adam,u_test)
            loss_adam.append(Loss.detach().cpu().numpy())
            Loss.backward()
            optimizer.step()
            scheduler.step()
        ypredict_adam = model(torch.from_numpy(xpredict.copy()).float().to(mydevice)).flatten().cpu().detach().numpy()
        
        #create prediction plots
        plt.figure()
        for j in range(len(optimization)):
            W0, b0,end = initWb(neurons_per_layer)

            W,b,MSE,n =  train(W0,b0,xtest,ytest,neurons_per_layer,end,functions,dfunctions,epochs,optimization[j],False,0,0,0)
            loss[j] = MSE
            nlist.append(n)
            ypredict = predict(xpredict,W,b,neurons_per_layer,functions)
            plt.plot(xpredict.flatten(),ypredict.flatten(),label=optimization[j])
        plt.plot(xpredict.flatten(),ypredict_adam,label='Adam')
        plt.xlabel(r'$t$',fontsize=12)
        plt.ylabel(r'$u$',fontsize=12)
        plt.legend()
        
        #create plots of evolution of training loss
        plt.figure()
        for j in range(len(optimization)):
            i = np.arange(1,nlist[j]+1)
            plt.plot(i,loss[j],label=optimization[j])
            plt.xlabel(r'$\it{epochs}$',fontsize=12)
            plt.ylabel(r'log($\mathcal{L})$',fontsize=12)
            plt.yscale("log")
        plt.plot(np.arange(1,epochs+1),loss_adam,label='Adam')
        plt.legend()
        plt.show()
#############
    #plots()
    
    #plot predictions and training loss
    W0, b0,end = initWb(neurons_per_layer)
    W,b,MSE,n =  train(W0,b0,xtest,ytest,neurons_per_layer,end,functions,dfunctions,epochs,optimization)
    ypredict = predict(xpredict,W,b,neurons_per_layer,functions)
    plt.figure()
    plt.plot(xpredict.flatten(),ypredict.flatten(),label=r'$\hat{u}$')
    plt.plot(xpredict.flatten(),ytrue.flatten(),label=r'$u$')
    plt.plot(xtest.flatten(),ytest.flatten(),'.',label='training data',markersize=1)
    plt.xlabel(r'$t$',fontsize=12)
    plt.ylabel(r'$u$',fontsize=12,labelpad=10,rotation='horizontal')
    plt.legend()
    plt.figure()
    i = np.arange(1,n+1)
    plt.plot(i,MSE)
    plt.xlabel(r'$\it{epochs}$',fontsize=12)
    plt.ylabel(r'$\mathcal{L}$',fontsize=12,labelpad=15,rotation='horizontal')
    plt.yscale("log")
    plt.show()
    print('Final training MSE: ',MSE[-1])
    print('Prediction MSE: ',loss(ytrue,ypredict)/len(ytrue))