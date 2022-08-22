'''
@author: Vlad Popa
'''

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.dpi'] = 600

np.random.seed(5113628)

#parameters to be changed
learning_rate_weight = 0.001 
learning_rate_bias = 0.001 
neurons_per_layer = np.array([1,4,4,1])
epochs = 10000

#activation functions per layer with respective derivative
functions = np.array(["tanh","tanh","x"])
dfunctions = np.array(list(map(lambda x: 'd' + x,functions)))
layers = len(neurons_per_layer)

#list of some well known activation functions and derivatives
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

#cost function
def loss(y_true, y_pred):
  return 0.5*np.sum(((y_true - y_pred) ** 2))

#target function
def testfunction(x):
    return x**2

#initialise W, b, dW, db matrices and list of MSE
W0 = [] 
b0 = [] 

for i in range(0,layers-1):
    W0.append(np.random.random((neurons_per_layer[i+1], neurons_per_layer[i])))
    b0.append(np.random.random((neurons_per_layer[i+1], 1)))

db = b0.copy()
dW = W0.copy()
MSE = np.ones(epochs)

#dictionary of some well known activation functions
activator_functions = {"sigmoid": sigmoid,"dsigmoid": dsigmoid,"tanh": tanh, "dtanh": dtanh,"x": x,"dx":dx,"relu": relu, "drelu": drelu}
'''
* Gives a method found in a select set of Activator functions
*
* @param key: string; name of the required activator function
*
* @return: callable; method
'''
def activator_finder(activatorname):
    #set activator function and check if activator exists
    try:
        activator = activator_functions[activatorname]
    except:
        activator_error = ValueError("given Activator function does not exist")
        raise activator_error
    return activator

#initialise A and Z list of matrices with correct dimensions
def init(x):
    Z = []
    for i in range(layers-1):
        Z.append(np.random.random((neurons_per_layer[i+1],len(x))))
    A = Z.copy()
    return A,Z

#forward propagation algorithm
'''
* Performs one forward propagation through the dense neural network
*
* @param x: ndarray(n0,N); input matrix with N samples
* @param W: list(ndarrays); list of nested matrices with different shapes, depending on the architecture
            of the neural network specified, of individual weights to be optimized
* @param b: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms to be optimized
* @param activatornames: ndarray(#layers); activation functions per layer
*
* @return: ndarray; prediction
'''
def forward(x,W,b,activatornames = functions):
    global Z,A
    activatorlist = [activator_finder(name) for name in activatornames]
    
    n = len(x)
    bdummy = b.copy()
    for i in range(layers-1):
        #change to correct shape s.t. bias term applied to each input in the same neuron
        bdummy[i] = np.tile(b[i],(1,n))
    Z[0] = np.dot(W[0],x)+bdummy[0] #preactivation
    A[0] = activatorlist[0](Z[0]) #activation
    
    for i in range(1,layers-1):
        Z[i] = np.dot(W[i],A[i-1])+bdummy[i]
        A[i] = activatorlist[i](Z[i])
    return A[-1]

#backwards propagation algorithm using gradient descent
'''
* Updates model parameters with gradient descent
*
* @param W: list(ndarrays); list of nested matrices with different shapes, depending on the architecture
            of the neural network specified, of individual weights to be optimized
* @param b: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms to be optimized
* @param x: ndarray(n0,N); input matrix with N samples
* @param y: ndarray(nL,N); matrix with true labels, needed for training
* @param b: list(ndarrays); list of vectors with different shapes, depending on the architecture
            of the neural network specified, of the bias terms to be optimized
* @param dactivatornames: ndarray(#layers); derivative of activation functions per layer
                                 
* @return: tuple; updated model parameters
'''
def train(W,b,x,y,dactivatornames = dfunctions):
    dactivatorlist = [activator_finder(name) for name in dactivatornames]
    delta = Z.copy() #initialise error matrices

    for i in range(epochs):
        yhat = forward(x,W,b)
        MSE[i] = loss(ytest,yhat)/len(ytest[0]) #MSE during training
        
        #backpropagation equations
        delta[-1] = -(y-yhat)*dactivatorlist[-1](Z[-1])
        for j in range(layers-2,0,-1):
            dW[j] = np.matmul(delta[j],A[j-1].T)
            delta[j-1] = np.dot(W[j].T,delta[j])*dactivatorlist[j-1](Z[j-1])

        dW[0] = np.dot(delta[0],x.T)
        
        #update model parameters
        for k in range(layers-1):
            W[k] -= learning_rate_weight*dW[k]
            b[k] -= learning_rate_bias*np.sum(delta[k],axis=1,keepdims=True)
    return W,b

if __name__ == "__main__":
    #trainingdata
    ntraining = 500 #amount of training samples
    #xtest = np.random.uniform(low=-2, high=2, size=ntraining).reshape(neurons_per_layer[0],ntraining)
    xtest = np.linspace(-2, 2, ntraining).reshape(neurons_per_layer[0],ntraining)
    ytest = testfunction(xtest)
    
    #initialise preactivation and activation matrices to correct shapes
    A,Z = init(xtest)
    
    #train netwerk
    W,b =  train(W0,b0,xtest,ytest)

    #new data, to be predicted
    npredict = 100 #amount of samples to be predicted
    xpredict0 = np.linspace(-2, 2, npredict).reshape(neurons_per_layer[0],npredict)
    A,Z = init(xpredict0)
    ypredict = forward(xpredict0,W,b).reshape(npredict,neurons_per_layer[-1])
    xpredict = xpredict0.reshape(npredict,neurons_per_layer[0])

    #true prediction values
    ytrue = testfunction(xpredict)

    #plot of target function and prediction
    plt.figure()
    plt.plot(xpredict,ypredict,'--',label=r'$\hat{y}$')
    plt.plot(xpredict,ytrue,label=r'$y$')
    #plt.plot(xtest.reshape(ntraining,neurons_per_layer[0]),ytest.reshape(ntraining,neurons_per_layer[0]),'x',label='training samples')
    plt.xlabel(r'$x$',fontsize=12)
    plt.ylabel(r'$y$',fontsize=12,labelpad=15,rotation='horizontal')
    plt.legend(loc='upper center')
    plt.show()
    
    #plot of evolution of MSE with time
    plt.figure()
    i = np.arange(1,epochs+1)
    plt.plot(i,MSE)
    plt.xlabel(r'$\it{epochs}$',fontsize=12)
    plt.ylabel(r'$\mathcal{L}$',fontsize=12,labelpad=15,rotation='horizontal')
    plt.yscale("log")
    plt.show()
    print("Final training MSE: ",MSE[-1])
    print("Prediction MSE: ",loss(ytrue,ypredict)/len(ytrue))