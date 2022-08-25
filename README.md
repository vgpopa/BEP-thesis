# Python codes BEP thesis

In an attempt to find alternatives for solving partial differential equations (PDEs) with traditional numerical methods, a new field has emerged which incorporates the residual of a PDE into the loss function of an Artificial Neural Network. This method is called Physics-Informed Neural Network (PINN). In this thesis, we study dense neural networks (DNNs), including codes developed in the context of this bachelor project and derive the backpropagation equations necessary for training. We distinguish between a-PINNs which use automatic differentiation to evaluate a PDE, and n-PINNs which approximate differential operators in a PDE with numerical differentiation. We compare both PINNs on the harmonic oscillator, the 1D heat equation and the 1-soliton and 2-soliton solutions of the Korteweg-De Vries (KdV) equation. Furthermore, we tested a highly nonlinear problem of the KdV equation, which can be described by a train of solitons. Additionally, PINNs were used for inverse problems to derive an unknown coefficient in a PDE.

The thesis has been written as part of the double bachelor's degree programme Applied Physics \& Applied Mathematics at Delft University of Technology. Link to thesis will become available soon.

In the folder Dense Neural Networks from scratch, we implement a dense network using gradient descent, and extend our model to be compatible with more complex optimization algorithms from scipy.optimize like BFGS. 

In the folder PINNs, we implement a-PINNs and n-PINNs with Pytorch and consider different PDEs.

In the folder Animations, some videos are included of the time evolution of the prediction for various PDE problems.

Necessary libraries to run all codes: os, time, numpy, matplotlib, and torch.
