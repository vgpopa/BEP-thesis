# Abstract

In an attempt to find alternatives for solving partial differential equations (PDEs) with traditional numerical methods, a new field has emerged which incorporates the residual of a PDE into the loss function of an Artificial Neural Network. This method is called Physics-Informed Neural Network (PINN). In this thesis, we study dense neural networks (DNNs), including codes developed in the context of this bachelor project. We derive the backpropagation equations necessary for training and use different configurations in a DNN to test its interpolating accuracy. We distinguish between a-PINNs which use automatic differentiation to evaluate a PDE, and n-PINNs which approximate differential operators in a PDE with numerical differentiation. We compare both PINNs on the harmonic oscillator, the 1D heat equation and the 1-soliton and 2-soliton solutions of the Korteweg-De Vries (KdV) equation. Both PINNs could accurately converge to the solution, except to the 2-soliton solution, where the a-PINN outperformed the n-PINN. Furthermore, we tested a highly nonlinear problem of the KdV equation, which can be described by a train of solitons. We observed that PINNs are inaccurate if insufficient training samples are used for training. Adding training samples on the interior from a numerical solution leads to a good qualitative agreement, though more effort is required to find a better network configuration to obtain more accurate predictions. 

Additionally, PINNs were used for inverse problems to derive an unknown coefficient in a PDE and proved to be highly accurate for noiseless data. When we generated training samples with 10\% noise from a uniform distribution, the PINN results' relative error stayed within a margin of under 2\%. However, inverse PINNs are much more inefficient compared to nonlinear least squares methods like the Levenberg–Marquardt algorithm. 

As of now, PINNs are still very early in development and stand no match against traditional numerical methods to a known PDE. They may, however, provide a useful alternative in the future as they are constantly being improved. 


The thesis has been written as part of the double bachelor's degree programme Applied Physics \& Applied Mathematics at Delft University of Technology. Link to thesis will become available soon.

In the folder Dense Neural Networks from scratch, we implement a dense network using gradient descent, and extend our model to be compatible with more complex optimization algorithms from scipy.optimize.

In the folder PINNs, we implement a-PINNs and n-PINNs with Pytorch and consider different PDEs.

Necessary libraries to run all codes: os, time, numpy, matplotlib, and torch.
