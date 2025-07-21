# Alejandro Valencia
# OpenCL Projects: 1-D Steady State Diffusion
# Start: 13 March, 2019
# Update: 13 March, 2019

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl


#/***********************************************************************
#* Implicit Finite Difference Method with OpenCL                        *
#***********************************************************************/

q2 = 1

#[A]:Mesh Properties
nx = 5         # Number of nodes
xf = 1.0        # Right bound for x
x0 = 0.0        # Left bound for x
dx = (xf-x0)/(nx-1) # Space increment

#[B]:Create Matrix
s = (nx,nx)         # Size of Matrix
A = np.zeros(s)     # Initialize Matrix
a = [1, -2, 1]      # The list of values that will be inserted into A
for i in range(1,nx-1):
    A[i,i-1:i+2] = a

#[C]:Apply Boundary Conditions
As = A[1:nx-1,1:nx-1]   # Reduce Matrix A by eliminating rows and columns
Ai = la.inv(As)         # Calculate the inverse of Matrix A

#[D]:Initialize Solution
us = np.zeros((nx-2))
us[0]  = -u[0]
us[-1] = -u[-1]

#/***********************************************************************
#* Parallelization Via OpenCL                                           *
#***********************************************************************/
sv = (1,1)      # Size of solution vector u i.e. 1 vector input
sm = (1,nx-2)   # Size of matrix in A inverse

print(Ai)
print(sm)


U = np.dot(Ai,us)

up = np.zeros(nx)
up[0] = u[0]
up[-1] = u[-1]
up[1:-1] = U


if q2 == 0:
    plt.plot(x,up)
    plt.show()
