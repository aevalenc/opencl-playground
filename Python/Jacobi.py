# Alejandro Valencia
# OpenCL Projects:Jacobi Iteration
# Start: 17 September, 2019
# Update: 17 September, 2019

#/***********************************************************************
#* This code utilizes the OpenCL Standard to parallelize the Jacobi     *
#*  iterative method to solve linear systems in the form Ax = b         *
#***********************************************************************/

import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
import time
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#/***********************************************************************
#* Main Program                                                         *
#***********************************************************************/

#[]:Problem Setup
n   = 101   # Number of nodes
bcl = 200   # left boundary condition
bcr = 400   # right boundary condition
tol = 0.001 # tolerance for iterative method

#construct Matrix A
A = np.zeros((n,n)).astype(np.float32)
band = [-1, 2, -1]
for i in range(1,n-1):
    A[i,i-1:i+2] = band
#end i
A[0,0:2] = [2, -1]
A[-1,-2:n] = [-1, 2]

#Right hand side
b = np.zeros(n).astype(np.float32)
b[0]  = bcl
b[-1] = bcr

#Calculate the Residuals once to intialize values
x = np.zeros(n).astype(np.float32) # initial guess of solution
iter = 1
RES  = np.asb(b - np.matmul(A,x))


#/***********************************************************************
#* Standard Jacobi Algorithm                                            *
#***********************************************************************/
#Start clock
start = time.time()

#Loop until Convergence
while np.max(RES) > tol:
    iter += 1
    xn = x.copy()
    for i in range(0,n):
        sum = 0
        for j in range(0,n):
            if j != i:
                sum += A[i,j]*xn[j]
            #end if
        #end j
        x[i] = (b[i]-sum)/A[i,i]
    #end i

    RES  = np.abs(b - np.matmul(A,x))
    #print("iter = %3d | Max Residual = %2.6f" % (iter,np.max(RES)))

#end while

#Stop clock
stop = time.time()
print("Standard Jacobi Algorithm compute time: %fs" %(stop-start))
print("iter = %3d | Max Residual = %2.6f" % (iter,np.max(RES)))


#/***********************************************************************
#* Parallelization Via OpenCL                                           *
#***********************************************************************/

#[]:Re-initialize the problem
x = np.zeros(n).astype(np.float32) # initial guess of solution
iter = 1
RES  = np.abs(b - np.matmul(A,x))


#Start clock
start = time.time()

#[A]:Retrieve Platforms
platforms = cl.get_platforms()
platform_extension = platforms[0].extensions
platform_name = platforms[0].name


#[B]:Retrieve Devices
# On this system platform 0 is the OpenCL Intel Compute Runtime. At the
#   current momment run OpenCL only on CPU
device = platforms[0].get_devices(cl.device_type.CPU) # Retrieve CPU
extensions  = device[0].extensions                    # Supported extensions
device_name = device[0].name                          # Device name


#[C]:Create Context and Queue
cxt = cl.Context([device[0]])
queue = cl.CommandQueue(cxt)


#[D]:Create Kernel to be Executed
prg = cl.Program(cxt,"""
    __kernel void jacobi(
    int n, __global const float *A, __global const float *xn, __global const float *b, __global float *x)
    {
        //[A]:Declarations
        int k;
        float sum,tmp;


        //[B]:Get Global ID Values
        int gid  = get_global_id(0); // Row number of result


        //[C]:Compute result @ Global ID Pair (gid,gid2)
        sum = 0.0;
        for (k = 0; k < n; k++){
            if (k != gid){
                tmp = A[k + n*gid] * xn[k];
                sum = sum + tmp;
            }
        }

        x[gid] = (b[gid] - sum)/A[gid + n*gid];

    }//end kernel
""").build()


#[F]:Deploy Kernel for Device Execution
# The following lines associate the arguments to the kernel and deploys
#   it for device execution by calling the method that PyOpenCL generates
#   in program with the built kernel name: backsub. Because back substi
#   tution can only parallelize the subtraction of the elements of U, we
#   loop through the number of columns in main program. Pre-assign the
#   memory flags and kernel variable.
#
mem_flags  = cl.mem_flags
krnl = prg.jacobi


#[G]:Loop Until Convergence
while np.max(RES) > tol:
    #Advance iteration counter
    iter += 1
    xn = x.copy()

    #[H]:Create Buffers
    # Allocate device memory and move input data from the host to the device
    #   memory. Also create a buffer for the result "x". Note A and b are
    #   static and will not change during kernel execution. We denote the
    #   memory flag as READ_ONLY. x will be both read and written to
    #   by OpenCL device and therefore uses the READ_WRITE memory flag
    #
    A_buf  = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A)
    b_buf  = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)
    xn_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=xn)
    x_buf  = cl.Buffer(cxt, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=x)

    #[I] Set Kernel Arguments
    # For arguments that are numpy arrays enter the created buffers
    krnl.set_args(np.int32(n), A_buf, xn_buf, b_buf,x_buf)

    #[J] Execute Kernel and Copy Result to Global Memory
    # The previously created queue is the first argument followed by
    #   kernel arguments, the shape of the resulting vector. When the
    #   Kernel is finsihed being executed the data stored in the result
    #   buffer will be copied into the result variable
    cl.enqueue_nd_range_kernel(queue, krnl, x.shape, None)
    cl.enqueue_copy(queue,x,x_buf)

    RES  = np.abs(b - np.matmul(A,x))
    #print("iter = %3d | Max Residual = %2.6f" % (iter,np.max(RES)))


#end while

#Stop clock
stop = time.time()
print("Parallelized Jacobi Algorithm compute time: %fs" %(stop-start))
print("iter = %3d | Max Residual = %2.6f" % (iter,np.max(RES)))

#[]:Plot
xvector = np.linspace(0,1,n)
plt.plot(xvector,x)
plt.show()
