# Alejandro Valencia
# OpenCL Projects: Conjugate Gradient Method
# Start: 19 April, 2019
# Update: 20 April, 2019

#/***********************************************************************
#* This program utiltizes OpenCL to parallelize the Conjugate Gradient  *
#*  Method via the steepest descent scheme to solve a system of linear  *
#*  equations iteratively                                               *
#***********************************************************************/


import pyopencl as cl
import numpy as np
import MatrixMultiply as MM
import MatrixMultiplyRowDom as MMRD
import time
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


#/***********************************************************************
#* Main Program                                                         *
#***********************************************************************/

#[A]:Setup Problem
tri = np.array([-1,2,-1]).astype(np.float32)
A   = np.zeros((200,200)).astype(np.float32)
A[0,0:2] = np.array([2,-1]).astype(np.float32)
A[199,-2:200] = np.array([-1,2]).astype(np.float32)
for i in range(1,199):
    A[i,i-1:i+2] = tri
#end i

b   = np.zeros((200,1)).astype(np.float32)
x   = 0.1*np.ones((200,1)).astype(np.float32)
n   = len(A)
tol = 0.01
b[0] = 200
b[-1] = 400

# // Start Clock //
start = time.time()

#/***********************************************************************
#* Parallelization Via OpenCL                                           *
#***********************************************************************/

#[B]:Retrieve Platforms
platforms = cl.get_platforms()
platform_extension = platforms[0].extensions
platform_name = platforms[0].name


#[C]:Retrieve Devices
# On this system platform 0 is the OpenCL Intel Compute Runtime. At the
#   current momment run OpenCL only on CPU
device = platforms[0].get_devices(cl.device_type.CPU) # Retrieve CPU
extensions  = device[0].extensions                    # Supported extensions
device_name = device[0].name                          # Device name


#[D]:Create Context and Queue
cxt = cl.Context([device[0]])
queue = cl.CommandQueue(cxt)


#[E]:Create Kernel to be Executed
pushstart = time.time()
prg = cl.Program(cxt,"""
    __kernel void ConjGradSD(
    int n, __global const float *RESn, __global const float *AR, __global float *x, __global float *RES)
    {
        //[A]:Get Global ID Values
        int gid = get_global_id(0); // Row number of result x


            int k;           // Counter
            float sumRR;     // sum for matrix multiplication b/w Residuals
            float sumARR;    // sum for matrix multiplication b/w Residuals & AR
            float s;         // Gradient Direction Scalar

            //[B]:Calculate Gradient Direction Scalar
            sumRR = 0;
            sumARR = 0;
            for (k = 0; k < n; k++){
                sumRR  = sumRR  + RESn[k]*RESn[k];
                sumARR = sumARR + RESn[k]*AR[k];
            }//end k

            s = sumRR/sumARR;


        //[C]:Compute result @ Global ID (gid)
        x[gid] = x[gid] + s*RESn[gid];
        RES[gid] = RESn[gid] - s*AR[gid];

    }//end kernel
""").build()

pushstop = time.time()
print("Push Time = %2.3f" % (pushstop-pushstart))


#[F]:Deploy Kernel for Device Execution
# The following lines associate the arguments to the kernel and deploys
#   it for device execution by calling the method that PyOpenCL generates
#   in program with the built kernel name: backsub. Because back substi
#   tution can only parallelize the subtraction of the elements of U, we
#   loop through the number of columns in main program. Pre-assign the
#   memory flags and kernel variable.
#
mem_flags  = cl.mem_flags
krnl = prg.ConjGradSD


#[G]:Loop Until Convergence
# Calculate the Residuals once to intialize values
iter = 1
RES  = b - np.matmul(A,x)
print("iter = %3d | Max Residual = %2.6f" % (iter,np.max(RES)))
while np.max(RES) > tol:
    iter += 1                               # Advance iteration counter
    AR    = MM.paraMatMult(A,RES)
    RESn  = RES.copy()

    #[H]:Create Buffers
    # Allocate device memory and move input data from the host to the device
    #   memory. Also create a buffer for the "result". Note U is static and
    #   will not change during kernel execution. We denote the memory flag
    #   as READ_ONLY. "result" will be both read and written to by OpenCL
    #   device and therefore uses the READ_WRITE memory flag
    #
    AR_buf   = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=AR)
    RESn_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=RESn)
    x_buf    = cl.Buffer(cxt, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=x)
    RES_buf  = cl.Buffer(cxt, mem_flags.WRITE_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=RES)

    #[I] Set Kernel Arguments
    # For arguments that are numpy arrays enter the created buffers
    krnl.set_args(np.int32(n), RESn_buf, AR_buf, x_buf, RES_buf)

    #[J] Execute Kernel and Copy Result to Global Memory
    # The previously created queue is the first argument followed by
    #   kernel arguments, the shape of the resulting vector. When the
    #   Kernel is finsihed being executed the data stored in the result
    #   buffer will be copied into the result variable
    cl.enqueue_nd_range_kernel(queue, krnl, x.shape, None)

    #pullstart = time.time()
    cl.enqueue_copy(queue,x,x_buf)
    cl.enqueue_copy(queue,RES,RES_buf)
    #pullstop = time.time()
    #print("Pull Time = %2.3f" % (pullstop-pullstart))

    # Output Metrics
    print("iter = %3d | Max Residual = %2.6f" % (iter,np.max(RES)))


# end while

# // Stop Clock //
end = time.time()
totaltime = end-start
print("Executed Time = %2.3f" % totaltime)

#print("result for 1 OpenCL job")
print(x[0:4])
print('success')
