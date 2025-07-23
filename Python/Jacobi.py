# Alejandro Valencia
# OpenCL Projects:Jacobi Iteration
# Start: 17 September, 2019
# Update: 17 September, 2019

# /***********************************************************************
# * This code utilizes the OpenCL Standard to parallelize the Jacobi     *
# *  iterative method to solve linear systems in the form Ax = b         *
# ***********************************************************************/

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
from numpy.linalg import norm

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def setup_problem(n, bcl, bcr, tol):
    """
    Setup the problem with given parameters.
    """
    # construct Matrix A
    A = np.zeros((n, n)).astype(np.float32)
    band = [-1, 2, -1]
    for i in range(1, n - 1):
        A[i, i - 1 : i + 2] = band
    A[0, 0:2] = [2, -1]
    A[-1, -2:n] = [-1, 2]

    # Right hand side
    b = np.zeros(n).astype(np.float32)
    b[0] = bcl
    b[-1] = bcr

    return A, b


def jacobi_cpu(A, b, x0, x, tolerance):

    iter = 1
    RES = np.abs(b - np.matmul(A, x0))

    # Loop until Convergence
    while np.max(RES) > tolerance:
        iter += 1
        # x = x0.copy()
        for i in range(0, len(x)):
            sum = 0
            for j in range(0, len(x)):
                if j != i:
                    sum += A[i, j] * x[j]
                # end if
            # end j
            x[i] = (b[i] - sum) / A[i, i]
        # end i

        RES = norm(b - np.matmul(A, x))
        x0 = x.copy()


def setup_opencl():
    # [A]:Retrieve Platforms
    platforms = cl.get_platforms()
    platform_extension = platforms[0].extensions
    platform_name = platforms[0].name

    # [B]:Retrieve Devices
    # On this system platform 0 is the OpenCL Intel Compute Runtime. At the
    #   current momment run OpenCL only on CPU
    cpu_devices = platforms[0].get_devices(cl.device_type.CPU)  # Retrieve CPU
    gpu_devices = platforms[0].get_devices(cl.device_type.GPU)  # Retrieve GPU if no CPU
    if len(cpu_devices) == 0 and len(gpu_devices) == 0:
        raise RuntimeError("No OpenCL devices found. Please check your OpenCL installation.")

    if len(cpu_devices) > 0:
        print("Using CPU devices for OpenCL execution.")
        device = cpu_devices[0]  # Use the first CPU device
        device_name = cpu_devices[0].name  # Device name
        extensions = cpu_devices[0].extensions  # Supported extensions
    else:
        print("Using GPU devices for OpenCL execution.")
        device = gpu_devices[0]  # Use the first GPU device
        device_name = gpu_devices[0].name  # Device name
        extensions = gpu_devices[0].extensions  # Supported extensions

    print("Using device:", device_name)

    # [C]:Create Context and Queue
    cxt = cl.Context([device])
    queue = cl.CommandQueue(cxt)

    # [D]:Create Kernel to be Executed
    prg = cl.Program(
        cxt,
        """
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
    """,
    ).build()

    # [F]:Deploy Kernel for Device Execution
    # The following lines associate the arguments to the kernel and deploys
    #   it for device execution by calling the method that PyOpenCL generates
    #   in program with the built kernel name: backsub. Because back substi
    #   tution can only parallelize the subtraction of the elements of U, we
    #   loop through the number of columns in main program. Pre-assign the
    #   memory flags and kernel variable.
    #
    mem_flags = cl.mem_flags
    krnl = prg.jacobi

    return cxt, queue, krnl, mem_flags


def jacobi_gpu(A, b, x0, x, tolerance, cxt, queue, krnl, mem_flags):
    """
    This function implements the Jacobi method on a GPU using OpenCL.
    It initializes the problem and iteratively computes the solution.
    """
    iter = 1
    RES = np.abs(b - np.matmul(A, x0))

    # Loop until Convergence
    while np.max(RES) > tolerance:
        iter += 1
        xn = x.copy()

        # Create Buffers
        A_buf = cl.Buffer(cxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(cxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        xn_buf = cl.Buffer(cxt, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=xn)
        x_buf = cl.Buffer(cxt, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)

        # Set Kernel Arguments
        krnl.set_args(np.int32(len(x)), A_buf, xn_buf, b_buf, x_buf)

        # Execute Kernel
        cl.enqueue_nd_range_kernel(queue, krnl, x.shape, None)
        cl.enqueue_copy(queue, x, x_buf)

        RES = np.abs(b - np.matmul(A, x))


def main():
    n = 101  # Number of nodes
    bcl = 200  # left boundary condition
    bcr = 400  # right boundary condition
    tolerance = 0.001  # tolerance for iterative method

    A, b = setup_problem(n, bcl, bcr, tolerance)

    x0 = np.zeros(n).astype(np.float32)  # initial guess of solution
    x = x0.copy()  # solution vector

    # Standard Jacobi Algorithm
    start = time.time()
    jacobi_cpu(A, b, x0, x, tolerance)
    stop = time.time()
    print("Standard Jacobi Algorithm compute time: %fs" % (stop - start))

    # Re-initialize the problem for GPU execution
    x = np.zeros(n).astype(np.float32)  # initial guess of solution
    iter = 1
    RES = np.abs(b - np.matmul(A, x))

    # Setup OpenCL
    cxt, queue, krnl, mem_flags = setup_opencl()
    # Jacobi Algorithm on GPU
    print("Running Jacobi Algorithm on GPU...")
    start = time.time()
    jacobi_gpu(A, b, x0, x, tolerance, cxt, queue, krnl, mem_flags)
    stop = time.time()
    print("Jacobi Algorithm on GPU compute time: %fs" % (stop - start))

    # []:Plot
    xvector = np.linspace(0, 1, n)
    plt.plot(xvector, x)
    plt.show()


if __name__ == "__main__":
    main()
