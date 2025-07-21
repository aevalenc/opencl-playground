# Alejandro Valencia
# OpenCL Projects: Conjugate Gradient Method
# Start: 19 April, 2019
# Update: 20 April, 2019

#/***********************************************************************
#* This program utiltizes OpenCL to parallelize the matrix multiplicat- *
#*  ion algorithm by turning the matrix A in a row dominated one        *                                               *
#***********************************************************************/

import pyopencl as cl
import numpy as np
import time
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#/***********************************************************************
#* Main Function                                                        *
#***********************************************************************/

def paraMatMultRD(A,B):

    m = len(A)
    An = len(A[0,:])
    n  = len(B)
    p  = len(B[0,:])

    if (An != n):
        return "Matrix dimensions do not match"

    else:

        A = A.reshape(n*n,1)
        b = np.zeros((n*n,1)).astype(np.float32)
        for i in range(0,n):
            b[n*i:n*i+n] = B
        #end i

        # Start CLock
        start = time.time()

        #/***********************************************************************
        #* Parallelization Via OpenCL                                           *
        #***********************************************************************/

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
            __kernel void array_mult(
            __global const float *A, __global const float *b,  __global float *result)
            {
                //[A]:Get Global ID Values
                int gid  = get_global_id(0); // Row number of result


                //[B]:Compute result @ Global ID Pair (gid,gid2)
                result[gid] = A[gid]*b[gid];

            }//end kernel
        """).build()


        #[E]:Create Buffers
        # Allocate device memory and move input data from the host to the device
        #   memory. Also initialize and create a buffer for the "result"
        mem_flags  = cl.mem_flags
        A_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)

        result = np.zeros((n*n,1),dtype=np.float32)
        result_buffer   = cl.Buffer(cxt, mem_flags.WRITE_ONLY, result.nbytes)


        #[F]:Deploy Kernel for Device Execution
        # The following line associates the arguments to the kernel and deploys
        #   it for device execution by calling the method that PyOpenCL generates
        #   in program with the built kernel name: matrixdotvector. The
        #   previously created queue is the first argument followed by all buffers
        #
        prg.array_mult(queue, result.shape, None, \
                            A_buf, b_buf,result_buffer)


        #[G]:Move the Kernel's Output to Host Memory
        # When the Kernel is finsihed being executed the data stored in the result
        #   buffer will be copied into the matrixdotvector variable
        cl.enqueue_copy(queue, result, result_buffer)

        c = np.zeros((n,p)).astype(np.float32)
        for i in range(0,n):
            c[i] = np.sum(result[n*i:i*n+n])
        #end i

        end = time.time()
        totaltime = end-start
        #print("%2.3f" % totaltime)

        #print('success')
        return c

    #end if

#end FUNCTION paraMatMult
