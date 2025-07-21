import pyopencl as cl
import numpy as np
import time
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'



#/***********************************************************************
#* Setup Problem                                                        *
#***********************************************************************/


def paraMatMult(A,B):

    m = len(A)
    An = len(A[0,:])
    n  = len(B)
    p  = len(B[0,:])

    if (An != n):
        return "Matrix dimensions do not match"

    else:

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
            __kernel void matrixdotvector(
            int m, int n, int p, __global const float *A, __global const float *B,  __global float *result)
            {
                //[A]:Declarations
                int k;
                float temp,sum;


                //[B]:Get Global ID Values
                int gid  = get_global_id(0); // Row number of result
                int gid2 = get_global_id(1); // Column number of result


                //[C]:Compute result @ Global ID Pair (gid,gid2)
                sum = 0.0;
                for (k = 0; k < n; k++){
                    temp = A[k + n*gid] * B[gid2 + p*k];
                    sum  = sum + temp;
                }

                result[gid2 + p*gid] = sum;



                //matrix[(gid2+k)+m*gid] A[gid2 + m*gid] * B[gid2 + p*gid];

            }//end kernel
        """).build()


        #[E]:Create Buffers
        # Allocate device memory and move input data from the host to the device
        #   memory. Also initialize and create a buffer for the "result"
        mem_flags  = cl.mem_flags
        A_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=B)

        matrixdotvector = np.zeros((m,p),dtype=np.float32)
        result_buffer   = cl.Buffer(cxt, mem_flags.WRITE_ONLY, matrixdotvector.nbytes)


        #[F]:Deploy Kernel for Device Execution
        # The following line associates the arguments to the kernel and deploys
        #   it for device execution by calling the method that PyOpenCL generates
        #   in program with the built kernel name: matrixdotvector. The
        #   previously created queue is the first argument followed by all buffers
        #
        prg.matrixdotvector(queue, matrixdotvector.shape, None,np.int32(m),np.int32(n),np.int32(p), \
                            A_buf, B_buf,result_buffer)


        #[G]:Move the Kernel's Output to Host Memory
        # When the Kernel is finsihed being executed the data stored in the result
        #   buffer will be copied into the matrixdotvector variable
        cl.enqueue_copy(queue, matrixdotvector, result_buffer)

        #print(matrixdotvector)

        end = time.time()
        totaltime = end-start
        #print("%2.3f" % totaltime)

        #print('success')
        return matrixdotvector

    #end if

#end FUNCTION paraMatMult
