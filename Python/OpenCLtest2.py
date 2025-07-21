import pyopencl as cl
from pyopencl import array
import numpy as np




#/***********************************************************************
#* Setup Problem                                                        *
#***********************************************************************/

# OpenCL is based on the vectorization of the problem
sv = (1,1) # Size of vector
sm = (1,4) # Size of matrix
vector = np.zeros(sv,cl.array.vec.float4) # Initialize vector
matrix = np.zeros(sm,cl.array.vec.float4) # Initialize matrix


# Assign Values
matrix[0,0] = (8, 9, 15, 23)
matrix[0,1] = (4, 5, 6, 2)
matrix[0,2] = (5, 5, 4, 5)
matrix[0,3] = (11, 4, 1, 6)
vector[0,0] = ( 2, 3, 4,5)




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
    __global const float *matrix, __global const float *vector, __global float *result)
    {
        int gid = get_global_id(0);
        printf("gid = %d \\n",gid);
        result[gid] = dot(matrix[gid], vector[0]);
    }
""").build()


#[E]:Create Buffers
# Allocate device memory and move input data from the host to the device
#   memory. Also initialize and create a buffer for the "result"
mem_flags  = cl.mem_flags
matrix_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
vector_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=vector)

matrixdotvector = np.zeros(4,np.float32)
result_buffer   = cl.Buffer(cxt, mem_flags.WRITE_ONLY, matrixdotvector.nbytes)


#[F]:Deploy Kernel for Device Execution
# The following line associates the arguments to the kernel and deploys
#   it for device execution by calling the method that PyOpenCL generates
#   in program with the built kernel name: matrixdotvector. The
#   previously created queue is the first argument followed by all buffers
#
prg.matrixdotvector(queue, matrixdotvector.shape, None, matrix_buf, vector_buf,result_buffer)


#[G]:Move the Kernel's Output to Host Memory
# When the Kernel is finsihed being executed the data stored in the result
#   buffer will be copied into the matrixdotvector variable
cl.enqueue_copy(queue, matrixdotvector, result_buffer)

print(matrixdotvector)




print('success')
