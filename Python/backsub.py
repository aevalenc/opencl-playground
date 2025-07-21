import pyopencl as cl
import numpy as np
import numpy.linalg as la
import time
import os

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


# /***********************************************************************
# * Setup Problem                                                        *
# ***********************************************************************/

# OpenCL is based on the vectorization of the problem
# Assign Values
U = np.array([(3, 2, 3), (0, 4, 5), (0, 0, 6)]).astype(np.float32)
y = np.array([7, 8, 9]).astype(np.float32).reshape([3, 1])
result = y.copy().astype(np.float32)
n = len(U)

# // Start Clock //
start = time.time()


# /***********************************************************************
# * Parallelization Via OpenCL                                           *
# ***********************************************************************/

# [A]:Retrieve Platforms
platforms = cl.get_platforms()
platform_extension = platforms[0].extensions
platform_name = platforms[0].name


# [B]:Retrieve Devices
# On this system platform 0 is the OpenCL Intel Compute Runtime. At the
#   current momment run OpenCL only on CPU
device = platforms[0].get_devices(cl.device_type.CPU)  # Retrieve CPU
if len(device) == 0:
    device = platforms[0].get_devices(cl.device_type.GPU)  # Retrieve GPU if no CPU
if len(device) == 0:
    raise RuntimeError("No OpenCL devices found. Please check your OpenCL installation.")
extensions = device[0].extensions  # Supported extensions
device_name = device[0].name  # Device name


# [C]:Create Context and Queue
cxt = cl.Context([device[0]])
queue = cl.CommandQueue(cxt)


# [D]:Create Kernel to be Executed
prg = cl.Program(
    cxt,
    """
    __kernel void backsub(
    int n, int j, __global const float *U, __global float *result)
    {
        //[A]:Get Global ID Values
        int gid  = get_global_id(0); // Row number of result

        //[B]:Compute result @ Global ID (gid)
        result[gid] = result[gid] - U[j + n*gid]*result[j];

    }//end kernel
""",
).build()


# [E]:Deploy Kernel for Device Execution
# The following lines associate the arguments to the kernel and deploys
#   it for device execution by calling the method that PyOpenCL generates
#   in program with the built kernel name: backsub. Because back substi
#   tution can only parallelize the subtraction of the elements of U, we
#   loop through the number of columns in main program. Pre-assign the
#   memory flags and kernel variable.
#
mem_flags = cl.mem_flags
krnl = prg.backsub

for j in range(n - 1, n - 4, -1):
    # Calculate answer and set U @ j,j to 0
    result[j] = result[j] / U[j, j]
    U[j, j] = 0

    # [F]:Create Buffers
    # Allocate device memory and move input data from the host to the device
    #   memory. Also create a buffer for the "result". Note U is static and
    #   will not change during kernel execution. We denote the memory flag
    #   as READ_ONLY. "result" will be both read and written to by OpenCL
    #   device and therefore uses the READ_WRITE memory flag
    #
    U_buf = cl.Buffer(cxt, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=U)
    result_buffer = cl.Buffer(cxt, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf=result)

    # [G] Set Kernel Arguments
    # For arguments that are numpy arrays enter the created buffers
    krnl.set_args(np.int32(n), np.int32(j), U_buf, result_buffer)

    # [H] Execute Kernel and Copy Result to Global Memory
    # The previously created queue is the first argument followed by
    #   kernel arguments, the shape of the resulting vector. When the
    #   Kernel is finsihed being executed the data stored in the result
    #   buffer will be copied into the result variable
    cl.enqueue_nd_range_kernel(queue, krnl, result.shape, None)
    cl.enqueue_copy(queue, result, result_buffer)

# end j


# // Stop Clock //
end = time.time()
totaltime = end - start
print("Executed Time = %2.3f" % totaltime)

# print("result for 1 OpenCL job")
print(result)
print("success")
