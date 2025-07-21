# Alejandro Valencia
# OpenCL Projects: test_bench
# Start: March 1, 2019
# Update: May 24, 2019

#/***********************************************************************
#* This code serves as a test bench for OpenCL projects 				*
#***********************************************************************/


import pyopencl as cl
import numpy as np
import os
import sys
import time
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


#/***********************************************************************
#* Main Program 														*
#***********************************************************************/

platform_id = int(sys.argv[1])

#[A]:Get Platform and CPU Device
platforms = cl.get_platforms()
platform = platforms[platform_id]
device   = platform.get_devices()[0]

#[B]:Create a context and Queue
# We create a queue in the context
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

#[C]:Create Arrays to be Added
a = np.linspace(0,4,5).astype(np.float32)
b = np.linspace(0,4,5).astype(np.float32)

#[D]:Create Memory Buffers
mf  = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

#[E]:Create Kernel
prg = cl.Program(ctx, """
__kernel void sum(
	__global const float *a_g, __global const float *b_g, __global float *res_g)
{
	//int i;
	int gid = get_global_id(0);
	int lid = get_local_id(0);

	//printf("gid = %d | lid = %d\\n", gid, lid);

	//int gid2 = get_global_id(1);
	//for (i = 0; i < 2; i++){
		//printf("gid = %d | gid2 = %d | b_g(%d,%d) = %.1f\\n", gid, gid2, gid, gid2, b_g[gid + 2*i]);
	//}
	res_g[gid] = a_g[gid] * b_g[gid];
}
""").build()

#[F]:Write the Result
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
res = np.empty_like(a)


start = time.time()
prg.sum(queue, a.shape, None, a_g, b_g, res_g)
cl.enqueue_copy(queue, res, res_g)
end   = time.time()

#[G]:Output Result
print(res[0:5])
print("Time = %f" %(end-start))







print("success")
