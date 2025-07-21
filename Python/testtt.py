import pyopencl as cl
import numpy as np

#this line would create a context
cntxt = cl.create_some_context()
#now create a command queue in the context
queue = cl.CommandQueue(cntxt)


# create some data array to give as input to Kernel and get output
a = np.random.rand(256**3).astype(np.float32)

# create the buffers to hold the values of the input
a_dev = cl.Buffer(cntxt, cl.mem_flags.READ_WRITE, size=a.nbytes)
#cl.enqueue_write_buffer(queue,a_dev,a)



# create output buffer
#out_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, out.nbytes)

# Kernel Program
code = cl.Program(cntxt,"""
__kernel void frst_prog(__global int* num1, __global int* num2,__global int* out)
{
    int i = get_global_id(0);
    out[i] = num1[i]*num1[i]+ num2[i]*num2[i];
}
""").build()


code.twice(queue,a.shape,(1,),a_dev)

result = np.empty_like(a)
cl.enqueue_read_buffer(queue, a_dev, result).wait()

import numpy.linalg as la

assert la.norm(result - 2*a) == 0

# build the Kernel
#bld = cl.Program(cntxt, code).build()
# Kernel is now launched
#launch = bld.frst_prog(queue, num1.shape, num1_buf,num2_buf,out_buf)
# wait till the process completes
#launch.wait()



# print the output
print "Number1:"#, num1
#print "Number2:", num2
#print "Output :", out
