import numpy as np
import time


#[A]:Matrix Dimensions
m = 400
n = 400
p = 200

#[B]: Create Matrix of Random Elements
A = np.random.rand(m,n).astype(np.float32)
B = np.random.rand(n,p).astype(np.float32)
C = np.zeros((m,p)).astype(np.float32)


#// Start Clock //
start = time.time()


#[C]:Multiply Matrices
for i in range(0,m):
    for j in range(0,p):
        sum = 0.0
        for k in range(0,n):
            temp = A[i,k]*B[k,j]
            sum = sum + temp
        #end k
    #end j
    C[i,j] = sum
#end i

#// Stop Clock //
end = time.time()

print(end-start)



#print("A: \n",A)
#print("B: \n",B)
#print("C: \n",C)
