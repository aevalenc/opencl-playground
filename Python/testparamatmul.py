import MatrixMultiplyRowDom as MMRD
import numpy as np

A = np.array([(1,2,3),(4, 5, 6),(7, 8, 9)]).astype(np.float32)
b = np.array([10,11,12]).reshape((3,1)).astype(np.float32)

C = MMRD.paraMatMultRD(A,b)
print(C)
