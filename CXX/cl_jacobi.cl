__kernel void cl_jacobi(int ny,
                        const __global double* A,
                        const __global double* b,
                        const __global double* x0,
                        __global double* x)
{
    //[A]:Get Global ID
    int gid = get_global_id(0);

    double sum = 0.0;
    for (int k = 0; k < ny; k++)
    {
        if (k != gid)
        {
            sum += A[k + ny * gid] * x0[k];
        }
    }

    x[gid] = (b[gid] - sum) / A[gid + ny * gid];
}
