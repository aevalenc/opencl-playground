__kernel void cl_jacobi(int ny,
                        const __global double* A,
                        const __global double* b,
                        const __global double* xn,
                        __global double* x)
{
    //[A]:Get Global ID
    int gid = get_global_id(0);
    // printf("gid = %d\n",gid);

    //[B]:Calculate sum
    int k;
    int j = gid;

    double sum = 0;
    for (k = 0; k < ny; k++)
    {
        if (k != gid)
        {
            // printf("xn[%d] = %f\n",k,xn[k]);
            // printf("A[%d+ny*%d] = %f\n",k,gid,A[k+ny*gid]);
            sum += A[k + ny * gid] * xn[k];
        } /*end if*/
    } /*end k*/

    // printf("sum = %f\n",sum);
    //[C]:Calculate next iteration
    x[gid] = (b[gid] - sum) / A[gid + ny * gid];
    // printf("x[%d] = %f\n",gid,x);
}
