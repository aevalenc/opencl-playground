/*
 * Basic C++ Helper Functions Library
 *
 * Author: Alejandro Valencia
 * Update: July 24, 2025
 */

#include "CXX/utils.h"
#include <cmath>
#include <cstdio>

namespace utils
{

std::int32_t linspace(double A[], double x0, double xf, std::int32_t nx)
{
    /* Declarations */
    double dx;
    std::int32_t i;

    /* Loop and create array */
    dx = (xf - x0) / (nx - 1);
    for (i = 0; i < nx; i++)
    {
        A[i] = x0 + i * dx;
    }

    return 0;
}

std::int32_t disparray(double A[], std::int32_t nx)
{

    /* Declarations */
    std::int32_t i;

    /* Loop and Print Every Element */
    for (i = 0; i < nx; i++)
    {
        printf("%f ", A[i]);
    }

    printf("\n");

    return 0;
}

std::int32_t plot2D(char name[], double x[], double y[], std::int32_t nx)
{

    /* Declarations */
    std::int32_t i;  // index
    FILE* fp;        // pointer to file

    /* Print to file */
    fp = fopen(name, "w");

    for (i = 0; i < nx; i++)
    {
        fprintf(fp, "%lf %lf\n", x[i], y[i]);
    }

    fclose(fp);

    return 0;
}

std::int32_t plot3D(char name[], double x[], double y[], double z[], std::int32_t ny)
{

    /* Declarations */
    std::int32_t i;  // index
    std::int32_t j;  // index
    FILE* fp;        // pointer to file

    /* Print to file */
    fp = fopen(name, "w");

    for (i = 0; i < ny; i++)
    {
        for (j = 0; j < ny; j++)
            fprintf(fp, "%lf %lf %lf\n", x[i], y[j], z[j + ny * i]);
    }

    fclose(fp);

    return 0;
}

std::int32_t DispMatrix(double* A, std::int32_t m, std::int32_t n)
{
    std::int32_t i, j;
    printf("\n");
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("   %f	", A[j + n * i]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}

std::int32_t zeros(double x[], std::int32_t n)
{

    /* Declarations */
    std::int32_t i;

    /* Main Algorithm */
    for (i = 0; i < n; i++)
    {
        x[i] = 0;
    }

    return 0;
}

std::int32_t ones(double x[], std::int32_t n)
{

    /* Declarations */
    std::int32_t i;

    /* Main Algorithm */
    for (i = 0; i < n; i++)
    {
        x[i] = 1.0;
    }

    return 0;
}

std::int32_t eyes(double x[], std::int32_t n)
{

    /* Declarations */
    std::int32_t i;

    zeros(x, n * n);
    /* Main Algorithm */
    for (i = 0; i < n; i++)
    {
        x[i + n * i] = 1.0;
    }

    return 0;
}

std::int32_t UpperTri(double A[], double b[], std::int32_t n)
{

    /* Declarations */
    std::int32_t k, i, j;  // Indicies

    /* Set Identity Matrix */
    double l[n * n];
    eyes(l, n);

    /* Main Algorithm */
    for (k = 0; k < n - 1; k++)
    {
        for (i = k + 1; i < n; i++)
        {
            l[k + n * i] = A[k + n * i] / A[k + n * k];
            b[i] = b[i] - (l[k + n * i] * b[k]);
            for (j = k; j < n; j++)
            {
                A[j + n * i] = A[j + n * i] - (l[k + n * i] * A[j + n * k]);
            }
        }
    }

    return 0;
}

std::int32_t backsub(double A[], double x[], double b[], std::int32_t n)
{

    /* Declarations */
    std::int32_t i, j;

    /* Main Algorithm */
    for (i = n - 1; i > -1; i--)
    {
        x[i] = b[i];
        if (i != n - 1)
        {
            for (j = i + 1; j < n; j++)
            {
                x[i] = x[i] - A[j + n * i] * x[j];
            }
        }
        x[i] = x[i] / A[i + n * i];
    }

    return 0;
}

std::int32_t forwardsub(double A[], double x[], double b[], std::int32_t n)
{

    /* Declarations */
    std::int32_t i, j;
    double temp, sum;

    /* Main Algorithm */
    x[0] = b[0] / A[0];
    for (i = 1; i < n; i++)
    {
        sum = 0.0;
        for (j = 0; j < i; j++)
        {
            temp = A[j + n * i] * x[j];
            sum = sum + temp;
        }

        x[i] = (b[i] - sum) / A[i + n * i];
    }

    return 0;
}

std::int32_t Doolittle(double A[], double L[], double U[], std::int32_t n)
{

    /* Declarations */
    std::int32_t k, m, i, j;
    double tempu, templ, sumu, suml;

    /* Set Identity Matrix */
    eyes(L, n);

    /* Main Algorithm */
    for (k = 0; k < n; k++)
    {

        // Upper Triangular Matrix
        for (m = k; m < n; m++)
        {
            sumu = 0.0;
            for (j = 0; j < k; j++)
            {
                tempu = L[j + n * k] * U[m + n * j];
                sumu = sumu + tempu;
            }

            U[m + n * k] = A[m + n * k] - sumu;
        }

        // Lower Triangular Matrix
        //  Recall principle diagonal (i,i) are 1s
        for (i = k + 1; i < n; i++)
        {
            suml = 0.0;
            for (j = 0; j < k; j++)
            {
                templ = L[j + n * i] * U[k + n * j];
                suml = suml + templ;
            }

            L[k + n * i] = (A[k + n * i] - suml) / U[k + n * k];
        }
    }

    return 0;
}

std::int32_t square(double x)
{
    std::int32_t duty, s, nodd;
    double PI = 4 * atan(1.0), tmp, w0;
    duty = 50;

    tmp = x - floor(x / (2 * PI)) * (2 * PI);

    // Compute normalized frequency for breaking up the interval (0,2*pi)
    w0 = 2 * PI * duty / 100;

    // Assign 1 values to normalized t between (0,w0), 0 elsewhere
    if (tmp < w0)
    {
        nodd = 1;
    }
    else
    {
        nodd = 0;
    }

    // The actual square wave computation
    s = 2 * nodd - 1;

    return s;
}

double triangle(double x)
{
    double ans;
    double PI = 4 * atan(1.0);
    // printf("x = %f\n",x);
    if (x <= PI / 2)
    {
        // printf("x == 0 && x <= PI/2\n");
        ans = (2 / PI) * x;
    }
    else if (x > PI / 2 && x <= 3 * PI / 2)
    {
        // printf("x > PI/2 && x <= 3*PI/2\n");
        ans = -(2 / PI) * x + 2;
    }
    else
    {
        // printf("x > 3*PI/2 && x <= 2*PI\n");
        ans = (2 / PI) * x - 4;
    }
    // printf("%f\n",ans);
    return ans;
}

double max(double x[], std::int32_t n)
{

    /* Declarations */
    std::int32_t i;
    double maxval = x[0];

    /* Main Algorithm */
    for (i = 1; i < n; i++)
    {
        if (x[i] > maxval)
        {
            maxval = x[i];
        }
    }

    return maxval;
}

std::int32_t matmult(double A[],
                     double B[],
                     double C[],
                     const std::int32_t M,
                     const std::int32_t N,
                     const std::int32_t P)
{

    /* Declarations */
    double sum{0.0};

    /* Main Algorithm */
    for (std::int32_t i = 0; i < M; ++i)
    {
        // Initialize sum
        for (std::int32_t j = 0; j < P; ++j)
        {
            sum = 0.0;
            for (std::int32_t k = 0; k < N; ++k)
            {
                sum += A[k + N * i] * B[j + P * k];
            }
            C[j + P * i] = sum;
        }
    }

    return 0;
}
}  // namespace utils
