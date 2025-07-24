/*
 * Basic C++ Helper Functions Library
 *
 * Author: Alejandro Valencia
 * Update: July 24, 2025
 */

#ifndef CXX_MYLIB_H
#define CXX_MYLIB_H

#include <cstdint>

// #define PI 4*atan(1.0)

namespace utils
{

/**
 * @brief Creates an evenly spaced array.
 *
 * @param A  Array to be evenly spaced (output).
 * @param x0 Start value.
 * @param xf Final value.
 * @param nx Number of spaces.
 * @return Status code (implementation-defined).
 */
std::int32_t linspace(double A[], double x0, double xf, std::int32_t nx);

/**
 * @brief Displays an array horizontally in the command window or terminal.
 *
 * This function does NOT print matrices.
 *
 * @param A  Array to be displayed.
 * @param nx Size of array.
 * @return Status code (implementation-defined).
 */
std::int32_t disparray(double A[], std::int32_t nx);

/**
 * @brief Creates a .dat file with two columns (x, y) for 2D plotting.
 *
 * The file can be plotted by a 3rd party program. Cannot plot 3-D.
 *
 * @param name Name of the .dat file (output, e.g., name.dat).
 * @param x    Array, independent variable.
 * @param y    Array, dependent variable.
 * @param nx   Size of both arrays.
 * @return Status code (implementation-defined).
 */
std::int32_t plot2D(char name[], double x[], double y[], std::int32_t nx);

/***
 * @brief Creates a .dat file with three columns (x, y, z) for 3D or contour plotting.
 *
 * The file can be plotted in 3-D or 2-D contours. Cannot plot 2D.
 *
 * @param name Name of the .dat file (output, e.g., name.dat).
 * @param x    Matrix, 1st independent variable.
 * @param y    Matrix, 2nd independent variable.
 * @param z    Matrix, dependent variable.
 * @param ny   Size of all arrays.
 * @return Status code (implementation-defined).
 */
std::int32_t plot3D(char name[], double x[], double y[], double z[], std::int32_t ny);

/**
 * @brief Displays a matrix in the command window or terminal.
 *
 * @param A Matrix to be displayed.
 * @param m Number of rows.
 * @param n Number of columns.
 * @return Status code (implementation-defined).
 */
std::int32_t DispMatrix(double* A, std::int32_t m, std::int32_t n);

/************************************************************************
 * Create Array of Zeros (Works for 2-D Matricies)                       *
 ************************************************************************/
/*
 !    This function takes an array and fills its values with zeros
 !
 !    The inputs are as follows
 !
 !        x: Array to be converted into zeros
 !        n: Size of desired array (for a 2-D matrix simply multiply rows
 !            and columns)
 !
*/

std::int32_t zeros(double x[], std::int32_t n);

/************************************************************************
 * Create Array of Ones (Works for 2-D Matricies)                       *
 ************************************************************************/
/*
 !    This function takes an array and fills its values with ones
 !
 !    The inputs are as follows
 !
 !        x: Array to be converted into zeros
 !        n: Size of desired array (for a 2-D matrix simply multiply rows
 !            and columns)
 !
*/

std::int32_t ones(double x[], std::int32_t n);

/************************************************************************
 * Create Identity Matrix                                                *
 ************************************************************************/
/*
 !    This function creates the identity matrix. Recall the identity matrix
 !    is a square matrix with 1s in the principle diagonal
 !
 !    The inputs are as follows
 !
 !        x: Matrix that will become the identity matrix
 !        n: The number of columns/rows
 !
*/

std::int32_t eyes(double x[], std::int32_t n);

/************************************************************************
 * Function To Turn Matrix Into Upper Triangular                         *
 ************************************************************************/
/*
 !    This function performs an algorithm that turns a SQUARE matrix into
 !    an upper triangular one
 !
 !    This function requires the following as inputs
 !
 !        A: The matrix that will be trurned into an upper triangular one
 !        b: The right hand side of the matrix equation
 !        n: Number of columns
 !
*/

std::int32_t UpperTri(double A[], double b[], std::int32_t n);

/************************************************************************
 * Backwards Substitution                                                *
 ************************************************************************/
/*
 !    This function performs a backwards substitution algorithm to solve
 !    the matrix equation Ax = b, where A is an UPPER triangular matrix.
 !
 !    This function requires the following inputs
 !
 !        A: The upper triangular matrix (square n x n)
 !        x: The array where the results will be placed (column n x 1)
 !        b: The right hand side of the matrix equation (column n x 1)
 !        n: The number of columns/rows of A
 !
*/

std::int32_t backsub(double A[], double x[], double b[], std::int32_t n);

/************************************************************************
 * Forward Substitution                                                  *
 ************************************************************************/
/*
 !    This function performs a forward substitution algorithm to solve
 !    the matrix equation Ax = b, where A is an LOWER triangular matrix.
 !
 !    NOTE:This function works for a dominant LOWER triangular matrix
 !       where elements in diagonal (a_ii) is not 0
 !
 !    This function requires the following inputs
 !
 !        A: The lower triangular matrix (square n x n)
 !        x: The array where the results will be placed (column n x 1)
 !        b: The right hand side of the matrix equation (column n x 1)
 !        n: The number of columns/rows of A
 !
*/

std::int32_t forwardsub(double A[], double x[], double b[], std::int32_t n);

/************************************************************************
 * Doolittle LU Decomposition                                            *
 ************************************************************************/
/*
 !   This function performs an LU Decomposition on a matrix A based on the
 !   Doolittle algorithm.
 !
 !   NOTE: L MUST be initialized as the identity matrix. Doolittle's
 !       algorithm is based on the lower triangular matrix having the
 !       values in the principle diagonal equal to 1.
 !
 !   The inputs are as follows
 !
 !       A: Square matrix to be decomposed
 !       L: Lower diagonal matrix
 !       U: Upper diagonal matrix
 !       n: Number of rows/columns
 !
*/

std::int32_t Doolittle(double A[], double L[], double U[], std::int32_t n);

/************************************************************************
 * Square Wave Function                                                  *
 ************************************************************************/
/*
 !   This function outputs a square wave. This is based on underlining
 !   fact that the square function can be thought of as the sign of the
 !   sine funciton; i.e. The square function is 1 when the sine function
 !   is positive and -1 when the sine function is -1
 !
 !   The folowing are inputs
 !
 !       x: value
 !
*/

std::int32_t square(double x);

/************************************************************************
 * Triangular Wave Function                                              *
 ************************************************************************/
/*
 !   This function returns the triangle wave
*/
double triangle(double x);

/************************************************************************
 * Max of Array Function                                                 *
 ************************************************************************/
/*
 ! This function find the max value of a number array
 !
 !   The inputs are as follows
 !
 !       x:  Number Array
 !       n:  Size of Array
 !
*/

double max(double x[], std::int32_t n);

/************************************************************************
 * Matrix Multiplication Function                                        *
 ************************************************************************/
/*
 ! This function finds the product of two Matrices
 !
 !   The inputs are as follows
 !
 !       A:  Matrix 1 with size m x n
 !       B:  Matrix 2 with size n x p
 !       C:  Resultant with size m x p
 !
*/

std::int32_t matmult(double A[], double B[], double C[], std::int32_t m, std::int32_t n, std::int32_t p);

}  // namespace utils

#endif  // CXX_MYLIB_H
