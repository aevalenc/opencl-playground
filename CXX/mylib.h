// Alejandro Valencia
// My Library
// Start: 31 October, 2018
// Update: 25 November, 2018

#ifndef mylib

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>

    //#define PI 4*atan(1.0)

    /************************************************************************
    * Linspace Function 													*
    ************************************************************************/
    /*
    !    This function creates an evenly spaced array
    !
    !    The inputs are as follows
    !
    !        A:  Array to be evenly spaced
    !        x0: Start value
    !        xf: Final value
    !        nx: Number of spaces
    !
    */

    int linspace(double A[],double x0, double xf, int nx){

    	/* Declarations */
    	double dx;
    	int i;

    	/* Loop and create array */
    	dx = (xf - x0)/(nx - 1);
    	for (i = 0; i < nx; i++){
    		A[i] = x0 + i*dx;
    	}//end for i

    	return 0;

    }// end function linspace



    /************************************************************************
    * Display Array Function 												*
    ************************************************************************/
    /*
    !   This function displays an array horizontally in the command window
    !   or terminal. This function does NOT print matrices!
    !
    !    the inputs are as follows
    !
    !        A:  Array to be displayed
    !        nx: Size of array
    !
    */

    int disparray(double A[], int nx){

    	/* Declarations */
    	int i;
        int x[9];

    	/* Loop and Print Every Element */
    	for (i = 0; i < nx; i++){
    		printf("%f ",A[i]);
    	}//end for i

    	printf("\n");

        return 0;

    }//end function disparray



    /************************************************************************
    * Data File To Plot Function 											*
    ************************************************************************/
    /*
    !    This function creates a .dat file with two columns, one x, one y to
    !    be plotted by 3rd party program. CANNOT plot 3-D
    !
    !    The inputs are as follows
    !
    !        name: A string that will be the name of the .dat file (name.dat)
    !        x:    Array, independent variable
    !        y:    Array, dependent variable
    !        nx:   Size of both arrays
    !
    */

    int plot2D(char name[], double x[], double y[],int nx){

        /* Declarations */
    	int i;     //index
    	FILE * fp; //pointer to file

        /* Print to file */
    	fp = fopen(name,"w");

    	for (i = 0; i < nx; i++){
    		fprintf(fp,"%lf %lf\n",x[i],y[i]);
    	}//end for i

    	fclose(fp);

        return 0;

    }//END FUNCTION plot2D



    /************************************************************************
    * Data File to Plot Contour Function                                    *
    ************************************************************************/
    /*
    !   This function creates a .dat file with three columns, one x, one y,
    !   and one z to be plotted in 3-D or 2-D contours. CANNOT plot 2D
    !
    !   The inputs are as follows:
    !
    !        name: A string that will be the name of the .dat file (name.dat)
    !        x:    Matrix, 1st independent variable
    !        y:    Matrix, 2nd independent variable
    !        z:    Matrix, dependent variable
    !        nx:   Size of all arrays
    !
    */

    int plot3D(char name[], double x[], double y[], double z[], int ny){

        /* Declarations */
    	int i;     //index
        int j;     //index
    	FILE * fp; //pointer to file

        /* Print to file */
    	fp = fopen(name,"w");

    	for (i = 0; i < ny; i++){
            for (j = 0; j < ny; j++)
    		fprintf(fp,"%lf %lf %lf\n",x[i],y[j],z[j+ny*i]);
    	}//end for i

    	fclose(fp);

        return 0;

    }//END FUNCTION plot3D



    /************************************************************************
    * Display Matrix Function 					     				        *
    ************************************************************************/
    /*
    !    This function displays a matrix in the command window/terminal
    !
    !    The inputs are as follows
    !
    !        A: Matrix to be displayed
    !        m: Number of rows
    !        n: Number of columns
    !
    */

    int DispMatrix(double *A, int m, int n){
    	int i,j;
    	printf("\n");
    		for (i = 0; i < m; i++){
    			for (j = 0; j < n; j++){
    				printf("   %f	",A[j+n*i]);
    			} // end j
    			printf("\n");
    		}// end i
    	printf("\n");

        return 0;

    }//END FUNCTION DispMatrix



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

    int zeros(double x[], int n){

        /* Declarations */
        int i;

        /* Main Algorithm */
        for (i = 0; i < n; i++){
            x[i] = 0;
        }//end for i

        return 0;

    }//END FUNCTION zeros



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

    int ones(double x[], int n){

        /* Declarations */
        int i;

        /* Main Algorithm */
        for (i = 0; i < n; i++){
            x[i] = 1.0;
        }//end for i

        return 0;

    }//END FUNCTION ones




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

    int eyes(double x[], int n){

        /* Declarations */
        int i;

        zeros(x,n*n);
        /* Main Algorithm */
        for (i = 0; i < n; i++){
            x[i + n*i] = 1.0;
        }//end for i

        return 0;

    }//END FUNCTION eyes



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

    int UpperTri(double A[],double b[], int n){

        /* Declarations */
        int k,i,j; //Indicies


        /* Set Identity Matrix */
        double l[n*n];
        eyes(l,n);

        /* Main Algorithm */
        for (k = 0; k < n - 1 ; k++) {
            for (i = k + 1; i < n; i++) {
                l[k + n*i] = A[k + n*i] / A[k + n*k];
                b[i] = b[i] - (l[k + n*i] * b[k]);
                    for (j = k ; j < n; j++) {
                        A[j + n*i] = A[j+n*i] - (l[k+n*i] * A[j+n*k]);
                    }// end for j
            }// end for i
        }// end for k

        return 0;

    }//END FUNCTION UpperTri



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

    int backsub(double A[], double x[], double b[],int n){

        /* Declarations */
        int i,j;


        /* Main Algorithm */
    	for (i = n-1; i > -1; i--){
    		x[i] = b[i];
    		if (i != n-1){
    			for ( j = i + 1; j < n; j++ ){
    				x[i] = x[i] - A[j+n*i] * x[j];
    			}// end for j
    		}// end if
    		x[i] = x[i] / A[i+n*i];
    	}// end for i

        return 0;

    }//END FUNCTION backsub



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

    int forwardsub(double A[], double x[], double b[], int n){

        /* Declarations */
        int     i,j;
        double  temp,sum;


        /* Main Algorithm */
        x[0] = b[0]/A[0];
        for (i = 1; i < n; i++){
            sum  = 0.0;
            for (j = 0; j < i; j++){
                temp = A[j+n*i]*x[j];
                sum  = sum + temp;

            }//end for j

            x[i] = (b[i] - sum)/A[i+n*i];

        }//end for i

        return 0;

    }//END FUNCTION forwardsub



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

    int Doolittle(double A[], double L[], double U[], int n){

        /* Declarations */
        int     k,m,i,j;
        double  tempu, templ, sumu, suml;


        /* Set Identity Matrix */
        eyes(L,n);


        /* Main Algorithm */
        for (k = 0; k < n; k++){

            //Upper Triangular Matrix
            for(m = k; m < n; m++){
                sumu = 0.0;
                for (j = 0; j < k; j++){
                    tempu = L[j + n*k]*U[m + n*j];
                    sumu  = sumu + tempu;
                }//end for j

                U[m + n*k] = A[m + n*k] - sumu;
            }//end for m

            //Lower Triangular Matrix
            // Recall principle diagonal (i,i) are 1s
            for (i = k+1; i < n; i++){
                suml = 0.0;
                for (j = 0; j < k; j++){
                    templ = L[j + n*i]*U[k + n*j];
                    suml  = suml + templ;
                }//end for j

                L[k + n*i] = (A[k + n*i] - suml)/U[k + n*k];
            }//end for i

        }//end for k

        return 0;

    }//END FUNCTION Doolittle



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

    int square(double x){
        int duty,s,nodd;
        double PI = 4*atan(1.0),tmp,w0;
        duty = 50;

        tmp = x - floor(x/(2*PI))*(2*PI);

        // Compute normalized frequency for breaking up the interval (0,2*pi)
        w0 = 2*PI*duty/100;

        // Assign 1 values to normalized t between (0,w0), 0 elsewhere
        if (tmp < w0){
            nodd = 1;
        } else {
            nodd = 0;
        }//end if

        // The actual square wave computation
        s = 2*nodd-1;

        return s;

    }//END FUNCTION square



    /************************************************************************
    * Triangular Wave Function                                              *
    ************************************************************************/
    /*
     !   This function returns the triangle wave
    */
    double triangle(double x){
        double ans;
        double PI = 4*atan(1.0);
        //printf("x = %f\n",x);
        if (x <= PI/2){
            //printf("x == 0 && x <= PI/2\n");
            ans = (2/PI)*x;
        } else if (x > PI/2 && x <= 3*PI/2){
            //printf("x > PI/2 && x <= 3*PI/2\n");
            ans = -(2/PI)*x + 2;
        } else {
            //printf("x > 3*PI/2 && x <= 2*PI\n");
            ans = (2/PI)*x - 4;
        }//end if
        //printf("%f\n",ans);
        return ans;
    }//END FUNCTION triangle



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

    double max(double x[],int n){

        /* Declarations */
        int    i;
        double maxval = x[0];


        /* Main Algorithm */
        for (i = 1; i < n; i++){
            if (x[i] > maxval){
                maxval = x[i];

            }//end fif
        }//end for i

        return maxval;
    }//END FUNCTION max



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

    int matmult(double A[], double B[], double C[],int m, int n, int p){

        /* Declarations */
        int i,j,k;
        double sum;

        /* Main Algorithm */
        for (i = 0; i < m; i++){
            for (j = 0; j < p; j++){
                sum = 0;
                for (k = 0; k < n; k++){
                    sum += A[k+n*i]*B[j+p*k];

                }//end k

            }//end j

            C[(j-1)+p*i] = sum;

        }//end i

        return 0;
    }//END FUNCTION max



#define mylib
#endif
