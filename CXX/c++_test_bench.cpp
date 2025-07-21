#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <stdio.h>
#include <CL/cl2.hpp>
#include <iostream>
#include "mylib.h"

int main(){
	int m = 2;
	int n = 2;
	int p = 1;

	string lo = "(dddprint("dd")dddd)"
	/*
	std::cout<<"Enter a: "<<std::endl;
	std::cin>>a;
	std::cout<<"a = "<<a<<std::endl;
	*/

	double A[m*n] = {1,0,0,2};
	double B[n*p] = {0,2};
	double C[2]   = {0,0};

	matmult(A,B,C,m,n,p);

	for (m = 0; m < 2; m++){
		std::cout<<C[m]<<std::endl;
	}


	return 0;

}//end program
