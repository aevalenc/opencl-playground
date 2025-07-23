// Alejandro Valencia
// OpenCL-Projects: opencl test C++
// Start: 25 May, 2019
// Update: 25 May, 2019

/************************************************************************
 * This code serves as a test bench for OpenCL Projects in C++ 			*
 ************************************************************************/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

/************************************************************************
 * Function Declarations *
 ************************************************************************/

int print_platforms(int, std::vector<cl::Platform>);

// end Function Declarations

/************************************************************************
 * Main Program
 **
 ************************************************************************/

int main()
{
    // printf("OpenCL C++ library found!\n");
    // std::cout<<"C++ OpenCL ver 2 check" << std::endl;

    cl_int err;

    // [A]:Problem Setup
    int a[4] = {1, 2, 3, 4};
    int b[4] = {1, 2, 3, 4};
    int c[4];

    // [B]:Create Platform Object
    // Filter for a 2.0 platform and set it as the default
    std::vector<cl::Platform> platforms;  // Declare "platforms" as std type vector cl type Platform
    cl::Platform::get(&platforms);        // Using scope cd::Platform get all platforms and place in variable platforms

    int num_platforms = platforms.size();  // Use platform.size() to get number of platforms
    std::cout << "Number of Platforms: " << num_platforms << std::endl;  // Print number of platforms

    // [B.1]:Print Platform Atributions
    print_platforms(num_platforms, platforms);

    // [C]:Get Devices for Said Platforms
    // NOTE: During debugging, platform[0], name "Clover", was found to have no devices.
    // 	Therefore, automatically set platform to platform[1]

    auto platform = platforms[0];

    std::vector<cl::Device> devices;                          // Declare "devices" as a std type vector cl type Device
    err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);  // Get ALL devices associated with Intel CPU Runtime
    std::cout << "Device on platform error = " << err << std::endl;

    // Print Number of devices on Platform 1 - Intel CPU Runtime
    std::cout << "Number of devices on " << platform.getInfo<CL_PLATFORM_NAME>() << " is " << devices.size() << "\n"
              << std::endl;

    // Get device
    auto device = devices[0];
    auto device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Device Name: " << device_name << "\n" << std::endl;

    // [D]:Create Context and Queue for Device
    cl::Context context(device);
    cl::CommandQueue queue(context, device, 0, &err);
    std::cout << "Creating queue error = " << err << std::endl;
    err = queue.getInfo<CL_QUEUE_PROPERTIES>();
    std::cout << "Queue properties error = " << err << std::endl;

    // [E]:Create Memory Buffers
    cl::Buffer a_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 4, a);
    cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 4, b);
    cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 4, c);

    // [F]:Program and Kernel
    // First write the kernel to be executed then build the program before
    // 	setting the kernel
    cl::Program program(context,
                        "__kernel void cl_add_arrays("
                        "const __global int *a, const __global int *b, __global int *c)"
                        "{"
                        "int gid = get_global_id(0);"
                        "c[gid] = a[gid] + b[gid];"
                        "}");
    program.build("-cl-std=CL2.0");
    //	program.build(device);

    cl::Kernel kernel(program, "cl_add_arrays");

    // [G]:Set Kernel Arguments
    kernel.setArg(0, a_buf);
    kernel.setArg(1, b_buf);
    kernel.setArg(2, c_buf);

    // [H]:Enqueue Kernel
    cl::NDRange global(4);
    cl::NDRange local(1);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    std::cout << "Execute kernel error number = " << err << std::endl;

    err = queue.enqueueReadBuffer(c_buf, CL_TRUE, 0, sizeof(int) * 4, c);
    std::cout << "Reading result buffer error = " << err << std::endl;

    // cl::finish();

    std::cout << "Code executed successfully!" << std::endl;

    int i;
    for (i = 0; i < 4; i++)
    {
        std::cout << c[i] << std::endl;
    }

    return 0;
}

/************************************************************************
 * Print All Platforms Functions 										*
 ************************************************************************/

int print_platforms(int num_platforms, std::vector<cl::Platform> platforms)
{

    int i;
    for (i = 0; i < num_platforms; i++)
    {

        auto platform_profile = platforms[i].getInfo<CL_PLATFORM_PROFILE>();
        auto platform_name = platforms[i].getInfo<CL_PLATFORM_NAME>();
        auto platform_vendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
        auto platform_version = platforms[i].getInfo<CL_PLATFORM_VERSION>();
        auto platform_extensions = platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>();

        std::cout << "\nPlatform Profile: " << platform_profile << std::endl;
        std::cout << "Platform Name: " << platform_name << std::endl;
        std::cout << "Platform Vendor: " << platform_vendor << std::endl;
        std::cout << "Platform Version " << platform_version << std::endl;
        std::cout << "Platform Extensions: " << platform_extensions << "\n" << std::endl;

    }  // end i

    return 0;

}  // end FUNCTION print_platforms
