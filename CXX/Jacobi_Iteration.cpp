// Alejandro Valencia
// OpenCL C++ Projects: Jacobi Method
// Start: 22 June, 2019
// Update: 22 June, 2019

/************************************************************************
 * This code parallelizes the Jacobi Method of solving systems in the  	*
 * 	form Ax = b via OpenCL in C++ 										*
 ************************************************************************/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define MAX_SOURCE_SIZE (0x100000)

#include "CXX/utils.h"
#include <CL/opencl.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace
{

std::int32_t print_platforms(std::int32_t num_platforms, std::vector<cl::Platform> platforms)
{

    std::int32_t i;
    for (i = 0; i < num_platforms; i++)
    {

        auto platform_profile = platforms[i].getInfo<CL_PLATFORM_PROFILE>();
        auto platform_name = platforms[i].getInfo<CL_PLATFORM_NAME>();
        auto platform_vendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
        auto platform_version = platforms[i].getInfo<CL_PLATFORM_VERSION>();
        auto platform_extensions = platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>();

        std::cout << "\nPlatform Profile	: " << platform_profile << std::endl;
        std::cout << "Platform Name		: " << platform_name << std::endl;
        std::cout << "Platform Vendor		: " << platform_vendor << std::endl;
        std::cout << "Platform Version	: " << platform_version << std::endl;
        std::cout << "Platform Extensions	: " << platform_extensions << "\n" << std::endl;
    }

    return 0;
}

}  // namespace

/************************************************************************
 * Main Program
 **
 ************************************************************************/

std::int32_t main()
{
    // printf("OpenCL C++ library found!\n");
    // std::cout<<"C++ OpenCL ver 2 check" << std::endl;

    cl_int err;

    // [A]:Problem Setup
    const std::int32_t nx = 9;
    const std::int32_t ny = 3;
    const std::int32_t maxiter = 200;
    std::array<double, nx> A = {2, -1, 0, -1, 2, -1, 0, -1, 2};  // Left Hand Side
    std::array<double, ny> b = {200, 0, 400};                    // Right Hand Side
    std::array<double, ny> x0 = {1, 1, 1};                       // Initial Guess
    auto x = x0;                                                 // Previous Iteration
    std::array<double, ny> residual{};                           // Residuals
    auto tmp = x0;                                               // Temporary Vector

    // std::array<double, ny> xn, RES, tmp;                         // Solution Vector
    const double tolerance = 0.001;

    // [B]:Create Platform Object
    // Filter for a 2.0 platform and set it as the default
    std::vector<cl::Platform> platforms;  // Declare "platforms" as std type vector cl type Platform
    cl::Platform::get(&platforms);        // Using scope cd::Platform get all platforms and place in variable platforms

    std::int32_t num_platforms = platforms.size();  // Use platform.size() to get number of platforms
    std::cout << "Number of Platforms: " << num_platforms << std::endl;  // Print number of platforms

    // [B.1]:Print Platform Atributions
    print_platforms(num_platforms, platforms);

    // [C]:Get Devices for Said Platforms
    // NOTE: During debugging, platform[0] is the "Intel CPU Compute Runtime",
    // 	while platform[1] is named "Portable Computing Language"

    const auto platform = platforms[1];

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
    cl::Buffer A_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * nx, A.data());
    cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * ny, b.data());

    // [F]:Program and Kernel
    // First write the kernel to be executed then build the program before
    // 	setting the kernel
    std::ifstream kernel_file("CXX/cl_jacobi.cl");
    if (!kernel_file)
    {
        std::cerr << "Failed to load kernel." << std::endl;
        exit(1);
    }
    std::string source_str((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());

    cl::Program program(context, source_str);

    try
    {
        program.build("-cl-std=CL2.0");
    }
    catch (...)
    {
        // Print build info for all devices
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto& pair : buildInfo)
        {
            std::cerr << pair.second << std::endl << std::endl;
        }

        return 1;
    }

    const auto build_info = program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &source_str);
    std::cout << "Program build Info: " << build_info << std::endl;

    try
    {
        cl::Kernel kernel(program, "cl_jacobi");

        // [G]:Iterate
        std::int32_t iter = 1;
        for (std::int32_t i = 0; i < ny; i++)
        {
            utils::matmult(A.data(), x0.data(), tmp.data(), ny, ny, 1);
            residual[i] = fabs(b[i] - tmp[i]);
        }

        printf("iter = %d | Max Residual = %f\n", iter, utils::max(residual.data(), ny));

        while (utils::max(residual.data(), ny) > tolerance)
        {
            std::copy(x.begin(), x.end(), x0.begin());

            // [H]:Advance Iteration Counter
            iter += 1;

            cl::Buffer x0_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * ny, x0.data());
            cl::Buffer x_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * ny, x.data());

            // [I]:Set Kernel Arguments
            kernel.setArg(0, ny);
            kernel.setArg(1, A_buf);
            kernel.setArg(2, b_buf);
            kernel.setArg(3, x0_buf);
            kernel.setArg(4, x_buf);

            // [J]:Enqueue Kernel
            cl::NDRange global(ny);
            cl::NDRange local(1);
            err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
            err = queue.enqueueReadBuffer(x_buf, CL_TRUE, 0, sizeof(double) * ny, x.data());

            for (std::int32_t i = 0; i < ny; i++)
            {
                utils::matmult(A.data(), x.data(), tmp.data(), ny, ny, 1);
                residual[i] = fabs(b[i] - tmp[i]);
            }

            const auto max_residual = utils::max(residual.data(), ny);
            printf("iter = %d | Max Residual = %f\n", iter, max_residual);

            if (iter == maxiter)
            {
                std::cout << "Maximum iterations reached: " << maxiter << std::endl;
                break;
            }
            if (max_residual <= tolerance)
            {
                std::cout << "Convergence achieved with tolerance: " << tolerance << std::endl;
                break;
            }

        }  // end while

        std::cout << "Code executed successfully!" << std::endl;

        // Display Result
        for (std::int32_t i = 0; i < ny; i++)
        {
            std::cout << x[i] << std::endl;
        }
    }

    catch (const cl::Error& e)
    {
        std::cerr << "Kernel creation failed: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    }

}  // END program
