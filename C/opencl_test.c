/*
 * Simple OpenCL Platform Detection Program
 *
 * Author: Alejandro Valencia
 * Date: July 21, 2025
 */

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/opencl.h>
#include <stdio.h>

int main()
{
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_int platform = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    printf("Number of Platforms = %d\n", ret_num_platforms);

    int i, j;
    char* info;
    size_t infoSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    const char* attributeNames[5] = {"Name", "Vendor", "Version", "Profile", "Extensions"};
    const cl_platform_info attributeTypes[5] = {
        CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS};
    const int attributeCount = sizeof(attributeNames) / sizeof(char*);

    // get platform count
    clGetPlatformIDs(5, NULL, &platformCount);

    // get all platforms
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    // for each platform print all attributes
    for (i = 0; i < platformCount; i++)
    {

        printf("\n %d. Platform \n", i + 1);

        for (j = 0; j < attributeCount; j++)
        {

            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*)malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

            printf("  %d.%d %-11s: %s\n", i + 1, j + 1, attributeNames[j], info);
            free(info);
        }

        printf("\n");
    }

    free(platforms);

    return 0;
}
