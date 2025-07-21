# Description:
#  OpenCL(TM) API Headers
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "opencl_headers",
    srcs = [
        "CL/cl.h",
        "CL/cl_d3d10.h",
        "CL/cl_d3d11.h",
        "CL/cl_dx9_media_sharing.h",
        "CL/cl_dx9_media_sharing_intel.h",
        "CL/cl_egl.h",
        "CL/cl_ext.h",
        "CL/cl_ext_intel.h",
        "CL/cl_function_types.h",
        "CL/cl_gl.h",
        "CL/cl_gl_ext.h",
        "CL/cl_half.h",
        "CL/cl_icd.h",
        "CL/cl_layer.h",
        "CL/cl_platform.h",
        "CL/cl_va_api_media_sharing_intel.h",
        "CL/cl_version.h",
        "CL/opencl.h",
    ],
    includes = ["."],
)
