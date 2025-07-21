"""
Module extension: opencl_headers
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _opencl_headers_impl(mctx):
    http_archive(
        name = "opencl_headers",
        # sha256 = "d4c3f8b1c5e0f2a6b7e8c9d1f2e3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1",
        strip_prefix = "OpenCL-Headers-2024.10.24",
        urls = ["https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.10.24.zip"],
        build_file = "//third_party/opencl_headers:opencl.BUILD",
    )

opencl_headers = module_extension(implementation = _opencl_headers_impl)
