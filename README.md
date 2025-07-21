# OpenCL Playground
This repository contains example projects and scripts for learning and experimenting with OpenCL in C, C++, and Python.

# Structure
C — OpenCL examples in C
CXX — OpenCL examples in C++
Python — OpenCL examples in Python
third_party — External dependencies

# Getting Started
Clone the repository:

```
git clone git@github.com:aevalenc/opencl-playground.git
```

Explore the examples in each language folder.
Make sure you have OpenCL drivers and development libraries installed. Additionally this repositoy utilizes the Bazel build system. Make sure Bazel is installed and you can proceed with the following Usage section


# Usage
```
bazel run path/to:target
```

# Adding third party pip dependencies
Pip dependencies are managed through `rules_python`. Write your dependency in `third_party/pip_deps/requirements.in` (with a specific version if necessary). Then run `bazel run third_party/pip_deps:requirements.update` to automatically update the `requirements_lock.txt` file. 


# Gazelle BUILD file generation
Gazelle has support for creating python BUILD files. However, dependencies must be defined in the `gazelle_python.yaml` file. This file is updated with the following command **after** updating the `requirements_lock.txt` file.

Update the yaml file: 
```
bazel run :gazelle_python_manifest.update
```

Then BUILD files can be maintained by running:

```
bazel run :gazelle_python
```
