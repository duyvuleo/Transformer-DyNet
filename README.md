# An Implementation of [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need) in [DyNet](https://github.com/clab/dynet)

### Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), e.g. 3.3.x

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

### Building

First, clone the repository

    git clone https://github.com/duyvuleo/Transformer-DyNet.git

As mentioned above, you'll need the latest [development] version of eigen

    hg clone https://bitbucket.org/eigen/eigen/

A modified version of latest [DyNet](https://github.com/clab/dynet) is already included (e.g., dynet folder).

#### CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH
    make -j 2 

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DMKL=TRUE -DMKL_ROOT=MKL_PATH -DENABLE_BOOST=TRUE

substituting in different paths to EIGEN_PATH and MKL_PATH if you have placed them in different directories. 

This will build the 3 binaries
    
    build_cpu/transformer-train
    build_cpu/transformer-decode

#### GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 7.5.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=EIGEN_PATH -DCUDA_TOOLKIT_ROOT_DIR=CUDA_PATH -DCUDNN_ROOT=CUDA_PATH -DENABLE_BOOST=TRUE
    make -j 2

substituting in your EIGEN_PATH and CUDA_PATH folders, as appropriate.

This will result in the 3 binaries

    build_gpu/transformer-train
    build_gpu/transformer-decode

#### Using the model

(to be updated)

## Contacts

Hoang Cong Duy Vu (vhoang2@student.unimelb.edu.au; duyvuleo@gmail.com)

---
Updated Nov 2017
