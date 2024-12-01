cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
#project(sobel_harness LANGUAGES CXX)

# 19 oct 2023: we are assuming the use of the NVHPC compiler on Perlmutter
# the compilation flags baked in below reflect that assumption
# set up your environment on Perlmutter with:
# module load PrgEnv-nvidia

set(CMAKE_BUILD_TYPE "Release")

add_executable(nbody_cpu nbody_cpu.cpp)
target_compile_options(nbody_cpu PRIVATE -fopenmp -Ofast)
target_link_options(nbody_cpu PRIVATE -fopenmp)
set(BENCHMARK "nbody_cpu")
configure_file(job.in job-${BENCHMARK})

# add_executable(sobel_gpu sobel_gpu.cu)
# generate CUDA code for an actual compute capability 7.0 device (Volta)
# set_property(TARGET sobel_gpu PROPERTY CUDA_ARCHITECTURES 70-real) 
