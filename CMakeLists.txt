cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
#project(sobel_harness LANGUAGES CXX)

# 19 oct 2023: we are assuming the use of the NVHPC compiler on Perlmutter
# the compilation flags baked in below reflect that assumption
# set up your environment on Perlmutter with:
# module load PrgEnv-nvidia

set(CMAKE_BUILD_TYPE "Release")

find_package(LIKWID QUIET)

# use cmake --log-level=debug to see DEBUG messages
if ( LIKWID_FOUND ) 
   message ( STATUS "Likwid: found")
else()
   message( STATUS  "Likwid: NOT found")
endif()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(nbody_cpu nbody_cpu.cpp)
if ( LIKWID_FOUND )
   target_include_directories(nbody_cpu PRIVATE ${LIKWID_INCLUDE_DIR})
   target_compile_definitions(nbody_cpu PRIVATE -DLIKWID_PERFMON)
   target_link_libraries(nbody_cpu PRIVATE ${LIKWID_LIBRARIES})
endif ()
target_compile_options(nbody_cpu PRIVATE -fopenmp -Ofast)
target_link_options(nbody_cpu PRIVATE -fopenmp)
set(BENCHMARK "nbody_cpu")
configure_file(job.in job-${BENCHMARK})

# add_executable(sobel_gpu sobel_gpu.cu)
# generate CUDA code for an actual compute capability 7.0 device (Volta)
# set_property(TARGET sobel_gpu PROPERTY CUDA_ARCHITECTURES 70-real)