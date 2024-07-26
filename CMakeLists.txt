cmake_minimum_required(VERSION 3.16)
project(IntegralImage)

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5")
set(CMAKE_CXX_STANDARD 14)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

set(CMAKE_CUDA_STANDARD 12)

add_executable(integralImage
        integralImage_par.cu
        integralImage_seq.cpp
        integralImage_par_efficient.cu
        scan.cu
)

target_link_libraries(integralImage cudart)