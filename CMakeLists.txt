cmake_minimum_required(VERSION 3.18)
project(cudaProva4 LANGUAGES C CXX CUDA)

# Specifica il percorso di installazione di CUDA
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5")

# Aggiungi i percorsi delle librerie e degli include di CUDA
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# Aggiungi il file sorgente CUDA
set(SOURCES main.cu
        integralImage_seq.cpp)

# Specifica la versione di CUDA
set(CMAKE_CUDA_STANDARD 11)

# Aggiungi l'eseguibile
add_executable(cudaProva4 ${SOURCES}
        integralImage_seq.cpp)

# Linka le librerie CUDA
target_link_libraries(cudaProva4 cudart)

