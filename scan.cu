#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#define CHECK_CUDA(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    }

__global__ void upsweep(int *data, int n, int twod, int twod1) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = index * twod1 + twod1 - 1;
    if (idx < n) {
        data[idx] += data[idx - twod];
    }
}

__global__ void downsweep(int *data, int n, int twod, int twod1) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = index * twod1 + twod1 - 1;
    if (idx < n) {
        int t = data[idx - twod];
        data[idx - twod] = data[idx];
        data[idx] += t;
    }
}

void scan(int *data, int n) {
    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;

    // Upsweep phase
    for (int twod = 1; twod < n; twod *= 2) {
        int twod1 = twod * 2;
        int numBlocks = (n + blockSize * twod1 - 1) / (blockSize * twod1);
        upsweep<<<numBlocks, blockSize>>>(d_data, n, twod, twod1);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Set the last element to 0
    CHECK_CUDA(cudaMemset(&d_data[n - 1], 0, sizeof(int)));

    // Downsweep phase
    for (int twod = n / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int numBlocks = (n + blockSize * twod1 - 1) / (blockSize * twod1);
        downsweep<<<numBlocks, blockSize>>>(d_data, n, twod, twod1);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
}

int main(void) {
    int n = 16; // Cambia questo valore per testare array di dimensioni maggiori
    int *data = new int[n];

    // Inizializza i dati
    for (int i = 0; i < n; i++) {
        data[i] = 1; // Puoi cambiare i valori come preferisci
    }

    printf("Input: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    scan(data, n);

    printf("Output: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    delete[] data;
    return 0;
}
