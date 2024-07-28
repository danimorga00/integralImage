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

__global__ void upsweep(int *data, int n, int m, int twod, int twod1) {
    int row = blockIdx.x;
    int index = threadIdx.x;
    int idx = row * m + index * twod1 + twod1 - 1;
    if (idx < row * m + m && idx >= row * m) {
        data[idx] += data[idx - twod];
    }
}

__global__ void downsweep(int *data, int n, int m, int twod, int twod1) {
    int row = blockIdx.x;
    int index = threadIdx.x;
    int idx = row * m + index * twod1 + twod1 - 1;
    if (idx < row * m + m && idx >= row * m) {
        int t = data[idx - twod];
        data[idx - twod] = data[idx];
        data[idx] += t;
    }
}

void scanMatrix(int *data, int n, int m) {
    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * m * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, data, n * m * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;

    // Upsweep phase
    for (int twod = 1; twod < m; twod *= 2) {
        int twod1 = twod * 2;
        int numBlocks = n; // Ogni riga è un blocco
        upsweep<<<numBlocks, blockSize>>>(d_data, n, m, twod, twod1);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Set the last element of each row to 0
    for (int i = 0; i < n; i++) {
        CHECK_CUDA(cudaMemset(&d_data[i * m + m - 1], 0, sizeof(int)));
    }

    // Downsweep phase
    for (int twod = m / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        int numBlocks = n; // Ogni riga è un blocco
        downsweep<<<numBlocks, blockSize>>>(d_data, n, m, twod, twod1);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(data, d_data, n * m * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
}

__global__ void transpose(int *input, int *output, int n, int m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < m) {
        output[y * n + x] = input[x * m + y];
    }
}
void transposeMatrix(int *data, int *result, int n, int m) {
    int *d_data, *d_result;
    CHECK_CUDA(cudaMalloc(&d_data, n * m * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result, n * m * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, data, n * m * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    transpose<<<gridSize, blockSize>>>(d_data, d_result, n, m);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(result, d_result, n * m * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));
}

int scan_sequential(int *data, int n) {
    int out=0;
    for(int i =0; i<n; i++) {
        out += data[i];
    }
    return out;
}

int main(void) {
    int print = 1;
    int n = 128;  // Numero di righe
    int m = 128;  // Numero di colonne
    int *data = new int[n * m];
    int *result = new int[n * m];

    // Inizializza i dati con valori casuali
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            data[i * m + j] = 1;
        }
    }
    printf("Input:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if(print == 1) printf("%d ", data[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    scanMatrix(data, n, n);
    transposeMatrix(data, result, n, n);
    scanMatrix(result, n, n);

    printf("Output:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if(print == 1) printf("%d ", result[i * m + j]);
        }
        printf("\n");
    }

    delete[] data;
    return 0;
}
