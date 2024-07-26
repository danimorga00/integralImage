#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Kernel per la scansione parallela (prefix sum)
__global__ void prefixSumKernel(int* d_data, int* d_sum, int n) {
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int offset = 1;

    temp[2 * thid] = d_data[2 * thid];
    temp[2 * thid + 1] = d_data[2 * thid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) {
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    d_sum[2 * thid] = temp[2 * thid];
    d_sum[2 * thid + 1] = temp[2 * thid + 1];
}

int main() {
    int n = 8;  // Dimensione dell'array
    std::vector<int> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = rand() % 10;
    }

    std::cout << "Original Data: ";
    for (int i = 0; i < n; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    int* d_data;
    int* d_sum;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_sum, n * sizeof(int));
    cudaMemcpy(d_data, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    prefixSumKernel<<<1, n / 2, n * sizeof(int)>>>(d_data, d_sum, n);

    std::vector<int> sum(n);
    cudaMemcpy(sum.data(), d_sum, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Prefix Sum: ";
    for (int i = 0; i < n; ++i) {
        std::cout << sum[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_data);
    cudaFree(d_sum);

    return 0;
}

