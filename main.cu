#include <iostream>
#include <cuda_runtime.h>

// Kernel CUDA per sommare due array
__global__ void add(int *a, int *b, int *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int N = 1000;
    int size = N * sizeof(int);

    // Alloca memoria sul host
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // Inizializza gli array
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Alloca memoria sul device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copia i dati dal host al device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Lancia il kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copia i risultati dal device al host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verifica i risultati
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Errore alla posizione " << i << std::endl;
            break;
        }
    }

    // Libera la memoria
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Calcolo completato correttamente!" << std::endl;

    return 0;
}
