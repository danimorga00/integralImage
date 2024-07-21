#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

__global__ void computeIntegralKernel(int* d_matrix, int* d_integralMatrix, int n, int m) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m) {
        int current = d_matrix[i * m + j];
        int left = (j > 0) ? d_integralMatrix[i * m + (j - 1)] : 0;
        int up = (i > 0) ? d_integralMatrix[(i - 1) * m + j] : 0;
        int upLeft = (i > 0 && j > 0) ? d_integralMatrix[(i - 1) * m + (j - 1)] : 0;

        d_integralMatrix[i * m + j] = current + left + up - upLeft;
    }
}

void computeIntegralMatrixCUDA(const std::vector<std::vector<int>>& matrix, std::vector<std::vector<int>>& integralMatrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    // Allocate memory on the device
    int* d_matrix;
    int* d_integralMatrix;
    cudaMalloc(&d_matrix, n * m * sizeof(int));
    cudaMalloc(&d_integralMatrix, n * m * sizeof(int));

    // Copy matrix to device
    std::vector<int> flat_matrix;
    for (const auto& row : matrix) {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
    cudaMemcpy(d_matrix, flat_matrix.data(), n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_integralMatrix, flat_matrix.data(), n * m * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    computeIntegralKernel<<<gridSize, blockSize>>>(d_matrix, d_integralMatrix, n, m);
    cudaDeviceSynchronize();

    // Copy result back to host
    std::vector<int> flat_integralMatrix(n * m);
    cudaMemcpy(flat_integralMatrix.data(), d_integralMatrix, n * m * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert flat vector to 2D vector
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            integralMatrix[i][j] = flat_integralMatrix[i * m + j];
        }
    }

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_integralMatrix);
}

// Funzione per generare una matrice di numeri casuali
std::vector<std::vector<int>> generateRandomMatrix(int n, int m, int maxValue) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = rand() % (maxValue + 1);  // Numeri casuali tra 0 e maxValue
        }
    }
    return matrix;
}

// Funzione per stampare una matrice
void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));  // Inizializza il generatore di numeri casuali

    int n = 50000;  // Dimensioni della matrice
    int m = 5000;
    int maxValue = 10;  // Valore massimo dei numeri casuali

    // Genera la matrice casuale
    std::vector<std::vector<int>> matrix = generateRandomMatrix(n, m, maxValue);

    std::cout << "Original Matrix:" << std::endl;
    //printMatrix(matrix);

    // Allocazione matrice integrale
    std::vector<std::vector<int>> integralMatrix(n, std::vector<int>(m, 0));

    // Misura il tempo di esecuzione
    auto start = std::chrono::high_resolution_clock::now();
    computeIntegralMatrixCUDA(matrix, integralMatrix);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Integral Matrix:" << std::endl;
    //printMatrix(integralMatrix);

    std::cout << "Time taken to compute the integral matrix using CUDA: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
