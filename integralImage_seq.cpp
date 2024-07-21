#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

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

// Funzione per calcolare la matrice integrale
std::vector<std::vector<int>> computeIntegralMatrix(const std::vector<std::vector<int>>& matrix) {
    int n = matrix.size();
    if (n == 0) return {};  // Se la matrice è vuota, restituisci una matrice vuota
    int m = matrix[0].size();
    if (m == 0) return {};  // Se la matrice ha righe vuote, restituisci una matrice vuota

    std::vector<std::vector<int>> integralMatrix(n, std::vector<int>(m, 0));

    // Calcola il valore per ogni elemento della matrice integrale
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int current = matrix[i][j];
            int left = (j > 0) ? integralMatrix[i][j - 1] : 0;
            int up = (i > 0) ? integralMatrix[i - 1][j] : 0;
            int upLeft = (i > 0 && j > 0) ? integralMatrix[i - 1][j - 1] : 0;

            integralMatrix[i][j] = current + left + up - upLeft;
        }
    }

    return integralMatrix;
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

    // Misura il tempo di esecuzione
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> integralMatrix = computeIntegralMatrix(matrix);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Integral Matrix:" << std::endl;
    //printMatrix(integralMatrix);

    std::cout << "Time taken to compute the integral matrix: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
