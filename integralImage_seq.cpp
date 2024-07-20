//
// Created by danie on 20/07/2024.
//
#include <iostream>
#include <vector>

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

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::cout << "Original Matrix:" << std::endl;
    printMatrix(matrix);

    std::vector<std::vector<int>> integralMatrix = computeIntegralMatrix(matrix);

    std::cout << "Integral Matrix:" << std::endl;
    printMatrix(integralMatrix);

    return 0;
}
