#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

// 2-D vector of ints
const int N = 10000;
vector<vector<int>> matrix(N, vector<int>(N, 1));

int main() {

    // ===== ROW BY ROW =====

    int row_sum = 0;

    // start timer
    auto row_start = chrono::steady_clock::now();

    // row-by-row loop through matrix
    int rows1 = matrix.size();
    int cols1 = matrix[0].size();
    for (int i = 0; i < rows1; ++i) {  // rows
        for (int j = 0; j < cols1; ++j) {  // cols
            row_sum += matrix[i][j];  // print element
        }
    }

    // end timer
    auto row_end = chrono::steady_clock::now();

    // elapsed time in seconds
    chrono::duration<double> row_elapsed = row_end - row_start;

    // ===== COLS BY COLS =====

    int cols_sum = 0;

    // start timer
    auto cols_start = chrono::steady_clock::now();

    // row-by-row loop through matrix
    int rows2 = matrix.size();
    int cols2 = matrix[0].size();
    for (int j = 0; j < cols2; ++j) {  // rows
        for (int i = 0; i < rows2; ++i) {  // cols
            cols_sum += matrix[i][j];  // print element
        }
    }

    // end timer
    auto cols_end = chrono::steady_clock::now();

    // elapsed time in seconds
    chrono::duration<double> cols_elapsed = cols_end - cols_start;

    cout << "\nTotal Runtime (Row-By-Row): "  << row_elapsed << " Sum: " << row_sum << endl;  // print runtime (slower)
    cout << "Total Runtime (Cols-By-Cols): " << cols_elapsed << " Sum: " << cols_sum << endl;  // print runtime (faster)

    // example output
    // Total Runtime (Row-By-Row): 0.165011s Sum: 100000000
    // Total Runtime (Cols-By-Cols): 0.626733s Sum: 100000000

}



