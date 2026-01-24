#include <iostream>
#include <vector>
#include <chrono>
#include <iterator>
#include <cmath>
using namespace std;

// 1-D array
int arr[40];

int main() {

    // ===== FILL ARRAY =====
    for (size_t i = 1; i <= 40; ++i) {
        arr[i] = i * 10;
    }

    cout << arr << endl;

    // ===== SEPERATE LOOPS - MEAN VS VARIANCE =====

    float sep_mean_sum = 0.0f;
    float sep_var_sum = 0.0f;

    // start timer
    auto sep_start = chrono::steady_clock::now();

    // loop for mean in array
    int sep_nums = size(arr);
    for (size_t i = 0; i < sep_nums; ++i) {
        sep_mean_sum += arr[i];
    }
    float sep_mean = sep_mean_sum / sep_nums;  // mean

    // loop for std in array
    for (size_t i = 0; i < sep_nums; ++i) {
        sep_var_sum += pow((arr[i] - sep_mean), 2);
    }
    float sep_var = sep_var_sum / (sep_nums - 1);  // variance

    // end timer
    auto sep_end = chrono::steady_clock::now();

    // elapsed time in seconds
    chrono::duration<double> sep_elapsed = sep_end - sep_start;

    // ===== WELFORD'S ALGO -  MEAN VS VARIANCE =====

    float tg_m = 0.0f;
    float tg_m_new;
    float tg_s = 0.0f;
    int tg_k = 0;
    float tg_var = 0.0f;

    // start timer
    auto tg_start = chrono::steady_clock::now();

    // row-by-row loop through matrix
    int tg_nums = size(arr);
    for (int i = 0; i < tg_nums; ++i) {
        tg_k += 1;
        tg_m_new = tg_m + (arr[i] - tg_m) / tg_k;
        tg_s = tg_s + (arr[i] - tg_m) * (arr[i] - tg_m_new);
        tg_m = tg_m_new;
    }

    float tg_mean = tg_m_new;  // mean
    if (tg_k > 1) {
        tg_var = tg_s / (tg_k - 1);  // variance
    }

    // end timer
    auto tg_end = chrono::steady_clock::now();

    // elapsed time in seconds
    chrono::duration<double> tg_elapsed = tg_end - tg_start;

    cout << "\nTotal Runtime (Seperate): "  << sep_elapsed << " Mean: " << sep_mean << " Sample Variance: " << sep_var << endl;  // print runtime (slower)
    cout << "Total Runtime (Welford): " << tg_elapsed << " Mean: " << tg_mean << " Sample Variance: " << tg_var << endl;  // print runtime (faster)

    // example output
    // Total Runtime (Seperate): 1.333e-06s Mean: 195 Sample Variance: 13666.7 (slower)
    // Total Runtime (Welford): 4.17e-07s Mean: 195 Sample Variance: 13666.7 (faster)

}



