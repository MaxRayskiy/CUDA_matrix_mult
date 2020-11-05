#include "../headers/test.h"

#include <iostream>
#include <cassert>
#include <cmath>


T CompareMatrixes(Matrix<T>* A, Matrix<T>* B) {
    T err = 0;
    assert(A->width_ == B->width_ && A->height_ == B->height_);
    for (int i = 0; i < A->height_ ; ++i){
        for (int j = 0; j < A->width_; ++j){
            if (USE_EPSILON) {
                if (std::abs(A->elements_[i * A->width_ + j] - B->elements_[i * B->width_ + j]) > EPSILON)
                    err += 1;
            } else {
                err += A->elements_[i * A->width_+ j] - B->elements_[i * B->width_ + j];
            }
        }
    }
    return err;
}


bool RunTest(size_t width, size_t mid_size, size_t height, bool print_time, bool is_debug) {
    std::cout  << width << "*" << height << " " << mid_size << std::endl;

    auto h_A = new Matrix<T>(mid_size, height);
    h_A->InitRandom();
    auto* h_B = new Matrix<T>(width, mid_size);
    h_B->InitRandom();

    auto* h_C = new Matrix<T>(width, height);
    auto* d_C = new Matrix<T>(width, height);

    MultMatrixNative(h_A, h_B, d_C, print_time);
    MultMatrixModNative(h_A, h_B, h_C, print_time);
    MultMatrixSharedMem(h_A, h_B, h_C, print_time);

    if (is_debug) {
        std::cout << std::endl;
        std::cout << "A = ";
        for (size_t i = 0; i < h_A->height_ * h_A->width_; ++i) {
            std::cout << h_A->elements_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "B = ";
        for (size_t i = 0; i < h_B->height_ * h_B->width_; ++i) {
            std::cout << h_B->elements_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "C1 = ";
        for (size_t i  = 0; i < h_C->height_ * h_C->width_; ++i) {
            std::cout << h_C->elements_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "C2 = ";
        for (size_t i  = 0; i < d_C->height_ * d_C->width_; ++i) {
            std::cout << d_C->elements_[i] << " ";
        }
        std::cout << std::endl;
    }

    T err = CompareMatrixes(h_C, d_C);
    if (err > EPSILON) {
        std::cout << " Status: FAILED!" << std::endl;
        std::cerr << "Error: " << err << " on " << width << "x" << mid_size << " " << height << std::endl;
        return false;
    }
    std::cout << std::endl;
    return true;
}
