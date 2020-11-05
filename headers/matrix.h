#pragma once

#include <chrono>
#include <random>

template <typename T>
struct Matrix {
    size_t width_;
    size_t height_;
    T* elements_;

    Matrix() = delete;

    Matrix(size_t width, size_t height)
            : width_(width)
            , height_(height)
    {
        elements_ = new T[width * height];
    }

    // fill with random values from -100 to 100
    void InitRandom() {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 rand_gen(seed);
        for (size_t i = 0; i < width_ * height_; ++i) {
            elements_[i] = static_cast <T> (rand_gen()) / static_cast <double> (std::mt19937::max() / 100.0);
        }
    }

    ~Matrix() {
        delete[] elements_;
    }
};
