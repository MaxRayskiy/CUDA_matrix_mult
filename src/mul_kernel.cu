#include "../headers/mul_kernel.h"

#include <cuda_runtime.h>
#include <iostream>

__global__
void MatrixMulDefault(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C) {
    for (size_t i = 0; i < C->height_; i += gridDim.y * blockDim.y) {
        for (size_t j = 0; j < C->width_; j += gridDim.x * blockDim.x) {

            size_t row = (i + blockIdx.x) * blockDim.x + threadIdx.x;
            size_t col = (j + blockIdx.y) * blockDim.y + threadIdx.y;

            if (row >= C->height_ || col >= C->width_)
                return;

            C->elements_[row * C->width_ + col] = static_cast<T>(0);

            for (int k = 0; k < A->width_; ++k) {
                C->elements_[row * C->width_ + col] += A->elements_[row * A->width_ + k] * B->elements_[k * C->width_ + col];
            }
        }
    }
}

__global__
void MatrixMulDefaultTransposed(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C) {
    for (size_t i = 0; i < C->height_; i += gridDim.y * blockDim.y) {
        for (size_t j = 0; j < C->width_; j += gridDim.x * blockDim.x) {

            size_t row = (i + blockIdx.y) * blockDim.y + threadIdx.y;
            size_t col = (j + blockIdx.x) * blockDim.x + threadIdx.x;

            if (row >= C->height_ || col >= C->width_)
                return;

            T loc_res = static_cast<T>(0);

            for (int k = 0; k < A->width_; ++k) {
                loc_res += A->elements_[row * A->width_ + k] * B->elements_[k * C->width_ + col];
            }
            C->elements_[row * C->width_ + col] = loc_res;
        }
    }
}

__global__
void MatrixMulSharedTransposed(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C) {
    size_t stride = blockDim.x;
    size_t mid_size = A->width_;
    __shared__ T shared_data_A[BLOCKSIZE * BLOCKSIZE];
    __shared__ T shared_data_B[BLOCKSIZE * BLOCKSIZE];

    size_t tidx = threadIdx.x;
    size_t tidy = threadIdx.y;
    size_t index = tidy * blockDim.x + tidx;

    for (size_t i = 0; i < C->height_; i += gridDim.y * blockDim.y) {
        for (size_t j = 0; j < C->width_; j += gridDim.x * blockDim.x) {

            size_t row = (i + blockIdx.y) * blockDim.y + threadIdx.y;
            size_t col = (j + blockIdx.x) * blockDim.x + threadIdx.x;

            if (row >= C->height_ || col >= C->width_)
                return;

            T loc_res = static_cast<T>(0);

            for (size_t factor = 0; factor < mid_size / stride; ++factor) {
                shared_data_A[index] = A->elements_[(row) * A->width_ + (factor * stride + tidx)];
                shared_data_B[index] = B->elements_[(factor * stride + tidy) * B->width_ + col];
                __syncthreads();
                for (size_t k = 0; k < stride; ++k) {
                    loc_res += shared_data_A[blockDim.x * tidy + k] * shared_data_B[k * blockDim.x + tidx];
                }
                __syncthreads();
            }
            C->elements_[row * C->width_ + col] = loc_res;
        }
    }
}

size_t GetGridSize(int dim) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return deviceProp.maxGridSize[dim] ;
}

void CudaDeepCopy(Matrix<T>* &src, Matrix<T>* &trg, T* &trg_elems) {
    cudaMalloc(&trg, sizeof(Matrix<T>));
    size_t size = src->width_ * src->height_ * sizeof(T);
    cudaMalloc(&trg_elems, size);
    // Copy up each piece separately
    cudaMemcpy(trg, src, sizeof(Matrix<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(trg_elems, src->elements_, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(trg->elements_), &trg_elems, sizeof(T*), cudaMemcpyHostToDevice);
}

void CudaDestruct(Matrix<T>* trg, T* trg_elems) {
    cudaFree(trg_elems);
    cudaFree(trg);
}

void MultMatrixNative(Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, bool print_time) {
    Matrix<T>* d_A = nullptr;
    T* A_elements = nullptr;
    CudaDeepCopy(A, d_A, A_elements);

    Matrix<T>* d_B = nullptr;
    T* B_elements = nullptr;
    CudaDeepCopy(B, d_B, B_elements);

    Matrix<T>* d_C = nullptr;
    T* C_elements = nullptr;
    CudaDeepCopy(C, d_C, C_elements);

    dim3 dim_block(BLOCKSIZE, BLOCKSIZE);
    size_t grid_x = std::min(GetGridSize(0), C->height_) / BLOCKSIZE;
    size_t grid_y = std::min(GetGridSize(1), C->width_)  / BLOCKSIZE;
    dim3 dim_grid(grid_x, grid_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulDefault<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
    float milliseconds = 0;
    cudaEventRecord(stop);
    if (cudaMemcpy(C->elements_, C_elements, C->height_ * C->width_ * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "device2host memcpy failed" << std::endl;
    }
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    if (print_time)
        std::cout << milliseconds << "ms - native" << std::endl;

    CudaDestruct(d_A, A_elements);
    CudaDestruct(d_B, B_elements);
    CudaDestruct(d_C, C_elements);
}

void MultMatrixModNative(Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, bool print_time) {
    Matrix<T>* d_A = nullptr;
    T* A_elements = nullptr;
    CudaDeepCopy(A, d_A, A_elements);

    Matrix<T>* d_B = nullptr;
    T* B_elements = nullptr;
    CudaDeepCopy(B, d_B, B_elements);

    Matrix<T>* d_C = nullptr;
    T* C_elements = nullptr;
    CudaDeepCopy(C, d_C, C_elements);

    dim3 dim_block(BLOCKSIZE, BLOCKSIZE);
    size_t grid_x = std::min(GetGridSize(0), C->width_)  / BLOCKSIZE;
    size_t grid_y = std::min(GetGridSize(1), C->height_) / BLOCKSIZE;
    dim3 dim_grid(grid_x, grid_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulDefaultTransposed<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
    float milliseconds = 0;
    cudaEventRecord(stop);
    if (cudaMemcpy(C->elements_, C_elements, C->height_ * C->width_ * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "device2host memcpy failed" << std::endl;
    }
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    if (print_time)
        std::cout << milliseconds << "ms - modified native" << std::endl;

    CudaDestruct(d_A, A_elements);
    CudaDestruct(d_B, B_elements);
    CudaDestruct(d_C, C_elements);
}

void MultMatrixSharedMem(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, bool print_time) {
    Matrix<T>* d_A = nullptr;
    T* A_elements = nullptr;
    CudaDeepCopy(A, d_A, A_elements);

    Matrix<T>* d_B = nullptr;
    T* B_elements = nullptr;
    CudaDeepCopy(B, d_B, B_elements);

    Matrix<T>* d_C = nullptr;
    T* C_elements = nullptr;
    CudaDeepCopy(C, d_C, C_elements);

    dim3 dim_block(BLOCKSIZE, BLOCKSIZE);
    size_t grid_x = std::min(GetGridSize(0), C->width_)  / BLOCKSIZE;
    size_t grid_y = std::min(GetGridSize(1), C->height_) / BLOCKSIZE;
    dim3 dim_grid(grid_x, grid_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MatrixMulSharedTransposed<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
    float milliseconds = 0;
    cudaEventRecord(stop);
    if (cudaMemcpy(C->elements_, C_elements, C->height_ * C->width_ * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "device2host memcpy failed" << std::endl;
    }
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    if (print_time)
        std::cout << milliseconds << "ms - with shared memory" << std::endl;

    CudaDestruct(d_A, A_elements);
    CudaDestruct(d_B, B_elements);
    CudaDestruct(d_C, C_elements);
}
