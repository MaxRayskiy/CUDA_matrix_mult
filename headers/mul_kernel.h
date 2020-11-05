#pragma once

#include "matrix.h"
#include "settings.h"

void MultMatrixNative(Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, bool print_time);

void MultMatrixModNative(Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, bool print_time);

void MultMatrixSharedMem(Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, bool print_time);
