
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"

void init_matrix(half *A, int lda, int m, int n);
void print_matrix(const half *A, const int lda, const int m, const int n, bool colMajor);
void cp_half_to_single(float* A, const half* B, int len);