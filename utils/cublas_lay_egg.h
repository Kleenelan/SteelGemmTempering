#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"

void verify(int M, int N, half* A, int lda, float* B, int ldb, double rerror);
void verify_blas(int M, int N, int K,
                 const half* Ah, int lda,
                 const half* Bh, int ldb,
                 half* Ch, int ldc,
                 const half alpha,
                 const half beta,
                 float* Dh);