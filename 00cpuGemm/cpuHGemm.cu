
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>



#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <omp.h>

#include "cublas_lay_egg.h"
#include "matrix_init_print.h"

void gemm_fp16_cpu(int M, int N, int K,
                   half* A, int lda,
                   half* B, int ldb,
                   half* C, int ldc,
                   half alpha, half beta)
{
    #pragma omp parallel for
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            half sigma = half(0.0f);
            //#pragma omp parallel for
            for(int k=0; k<K; k++)
            {
                sigma += A[i*lda + k] * B[k + j*ldb];
            }
            C[i + j*ldc] = alpha*sigma + beta*C[i + j*ldc];
        }
    }
}

int main()
{
#if 1
    int M = 128;
    int N = 128;
    int K = 128;
#else
    int M = 7;      // \forall M > 0;
    int N = 7;      // \forall N > 0;
    int K = 4*16;   // \forall K > 0;
#endif
    int lda = K;// A raw major
    int ldb = K;// B col major
    int ldc = M;// C col major

	half *A_h = nullptr;
    half *B_h = nullptr;
    half *C_h = nullptr;
    half *D_h_cpu = nullptr;
    half *D_h_cublas = nullptr;

    half alpha = half(1.0);
    half beta  = half(0.0);

    A_h = (half*)malloc(M * lda * sizeof(half));
    B_h = (half*)malloc(ldb * N * sizeof(half));
    C_h = (half*)malloc(ldc * N * sizeof(half));
    D_h_cpu = (half*)malloc(ldc * N * sizeof(half));
    D_h_cublas = (half*)malloc(ldc * N * sizeof(half));

    init_matrix(A_h, lda, M, K, false);
    init_matrix(B_h, ldb, K, N, true);
    init_matrix(C_h, ldc, M, N, true);

    memcpy(D_h_cpu, C_h, ldc * N * sizeof(half));
    memcpy(D_h_cublas, C_h, ldc * N * sizeof(half));

#if 0
    printf("A_h =");
    print_matrix(A_h, lda, M, K, false);
    printf("B_h =");
    print_matrix(B_h, ldb, K, N, true);
    printf("C_h =");
    print_matrix(C_h, ldc, M, N, true);
#endif

    gemm_fp16_cpu(M, N, K, A_h, lda, B_h, ldb, D_h_cpu, ldc, alpha, beta);// Arow Bcol Crow Major;
    verify_blas(M, N, K, A_h, lda, B_h, ldb, C_h, ldc, alpha, beta, D_h_cublas);

#if 0
    printf("D_h_cpu = cpuGemm(A, B) =\n");
    print_matrix(D_h_cpu, ldc, M, N, true);

    printf("D_h_very=\n");
    print_matrix(D_h_cublas, ldc, M, N, true);
    printf("=====\n");
#endif

    verify(M, N, D_h_cpu, ldc, D_h_cublas, ldc, 0.01);

    free(A_h);
    free(B_h);
    free(C_h);
    free(D_h_cpu);
    free(D_h_cublas);

    return 0;
}
