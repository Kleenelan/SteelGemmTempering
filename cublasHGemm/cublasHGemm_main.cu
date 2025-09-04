
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

///############################################
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>


void init_matrix(half *A, int lda, int m, int n, bool colMajor)
{
    if(colMajor)
    {
        for(int j=0; j<n; j++)
        {
            for(int i=0; i<m; i++)
            {
                half x = half(rand()*1.0f/RAND_MAX);
                A[i + j*lda] = x;
                //printf(" %f",  float(x));
            }
        }
        //printf("\n\n");
    }
    else
    {
        for(int i=0; i<m; i++)
        {
            for(int j=0; j<n; j++)
            {
                half x = half(rand()*1.0f/RAND_MAX);
                A[i*lda + j] = x;
                //printf(" %f",  float(x));
            }
        }
    }
}

void print_matrix(const half *A, const int lda, const int m, const int n, bool colMajor)
{
    printf("[ ...\n");
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(colMajor)
                printf(" %5.4f,", float(A[i + j*lda]));
            else
                printf(" %5.4f,", float(A[i*lda + j]));
        }
        printf(" ; ...\n");
    }
    printf("]\n");
}


void verify_blas(int M, int N, int K,
                 const half* Ah, int lda,
                 const half* Bh, int ldb,
                 half* Ch, int ldc,
                 const half alpha,
                 const half beta,
                 half* Dh)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    half *d_A = nullptr;
    half *d_B = nullptr;
    half *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), M*lda * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), ldb*K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), ldc*N * sizeof(half)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, Ah, M*lda * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, Bh, ldb*K * sizeof(half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, Ch, ldc*N * sizeof(half), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasGemmEx(cublasH, transa, transb,
                              M, N, K,
                              &alpha,
                              d_A, CUDA_R_16F, lda,
                              d_B, CUDA_R_16F, ldb,
                              &beta,
                              d_C, CUDA_R_16F, ldc,
                              CUBLAS_COMPUTE_16F,
                              CUBLAS_GEMM_DEFAULT));

    CUDA_CHECK(cudaMemcpyAsync(Dh, d_C, ldc*N * sizeof(half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

#if 0
    printf("Dh_cublasGemm =\n");
    print_matrix(Dh, ldc, M, N, true);
    printf("=====\n");
#endif

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

}


int main()
{

#if 0
    int M = 64;
    int N = 64;
    int K = 64;
#else
    int M = 7;      // \forall M > 0;
    int N = 7;      // \forall N > 0;
    int K = 4*16;   // \forall K > 0;
#endif
    int lda = K;// A raw major
    int ldb = K;// B col major
    int ldc = M;// C col major

	half *A_h;
    half *B_h;
    half *C_h;
    half *D_h;

    half alpha = half(1.0);
    half beta  = half(0.0);

    A_h = (half*)malloc(M * lda * sizeof(half));
    B_h = (half*)malloc(ldb * N * sizeof(half));
    C_h = (half*)malloc(ldc * N * sizeof(half));
    D_h = (half*)malloc(ldc * N * sizeof(half));

    init_matrix(A_h, lda, M, K, false);
    init_matrix(B_h, ldb, K, N, true);
    init_matrix(C_h, ldc, M, N, true);
    memcpy(D_h, C_h, ldc * N * sizeof(half));

#if 0
    printf("A_h =");
    print_matrix(A_h, lda, M, K, false);
    printf("B_h =");
    print_matrix(B_h, ldb, K, N, true);
    printf("C_h =");
    print_matrix(C_h, ldc, M, N, true);
#endif

    verify_blas(M, N, K, A_h, lda, B_h, ldb, C_h, ldc, alpha, beta, D_h);

    printf("D_h_very=\n");
    print_matrix(D_h, ldc, M, N, true);
    printf("=====\n");


    free(A_h);
    free(B_h);
    free(C_h);
    free(D_h);

    return 0;
}
