
#include "cublas_lay_egg.h"

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

void verify(int M, int N, half* A, int lda, half* B, int ldb, double rerror)
{
    long long count = 0;

    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            half a = A[i + j*lda];
            half b = B[ i + j*ldb];

            if(abs(float(a-b)/float(a)) > rerror)
            {
                printf("<%5.4f, %5.4f > ", float(a), float(b));
                count++ ;
            }
        }
    }
    printf("\n diff count = %lld\n", count);
}
