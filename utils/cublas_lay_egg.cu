#include "matrix_init_print.h"
#include "cublas_lay_egg.h"

void verify_blas(int M, int N, int K,
                 const half* Ah, int lda,
                 const half* Bh, int ldb,
                 half* Ch, int ldc,
                 const half alpha_ori,
                 const half beta_ori,
                 float* Dh)
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

    float* sA = (float*)malloc(lda*M * sizeof(float));
    float* sB = (float*)malloc(ldb*N * sizeof(float));
    float* sC = (float*)malloc(ldc*N * sizeof(float));

    cp_half_to_single(sA, Ah, lda*M);
    cp_half_to_single(sB, Bh, ldb*N);
    cp_half_to_single(sC, Ch, ldc*N);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), lda*M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), ldb*N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), ldc*N * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, sA, lda*M * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, sB, ldb*N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, sC, ldc*N * sizeof(float), cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaStreamSynchronize(stream));//LL:: add

    float alpha = float(alpha_ori);
    float beta  = float(beta_ori);

    CUBLAS_CHECK(cublasSgemmEx(cublasH, transa, transb,
                              M, N, K,
                              &alpha,
                              d_A, CUDA_R_32F, lda,
                              d_B, CUDA_R_32F, ldb,
                              &beta,
                              d_C, CUDA_R_32F, ldc//,
                              //CUBLAS_COMPUTE_32F,
                            //  CUBLAS_GEMM_DEFAULT
                            ));

    CUDA_CHECK(cudaMemcpyAsync(Dh, d_C, ldc*N * sizeof(float), cudaMemcpyDeviceToHost, stream));
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

void verify(int M, int N, half* Ca, int lda, float* Cb, int ldb, double rerror)// all col
{
    long long count = 0;
    #pragma omp parallel for
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            half a = Ca[i + j*lda];
            half b = Cb[ i + j*ldb];

            if(abs(float(a-b)/float(a)) > rerror)
            {
                printf("<%5.4f, %5.4f > ", float(a), float(b));
                count++ ;
            }
        }
    }
    printf("\n diff count = %lld\n", count);
}
