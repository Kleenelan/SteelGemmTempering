
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
//#include <cuda_runtime.h>


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

void print_matrix(half *A, int lda, int m, int n, bool colMajor)
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

void gemm_fp16_cpu(int M, int N, int K,
                   half* A, int lda,
                   half* B, int ldb,
                   half* C, int ldc,
                   half alpha, half beta)
{
    for(int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            half sigma = half(0.0f);
            for(int k=0; k<K; k++)
            {
                sigma += A[i*lda + k] * B[k + j*ldb];
            }
            C[i + j*ldc] = alpha*sigma + beta*C[i + j*ldc];
        }
    }
}

//A-rowMajor; B-colMajor; C-col Major;
__global__ void gemm_v01_fp16_all(int M, int N, int K,
                                  half* Ad, int lda,
                                  half* Bd, int ldb,
                                  half* Cd, int ldc,
                                  half alpha, half beta)
{
    //C(i, j)
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;
    //if(i==16) printf("%d ", j);printf("threadIdx.x=%d  ", threadIdx.x);
    // A_subblock start index
    unsigned int start_row_C = blockDim.x*blockIdx.x;// start_row_C too;
    unsigned int start_col_C = blockDim.y*blockIdx.y;// start_col_C too;

    half* C = Cd + start_row_C     + start_col_C*ldc;
    half* A = Ad + start_row_C*lda +               0;
    half* B = Bd + 0               + start_col_C*ldb;

    half sigma = half(0.0);
    for(unsigned int k = 0; k<K; k++)
    {
        sigma += A[i*lda + k]*B[k + j*ldb];
//        sigma += A[i + k*lda]*B[k *ldb+ j];
    }

    C[i + j*ldc] = alpha*sigma + beta*C[i + j*ldc];
}



void gemm_v01_cuda(int m, int n, int k,
                   half* Ah, int lda,
                   half* Bh, int ldb,
                   half* Ch, int ldc,
                   half alpha, half beta,
                   half* Dh)
{
    //1. alloc ABC_d
    half * Ad = nullptr;
    half * Bd = nullptr;
    half * Cd = nullptr;
    cudaMalloc((void**)&Ad, m*lda*sizeof(half));
    cudaMalloc((void**)&Bd, ldb*n*sizeof(half));
    cudaMalloc((void**)&Cd, ldc*n*sizeof(half));

    //2. cpy H2D
    cudaMemcpy(Ad, Ah, m*lda*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bh, ldb*n*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, Ch, ldc*n*sizeof(half), cudaMemcpyHostToDevice);
    //3. Gemm_v01, simple cuda core gemm
    dim3 block_;
    dim3 grid_;
    block_.x = 16;
    block_.y = 16;
    grid_.x = (m+block_.x-1)/block_.x;
    grid_.y = (n+block_.y-1)/block_.y;

printf("__00________\n");
    gemm_v01_fp16_all<<<grid_,block_>>>(m, n, k, Ad, lda, Bd, ldb, Cd, ldc, alpha, beta);
printf("##11########\n");
    //4. cpy D2H
    cudaMemcpy(Dh, Cd, ldc*n*sizeof(half), cudaMemcpyDeviceToHost);
    //5. free ABC_d
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}


int main()
{
#if 0
    int M = 64;
    int N = 64;
    int K = 64;
#else
    int M = 2*16;
    int N = 2*16;
    int K = 7*16;
#endif
    int lda = K;// A raw major
    int ldb = K;// B col major
    int ldc = M;// C col major

	half *A_h;
    half *B_h;
    half *C_h;
    half *D_h;
    half *V_h;


    half alpha = half(1.0);
    half beta  = half(0.0);

    A_h = (half*)malloc(M * lda * sizeof(half));
    B_h = (half*)malloc(ldb * N * sizeof(half));
    C_h = (half*)malloc(ldc * N * sizeof(half));
    D_h = (half*)malloc(ldc * N * sizeof(half));
    V_h = (half*)malloc(ldc * N * sizeof(half));

    init_matrix(A_h, lda, M, K, false);
    init_matrix(B_h, ldb, K, N, true);
    init_matrix(C_h, ldc, M, N, true);
    memcpy(D_h, C_h, M * ldc * sizeof(half));

#if 1
    printf("A_h =");
    print_matrix(A_h, lda, M, K, false);
    printf("B_h =");
    print_matrix(B_h, ldb, K, N, true);
    printf("C_h =");
    print_matrix(C_h, ldc, M, N, true);
    printf("D_h =");
    print_matrix(D_h, ldc, M, N, true);
#endif

    gemm_fp16_cpu(M, N, K, A_h, lda, B_h, ldb, D_h, ldc, alpha, beta);// Arow Bcol Crow Major;

    printf("D_h = cpuGemm(A, B) =\n");
    print_matrix(D_h, ldc, M, N, true);
    memset(D_h, 0x00, ldc * N * sizeof(half));

    gemm_v01_cuda(M, N, K, A_h, lda, B_h, ldb, C_h, ldc, alpha, beta, V_h);
    printf("V_h = cudaCoreGemm(A, B) =\n");
    print_matrix(V_h, ldc, M, N, true);


    free(D_h);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}

