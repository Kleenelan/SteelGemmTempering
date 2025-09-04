
#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
//#include <cuda_runtime.h>

#include "cublas_lay_egg.h"
#include "matrix_init_print.h"

#include <mma.h>

using namespace nvcuda;

// Tensor Core GEMM 内核
__global__ void wmma_gemm(half *a, half *b, half *c, int M, int N, int K) {

    // 声明矩阵分片
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 计算分片在全局内存中的位置
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // 纵向 warp i
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);            // 横向 warp j

    const int a_row = warpM * 16; //本 warp 负责的数据的 A/C 第一个行号； start_i
    const int b_col = warpN * 16; //本 warp 负责的数据的 B/C 第一个列号； start_j
    const int start_i = a_row;
    const int start_j = b_col;

    // 分块矩阵乘法
    for (int k = 0; k < K; k += 16) {
        int a_col = k; // 本轮 subA(16x16) 中，A 的第一个列号；
        int b_row = k; // 本轮 subB(16x16) 中, B 的第一个行号；

        // 检查边界
        if (a_row < M && a_col + 16 <= K) {
            wmma::load_matrix_sync(a_frag, a + a_row * K + a_col, K);// dst, 起始地址， 主维度； 自动取一个 16x16 的子矩阵。
        } else {
            // 处理边界情况
            wmma::fill_fragment(a_frag, 0.0f);
        }

        if (b_row + 16 <= K && b_col < N) {
            wmma::load_matrix_sync(b_frag, b + b_row  + b_col*K, K);
        } else {
            wmma::fill_fragment(b_frag, 0.0f);
        }

        // Tensor Core 矩阵乘加
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    if (a_row < M && b_col < N) {
        wmma::store_matrix_sync(c + start_i + start_j*M, c_frag, M, wmma::mem_col_major);
        //wmma::store_matrix_sync(c + a_row * N + b_col, c_frag, N, wmma::mem_row_major);
        //wmma::store_matrix_sync(c + a_row  + b_col*M, c_frag, M, wmma::mem_col_major);
    }
}

// 调用示例
void launch_wmma_gemm(half *A, half *B, half *C, int M, int N, int K) {
    dim3 gridDim((M + 15) / 16, (N + 15) / 16);
    dim3 blockDim(32, 4); // 128 threads per block

    wmma_gemm<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}


void gemm_v02_test(int m, int n, int k,
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
    printf("__1000________\n");
    cudaMalloc((void**)&Ad, m*lda*sizeof(half));
    cudaMalloc((void**)&Bd, ldb*n*sizeof(half));
    cudaMalloc((void**)&Cd, ldc*n*sizeof(half));

    //2. cpy H2D
    cudaMemcpy(Ad, Ah, m*lda*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bh, ldb*n*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, Ch, ldc*n*sizeof(half), cudaMemcpyHostToDevice);
    //3. Gemm_v01, simple cuda core gemm

printf("__01________\n");
    launch_wmma_gemm(Ad, Bd, Cd, m, n, k);
//    gemm_v01_fp16_all<<<grid_,block_>>>(m, n, k, Ad, lda, Bd, ldb, Cd, ldc, alpha, beta);
printf("##22########\n");
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
    half *D_h_tcu;
    half *D_h_cublas;


    half alpha = half(1.0);
    half beta  = half(0.0);

    A_h = (half*)malloc(M * lda * sizeof(half));
    B_h = (half*)malloc(ldb * N * sizeof(half));
    C_h = (half*)malloc(ldc * N * sizeof(half));
    D_h_tcu = (half*)malloc(ldc * N * sizeof(half));
    D_h_cublas = (half*)malloc(ldc * N * sizeof(half));

    init_matrix(A_h, lda, M, K, false);
    init_matrix(B_h, ldb, K, N, true);
    init_matrix(C_h, ldc, M, N, true);
    memcpy(D_h_tcu, C_h, M * ldc * sizeof(half));
    memcpy(D_h_cublas, C_h, M * ldc * sizeof(half));

#if 0
    printf("A_h =");
    print_matrix(A_h, lda, M, K, false);
    printf("B_h =");
    print_matrix(B_h, ldb, K, N, true);
    printf("C_h =");
    print_matrix(C_h, ldc, M, N, true);
    printf("D_h_tcu =");
    print_matrix(D_h_tcu, ldc, M, N, true);
#endif

    gemm_v02_test(M, N, K, A_h, lda, B_h, ldb, C_h, ldc, alpha, beta, D_h_tcu);
    printf("D_h_tcu = tensorCoreGemm(A, B) =\n");
    print_matrix(D_h_tcu, ldc, M, N, true);

    verify_blas(M, N, K, A_h, lda, B_h, ldb, C_h, ldc, alpha, beta, D_h_cublas);
    verify(M, N, D_h_tcu, ldc, D_h_cublas, ldc, 0.001);// relative error

    free(D_h_tcu);
    free(D_h_cublas);
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}

