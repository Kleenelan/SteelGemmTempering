#include "matrix_init_print.h"

void init_matrix(half *A, int lda, int m, int n)
{
    for(int j=0; j<n; j++)
    {
        for(int i=0; i<m; i++)
        {
            half x = half(rand()*1.0f/RAND_MAX);
            A[i + j*lda] = x;
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

void cp_half_to_single(float* A, const half* B, int len)
{
    for(int i=0; i<len; i++)
        A[i] = float(B[i]);
}
