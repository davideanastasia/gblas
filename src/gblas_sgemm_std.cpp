/*
 *  gblas_sgemm_std.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 07/07/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include "gblas_kernels.h"
#include "gblas_matrix_utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void gblas_sgemm_trivial(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
               const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
               const int K, const float alpha, const float *A /* M x K */,
               const int lda, const float *B /* K x N */, const int ldb,
               const float beta, float *C /* M x N*/, const int ldc)
{
  float t;
  for (int i = 0; i < M; i++) // rows A
  {
    for (int j = 0; j < N; j++) // cols B
    {
      //C[i][j] *= beta;
      t = C[i*N + j] * beta;
      for (int k = 0; k < K; k++) // cols A - rows B
      {
        t +=  alpha * A[i*K + k] * B[k*N + j];
      }
      C[i*N + j] = t;
    }
  }
}

void gblas_sgemm_plain(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const float alpha, const float *A /* M x K */,
                  const int lda, const float *B /* K x N */, const int ldb,
                  const float beta, float *C /* M x N*/, const int ldc)
{ 
  const int BLOCK_ELEMS  = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  float *__A = (float*)_mm_malloc(sizeof(float)*K*M, 16);
  float *__B = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  
  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A);
  //exit(-1);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B);
  //exit(-1);
  
  float *p_A = __A;
  float *p_B = __B;
  float *p_C = C;
  
#pragma omp parallel for private(p_C, p_A, p_B)
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
    //#pragma omp parallel for private(p_C, p_A, p_B)
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B 
    {
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      p_A = &__A[(ii*K_blocks)*BLOCK_ELEMS];
      p_B = &__B[(jj*K_blocks)*BLOCK_ELEMS];
      
      if ( beta == 0.0f ) {
        KERNEL_std_sgemm_v6_B0(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, p_C, ldc);
      } else if (beta == 1.0f) {
        KERNEL_std_sgemm_v6_B1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, p_C, ldc);
      } else {
        KERNEL_std_sgemm_v6_BX(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc);
      }
      
      //KERNEL_std_sgemm_v6(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc);
      
      for (int kk = 1; kk < K_blocks; kk++)
      {      
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        KERNEL_std_sgemm_v6_B1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, p_C, ldc);
      }
    }
  }
  
  _mm_free(__A);
  _mm_free(__B);
}
