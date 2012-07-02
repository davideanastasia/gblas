/*
 *  dgemm_std.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 30/06/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include "gblas_kernels.h"
#include "gblas_matrix_utils.h"

void gblas_dgemm_trivial(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
               const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
               const int K, const double alpha, const double *A /* M x K */,
               const int lda, const double *B /* K x N */, const int ldb,
               const double beta, double *C /* M x N*/, const int ldc)
{
  for (int i = 0; i < M; i++) // rows A
  {
    for (int j = 0; j < N; j++) // cols B
    {
      //C[i][j] *= beta;
      C[i*N + j] *= beta;
      for (int k = 0; k < K; k++) // cols A - rows B
      {
        //C[i][j] +=  alpha * A[i][k] * B[k][j];
        C[i*N + j] +=  alpha * A[i*K + k] * B[k*N + j];
      }
    }
  }
}


void gblas_dgemm_plain(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const double alpha, const double *A /* M x K */,
                  const int lda, const double *B /* K x N */, const int ldb,
                  const double beta, double *C /* M x N*/, const int ldc)
{
  double *__A = (double*)_mm_malloc(sizeof(double)*K*M, 16); 
  double *__B = (double*)_mm_malloc(sizeof(double)*K*N, 16);

  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B);
  //exit(-1);
  
  const int BLOCK_ELEMS  = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  double *p_A;
  double *p_B;
  double *p_C;
  
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B
    {
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      p_A = &__A[(ii*K_blocks)*BLOCK_ELEMS];
      p_B = &__B[(jj*K_blocks)*BLOCK_ELEMS];
      
      KERNEL_std_dgemm_v1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc);
      
      for (int kk = 1; kk < K_blocks; kk++)
      {   
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        KERNEL_std_dgemm_v1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0), p_C, ldc);
      }
    }
  }
  
  _mm_free(__A);
  _mm_free(__B);
}