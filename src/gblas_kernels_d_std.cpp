/*
 *  gblas_kernels_d_std.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 30/08/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <cmath>

#include "gblas_kernels.h"

void KERNEL_std_dgemm_v1(const int M, const int N,
                         const int K, const double alpha, const double *A,
                         const int lda, const double *B, const int ldb,
                         const double beta, double *C, const int ldc)
{  
  const __m128d __beta = _mm_set1_pd(beta);
  
  __m128d A0, A1;
  
  __m128d B0, B1;
  __m128d b0, b1;  // temp
  __m128d C0_0, C0_1;
  __m128d C1_0, C1_1; 
  
  __m128d acc0;
  
  const double* pA0 = A;
  const double* pA1 = A + GBLAS_KERNEL_SIZE;
  
  const double* pB0 = B;
  const double* pB1 = B + GBLAS_KERNEL_SIZE;
  
  const double* stB = B + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const double* stA = A + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  double* pC0       = C;
  double* pC1       = C + ldc;
  
  do
  {
    do
    {
      B0 = _mm_load_pd(pB0);                            pB0 += 2;
      B1 = _mm_load_pd(pB1);                            pB1 += 2;
      
      A0 = _mm_load_pd(pA0);                            pA0 += 2;
      C0_0 = B0;
      C0_0 = _mm_mul_pd(C0_0, A0);
      C0_1 = B1;
      C0_1 = _mm_mul_pd(C0_1, A0);
      
      A1 = _mm_load_pd(pA1);                            pA1 += 2;
      C1_0 = B0;
      C1_0 = _mm_mul_pd(C1_0, A1);
      C1_1 = B1;
      C1_1 = _mm_mul_pd(C1_1, A1);
      
      for ( int k = ((GBLAS_KERNEL_SIZE >> 1)-1); k; k--) // cols A - rows B
      {
        B0 = _mm_load_pd(pB0);                            pB0 += 2;
        B1 = _mm_load_pd(pB1);                            pB1 += 2;
        
        A0 = _mm_load_pd(pA0);                            pA0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, A0);
        C0_0 = _mm_add_pd(C0_0, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A0);
        C0_1 = _mm_add_pd(C0_1, b1);
        
        A1 = _mm_load_pd(pA1);                            pA1 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, A1);
        C1_0 = _mm_add_pd(C1_0, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A1);
        C1_1 = _mm_add_pd(C1_1, b1);
      }
      
      acc0 = _mm_load_pd(pC0);
      acc0 = _mm_mul_pd(acc0, __beta);
      
      C0_0 = _mm_hadd_pd(C0_0, C0_1);

      acc0 = _mm_add_pd(acc0, C0_0);
      _mm_store_pd(pC0, acc0);              pC0 += 2;
      
      acc0 = _mm_loadu_pd(pC1);
      acc0 = _mm_mul_pd(acc0, __beta);

      C1_0 = _mm_hadd_pd(C1_0, C1_1);

      acc0 = _mm_add_pd(acc0, C1_0);      
      _mm_storeu_pd(pC1, acc0);              pC1 += 2;
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 -= GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE;    
      pB1 += GBLAS_KERNEL_SIZE;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;               //  next 2 rows
    pA1 += GBLAS_KERNEL_SIZE*2;               //  next 2 rows
    
    pB0 = B;                                  //  turn back to the begin
    pB1 = B + GBLAS_KERNEL_SIZE;              //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
    pC1 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
  }
  while ( pA0 != stA );
}

