/*
 *  gblas_kernels_s_std.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 28/07/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */


#include <cmath>
#include <string.h>

#include "gblas_kernels.h"

void KERNEL_std_sgemm_v6(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         const float beta, float *C, const int ldc)
{  
  //const int K_SSE = (GBLAS_KERNEL_SIZE/4);
  const __m128 BETA = _mm_set1_ps(beta);
  
  __m128 A0, b; //A1;
  
  __m128 B0, B1, B2, B3;
  __m128 C0_0, C0_1, C0_2, C0_3;
  __m128 C1_0, C1_1, C1_2, C1_3; 
  
  const float* pA0 = A;
  const float* pA1 = pA0 + GBLAS_KERNEL_SIZE;
  
  const float* pB0 = B;
  const float* pB1 = pB0 + GBLAS_KERNEL_SIZE;
  const float* pB2 = pB1 + GBLAS_KERNEL_SIZE;
  const float* pB3 = pB2 + GBLAS_KERNEL_SIZE;
  
  const float* stB = B + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const float* stA = A + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  float* pC0       = C;
  float* pC1       = pC0 + ldc;
  
  do
  //for (int mm = 0; mm < (GBLAS_KERNEL_SIZE/2); mm++)
  {
    do
    //for (int nn = 0; nn < (GBLAS_KERNEL_SIZE/4); nn++) 
    //for (int nn = (GBLAS_KERNEL_SIZE >> 2); nn ; nn--) 
    {
      B0 = _mm_load_ps(pB0);                            pB0 += 4;
      B1 = _mm_load_ps(pB1);                            pB1 += 4;
      B2 = _mm_load_ps(pB2);                            pB2 += 4;
      B3 = _mm_load_ps(pB3);                            pB3 += 4;
      
      A0 = _mm_load_ps(pA0);                            pA0 += 4;
      C0_0 = _mm_mul_ps(B0, A0);
      C0_1 = _mm_mul_ps(B1, A0);
      C0_2 = _mm_mul_ps(B2, A0);
      C0_3 = _mm_mul_ps(B3, A0); 
          
      A0 = _mm_load_ps(pA1);                            pA1 += 4;
      C1_0 = _mm_mul_ps(B0, A0);
      C1_1 = _mm_mul_ps(B1, A0);
      C1_2 = _mm_mul_ps(B2, A0);
      C1_3 = _mm_mul_ps(B3, A0);

      for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k; k--)        // V.3
      // for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k > 0; k--)     // V.2
      //for (int k = 1; k < K_SSE; k++) // cols A - rows B          // V.1
      {
        B0 = _mm_load_ps(pB0);                            pB0 += 4;
        B1 = _mm_load_ps(pB1);                            pB1 += 4;
        B2 = _mm_load_ps(pB2);                            pB2 += 4;
        B3 = _mm_load_ps(pB3);                            pB3 += 4;
        
        A0 = _mm_load_ps(pA0);                            pA0 += 4;
        b = B0;
        b = _mm_mul_ps(b, A0);
        C0_0 = _mm_add_ps(C0_0, b);
        b = B1;
        b = _mm_mul_ps(b, A0);
        C0_1 = _mm_add_ps(C0_1, b);
        b = B2;
        b = _mm_mul_ps(b, A0);
        C0_2 = _mm_add_ps(C0_2, b);
        b = B3;
        b = _mm_mul_ps(b, A0);
        C0_3 = _mm_add_ps(C0_3, b);
        
        A0 = _mm_load_ps(pA1);                            pA1 += 4;
        B0 = _mm_mul_ps(B0, A0);
        C1_0 = _mm_add_ps(C1_0, B0);
        B1 = _mm_mul_ps(B1, A0);
        C1_1 = _mm_add_ps(C1_1, B1);
        B2 = _mm_mul_ps(B2, A0);
        C1_2 = _mm_add_ps(C1_2, B2);
        B3 = _mm_mul_ps(B3, A0);
        C1_3 = _mm_add_ps(C1_3, B3);
      }
      
      // --- horizontal ADD ---     
      C0_0 = _mm_hadd_ps(C0_0, C0_1);
      C0_2 = _mm_hadd_ps(C0_2, C0_3);
      C0_0 = _mm_hadd_ps(C0_0, C0_2);
      // ---
      
      // --- horizontal ADD ---     
      C1_0 = _mm_hadd_ps(C1_0, C1_1);
      C1_2 = _mm_hadd_ps(C1_2, C1_3);
      C1_0 = _mm_hadd_ps(C1_0, C1_2);
      // ---
      
      if ( beta != 0.0f )
      {
        C0_3 = _mm_loadu_ps(pC0);     // UN-aligned LOAD
        C0_3 = _mm_mul_ps(C0_3, BETA);
        C0_0 = _mm_add_ps(C0_0, C0_3);

        C1_3 = _mm_loadu_ps(pC1);     // UN-aligned LOAD
        C1_3 = _mm_mul_ps(C1_3, BETA);
        C1_0 = _mm_add_ps(C1_0, C1_3);
      }
            
      // Unaligned store of output
      _mm_storeu_ps(pC0, C0_0);
      _mm_storeu_ps(pC1, C1_0);
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 = pA0 + GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE*3;    
      pB1 = pB0 + GBLAS_KERNEL_SIZE;
      pB2 = pB1 + GBLAS_KERNEL_SIZE;
      pB3 = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0 += 4;
      pC1 += 4;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;              //  next 2 rows
    pA1 = pA0 + GBLAS_KERNEL_SIZE;           //  next 2 rows
    
    pB0 = B;                                 //  turn back to the begin
    pB1 = pB0 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB2 = pB1 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB3 = pB2 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);      //  next 2 rows
    pC1 = pC0 + ldc;                         //  next 2 rows
  }
  while ( pA0 != stA );
}

void KERNEL_std_sgemm_v6_B0(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         float *C, const int ldc)
{ 
  __m128 A0, b; //A1;
  
  __m128 B0, B1, B2, B3;
  __m128 C0_0, C0_1, C0_2, C0_3;
  __m128 C1_0, C1_1, C1_2, C1_3; 
  
  const float* pA0 = A;
  const float* pA1 = pA0 + GBLAS_KERNEL_SIZE;
  
  const float* pB0 = B;
  const float* pB1 = pB0 + GBLAS_KERNEL_SIZE;
  const float* pB2 = pB1 + GBLAS_KERNEL_SIZE;
  const float* pB3 = pB2 + GBLAS_KERNEL_SIZE;
  
  const float* stB = B + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const float* stA = A + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  float* pC0       = C;
  float* pC1       = pC0 + ldc;
  
  do
  {
    do
    {
      B0 = _mm_load_ps(pB0);                            pB0 += 4;
      B1 = _mm_load_ps(pB1);                            pB1 += 4;
      B2 = _mm_load_ps(pB2);                            pB2 += 4;
      B3 = _mm_load_ps(pB3);                            pB3 += 4;
      
      A0 = _mm_load_ps(pA0);                            pA0 += 4;
      C0_0 = _mm_mul_ps(B0, A0);
      C0_1 = _mm_mul_ps(B1, A0);
      C0_2 = _mm_mul_ps(B2, A0);
      C0_3 = _mm_mul_ps(B3, A0); 
      
      A0 = _mm_load_ps(pA1);                            pA1 += 4;
      C1_0 = _mm_mul_ps(B0, A0);
      C1_1 = _mm_mul_ps(B1, A0);
      C1_2 = _mm_mul_ps(B2, A0);
      C1_3 = _mm_mul_ps(B3, A0);
      
      for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k; k--)        // V.3
      {
        B0 = _mm_load_ps(pB0);                            pB0 += 4;
        B1 = _mm_load_ps(pB1);                            pB1 += 4;
        B2 = _mm_load_ps(pB2);                            pB2 += 4;
        B3 = _mm_load_ps(pB3);                            pB3 += 4;
        
        A0 = _mm_load_ps(pA0);                            pA0 += 4;
        b = B0;
        b = _mm_mul_ps(b, A0);
        C0_0 = _mm_add_ps(C0_0, b);
        b = B1;
        b = _mm_mul_ps(b, A0);
        C0_1 = _mm_add_ps(C0_1, b);
        b = B2;
        b = _mm_mul_ps(b, A0);
        C0_2 = _mm_add_ps(C0_2, b);
        b = B3;
        b = _mm_mul_ps(b, A0);
        C0_3 = _mm_add_ps(C0_3, b);
        
        A0 = _mm_load_ps(pA1);                            pA1 += 4;
        B0 = _mm_mul_ps(B0, A0);
        C1_0 = _mm_add_ps(C1_0, B0);
        B1 = _mm_mul_ps(B1, A0);
        C1_1 = _mm_add_ps(C1_1, B1);
        B2 = _mm_mul_ps(B2, A0);
        C1_2 = _mm_add_ps(C1_2, B2);
        B3 = _mm_mul_ps(B3, A0);
        C1_3 = _mm_add_ps(C1_3, B3);
      }
      
      // --- horizontal ADD ---     
      C0_0 = _mm_hadd_ps(C0_0, C0_1);
      C0_2 = _mm_hadd_ps(C0_2, C0_3);
      C0_0 = _mm_hadd_ps(C0_0, C0_2);
      // ---
      
      // --- horizontal ADD ---     
      C1_0 = _mm_hadd_ps(C1_0, C1_1);
      C1_2 = _mm_hadd_ps(C1_2, C1_3);
      C1_0 = _mm_hadd_ps(C1_0, C1_2);
      // ---
            
      // Unaligned store of output
      _mm_storeu_ps(pC0, C0_0);
      _mm_storeu_ps(pC1, C1_0);
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 = pA0 + GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE*3;    
      pB1 = pB0 + GBLAS_KERNEL_SIZE;
      pB2 = pB1 + GBLAS_KERNEL_SIZE;
      pB3 = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0 += 4;
      pC1 += 4;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;              //  next 2 rows
    pA1 = pA0 + GBLAS_KERNEL_SIZE;           //  next 2 rows
    
    pB0 = B;                                 //  turn back to the begin
    pB1 = pB0 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB2 = pB1 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB3 = pB2 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);      //  next 2 rows
    pC1 = pC0 + ldc;                         //  next 2 rows
  }
  while ( pA0 != stA );
}

void KERNEL_std_sgemm_v6_B1(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         float *C, const int ldc)
{  
  __m128 A0, b; //A1;
  
  __m128 B0, B1, B2, B3;
  __m128 C0_0, C0_1, C0_2, C0_3;
  __m128 C1_0, C1_1, C1_2, C1_3; 
  
  const float* pA0 = A;
  const float* pA1 = pA0 + GBLAS_KERNEL_SIZE;
  
  const float* pB0 = B;
  const float* pB1 = pB0 + GBLAS_KERNEL_SIZE;
  const float* pB2 = pB1 + GBLAS_KERNEL_SIZE;
  const float* pB3 = pB2 + GBLAS_KERNEL_SIZE;
  
  const float* stB = B + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const float* stA = A + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  float* pC0       = C;
  float* pC1       = pC0 + ldc;
  
  do
  {
    do
    {
      B0 = _mm_load_ps(pB0);                            pB0 += 4;
      B1 = _mm_load_ps(pB1);                            pB1 += 4;
      B2 = _mm_load_ps(pB2);                            pB2 += 4;
      B3 = _mm_load_ps(pB3);                            pB3 += 4;
      
      A0 = _mm_load_ps(pA0);                            pA0 += 4;
      C0_0 = _mm_mul_ps(B0, A0);
      C0_1 = _mm_mul_ps(B1, A0);
      C0_2 = _mm_mul_ps(B2, A0);
      C0_3 = _mm_mul_ps(B3, A0); 
      
      A0 = _mm_load_ps(pA1);                            pA1 += 4;
      C1_0 = _mm_mul_ps(B0, A0);
      C1_1 = _mm_mul_ps(B1, A0);
      C1_2 = _mm_mul_ps(B2, A0);
      C1_3 = _mm_mul_ps(B3, A0);
      
      for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k; k--)        // V.3
      {
        B0 = _mm_load_ps(pB0);                            pB0 += 4;
        B1 = _mm_load_ps(pB1);                            pB1 += 4;
        B2 = _mm_load_ps(pB2);                            pB2 += 4;
        B3 = _mm_load_ps(pB3);                            pB3 += 4;
        
        A0 = _mm_load_ps(pA0);                            pA0 += 4;
        b = B0;
        b = _mm_mul_ps(b, A0);
        C0_0 = _mm_add_ps(C0_0, b);
        b = B1;
        b = _mm_mul_ps(b, A0);
        C0_1 = _mm_add_ps(C0_1, b);
        b = B2;
        b = _mm_mul_ps(b, A0);
        C0_2 = _mm_add_ps(C0_2, b);
        b = B3;
        b = _mm_mul_ps(b, A0);
        C0_3 = _mm_add_ps(C0_3, b);
        
        A0 = _mm_load_ps(pA1);                            pA1 += 4;
        B0 = _mm_mul_ps(B0, A0);
        C1_0 = _mm_add_ps(C1_0, B0);
        B1 = _mm_mul_ps(B1, A0);
        C1_1 = _mm_add_ps(C1_1, B1);
        B2 = _mm_mul_ps(B2, A0);
        C1_2 = _mm_add_ps(C1_2, B2);
        B3 = _mm_mul_ps(B3, A0);
        C1_3 = _mm_add_ps(C1_3, B3);
      }
      
      // --- horizontal ADD ---     
      C0_0 = _mm_hadd_ps(C0_0, C0_1);
      C0_2 = _mm_hadd_ps(C0_2, C0_3);
      C0_0 = _mm_hadd_ps(C0_0, C0_2);
      // ---
      
      // --- horizontal ADD ---     
      C1_0 = _mm_hadd_ps(C1_0, C1_1);
      C1_2 = _mm_hadd_ps(C1_2, C1_3);
      C1_0 = _mm_hadd_ps(C1_0, C1_2);
      // ---
      
      C0_3 = _mm_loadu_ps(pC0);     // UN-aligned LOAD
      C0_0 = _mm_add_ps(C0_0, C0_3);
      _mm_storeu_ps(pC0, C0_0);     // Un-aligned STORE
      
      C1_3 = _mm_loadu_ps(pC1);     // UN-aligned LOAD
      C1_0 = _mm_add_ps(C1_0, C1_3);
      _mm_storeu_ps(pC1, C1_0);     // Un-aligned STORE
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 = pA0 + GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE*3;    
      pB1 = pB0 + GBLAS_KERNEL_SIZE;
      pB2 = pB1 + GBLAS_KERNEL_SIZE;
      pB3 = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0 += 4;
      pC1 += 4;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;              //  next 2 rows
    pA1 = pA0 + GBLAS_KERNEL_SIZE;           //  next 2 rows
    
    pB0 = B;                                 //  turn back to the begin
    pB1 = pB0 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB2 = pB1 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB3 = pB2 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);      //  next 2 rows
    pC1 = pC0 + ldc;                         //  next 2 rows
  }
  while ( pA0 != stA );
}

void KERNEL_std_sgemm_v6_BX(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         const float beta, float *C, const int ldc)
{  
  //const int K_SSE = (GBLAS_KERNEL_SIZE/4);
  const __m128 BETA = _mm_set1_ps(beta);
  
  __m128 A0, b; //A1;
  
  __m128 B0, B1, B2, B3;
  __m128 C0_0, C0_1, C0_2, C0_3;
  __m128 C1_0, C1_1, C1_2, C1_3; 
  
  const float* pA0 = A;
  const float* pA1 = pA0 + GBLAS_KERNEL_SIZE;
  
  const float* pB0 = B;
  const float* pB1 = pB0 + GBLAS_KERNEL_SIZE;
  const float* pB2 = pB1 + GBLAS_KERNEL_SIZE;
  const float* pB3 = pB2 + GBLAS_KERNEL_SIZE;
  
  const float* stB = B + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const float* stA = A + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  float* pC0       = C;
  float* pC1       = pC0 + ldc;
  
  do
  {
    do
    {
      B0 = _mm_load_ps(pB0);                            pB0 += 4;
      B1 = _mm_load_ps(pB1);                            pB1 += 4;
      B2 = _mm_load_ps(pB2);                            pB2 += 4;
      B3 = _mm_load_ps(pB3);                            pB3 += 4;
      
      A0 = _mm_load_ps(pA0);                            pA0 += 4;
      C0_0 = _mm_mul_ps(B0, A0);
      C0_1 = _mm_mul_ps(B1, A0);
      C0_2 = _mm_mul_ps(B2, A0);
      C0_3 = _mm_mul_ps(B3, A0); 
      
      A0 = _mm_load_ps(pA1);                            pA1 += 4;
      C1_0 = _mm_mul_ps(B0, A0);
      C1_1 = _mm_mul_ps(B1, A0);
      C1_2 = _mm_mul_ps(B2, A0);
      C1_3 = _mm_mul_ps(B3, A0);
      
      for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k; k--)        // V.3
      {
        B0 = _mm_load_ps(pB0);                            pB0 += 4;
        B1 = _mm_load_ps(pB1);                            pB1 += 4;
        B2 = _mm_load_ps(pB2);                            pB2 += 4;
        B3 = _mm_load_ps(pB3);                            pB3 += 4;
        
        A0 = _mm_load_ps(pA0);                            pA0 += 4;
        b = B0;
        b = _mm_mul_ps(b, A0);
        C0_0 = _mm_add_ps(C0_0, b);
        b = B1;
        b = _mm_mul_ps(b, A0);
        C0_1 = _mm_add_ps(C0_1, b);
        b = B2;
        b = _mm_mul_ps(b, A0);
        C0_2 = _mm_add_ps(C0_2, b);
        b = B3;
        b = _mm_mul_ps(b, A0);
        C0_3 = _mm_add_ps(C0_3, b);
        
        A0 = _mm_load_ps(pA1);                            pA1 += 4;
        B0 = _mm_mul_ps(B0, A0);
        C1_0 = _mm_add_ps(C1_0, B0);
        B1 = _mm_mul_ps(B1, A0);
        C1_1 = _mm_add_ps(C1_1, B1);
        B2 = _mm_mul_ps(B2, A0);
        C1_2 = _mm_add_ps(C1_2, B2);
        B3 = _mm_mul_ps(B3, A0);
        C1_3 = _mm_add_ps(C1_3, B3);
      }
      
      // --- horizontal ADD ---     
      C0_0 = _mm_hadd_ps(C0_0, C0_1);
      C0_2 = _mm_hadd_ps(C0_2, C0_3);
      C0_0 = _mm_hadd_ps(C0_0, C0_2);
      // ---
      
      // --- horizontal ADD ---     
      C1_0 = _mm_hadd_ps(C1_0, C1_1);
      C1_2 = _mm_hadd_ps(C1_2, C1_3);
      C1_0 = _mm_hadd_ps(C1_0, C1_2);
      // ---

      C0_3 = _mm_loadu_ps(pC0);       // UN-aligned LOAD
      C0_3 = _mm_mul_ps(C0_3, BETA);
      C0_0 = _mm_add_ps(C0_0, C0_3);
      _mm_storeu_ps(pC0, C0_0);       // Un-aligned STORE
      
      C1_3 = _mm_loadu_ps(pC1);       // UN-aligned LOAD
      C1_3 = _mm_mul_ps(C1_3, BETA);
      C1_0 = _mm_add_ps(C1_0, C1_3);
      _mm_storeu_ps(pC1, C1_0);       // Un-aligned STORE
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 = pA0 + GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE*3;    
      pB1 = pB0 + GBLAS_KERNEL_SIZE;
      pB2 = pB1 + GBLAS_KERNEL_SIZE;
      pB3 = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0 += 4;
      pC1 += 4;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;              //  next 2 rows
    pA1 = pA0 + GBLAS_KERNEL_SIZE;           //  next 2 rows
    
    pB0 = B;                                 //  turn back to the begin
    pB1 = pB0 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB2 = pB1 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB3 = pB2 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);      //  next 2 rows
    pC1 = pC0 + ldc;                         //  next 2 rows
  }
  while ( pA0 != stA );
}

//without SSE quantization
void KERNEL_std_sgemm_v7(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         const float beta, float *C, const int ldc)
{
  const float* A_l;
  const float* B_l;
  float* C_l;
  register float Curr_C;
  for (int m = 0; m < GBLAS_KERNEL_SIZE; m++) // rows A = M
  {
    C_l = &C[m*ldc];
    for (int n = 0; n < GBLAS_KERNEL_SIZE; n++) // cols B = N
    {
      Curr_C  = C_l[n] * beta;
      A_l     = &A[m*GBLAS_KERNEL_SIZE];
      B_l     = &B[n*GBLAS_KERNEL_SIZE];
      
      for (int k = 0; k < GBLAS_KERNEL_SIZE; k++) // cols A - rows B = K
      {
        Curr_C += (A_l[k] * B_l[k]);
      }
      C_l[n] = Curr_C;
    }
  }
}

void KERNEL_std_sgemm_v6_double_elaboration(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         const float beta, float *C, const int ldc)
{  
  const __m128 BETA = _mm_set1_ps(beta);
  
  __m128 A0, b; //A1;
  
  __m128 B0, B1, B2, B3;
  __m128 C0_0, C0_1, C0_2, C0_3;
  __m128 C1_0, C1_1, C1_2, C1_3; 
  
  const float* pA0 = A;
  const float* pA1 = pA0 + GBLAS_KERNEL_SIZE;
  
  const float* pB0 = B;
  const float* pB1 = pB0 + GBLAS_KERNEL_SIZE;
  const float* pB2 = pB1 + GBLAS_KERNEL_SIZE;
  const float* pB3 = pB2 + GBLAS_KERNEL_SIZE;
  
  const float* stB = B + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const float* stA = A + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  float* pC0       = C;
  float* pC1       = pC0 + ldc;
  
  do
    //for (int mm = 0; mm < (GBLAS_KERNEL_SIZE/2); mm++)
  {
    do
      //for (int nn = 0; nn < (GBLAS_KERNEL_SIZE/4); nn++) 
    {
      B0 = _mm_load_ps(pB0);                            pB0 += 4;
      B1 = _mm_load_ps(pB1);                            pB1 += 4;
      B2 = _mm_load_ps(pB2);                            pB2 += 4;
      B3 = _mm_load_ps(pB3);                            pB3 += 4;
      
      A0 = _mm_load_ps(pA0);                            pA0 += 4;
      C0_0 = _mm_mul_ps(B0, A0);
      C0_1 = _mm_mul_ps(B1, A0);
      C0_2 = _mm_mul_ps(B2, A0);
      C0_3 = _mm_mul_ps(B3, A0); 
      
      A0 = _mm_load_ps(pA1);                            pA1 += 4;
      C1_0 = _mm_mul_ps(B0, A0);
      C1_1 = _mm_mul_ps(B1, A0);
      C1_2 = _mm_mul_ps(B2, A0);
      C1_3 = _mm_mul_ps(B3, A0);
      
      for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k; k--)        // V.3
        // for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k > 0; k--)     // V.2
        //for (int k = 1; k < K_SSE; k++) // cols A - rows B          // V.1
      {
        B0 = _mm_load_ps(pB0);                            pB0 += 4;
        B1 = _mm_load_ps(pB1);                            pB1 += 4;
        B2 = _mm_load_ps(pB2);                            pB2 += 4;
        B3 = _mm_load_ps(pB3);                            pB3 += 4;
        
        A0 = _mm_load_ps(pA0);                            pA0 += 4;
        b = B0;
        b = _mm_mul_ps(b, A0);
        C0_0 = _mm_add_ps(C0_0, b);
        b = B1;
        b = _mm_mul_ps(b, A0);
        C0_1 = _mm_add_ps(C0_1, b);
        b = B2;
        b = _mm_mul_ps(b, A0);
        C0_2 = _mm_add_ps(C0_2, b);
        b = B3;
        b = _mm_mul_ps(b, A0);
        C0_3 = _mm_add_ps(C0_3, b);
        
        A0 = _mm_load_ps(pA1);                            pA1 += 4;
        B0 = _mm_mul_ps(B0, A0);
        C1_0 = _mm_add_ps(C1_0, B0);
        B1 = _mm_mul_ps(B1, A0);
        C1_1 = _mm_add_ps(C1_1, B1);
        B2 = _mm_mul_ps(B2, A0);
        C1_2 = _mm_add_ps(C1_2, B2);
        B3 = _mm_mul_ps(B3, A0);
        C1_3 = _mm_add_ps(C1_3, B3);
      }
      
      // --- horizontal ADD ---     
      C0_0 = _mm_hadd_ps(C0_0, C0_1);
      C0_2 = _mm_hadd_ps(C0_2, C0_3);
      C0_0 = _mm_hadd_ps(C0_0, C0_2);
      // ---
      
      // --- horizontal ADD ---     
      C1_0 = _mm_hadd_ps(C1_0, C1_1);
      C1_2 = _mm_hadd_ps(C1_2, C1_3);
      C1_0 = _mm_hadd_ps(C1_0, C1_2);
      // ---
      
      if ( beta != 0.0f )
      {
        C0_3 = _mm_loadu_ps(pC0);     // UN-aligned LOAD
        C0_3 = _mm_mul_ps(C0_3, BETA);
        C0_0 = _mm_add_ps(C0_0, C0_3);
        
        C1_3 = _mm_loadu_ps(pC1);     // UN-aligned LOAD
        C1_3 = _mm_mul_ps(C1_3, BETA);
        C1_0 = _mm_add_ps(C1_0, C1_3);
      }
      
      // Unaligned store of output
      _mm_storeu_ps(pC0, C0_0);
      _mm_storeu_ps(pC1, C1_0);
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 = pA0 + GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE*3;    
      pB1 = pB0 + GBLAS_KERNEL_SIZE;
      pB2 = pB1 + GBLAS_KERNEL_SIZE;
      pB3 = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0 += 4;
      pC1 += 4;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;              //  next 2 rows
    pA1 = pA0 + GBLAS_KERNEL_SIZE;           //  next 2 rows
    
    pB0 = B;                                 //  turn back to the begin
    pB1 = pB0 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB2 = pB1 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB3 = pB2 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);      //  next 2 rows
    pC1 = pC0 + ldc;                         //  next 2 rows
  }
  while ( pA0 != stA );
  
  
  float M_A2[GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float M_B2[GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  memcpy(M_A2, A, sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE);
  memcpy(M_B2, B, sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE);  
  
  pA0 = M_A2;
  pA1 = pA0 + GBLAS_KERNEL_SIZE;
  
  pB0 = M_B2;
  pB1 = pB0 + GBLAS_KERNEL_SIZE;
  pB2 = pB1 + GBLAS_KERNEL_SIZE;
  pB3 = pB2 + GBLAS_KERNEL_SIZE;

  stA = M_A2 + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;  
  stB = M_B2 + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  
  pC0       = C;
  pC1       = pC0 + ldc;
  
  do
  {
    do
    {
      B0 = _mm_load_ps(pB0);                            pB0 += 4;
      B1 = _mm_load_ps(pB1);                            pB1 += 4;
      B2 = _mm_load_ps(pB2);                            pB2 += 4;
      B3 = _mm_load_ps(pB3);                            pB3 += 4;
      
      A0 = _mm_load_ps(pA0);                            pA0 += 4;
      C0_0 = _mm_mul_ps(B0, A0);
      C0_1 = _mm_mul_ps(B1, A0);
      C0_2 = _mm_mul_ps(B2, A0);
      C0_3 = _mm_mul_ps(B3, A0); 
      
      A0 = _mm_load_ps(pA1);                            pA1 += 4;
      C1_0 = _mm_mul_ps(B0, A0);
      C1_1 = _mm_mul_ps(B1, A0);
      C1_2 = _mm_mul_ps(B2, A0);
      C1_3 = _mm_mul_ps(B3, A0);
      
      for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k; k--)        // V.3
        // for (int k = (GBLAS_KERNEL_SIZE >> 2) - 1; k > 0; k--)     // V.2
        //for (int k = 1; k < K_SSE; k++) // cols A - rows B          // V.1
      {
        B0 = _mm_load_ps(pB0);                            pB0 += 4;
        B1 = _mm_load_ps(pB1);                            pB1 += 4;
        B2 = _mm_load_ps(pB2);                            pB2 += 4;
        B3 = _mm_load_ps(pB3);                            pB3 += 4;
        
        A0 = _mm_load_ps(pA0);                            pA0 += 4;
        b = B0;
        b = _mm_mul_ps(b, A0);
        C0_0 = _mm_add_ps(C0_0, b);
        b = B1;
        b = _mm_mul_ps(b, A0);
        C0_1 = _mm_add_ps(C0_1, b);
        b = B2;
        b = _mm_mul_ps(b, A0);
        C0_2 = _mm_add_ps(C0_2, b);
        b = B3;
        b = _mm_mul_ps(b, A0);
        C0_3 = _mm_add_ps(C0_3, b);
        
        A0 = _mm_load_ps(pA1);                            pA1 += 4;
        B0 = _mm_mul_ps(B0, A0);
        C1_0 = _mm_add_ps(C1_0, B0);
        B1 = _mm_mul_ps(B1, A0);
        C1_1 = _mm_add_ps(C1_1, B1);
        B2 = _mm_mul_ps(B2, A0);
        C1_2 = _mm_add_ps(C1_2, B2);
        B3 = _mm_mul_ps(B3, A0);
        C1_3 = _mm_add_ps(C1_3, B3);
      }
      
      // --- horizontal ADD ---     
      C0_0 = _mm_hadd_ps(C0_0, C0_1);
      C0_2 = _mm_hadd_ps(C0_2, C0_3);
      C0_0 = _mm_hadd_ps(C0_0, C0_2);
      // ---
      
      // --- horizontal ADD ---     
      C1_0 = _mm_hadd_ps(C1_0, C1_1);
      C1_2 = _mm_hadd_ps(C1_2, C1_3);
      C1_0 = _mm_hadd_ps(C1_0, C1_2);
      // ---
      
      if ( beta != 0.0f )
      {
        C0_3 = _mm_loadu_ps(pC0);     // UN-aligned LOAD
        C0_3 = _mm_mul_ps(C0_3, BETA);
        C0_0 = _mm_add_ps(C0_0, C0_3);
        
        C1_3 = _mm_loadu_ps(pC1);     // UN-aligned LOAD
        C1_3 = _mm_mul_ps(C1_3, BETA);
        C1_0 = _mm_add_ps(C1_0, C1_3);
      }
      
      // Unaligned store of output
      _mm_storeu_ps(pC0, C0_0);
      _mm_storeu_ps(pC1, C1_0);
      
      pA0 -= GBLAS_KERNEL_SIZE;
      pA1 = pA0 + GBLAS_KERNEL_SIZE;
      
      pB0 += GBLAS_KERNEL_SIZE*3;    
      pB1 = pB0 + GBLAS_KERNEL_SIZE;
      pB2 = pB1 + GBLAS_KERNEL_SIZE;
      pB3 = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0 += 4;
      pC1 += 4;
    }
    while ( pB0 != stB );
    
    pA0 += GBLAS_KERNEL_SIZE*2;              //  next 2 rows
    pA1 = pA0 + GBLAS_KERNEL_SIZE;           //  next 2 rows
    
    pB0 = M_B2;                                //  turn back to the begin
    pB1 = pB0 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB2 = pB1 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    pB3 = pB2 + GBLAS_KERNEL_SIZE;           //  turn back to the begin
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);      //  next 2 rows
    pC1 = pC0 + ldc;                         //  next 2 rows
  }
  while ( pA0 != stA );
}