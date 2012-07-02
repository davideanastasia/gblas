/*
 *  gblas_kernels_packed.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 05/08/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include "gblas_kernels.h"
#include "gblas_matrix_utils.h"

#include <cmath>
#include <stdio.h>

#define K_p            (GBLAS_KERNEL_SIZE >> 1)
#define K_LOOPS        (K_p >> 2)

void KERNEL_p_sgemm_v1_r3(const int M, const int N,
                          const int K, const float alpha, const float *A,
                          const int lda, const float *B, const int ldb,
                          const float beta, float *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  __m128 cmp, disp; // unpacking!
  const __m128 __beta     =   _mm_set1_ps(beta);
  
  const float TIGHT_PACKING_A_MAX = 2*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const float INV_EPS     =   ceil(2*TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  //const __m128  wellsee   =   _mm_set1_ps(TIGHT_PACKING_A_MAX);
  
  //float *A_p = (float*)_mm_malloc(sizeof(float)*(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE, 16);
  //float *B_p = (float*)_mm_malloc(sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE, 16);
  
  float A_p[(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  const __m128 __eps      = _mm_set1_ps(EPS);
  const __m128 __inv_eps  = _mm_set1_ps(INV_EPS);
  
  for (int i = 0, ih = 0; i < GBLAS_KERNEL_SIZE; i+=2, ih++)
  {
    for (int k = 0; k < GBLAS_KERNEL_SIZE; k+=4)
    {
      // quantization and packing of A
      cmp  = _mm_load_ps(&A[(i+1)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qa.quantize_sample(cmp);
      cmp  = _mm_mul_ps(cmp, __eps);
      
      disp = _mm_load_ps(&A[i*GBLAS_KERNEL_SIZE + k]);
      disp = Qa.quantize_sample(disp);
      
      cmp  = _mm_add_ps(cmp, disp);
      _mm_store_ps(&A_p[ih*GBLAS_KERNEL_SIZE + k], cmp);
      
      // quantization of B
      cmp  = _mm_load_ps(&B[i*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_ps(&B_p[i*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_ps(&B[(i+1)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_ps(&B_p[(i+1)*GBLAS_KERNEL_SIZE + k], cmp);
    }
  }
  
#ifdef DEBUG_PRINT
  cout << "A_p = "; print_matrix_matlab_notation(A_p, (GBLAS_KERNEL_SIZE >> 1), GBLAS_KERNEL_SIZE);
  cout << "B_p = "; print_matrix_matlab_notation(B_p, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE);
#endif
  
  /* Packing */
  //  for (int i = 0, ih = 0; i < GBLAS_KERNEL_SIZE; i+=2, ih++)
  //  {
  //    for (int k = 0; k < GBLAS_KERNEL_SIZE; k++)
  //    {
  //      A_p[ih*GBLAS_KERNEL_SIZE+k] = A_p[i*GBLAS_KERNEL_SIZE+k] + eps*A_p[i*GBLAS_KERNEL_SIZE + k + GBLAS_KERNEL_SIZE];
  //    }
  //  }
  /* End Packing */
  
  __m128 mm_A0, mm_A1;
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;  // temp
  
  __m128 mm_p00, mm_p01, mm_p02, mm_p03, mm_p10, mm_p11, mm_p12, mm_p13;
  
  const float* stB = B_p + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const float* stA = A_p + GBLAS_KERNEL_SIZE*(GBLAS_KERNEL_SIZE >> 1);
  
  const float* pB0 = B_p;
  const float* pB1 = pB0 + GBLAS_KERNEL_SIZE;
  const float* pB2 = pB1 + GBLAS_KERNEL_SIZE;
  const float* pB3 = pB2 + GBLAS_KERNEL_SIZE;
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = pA_p0 + GBLAS_KERNEL_SIZE;
  
  float* pC0       = C;
  float* pC1       = pC0 + ldc;
  float* pC2       = pC1 + ldc;
  float* pC3       = pC2 + ldc;
  
  //  const int M_p       = (GBLAS_KERNEL_SIZE >> 1); // GBLAS_KERNEL_SIZE/2  
  //for (int i = (M_p >> 1); i; i--)  
  do
  {
    //for (int j = (GBLAS_KERNEL_SIZE >> 2); j ; j-- )
    do
    {
      B0 = _mm_load_ps(pB0);
      pB0 += 4;
      B1 = _mm_load_ps(pB1);
      pB1 += 4;
      B2 = _mm_load_ps(pB2);
      pB2 += 4;
      B3 = _mm_load_ps(pB3);
      pB3 += 4;
      
      mm_A0 = _mm_load_ps(pA_p0);
      pA_p0 += 4;
      mm_p00 = B0; 
      mm_p00 = _mm_mul_ps(mm_p00, mm_A0);
      mm_p01 = B1;
      mm_p01 = _mm_mul_ps(mm_p01, mm_A0);
      mm_p02 = B2;
      mm_p02 = _mm_mul_ps(mm_p02, mm_A0);
      mm_p03 = B3;
      mm_p03 = _mm_mul_ps(mm_p03, mm_A0);
      
      mm_A1 = _mm_load_ps(pA_p1);
      pA_p1 += 4;
      mm_p10 = B0; 
      mm_p10 = _mm_mul_ps(mm_p10, mm_A1);
      mm_p11 = B1;
      mm_p11 = _mm_mul_ps(mm_p11, mm_A1);
      mm_p12 = B2;
      mm_p12 = _mm_mul_ps(mm_p12, mm_A1);
      mm_p13 = B3;
      mm_p13 = _mm_mul_ps(mm_p13, mm_A1);
      
      for (int k = ((GBLAS_KERNEL_SIZE >> 2)-1); k ; k-- )
      {
        B0 = _mm_load_ps(pB0);
        pB0 += 4;
        B1 = _mm_load_ps(pB1);
        pB1 += 4;
        B2 = _mm_load_ps(pB2);
        pB2 += 4;
        B3 = _mm_load_ps(pB3);
        pB3 += 4;
        
        mm_A0 = _mm_load_ps(pA_p0);
        pA_p0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mm_A0);
        mm_p00 = _mm_add_ps(mm_p00, b0);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mm_A0);
        mm_p01 = _mm_add_ps(mm_p01, b1);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mm_A0);
        mm_p02 = _mm_add_ps(mm_p02, b2);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mm_A0);
        mm_p03 = _mm_add_ps(mm_p03, b3);
        
        mm_A1 = _mm_load_ps(pA_p1);
        pA_p1 += 4;        
        b0 = B0;
        b0 = _mm_mul_ps(b0, mm_A1);
        mm_p10 = _mm_add_ps(mm_p10, b0);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mm_A1);
        mm_p11 = _mm_add_ps(mm_p11, b1);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mm_A1);
        mm_p12 = _mm_add_ps(mm_p12, b2);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mm_A1);
        mm_p13 = _mm_add_ps(mm_p13, b3);
      }
      
      // --- horizontal ADD ---     
      mm_p00 = _mm_hadd_ps(mm_p00, mm_p01);
      mm_p02 = _mm_hadd_ps(mm_p02, mm_p03);
      mm_p00 = _mm_hadd_ps(mm_p00, mm_p02);
      // ---
      
      // unpacking #1
      //mm_p01 = mm_full_round_v2(mm_p00);                          //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_ps(mm_p00, _MM_ZERO_S);
      disp     = _mm_and_ps(cmp, _MM_MASK_ONE_S);
      disp     = _mm_sub_ps(disp, _MM_ZERO_DOT_FIVE_S);
      mm_p01   = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(mm_p00, disp)));
      
      b0 = gblas_quantizer::dequantize_sample(mm_p01, DE_Q_FACTOR);
      
      mm_p02 = _mm_loadu_ps(pC0);                                    // &C[i*KERNEL_SIZE + j]
      mm_p02 = _mm_mul_ps(mm_p02, __beta);                          // C[i*N + j] *= beta;
      mm_p02 = _mm_add_ps(mm_p02, b0);
      // C[i*N + j] += sample_i;
      _mm_storeu_ps(pC0, mm_p02);                                    // &C[i*KERNEL_SIZE + j]
      
      mm_p00 = _mm_mul_ps(_mm_sub_ps(mm_p00, mm_p01), __inv_eps);   //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                          //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_ps(mm_p00, _MM_ZERO_S);
      disp     = _mm_and_ps(cmp, _MM_MASK_ONE_S);
      disp     = _mm_sub_ps(disp, _MM_ZERO_DOT_FIVE_S);      
      mm_p01   = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(mm_p00, disp)));
      b0   = gblas_quantizer::dequantize_sample(mm_p01, DE_Q_FACTOR);
      
      
      mm_p02 = _mm_loadu_ps(pC1);
      mm_p02 = _mm_mul_ps(mm_p02, __beta);                          // C[(i+1)*N + j] *= beta;
      mm_p02 = _mm_add_ps(mm_p02, b0);
      // C[i*N + j] += sample_i;
      _mm_storeu_ps(pC1, mm_p02);                                    // &C[(i+1)*KERNEL_SIZE + j]
      
      // --- horizontal ADD ---     
      mm_p10 = _mm_hadd_ps(mm_p10, mm_p11);
      mm_p12 = _mm_hadd_ps(mm_p12, mm_p13);
      mm_p10 = _mm_hadd_ps(mm_p10, mm_p12);
      // ---
      
      // unpacking #2
      //mm_p11 = mm_full_round_v2(mm_p10);                          //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_ps(mm_p10, _MM_ZERO_S);
      disp     = _mm_and_ps(cmp, _MM_MASK_ONE_S);
      disp     = _mm_sub_ps(disp, _MM_ZERO_DOT_FIVE_S);      
      mm_p11   = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(mm_p10, disp)));
      b1   = gblas_quantizer::dequantize_sample(mm_p11, DE_Q_FACTOR);
      
      mm_p12 = _mm_loadu_ps(pC2);
      mm_p12 = _mm_mul_ps(mm_p12, __beta);                          // C[i*N + j] *= beta;
      mm_p12 = _mm_add_ps(mm_p12, b1);
      // C[i*N + j] += sample_i;
      _mm_storeu_ps(pC2, mm_p12);                                    // &C[(i+2)*KERNEL_SIZE + j]
      
      mm_p10 = _mm_mul_ps(_mm_sub_ps(mm_p10, mm_p11), __inv_eps);     //sample_d = (sample_d - sample_i)/eps;
      
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_ps(mm_p10, _MM_ZERO_S);
      disp     = _mm_and_ps(cmp, _MM_MASK_ONE_S);
      disp     = _mm_sub_ps(disp, _MM_ZERO_DOT_FIVE_S);
      mm_p11   = _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(mm_p10, disp)));
      b1  = gblas_quantizer::dequantize_sample(mm_p11, DE_Q_FACTOR);
      
      
      mm_p12 = _mm_loadu_ps(pC3);
      mm_p12 = _mm_mul_ps(mm_p12, __beta);                          // C[(i+1)*N + j] *= beta;
      mm_p12 = _mm_add_ps(mm_p12, b1);
      // C[i*N + j] += sample_i;
      _mm_storeu_ps(pC3, mm_p12);
      
      pA_p0  -= GBLAS_KERNEL_SIZE;
      pA_p1  -= GBLAS_KERNEL_SIZE;
      
      pB0    += GBLAS_KERNEL_SIZE*3;
      pB1    = pB0 + GBLAS_KERNEL_SIZE;
      pB2    = pB1 + GBLAS_KERNEL_SIZE;
      pB3    = pB2 + GBLAS_KERNEL_SIZE;
      
      pC0    += 4;
      pC1    += 4;
      pC2    += 4;
      pC3    += 4;
    }
    while ( pB0 != stB );
    
    pA_p0 += GBLAS_KERNEL_SIZE*2;
    pA_p1 = pA_p0 + GBLAS_KERNEL_SIZE;
    
    pB0 = B_p;
    pB1 = B_p + GBLAS_KERNEL_SIZE;
    pB2 = B_p + GBLAS_KERNEL_SIZE*2;
    pB3 = B_p + GBLAS_KERNEL_SIZE*3;
    
    pC0 -= GBLAS_KERNEL_SIZE;                 //  roll back!
    pC0 += ldc*4;                             //  next 4 rows
    pC1 = pC0 + ldc;                          //  next 4 rows
    pC2 = pC1 + ldc;                          //  next 4 rows
    pC3 = pC2 + ldc;                          //  next 4 rows
    
  }
  while ( pA_p0 != stA );
  
  //_mm_free(A_p);
}


void KERNEL_p_sgemm_v2_r5(const int M, const int N,
                          const int K, const float alpha, const float *A,
                          const int lda, const float *B, const int ldb,
                          const float beta, float *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb)
{ 
  const __m128 __beta     =   _mm_set1_ps(beta);
  
  const float TIGHT_PACKING_A_MAX = 2*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const float INV_EPS     =   ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  float A_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128 acc0, acc1;
  
  const __m128 a_factor = _mm_set_ps(EPS, 1.0f, EPS, 1.0f);
  const __m128 b_factor = _mm_set_ps(INV_EPS, 1.0f, INV_EPS, 1.0f);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, a_factor);
    
    acc1 = _mm_load_ps(&A[k + 4]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&A_p[kh], acc0);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, b_factor);
    
    acc1 = _mm_load_ps(&B[k + 4]);
    acc1 = Qb.quantize_sample(acc1);    
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&B_p[kh], acc0);
  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 mmA0, mmA1;
  
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;
  
  __m128 mmC00, mmC01, mmC02, mmC03, mmC10, mmC11, mmC12, mmC13; 
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = pA_p0 + K_p;
  
  const float* pB_p0 = B_p;
  const float* pB_p1 = pB_p0 + K_p;
  const float* pB_p2 = pB_p1 + K_p;
  const float* pB_p3 = pB_p2 + K_p;
  
  float* pC = C;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
      B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
      B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
      B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
      
      mmA0 = _mm_load_ps(pA_p0);
      pA_p0 += 4;
      mmC00 = B0;
      mmC00 = _mm_mul_ps(mmC00, mmA0);
      mmC01 = B1;
      mmC01 = _mm_mul_ps(mmC01, mmA0);
      mmC02 = B2;
      mmC02 = _mm_mul_ps(mmC02, mmA0);
      mmC03 = B3;
      mmC03 = _mm_mul_ps(mmC03, mmA0);
      
      mmA1 = _mm_load_ps(pA_p1);
      pA_p1 += 4;
      mmC10 = B0;
      mmC10 = _mm_mul_ps(mmC10, mmA1);
      mmC11 = B1;
      mmC11 = _mm_mul_ps(mmC11, mmA1);
      mmC12 = B2;
      mmC12 = _mm_mul_ps(mmC12, mmA1);
      mmC13 = B3;
      mmC13 = _mm_mul_ps(mmC13, mmA1);
      
      for (int k = K_LOOPS-1; k; k--)  
      {
        B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
        B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
        B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
        B3 = _mm_load_ps(pB_p3); pB_p3 += 4; 
        
        mmA0 = _mm_load_ps(pA_p0);
        pA_p0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mmA0);
        mmC00 = _mm_add_ps(b0, mmC00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mmA0);
        mmC01 = _mm_add_ps(b1, mmC01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mmA0);
        mmC02 = _mm_add_ps(b2, mmC02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mmA0);
        mmC03 = _mm_add_ps(b3, mmC03);
        
        mmA1 = _mm_load_ps(pA_p1);
        pA_p1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mmA1);
        mmC10 = _mm_add_ps(b0, mmC10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mmA1);
        mmC11 = _mm_add_ps(b1, mmC11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mmA1);
        mmC12 = _mm_add_ps(b2, mmC12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mmA1);
        mmC13 = _mm_add_ps(b3, mmC13);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---     
      mmC00 = _mm_hadd_ps(mmC00, mmC01);
      mmC02 = _mm_hadd_ps(mmC02, mmC03);
      mmC00 = _mm_hadd_ps(mmC00, mmC02);
      // ---
      
      fast_unpack_tight_v2(mmC00, EPS, INV_EPS);
      
      mmC00 = gblas_quantizer::dequantize_sample(mmC00, DE_Q_FACTOR);
      
      // SECOND ROW
      // --- horizontal ADD ---     
      mmC10 = _mm_hadd_ps(mmC10, mmC11);
      mmC12 = _mm_hadd_ps(mmC12, mmC13);
      mmC10 = _mm_hadd_ps(mmC10, mmC12);
      // ---
      
      fast_unpack_tight_v2(mmC10, EPS, INV_EPS);
      
      mmC10 = gblas_quantizer::dequantize_sample(mmC10, DE_Q_FACTOR);
      
      if ( beta != 0.0f )
      {
        acc0 = _mm_loadu_ps(&pC[0]);
        acc0 = _mm_mul_ps(acc0, __beta);
        mmC00  = _mm_add_ps(acc0, mmC00);
        
        acc1 = _mm_loadu_ps(&pC[ldc]);
        acc1 = _mm_mul_ps(acc1, __beta);
        mmC10  = _mm_add_ps(acc1, mmC10);
      }
      
      _mm_storeu_ps(&pC[0],      mmC00);
      _mm_storeu_ps(&pC[ldc],    mmC10);
      
      pA_p0 -= K_p;
      pA_p1 = pA_p0 + K_p;
      
      pB_p0 += K_p*3;
      pB_p1 = pB_p0 + K_p;
      pB_p2 = pB_p1 + K_p;
      pB_p3 = pB_p2 + K_p;
      
      pC += 4;
    }
    pA_p0 += K_p*2;
    pA_p1 += K_p*2;
    
    pB_p0 = B_p;
    pB_p1 = pB_p0 + K_p;
    pB_p2 = pB_p1 + K_p;
    pB_p3 = pB_p2 + K_p;
    
    pC   += (ldc*2 - GBLAS_KERNEL_SIZE);
  }
}

void KERNEL_p_sgemm_v2_r5_EC(const int M, const int N,
                             const int K, const float alpha, const float *A,
                             const int lda, const float *B, const int ldb,
                             const float beta, float *C, const int ldc,
                             gblas_quantizer& Qa, gblas_quantizer& Qb)
{ 
  const __m128 __beta     =   _mm_set1_ps(beta);
  
  const float TIGHT_PACKING_A_MAX = 2*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const float INV_EPS     =   ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  float A_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p_EC[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128 acc0, acc1, acc0_EC, acc1_EC;
  
  const __m128 scaling_factor = _mm_set1_ps(INV_EPS);
  const __m128 a_factor = _mm_set_ps(EPS, 1.0f, EPS, 1.0f);
  const __m128 b_factor = _mm_set_ps(INV_EPS, 1.0f, INV_EPS, 1.0f);
  const __m128 b_factor_EC = _mm_set_ps(1.0f, EPS, 1.0f, EPS);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, a_factor);
    
    acc1 = _mm_load_ps(&A[k + 4]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&A_p[kh], acc0);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0_EC = _mm_mul_ps(acc0, b_factor_EC); // this one first because the next line will modify the value of acc0
    acc0 = _mm_mul_ps(acc0, b_factor);
    
    acc1 = _mm_load_ps(&B[k + 4]);
    acc1 = Qb.quantize_sample(acc1);
    acc1_EC = _mm_mul_ps(acc1, b_factor_EC); // this one first because the next line will modify the value of acc0
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    acc0_EC = _mm_hadd_ps(acc0_EC, acc1_EC);
    
    _mm_store_ps(&B_p[kh], acc0);
    _mm_store_ps(&B_p_EC[kh], acc0_EC);
  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 A0, A1;
  
  __m128 b0, b1, b2, b3;
  
  __m128 B0, B1, B2, B3;
  __m128 B0_EC, B1_EC, B2_EC, B3_EC;
  
  __m128 C00, C01, C02, C03, C10, C11, C12, C13; 
  __m128 C00_EC, C01_EC, C02_EC, C03_EC,
  C10_EC, C11_EC, C12_EC, C13_EC;
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = A_p + K_p;
  
  const float* pB_p0 = B_p;
  const float* pB_p1 = pB_p0 + K_p;
  const float* pB_p2 = pB_p1 + K_p;
  const float* pB_p3 = pB_p2 + K_p;
  
  const float* pB_p0_EC = B_p_EC;
  const float* pB_p1_EC = pB_p0_EC + K_p;
  const float* pB_p2_EC = pB_p1_EC + K_p;
  const float* pB_p3_EC = pB_p2_EC + K_p;
  
  float* pC = C;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      // Normal Section
      B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
      B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
      B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
      B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
      // Error Control Section
      B0_EC = _mm_load_ps(pB_p0_EC); pB_p0_EC += 4;
      B1_EC = _mm_load_ps(pB_p1_EC); pB_p1_EC += 4;
      B2_EC = _mm_load_ps(pB_p2_EC); pB_p2_EC += 4;
      B3_EC = _mm_load_ps(pB_p3_EC); pB_p3_EC += 4;
      
      // Trailer
      A0 = _mm_load_ps(pA_p0);
      pA_p0 += 4;
      // Normal Section
      C00 = B0;
      C00 = _mm_mul_ps(C00, A0);
      C01 = B1;
      C01 = _mm_mul_ps(C01, A0);
      C02 = B2;
      C02 = _mm_mul_ps(C02, A0);
      C03 = B3;
      C03 = _mm_mul_ps(C03, A0);
      // Error Control Section
      C00_EC = B0_EC;
      C00_EC = _mm_mul_ps(C00_EC, A0);
      C01_EC = B1_EC;
      C01_EC = _mm_mul_ps(C01_EC, A0);
      C02_EC = B2_EC;
      C02_EC = _mm_mul_ps(C02_EC, A0);
      C03_EC = B3_EC;
      C03_EC = _mm_mul_ps(C03_EC, A0);
      
      // Trailer
      A1 = _mm_load_ps(pA_p1);
      pA_p1 += 4;
      // Normal Section
      C10 = B0;
      C10 = _mm_mul_ps(C10, A1);
      C11 = B1;
      C11 = _mm_mul_ps(C11, A1);
      C12 = B2;
      C12 = _mm_mul_ps(C12, A1);
      C13 = B3;
      C13 = _mm_mul_ps(C13, A1);
      // Error Control Section
      C10_EC = B0_EC;
      C10_EC = _mm_mul_ps(C10_EC, A1);
      C11_EC = B1_EC;
      C11_EC = _mm_mul_ps(C11_EC, A1);
      C12_EC = B2_EC;
      C12_EC = _mm_mul_ps(C12_EC, A1);
      C13_EC = B3_EC;
      C13_EC = _mm_mul_ps(C13_EC, A1);
      
      for (int k = K_LOOPS-1; k; k--)  
      {
        // Normal Section
        B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
        B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
        B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
        B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
        // Error Control Section
        B0_EC = _mm_load_ps(pB_p0_EC); pB_p0_EC += 4;
        B1_EC = _mm_load_ps(pB_p1_EC); pB_p1_EC += 4;
        B2_EC = _mm_load_ps(pB_p2_EC); pB_p2_EC += 4;
        B3_EC = _mm_load_ps(pB_p3_EC); pB_p3_EC += 4;
        
        // Trailer
        A0 = _mm_load_ps(pA_p0);
        pA_p0 += 4;
        // Normal Section
        b0 = B0;
        b0 = _mm_mul_ps(b0, A0);
        C00 = _mm_add_ps(b0, C00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A0);
        C01 = _mm_add_ps(b1, C01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A0);
        C02 = _mm_add_ps(b2, C02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A0);
        C03 = _mm_add_ps(b3, C03);
        // Error Control Section
        b0 = B0_EC;
        b0 = _mm_mul_ps(b0, A0);
        C00_EC = _mm_add_ps(b0, C00_EC);
        b1 = B1_EC;
        b1 = _mm_mul_ps(b1, A0);
        C01_EC = _mm_add_ps(b1, C01_EC);
        b2 = B2_EC;
        b2 = _mm_mul_ps(b2, A0);
        C02_EC = _mm_add_ps(b2, C02_EC);
        b3 = B3_EC;
        b3 = _mm_mul_ps(b3, A0);
        C03_EC = _mm_add_ps(b3, C03_EC);
        
        // Trailer
        A1 = _mm_load_ps(pA_p1);
        pA_p1 += 4;
        // Normal Section
        b0 = B0;
        b0 = _mm_mul_ps(b0, A1);
        C10 = _mm_add_ps(b0, C10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A1);
        C11 = _mm_add_ps(b1, C11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A1);
        C12 = _mm_add_ps(b2, C12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A1);
        C13 = _mm_add_ps(b3, C13);
        // Error Control Section
        b0 = B0_EC;
        b0 = _mm_mul_ps(b0, A1);
        C10_EC = _mm_add_ps(b0, C10_EC);
        b1 = B1_EC;
        b1 = _mm_mul_ps(b1, A1);
        C11_EC = _mm_add_ps(b1, C11_EC);
        b2 = B2_EC;
        b2 = _mm_mul_ps(b2, A1);
        C12_EC = _mm_add_ps(b2, C12_EC);
        b3 = B3_EC;
        b3 = _mm_mul_ps(b3, A1);
        C13_EC = _mm_add_ps(b3, C13_EC);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---
      // Normal Section
      C00 = _mm_hadd_ps(C00, C01);
      C02 = _mm_hadd_ps(C02, C03);
      C00 = _mm_hadd_ps(C00, C02);
      // Error Control Section
      C00_EC = _mm_hadd_ps(C00_EC, C01_EC);
      C02_EC = _mm_hadd_ps(C02_EC, C03_EC);
      C00_EC = _mm_hadd_ps(C00_EC, C02_EC);      
      // ---
      
      // Error Control Cross Check
      //
      //
      // NOTE : be carefully here, I'm doing something that I should not be doing
      // just to make a true measurement of the processing time
      //fast_unpack_tight_v2(C00, EPS, INV_EPS);
      //C00 = gblas_quantizer::dequantize_sample(C00, DE_Q_FACTOR);
      
      C00_EC = _mm_mul_ps(C00_EC, scaling_factor);
      
      // NOTE : necessary to make the compiler think that I need both!
      C00_EC = _mm_sub_ps(C00_EC, C00);
      
      fast_unpack_tight_v2(C00_EC, EPS, INV_EPS);
      C00_EC = gblas_quantizer::dequantize_sample(C00_EC, DE_Q_FACTOR);
      
      // SECOND ROW
      // --- horizontal ADD ---
      // Normal Section
      C10 = _mm_hadd_ps(C10, C11);
      C12 = _mm_hadd_ps(C12, C13);
      C10 = _mm_hadd_ps(C10, C12);
      // Error Control Section
      C10_EC = _mm_hadd_ps(C10_EC, C11_EC);
      C12_EC = _mm_hadd_ps(C12_EC, C13_EC);
      C10_EC = _mm_hadd_ps(C10_EC, C12_EC);
      // ---
      
      // Error Control Cross Check
      //
      //
      
      
      // NOTE : be carefully here, I'm doing something that I should not be doing
      // just to make a true measurement of the processing time
      //fast_unpack_tight_v2(C10, EPS, INV_EPS);
      //C10 = gblas_quantizer::dequantize_sample(C10, DE_Q_FACTOR);
      
      C10_EC = _mm_mul_ps(C10_EC, scaling_factor);   
      
      // NOTE : necessary to make the compiler think that I need both!      
      C10_EC = _mm_sub_ps(C10_EC, C10);
      
      fast_unpack_tight_v2(C10_EC, EPS, INV_EPS);
      C10_EC = gblas_quantizer::dequantize_sample(C10_EC, DE_Q_FACTOR);
      
      
      
      if ( beta != 0.0f )
      {
        acc0 = _mm_loadu_ps(&pC[0]);
        acc0 = _mm_mul_ps(acc0, __beta);
        //C00  = _mm_add_ps(acc0, C00);
        C00_EC  = _mm_add_ps(acc0, C00_EC);
        
        acc1 = _mm_loadu_ps(&pC[ldc]);
        acc1 = _mm_mul_ps(acc1, __beta);
        //C10  = _mm_add_ps(acc1, C10);
        C10_EC  = _mm_add_ps(acc1, C10_EC);
      }
      
      //_mm_storeu_ps(&pC[0],      C00);
      //_mm_storeu_ps(&pC[ldc],    C10);
      
      _mm_storeu_ps(&pC[0],      C00_EC);
      _mm_storeu_ps(&pC[ldc],    C10_EC);
      
      pA_p0 -= K_p;
      pA_p1 -= K_p;
      
      pB_p0 += K_p*3;
      pB_p1 = pB_p0 + K_p;
      pB_p2 = pB_p1 + K_p;
      pB_p3 = pB_p2 + K_p;
      
      pB_p0_EC += K_p*3;
      pB_p1_EC = pB_p0_EC + K_p;
      pB_p2_EC = pB_p1_EC + K_p;
      pB_p3_EC = pB_p2_EC + K_p;
      
      pC += 4;
    }
    pA_p0 += K_p*2;
    pA_p1 = pA_p0 + K_p;
    
    pB_p0 = B_p;
    pB_p1 = pB_p0 + K_p;
    pB_p2 = pB_p1 + K_p;
    pB_p3 = pB_p2 + K_p;
    
    pB_p0_EC = B_p_EC;
    pB_p1_EC = pB_p0_EC + K_p;
    pB_p2_EC = pB_p1_EC + K_p;
    pB_p3_EC = pB_p2_EC + K_p;
    
    pC   += (ldc*2 - GBLAS_KERNEL_SIZE);
  }
}

void KERNEL_p_sgemm_v2_r5_EC_v2(const int M, const int N,
                                const int K, const float alpha, const float *A,
                                const int lda, const float *B, const int ldb,
                                const float beta, float *C, const int ldc,
                                gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  //printf(".");
  
  const __m128 __beta     =   _mm_set1_ps(beta);
  
  const float TIGHT_PACKING_A_MAX = 2.0f*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  //const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const float INV_EPS     =   ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  const float INV_EPS_EC  =   ceil(2.0f*TIGHT_PACKING_A_MAX + DELTA);
  const float EPS_EC      =   1.0f/INV_EPS_EC;
  
  float A_p[(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  float A_p_EC[(GBLAS_KERNEL_SIZE >> 2)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p_EC[(GBLAS_KERNEL_SIZE >> 2)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  // packing
  __m128 acc0, acc1, acc2, acc3;
  __m128 acc4, acc5;
  __m128 o1, o2, o3;
  
  const __m128 A_FACTOR = _mm_set_ps(EPS, 1.0f, EPS, 1.0f);
  const __m128 B_FACTOR = _mm_set_ps(INV_EPS, 1.0f, INV_EPS, 1.0f);
  const __m128 A_FACTOR_EC = _mm_set_ps(EPS_EC, 1.0f, EPS_EC, 1.0f);
  const __m128 B_FACTOR_EC = _mm_set_ps(INV_EPS_EC, 1.0f, INV_EPS_EC, 1.0f);
  //const __m128 EPS_FACTOR = _mm_set1_ps(EPS);
  //const __m128 INV_EPS_FACTOR = _mm_set1_ps(INV_EPS);  
  
  for ( int k = 0, kh = 0, kh2 = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=16, kh+=8, kh2+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc1 = _mm_load_ps(&A[k + 4]);
    acc2 = _mm_load_ps(&A[k + 8]);
    acc3 = _mm_load_ps(&A[k + 12]);
    
    acc0 = Qa.quantize_sample(acc0);
    acc1 = Qa.quantize_sample(acc1);
    acc2 = Qa.quantize_sample(acc2);
    acc3 = Qa.quantize_sample(acc3);
    
    // A_p_EC
    acc4 = _mm_hadd_ps(acc0, acc1);
    acc5 = _mm_hadd_ps(acc2, acc3);
    
    acc4 = _mm_mul_ps(acc4, A_FACTOR_EC);
    acc5 = _mm_mul_ps(acc5, A_FACTOR_EC);
    
    acc4 = _mm_hadd_ps(acc4, acc5); //*
    
    _mm_store_ps(&A_p_EC[kh2], acc4);
    
    // A_p
    acc0 = _mm_mul_ps(acc0, A_FACTOR);
    acc1 = _mm_mul_ps(acc1, A_FACTOR);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    _mm_store_ps(&A_p[kh], acc0);
    
    acc2 = _mm_mul_ps(acc2, A_FACTOR);
    acc3 = _mm_mul_ps(acc3, A_FACTOR);
    
    acc2 = _mm_hadd_ps(acc2, acc3);
    _mm_store_ps(&A_p[kh + 4], acc2);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc1 = _mm_load_ps(&B[k + 4]);
    acc2 = _mm_load_ps(&B[k + 8]);
    acc3 = _mm_load_ps(&B[k + 12]);
    
    acc0 = Qb.quantize_sample(acc0);
    acc1 = Qb.quantize_sample(acc1);
    acc2 = Qb.quantize_sample(acc2);
    acc3 = Qb.quantize_sample(acc3);
    
    // B_p_EC
    acc4 = _mm_hadd_ps(acc0, acc1);
    acc5 = _mm_hadd_ps(acc2, acc3);
    
    acc4 = _mm_mul_ps(acc4, B_FACTOR_EC);
    acc5 = _mm_mul_ps(acc5, B_FACTOR_EC);
    
    acc4 = _mm_hadd_ps(acc4, acc5); //*
    
    _mm_store_ps(&B_p_EC[kh2], acc4);
    
    // B_p
    acc0 = _mm_mul_ps(acc0, B_FACTOR);
    acc1 = _mm_mul_ps(acc1, B_FACTOR);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    _mm_store_ps(&B_p[kh], acc0);
    
    acc2 = _mm_mul_ps(acc2, B_FACTOR);
    acc3 = _mm_mul_ps(acc3, B_FACTOR);
    
    acc2 = _mm_hadd_ps(acc2, acc3);
    _mm_store_ps(&B_p[kh + 4], acc2);
  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  const int _K_P2 = GBLAS_KERNEL_SIZE >> 1;
  const int _K_P4 = GBLAS_KERNEL_SIZE >> 2;
  
  __m128 A0, A1;
  
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;
  
  __m128 C00, C01, C02, C03;
  __m128 C10, C11, C12, C13; 
  
  // Packed Computation
  const float* pA0 = A_p;
  const float* pA1 = pA0 + _K_P2;
  
  const float* pB0 = B_p;
  const float* pB1 = pB0 + _K_P2;
  const float* pB2 = pB1 + _K_P2;
  const float* pB3 = pB2 + _K_P2;
  
  // Error Correction Computation
  const float* pA0_EC = A_p_EC;
  const float* pA1_EC = pA0_EC + _K_P4;
  
  const float* pB0_EC = B_p_EC;
  const float* pB1_EC = pB0_EC + _K_P4;
  const float* pB2_EC = pB1_EC + _K_P4;
  const float* pB3_EC = pB2_EC + _K_P4;
  
  // Final result
  float* pC0 = C;
  float* pC1 = pC0 + ldc;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A (2 at the same time!)
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  // N // rows B (4 at the same time!)
    {
      // Core computation "classic"
      B0 = _mm_load_ps(pB0); pB0 += 4;
      B1 = _mm_load_ps(pB1); pB1 += 4;
      B2 = _mm_load_ps(pB2); pB2 += 4;
      B3 = _mm_load_ps(pB3); pB3 += 4;
      
      A0 = _mm_load_ps(pA0); pA0 += 4;
      C00 = B0;
      C00 = _mm_mul_ps(C00, A0);
      C01 = B1;
      C01 = _mm_mul_ps(C01, A0);
      C02 = B2;
      C02 = _mm_mul_ps(C02, A0);
      C03 = B3;
      C03 = _mm_mul_ps(C03, A0);
      
      A1 = _mm_load_ps(pA1); pA1 += 4;
      C10 = B0;
      C10 = _mm_mul_ps(C10, A1);
      C11 = B1;
      C11 = _mm_mul_ps(C11, A1);
      C12 = B2;
      C12 = _mm_mul_ps(C12, A1);
      C13 = B3;
      C13 = _mm_mul_ps(C13, A1);
      
      for (int k = (_K_P2 >> 2) -1; k; k--)  
      {
        B0 = _mm_load_ps(pB0); pB0 += 4;
        B1 = _mm_load_ps(pB1); pB1 += 4;
        B2 = _mm_load_ps(pB2); pB2 += 4;
        B3 = _mm_load_ps(pB3); pB3 += 4; 
        
        A0 = _mm_load_ps(pA0); pA0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A0);
        C00 = _mm_add_ps(b0, C00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A0);
        C01 = _mm_add_ps(b1, C01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A0);
        C02 = _mm_add_ps(b2, C02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A0);
        C03 = _mm_add_ps(b3, C03);
        
        A1 = _mm_load_ps(pA1); pA1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A1);
        C10 = _mm_add_ps(b0, C10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A1);
        C11 = _mm_add_ps(b1, C11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A1);
        C12 = _mm_add_ps(b2, C12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A1);
        C13 = _mm_add_ps(b3, C13);
      }
      
      // --- horizontal ADD ---       
      // FIRST ROW
      C00 = _mm_hadd_ps(C00, C01);
      C02 = _mm_hadd_ps(C02, C03);
      acc0 = _mm_hadd_ps(C00, C02);
      // SECOND ROW   
      C10 = _mm_hadd_ps(C10, C11);
      C12 = _mm_hadd_ps(C12, C13);
      acc1 = _mm_hadd_ps(C10, C12);
      // ---
      
      // Core computation "error correction"
      B0 = _mm_load_ps(pB0_EC); pB0_EC += 4;
      B1 = _mm_load_ps(pB1_EC); pB1_EC += 4;
      B2 = _mm_load_ps(pB2_EC); pB2_EC += 4;
      B3 = _mm_load_ps(pB3_EC); pB3_EC += 4;
      
      A0 = _mm_load_ps(pA0_EC); pA0_EC += 4;
      C00 = B0;
      C00 = _mm_mul_ps(C00, A0);
      C01 = B1;
      C01 = _mm_mul_ps(C01, A0);
      C02 = B2;
      C02 = _mm_mul_ps(C02, A0);
      C03 = B3;
      C03 = _mm_mul_ps(C03, A0);
      
      A1 = _mm_load_ps(pA1_EC); pA1_EC += 4;
      C10 = B0;
      C10 = _mm_mul_ps(C10, A1);
      C11 = B1;
      C11 = _mm_mul_ps(C11, A1);
      C12 = B2;
      C12 = _mm_mul_ps(C12, A1);
      C13 = B3;
      C13 = _mm_mul_ps(C13, A1);
      
      for (int k = (_K_P4 >> 2)-1; k; k--)  
      {
        B0 = _mm_load_ps(pB0_EC); pB0_EC += 4;
        B1 = _mm_load_ps(pB1_EC); pB1_EC += 4;
        B2 = _mm_load_ps(pB2_EC); pB2_EC += 4;
        B3 = _mm_load_ps(pB3_EC); pB3_EC += 4; 
        
        A0 = _mm_load_ps(pA0_EC); pA0_EC += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A0);
        C00 = _mm_add_ps(b0, C00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A0);
        C01 = _mm_add_ps(b1, C01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A0);
        C02 = _mm_add_ps(b2, C02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A0);
        C03 = _mm_add_ps(b3, C03);
        
        A1 = _mm_load_ps(pA1_EC); pA1_EC += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A1);
        C10 = _mm_add_ps(b0, C10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A1);
        C11 = _mm_add_ps(b1, C11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A1);
        C12 = _mm_add_ps(b2, C12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A1);
        C13 = _mm_add_ps(b3, C13);
      }
      
      // --- horizontal ADD ---       
      // FIRST ROW
      C00 = _mm_hadd_ps(C00, C01);
      C02 = _mm_hadd_ps(C02, C03);
      acc2 = _mm_hadd_ps(C00, C02);
      // SECOND ROW   
      C10 = _mm_hadd_ps(C10, C11);
      C12 = _mm_hadd_ps(C12, C13);
      acc3 = _mm_hadd_ps(C10, C12);
      // ---
      
      // Previous version
      /*
       // Line 1
       //fast_unpack_tight_v2(acc0, EPS, INV_EPS);
       unpack_complete_tight_v1(acc0, EPS, INV_EPS, o1, o2, o3);
       fast_unpack_tight_v2(acc2, EPS_EC, INV_EPS_EC);    
       
       acc2 = _mm_sub_ps(acc2, o2);
       acc2 = gblas_quantizer::dequantize_sample(acc2, DE_Q_FACTOR);
       // Line 2
       //fast_unpack_tight_v2(acc1, EPS, INV_EPS);
       unpack_complete_tight_v1(acc1, EPS, INV_EPS, o1, o2, o3);
       fast_unpack_tight_v2(acc3, EPS_EC, INV_EPS_EC); 
       acc3 = _mm_sub_ps(acc3, o2);
       acc3 = gblas_quantizer::dequantize_sample(acc3, DE_Q_FACTOR);      
       */
      
      // Second version
      unpack_complete_tight_v1(acc0, EPS, INV_EPS, o1, o2, o3);
      o1 = _mm_add_ps(o1, o2);
      o1 = _mm_add_ps(o1, o3);
      fast_unpack_tight_v2(acc2, EPS_EC, INV_EPS_EC);
      acc2 = _mm_sub_ps(acc2, o1);
      
      //acc2 = gblas_quantizer::dequantize_sample(acc2, DE_Q_FACTOR);
      
      unpack_complete_tight_v1(acc1, EPS, INV_EPS, o1, o2, o3);
      o1 = _mm_add_ps(o1, o2);
      o1 = _mm_add_ps(o1, o3);
      fast_unpack_tight_v2(acc3, EPS_EC, INV_EPS_EC);
      acc3 = _mm_sub_ps(acc3, o1);
      
      //acc3 = gblas_quantizer::dequantize_sample(acc3, DE_Q_FACTOR);
      
      if ( beta != 0.0f )
      {
        // C03 & C13 are just a temporary variable used as accumulator
        acc4 = _mm_loadu_ps(pC0);
        acc4 = _mm_mul_ps(acc4, __beta);
        acc2  = _mm_add_ps(acc2, acc4);
        
        acc5 = _mm_loadu_ps(pC1);
        acc5 = _mm_mul_ps(acc5, __beta);
        acc3  = _mm_add_ps(acc3, acc5);
      }
      
      _mm_storeu_ps(pC0, acc2);
      _mm_storeu_ps(pC1, acc3);
      
      //printf("%f %f %f %f\n", pC0[0], pC0[1], pC0[2], pC0[3]);
      
      // TAIL!
      // Packed Computation
      pA0 -= _K_P2;
      pA1 = pA0 + _K_P2;
      
      pB0 += _K_P2*3;
      pB1 = pB0 + _K_P2;
      pB2 = pB1 + _K_P2;
      pB3 = pB2 + _K_P2;
      
      // Error Correction Computation
      pA0_EC -= _K_P4;
      pA1_EC = pA0_EC + _K_P4;
      
      pB0_EC += _K_P4*3;
      pB1_EC = pB0_EC + _K_P4;
      pB2_EC = pB1_EC + _K_P4;
      pB3_EC = pB2_EC + _K_P4;
      
      // output
      pC0 += 4;
      pC1 += 4;
    }
    // Packed Computation
    pA0 += _K_P2*2;
    pA1 += _K_P2*2;
    
    pB0 = B_p;
    pB1 = pB0 + _K_P2;
    pB2 = pB1 + _K_P2;
    pB3 = pB2 + _K_P2;
    
    // Error Correction Computation
    pA0_EC += _K_P4*2;
    pA1_EC += _K_P4*2;
    
    pB0_EC = B_p_EC;
    pB1_EC = pB0_EC + _K_P4;
    pB2_EC = pB1_EC + _K_P4;
    pB3_EC = pB2_EC + _K_P4;
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);
    pC1 = pC0 + ldc;
  }
}


void KERNEL_p_sgemm_v2_r5_EC_v3(const int M, const int N,
                                const int K, const float alpha, const float *A,
                                const int lda, const float *B, const int ldb,
                                const float beta, float *C, const int ldc,
                                gblas_quantizer& Qa, gblas_quantizer& Qb)
{ 
  const __m128 __beta     =   _mm_set1_ps(beta);
  
  const float TIGHT_PACKING_A_MAX = 2*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const float INV_EPS     =   ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  float A_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float A_p_EC[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p_EC[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128 acc0, acc1, acc0_EC, acc1_EC;
  
  const __m128 SCALE_FACTOR = _mm_set1_ps(0.5f);
  const __m128 a_factor = _mm_set_ps(EPS, 1.0f, EPS, 1.0f);
  const __m128 b_factor = _mm_set_ps(INV_EPS, 1.0f, INV_EPS, 1.0f);
  const __m128 a_factor_EC = _mm_set_ps(EPS, -1.0f, EPS, -1.0f);  
  const __m128 b_factor_EC = _mm_set_ps(INV_EPS, -1.0f, INV_EPS, -1.0f);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0_EC = _mm_mul_ps(acc0, a_factor_EC);
    acc0 = _mm_mul_ps(acc0, a_factor);
    
    acc1 = _mm_load_ps(&A[k + 4]);
    acc1 = Qa.quantize_sample(acc1);
    acc1_EC = _mm_mul_ps(acc1, a_factor_EC);    
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    acc0_EC = _mm_hadd_ps(acc0_EC, acc1_EC);
    
    _mm_store_ps(&A_p[kh], acc0);
    _mm_store_ps(&A_p_EC[kh], acc0_EC);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0_EC = _mm_mul_ps(acc0, b_factor_EC); // this one first because the next line will modify the value of acc0
    acc0 = _mm_mul_ps(acc0, b_factor);
    
    acc1 = _mm_load_ps(&B[k + 4]);
    acc1 = Qb.quantize_sample(acc1);
    acc1_EC = _mm_mul_ps(acc1, b_factor_EC); // this one first because the next line will modify the value of acc0
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    acc0_EC = _mm_hadd_ps(acc0_EC, acc1_EC);
    
    _mm_store_ps(&B_p[kh], acc0);
    _mm_store_ps(&B_p_EC[kh], acc0_EC);
  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 A0, A1;
  
  __m128 b0, b1, b2, b3;
  
  __m128 B0, B1, B2, B3;
  __m128 B0_EC, B1_EC, B2_EC, B3_EC;
  
  __m128 C00, C01, C02, C03, C10, C11, C12, C13; 
  __m128 C00_EC, C01_EC, C02_EC, C03_EC,
  C10_EC, C11_EC, C12_EC, C13_EC;
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = A_p + K_p;
  
  const float* pA_p0_EC = A_p_EC;
  const float* pA_p1_EC = A_p_EC + K_p;
  
  const float* pB_p0 = B_p;
  const float* pB_p1 = pB_p0 + K_p;
  const float* pB_p2 = pB_p1 + K_p;
  const float* pB_p3 = pB_p2 + K_p;
  
  const float* pB_p0_EC = B_p_EC;
  const float* pB_p1_EC = pB_p0_EC + K_p;
  const float* pB_p2_EC = pB_p1_EC + K_p;
  const float* pB_p3_EC = pB_p2_EC + K_p;
  
  float* pC = C;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      // Normal Section
      B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
      B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
      B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
      B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
      // Error Control Section
      B0_EC = _mm_load_ps(pB_p0_EC); pB_p0_EC += 4;
      B1_EC = _mm_load_ps(pB_p1_EC); pB_p1_EC += 4;
      B2_EC = _mm_load_ps(pB_p2_EC); pB_p2_EC += 4;
      B3_EC = _mm_load_ps(pB_p3_EC); pB_p3_EC += 4;
      
      // Normal Section      
      A0 = _mm_load_ps(pA_p0);
      pA_p0 += 4;
      C00 = B0;
      C00 = _mm_mul_ps(C00, A0);
      C01 = B1;
      C01 = _mm_mul_ps(C01, A0);
      C02 = B2;
      C02 = _mm_mul_ps(C02, A0);
      C03 = B3;
      C03 = _mm_mul_ps(C03, A0);
      // Error Control Section
      A0 = _mm_load_ps(pA_p0_EC);
      pA_p0_EC += 4;
      C00_EC = B0_EC;
      C00_EC = _mm_mul_ps(C00_EC, A0);
      C01_EC = B1_EC;
      C01_EC = _mm_mul_ps(C01_EC, A0);
      C02_EC = B2_EC;
      C02_EC = _mm_mul_ps(C02_EC, A0);
      C03_EC = B3_EC;
      C03_EC = _mm_mul_ps(C03_EC, A0);
      
      // Normal Section      
      A1 = _mm_load_ps(pA_p1);
      pA_p1 += 4;
      C10 = B0;
      C10 = _mm_mul_ps(C10, A1);
      C11 = B1;
      C11 = _mm_mul_ps(C11, A1);
      C12 = B2;
      C12 = _mm_mul_ps(C12, A1);
      C13 = B3;
      C13 = _mm_mul_ps(C13, A1);
      // Error Control Section
      A1 = _mm_load_ps(pA_p1_EC);
      pA_p1_EC += 4;      
      C10_EC = B0_EC;
      C10_EC = _mm_mul_ps(C10_EC, A1);
      C11_EC = B1_EC;
      C11_EC = _mm_mul_ps(C11_EC, A1);
      C12_EC = B2_EC;
      C12_EC = _mm_mul_ps(C12_EC, A1);
      C13_EC = B3_EC;
      C13_EC = _mm_mul_ps(C13_EC, A1);
      
      for (int k = K_LOOPS-1; k; k--)  
      {
        // Normal Section
        B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
        B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
        B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
        B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
        // Error Control Section
        B0_EC = _mm_load_ps(pB_p0_EC); pB_p0_EC += 4;
        B1_EC = _mm_load_ps(pB_p1_EC); pB_p1_EC += 4;
        B2_EC = _mm_load_ps(pB_p2_EC); pB_p2_EC += 4;
        B3_EC = _mm_load_ps(pB_p3_EC); pB_p3_EC += 4;
        
        // Normal Section
        A0 = _mm_load_ps(pA_p0);
        pA_p0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A0);
        C00 = _mm_add_ps(b0, C00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A0);
        C01 = _mm_add_ps(b1, C01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A0);
        C02 = _mm_add_ps(b2, C02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A0);
        C03 = _mm_add_ps(b3, C03);
        // Error Control Section
        A0 = _mm_load_ps(pA_p0_EC);
        pA_p0_EC += 4;
        b0 = B0_EC;
        b0 = _mm_mul_ps(b0, A0);
        C00_EC = _mm_add_ps(b0, C00_EC);
        b1 = B1_EC;
        b1 = _mm_mul_ps(b1, A0);
        C01_EC = _mm_add_ps(b1, C01_EC);
        b2 = B2_EC;
        b2 = _mm_mul_ps(b2, A0);
        C02_EC = _mm_add_ps(b2, C02_EC);
        b3 = B3_EC;
        b3 = _mm_mul_ps(b3, A0);
        C03_EC = _mm_add_ps(b3, C03_EC);
        
        // Normal Section
        A1 = _mm_load_ps(pA_p1);
        pA_p1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A1);
        C10 = _mm_add_ps(b0, C10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A1);
        C11 = _mm_add_ps(b1, C11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A1);
        C12 = _mm_add_ps(b2, C12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A1);
        C13 = _mm_add_ps(b3, C13);
        // Error Control Section
        A1 = _mm_load_ps(pA_p1_EC);
        pA_p1_EC += 4;
        b0 = B0_EC;
        b0 = _mm_mul_ps(b0, A1);
        C10_EC = _mm_add_ps(b0, C10_EC);
        b1 = B1_EC;
        b1 = _mm_mul_ps(b1, A1);
        C11_EC = _mm_add_ps(b1, C11_EC);
        b2 = B2_EC;
        b2 = _mm_mul_ps(b2, A1);
        C12_EC = _mm_add_ps(b2, C12_EC);
        b3 = B3_EC;
        b3 = _mm_mul_ps(b3, A1);
        C13_EC = _mm_add_ps(b3, C13_EC);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---
      // Normal Section
      C00 = _mm_hadd_ps(C00, C01);
      C02 = _mm_hadd_ps(C02, C03);
      C00 = _mm_hadd_ps(C00, C02);
      // Error Control Section
      C00_EC = _mm_hadd_ps(C00_EC, C01_EC);
      C02_EC = _mm_hadd_ps(C02_EC, C03_EC);
      C00_EC = _mm_hadd_ps(C00_EC, C02_EC);      
      // ---
      
      // Error Control Cross Check      
      C00 = _mm_add_ps(C00, C00_EC);
      C00 = _mm_mul_ps(C00, SCALE_FACTOR);
      
      C00 = gblas_quantizer::dequantize_sample(C00, DE_Q_FACTOR);
      
      // SECOND ROW
      // --- horizontal ADD ---
      // Normal Section
      C10 = _mm_hadd_ps(C10, C11);
      C12 = _mm_hadd_ps(C12, C13);
      C10 = _mm_hadd_ps(C10, C12);
      // Error Control Section
      C10_EC = _mm_hadd_ps(C10_EC, C11_EC);
      C12_EC = _mm_hadd_ps(C12_EC, C13_EC);
      C10_EC = _mm_hadd_ps(C10_EC, C12_EC);
      // ---
      
      // Error Control Cross Check
      C10 = _mm_add_ps(C10, C10_EC);
      C10 = _mm_mul_ps(C10, SCALE_FACTOR);
      
      C10 = gblas_quantizer::dequantize_sample(C10, DE_Q_FACTOR);
      
      if ( beta != 0.0f )
      {
        acc0 = _mm_loadu_ps(&pC[0]);
        acc0 = _mm_mul_ps(acc0, __beta);
        C00  = _mm_add_ps(acc0, C00);
        
        acc1 = _mm_loadu_ps(&pC[ldc]);
        acc1 = _mm_mul_ps(acc1, __beta);
        C10  = _mm_add_ps(acc1, C10);
      }
      
      _mm_storeu_ps(&pC[0],      C00);
      _mm_storeu_ps(&pC[ldc],    C10);
      
      pA_p0 -= K_p;
      pA_p1 -= K_p;
      
      pA_p0_EC -= K_p;
      pA_p1_EC -= K_p;
      
      pB_p0 += K_p*3;
      pB_p1 = pB_p0 + K_p;
      pB_p2 = pB_p1 + K_p;
      pB_p3 = pB_p2 + K_p;
      
      pB_p0_EC += K_p*3;
      pB_p1_EC = pB_p0_EC + K_p;
      pB_p2_EC = pB_p1_EC + K_p;
      pB_p3_EC = pB_p2_EC + K_p;
      
      pC += 4;
    }
    pA_p0 += K_p*2;
    pA_p1 = pA_p0 + K_p;
    
    pA_p0_EC += K_p*2;
    pA_p1_EC = pA_p0_EC + K_p;
    
    pB_p0 = B_p;
    pB_p1 = pB_p0 + K_p;
    pB_p2 = pB_p1 + K_p;
    pB_p3 = pB_p2 + K_p;
    
    pB_p0_EC = B_p_EC;
    pB_p1_EC = pB_p0_EC + K_p;
    pB_p2_EC = pB_p1_EC + K_p;
    pB_p3_EC = pB_p2_EC + K_p;
    
    pC   += (ldc*2 - GBLAS_KERNEL_SIZE);
  }
}



/*
 * Slowest parallel kernel of the history: forget about it! :D
 */
void KERNEL_p_sgemm_v2_r5_EC_v3_MT(const int M, const int N,
                                   const int K, const float alpha, const float *A,
                                   const int lda, const float *B, const int ldb,
                                   const float beta, float *C, const int ldc,
                                   gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  const __m128 __beta     =   _mm_set1_ps(beta);
  
  const float TIGHT_PACKING_A_MAX = 2.0f*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const float INV_EPS     =   ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  float A_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float A_p_EC[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p_EC[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128 acc0, acc1, acc0_EC, acc1_EC;
  
  const __m128 SCALE_FACTOR = _mm_set1_ps(0.5f);
  const __m128 a_factor = _mm_set_ps(EPS, 1.0f, EPS, 1.0f);
  const __m128 b_factor = _mm_set_ps(INV_EPS, 1.0f, INV_EPS, 1.0f);
  const __m128 a_factor_EC = _mm_set_ps(EPS, -1.0f, EPS, -1.0f);  
  const __m128 b_factor_EC = _mm_set_ps(INV_EPS, -1.0f, INV_EPS, -1.0f);
  
  int kh;
#pragma omp parallel for private(kh, acc0, acc1, acc0_EC, acc1_EC)
  for ( int k = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8 )
  {
    kh = k >> 1;
    
    //printf("%i %i\n", k, kh);
    
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0_EC = _mm_mul_ps(acc0, a_factor_EC);
    acc0 = _mm_mul_ps(acc0, a_factor);
    
    acc1 = _mm_load_ps(&A[k + 4]);
    acc1 = Qa.quantize_sample(acc1);
    acc1_EC = _mm_mul_ps(acc1, a_factor_EC);    
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    acc0_EC = _mm_hadd_ps(acc0_EC, acc1_EC);
    
    _mm_store_ps(&A_p[kh], acc0);
    _mm_store_ps(&A_p_EC[kh], acc0_EC);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0_EC = _mm_mul_ps(acc0, b_factor_EC); // this one first because the next line will modify the value of acc0
    acc0 = _mm_mul_ps(acc0, b_factor);
    
    acc1 = _mm_load_ps(&B[k + 4]);
    acc1 = Qb.quantize_sample(acc1);
    acc1_EC = _mm_mul_ps(acc1, b_factor_EC); // this one first because the next line will modify the value of acc0
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    acc0_EC = _mm_hadd_ps(acc0_EC, acc1_EC);
    
    _mm_store_ps(&B_p[kh], acc0);
    _mm_store_ps(&B_p_EC[kh], acc0_EC);
  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 A0, A1;
  
  __m128 b0, b1, b2, b3;
  
  __m128 B0, B1, B2, B3;
  __m128 B0_EC, B1_EC, B2_EC, B3_EC;
  
  __m128 C00, C01, C02, C03, C10, C11, C12, C13; 
  __m128 C00_EC, C01_EC, C02_EC, C03_EC,
  C10_EC, C11_EC, C12_EC, C13_EC;
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = A_p + K_p;
  
  const float* pA_p0_EC = A_p_EC;
  const float* pA_p1_EC = A_p_EC + K_p;
  
  const float* pB_p0 = B_p;
  const float* pB_p1 = pB_p0 + K_p;
  const float* pB_p2 = pB_p1 + K_p;
  const float* pB_p3 = pB_p2 + K_p;
  
  const float* pB_p0_EC = B_p_EC;
  const float* pB_p1_EC = pB_p0_EC + K_p;
  const float* pB_p2_EC = pB_p1_EC + K_p;
  const float* pB_p3_EC = pB_p2_EC + K_p;
  
  float* pC = C;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
#pragma omp parallel sections private(A0, A1, b0, b1, b2, b3)
      {
#pragma omp section
        {
          // Normal Section
          B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
          B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
          B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
          B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
          
          // Normal Section      
          A0 = _mm_load_ps(pA_p0);
          pA_p0 += 4;
          C00 = B0;
          C00 = _mm_mul_ps(C00, A0);
          C01 = B1;
          C01 = _mm_mul_ps(C01, A0);
          C02 = B2;
          C02 = _mm_mul_ps(C02, A0);
          C03 = B3;
          C03 = _mm_mul_ps(C03, A0);
          
          // Normal Section      
          A1 = _mm_load_ps(pA_p1);
          pA_p1 += 4;
          C10 = B0;
          C10 = _mm_mul_ps(C10, A1);
          C11 = B1;
          C11 = _mm_mul_ps(C11, A1);
          C12 = B2;
          C12 = _mm_mul_ps(C12, A1);
          C13 = B3;
          C13 = _mm_mul_ps(C13, A1);
          
          for (int k = K_LOOPS-1; k; k--)  
          {
            // Normal Section
            B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
            B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
            B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
            B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
            
            // Normal Section
            A0 = _mm_load_ps(pA_p0);
            pA_p0 += 4;
            b0 = B0;
            b0 = _mm_mul_ps(b0, A0);
            C00 = _mm_add_ps(b0, C00);
            b1 = B1;
            b1 = _mm_mul_ps(b1, A0);
            C01 = _mm_add_ps(b1, C01);
            b2 = B2;
            b2 = _mm_mul_ps(b2, A0);
            C02 = _mm_add_ps(b2, C02);
            b3 = B3;
            b3 = _mm_mul_ps(b3, A0);
            C03 = _mm_add_ps(b3, C03);
            
            // Normal Section
            A1 = _mm_load_ps(pA_p1);
            pA_p1 += 4;
            b0 = B0;
            b0 = _mm_mul_ps(b0, A1);
            C10 = _mm_add_ps(b0, C10);
            b1 = B1;
            b1 = _mm_mul_ps(b1, A1);
            C11 = _mm_add_ps(b1, C11);
            b2 = B2;
            b2 = _mm_mul_ps(b2, A1);
            C12 = _mm_add_ps(b2, C12);
            b3 = B3;
            b3 = _mm_mul_ps(b3, A1);
            C13 = _mm_add_ps(b3, C13);
          }
          
          // Normal Section
          C00 = _mm_hadd_ps(C00, C01);
          C02 = _mm_hadd_ps(C02, C03);
          C00 = _mm_hadd_ps(C00, C02);
          
          // Normal Section
          C10 = _mm_hadd_ps(C10, C11);
          C12 = _mm_hadd_ps(C12, C13);
          C10 = _mm_hadd_ps(C10, C12);
        }
        
#pragma omp section
        {
          // Error Control Section
          B0_EC = _mm_load_ps(pB_p0_EC); pB_p0_EC += 4;
          B1_EC = _mm_load_ps(pB_p1_EC); pB_p1_EC += 4;
          B2_EC = _mm_load_ps(pB_p2_EC); pB_p2_EC += 4;
          B3_EC = _mm_load_ps(pB_p3_EC); pB_p3_EC += 4;
          
          // Error Control Section
          A0 = _mm_load_ps(pA_p0_EC);
          pA_p0_EC += 4;
          C00_EC = B0_EC;
          C00_EC = _mm_mul_ps(C00_EC, A0);
          C01_EC = B1_EC;
          C01_EC = _mm_mul_ps(C01_EC, A0);
          C02_EC = B2_EC;
          C02_EC = _mm_mul_ps(C02_EC, A0);
          C03_EC = B3_EC;
          C03_EC = _mm_mul_ps(C03_EC, A0);
          
          // Error Control Section
          A1 = _mm_load_ps(pA_p1_EC);
          pA_p1_EC += 4;      
          C10_EC = B0_EC;
          C10_EC = _mm_mul_ps(C10_EC, A1);
          C11_EC = B1_EC;
          C11_EC = _mm_mul_ps(C11_EC, A1);
          C12_EC = B2_EC;
          C12_EC = _mm_mul_ps(C12_EC, A1);
          C13_EC = B3_EC;
          C13_EC = _mm_mul_ps(C13_EC, A1);
          
          for (int k = K_LOOPS-1; k; k--)  
          {
            // Error Control Section
            B0_EC = _mm_load_ps(pB_p0_EC); pB_p0_EC += 4;
            B1_EC = _mm_load_ps(pB_p1_EC); pB_p1_EC += 4;
            B2_EC = _mm_load_ps(pB_p2_EC); pB_p2_EC += 4;
            B3_EC = _mm_load_ps(pB_p3_EC); pB_p3_EC += 4;
            
            // Error Control Section
            A0 = _mm_load_ps(pA_p0_EC);
            pA_p0_EC += 4;
            b0 = B0_EC;
            b0 = _mm_mul_ps(b0, A0);
            C00_EC = _mm_add_ps(b0, C00_EC);
            b1 = B1_EC;
            b1 = _mm_mul_ps(b1, A0);
            C01_EC = _mm_add_ps(b1, C01_EC);
            b2 = B2_EC;
            b2 = _mm_mul_ps(b2, A0);
            C02_EC = _mm_add_ps(b2, C02_EC);
            b3 = B3_EC;
            b3 = _mm_mul_ps(b3, A0);
            C03_EC = _mm_add_ps(b3, C03_EC);
            
            // Error Control Section
            A1 = _mm_load_ps(pA_p1_EC);
            pA_p1_EC += 4;
            b0 = B0_EC;
            b0 = _mm_mul_ps(b0, A1);
            C10_EC = _mm_add_ps(b0, C10_EC);
            b1 = B1_EC;
            b1 = _mm_mul_ps(b1, A1);
            C11_EC = _mm_add_ps(b1, C11_EC);
            b2 = B2_EC;
            b2 = _mm_mul_ps(b2, A1);
            C12_EC = _mm_add_ps(b2, C12_EC);
            b3 = B3_EC;
            b3 = _mm_mul_ps(b3, A1);
            C13_EC = _mm_add_ps(b3, C13_EC);
          }
          
          // Error Control Section
          C00_EC = _mm_hadd_ps(C00_EC, C01_EC);
          C02_EC = _mm_hadd_ps(C02_EC, C03_EC);
          C00_EC = _mm_hadd_ps(C00_EC, C02_EC);  
          
          // Error Control Section
          C10_EC = _mm_hadd_ps(C10_EC, C11_EC);
          C12_EC = _mm_hadd_ps(C12_EC, C13_EC);
          C10_EC = _mm_hadd_ps(C10_EC, C12_EC);
        }
      }
      
      // FIRST ROW
      // Error Control Cross Check
      C00 = _mm_add_ps(C00, C00_EC);
      C00 = _mm_mul_ps(C00, SCALE_FACTOR);
      
      C00 = gblas_quantizer::dequantize_sample(C00, DE_Q_FACTOR);
      
      // SECOND ROW      
      // Error Control Cross Check
      C10 = _mm_add_ps(C10, C10_EC);
      C10 = _mm_mul_ps(C10, SCALE_FACTOR);
      
      C10 = gblas_quantizer::dequantize_sample(C10, DE_Q_FACTOR);
      
      if ( beta != 0.0f )
      {
        acc0 = _mm_loadu_ps(&pC[0]);
        acc0 = _mm_mul_ps(acc0, __beta);
        C00  = _mm_add_ps(acc0, C00);
        
        acc1 = _mm_loadu_ps(&pC[ldc]);
        acc1 = _mm_mul_ps(acc1, __beta);
        C10  = _mm_add_ps(acc1, C10);
      }
      
      _mm_storeu_ps(&pC[0],      C00);
      _mm_storeu_ps(&pC[ldc],    C10);
      
      pA_p0 -= K_p;
      pA_p1 -= K_p;
      
      pA_p0_EC -= K_p;
      pA_p1_EC -= K_p;
      
      pB_p0 += K_p*3;
      pB_p1 = pB_p0 + K_p;
      pB_p2 = pB_p1 + K_p;
      pB_p3 = pB_p2 + K_p;
      
      pB_p0_EC += K_p*3;
      pB_p1_EC = pB_p0_EC + K_p;
      pB_p2_EC = pB_p1_EC + K_p;
      pB_p3_EC = pB_p2_EC + K_p;
      
      pC += 4;
    }
    pA_p0 += K_p*2;
    pA_p1 = pA_p0 + K_p;
    
    pA_p0_EC += K_p*2;
    pA_p1_EC = pA_p0_EC + K_p;
    
    pB_p0 = B_p;
    pB_p1 = pB_p0 + K_p;
    pB_p2 = pB_p1 + K_p;
    pB_p3 = pB_p2 + K_p;
    
    pB_p0_EC = B_p_EC;
    pB_p1_EC = pB_p0_EC + K_p;
    pB_p2_EC = pB_p1_EC + K_p;
    pB_p3_EC = pB_p2_EC + K_p;
    
    pC   += (ldc*2 - GBLAS_KERNEL_SIZE);
  }
}

inline void KERNEL_p_sgemm_v2_r5_P(const float *A, const float *B, float *C, gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  //const float TIGHT_PACKING_A_MAX = 2.0f*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const float INV_EPS     =   sqrt(2.0f); //ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  float A_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128 acc0, acc1;
  
  const __m128 a_factor = _mm_set_ps(EPS, 1.f, EPS, 1.f);
  const __m128 b_factor = _mm_set_ps(INV_EPS, 1.f, INV_EPS, 1.f);
  
  // V.1
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    //acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, a_factor);
    
    acc1 = _mm_load_ps(&A[k + 4]);
    //acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&A_p[kh], acc0);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    //acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, b_factor);
    
    acc1 = _mm_load_ps(&B[k + 4]);
    //acc1 = Qb.quantize_sample(acc1); 
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&B_p[kh], acc0);
  }
  
  // V.2
//  const float* pA0 = A;
//  const float* pB0 = B;
//  float* pA1 = A_p;
//  float* pB1 = B_p;
//  
//  //for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
//  for ( int k = ((GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE) >> 3); k ; k-- )
//  {
//    /* packing A */
//    acc0 = _mm_load_ps(pA0);  pA0 += 4;
//    acc0 = Qa.quantize_sample(acc0);
//    acc0 = _mm_mul_ps(acc0, a_factor);
//    
//    acc1 = _mm_load_ps(pA0);  pA0 += 4;
//    acc1 = Qa.quantize_sample(acc1);
//    acc1 = _mm_mul_ps(acc1, a_factor);
//    
//    acc0 = _mm_hadd_ps(acc0, acc1);
//    
//    _mm_store_ps(pA1, acc0);  pA1 += 4;
//    
//    /* packing B */
//    acc0 = _mm_load_ps(pB0);  pB0 += 4;
//    acc0 = Qb.quantize_sample(acc0);
//    acc0 = _mm_mul_ps(acc0, b_factor);
//    
//    acc1 = _mm_load_ps(pB0);  pB0 += 4;
//    acc1 = Qb.quantize_sample(acc1); 
//    acc1 = _mm_mul_ps(acc1, b_factor);
//    
//    acc0 = _mm_hadd_ps(acc0, acc1);
//    
//    _mm_store_ps(pB1, acc0);  pB1 += 4;
//  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 mmA0, mmA1;
  
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;
  
  __m128 mmC00, mmC01, mmC02, mmC03, mmC10, mmC11, mmC12, mmC13; 
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = pA_p0 + K_p;
  
  const float* pB_p0 = B_p;
  const float* pB_p1 = pB_p0 + K_p;
  const float* pB_p2 = pB_p1 + K_p;
  const float* pB_p3 = pB_p2 + K_p;
  
  float* pC0 = C;
  float* pC1 = pC0 + GBLAS_KERNEL_SIZE;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
      B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
      B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
      B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
      
      mmA0 = _mm_load_ps(pA_p0);
      pA_p0 += 4;
      mmC00 = B0;
      mmC00 = _mm_mul_ps(mmC00, mmA0);
      mmC01 = B1;
      mmC01 = _mm_mul_ps(mmC01, mmA0);
      mmC02 = B2;
      mmC02 = _mm_mul_ps(mmC02, mmA0);
      mmC03 = B3;
      mmC03 = _mm_mul_ps(mmC03, mmA0);
      
      mmA1 = _mm_load_ps(pA_p1);
      pA_p1 += 4;
      mmC10 = B0;
      mmC10 = _mm_mul_ps(mmC10, mmA1);
      mmC11 = B1;
      mmC11 = _mm_mul_ps(mmC11, mmA1);
      mmC12 = B2;
      mmC12 = _mm_mul_ps(mmC12, mmA1);
      mmC13 = B3;
      mmC13 = _mm_mul_ps(mmC13, mmA1);
      
      for (int k = K_LOOPS-1; k; k--)  
      {
        B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
        B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
        B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
        B3 = _mm_load_ps(pB_p3); pB_p3 += 4; 
        
        mmA0 = _mm_load_ps(pA_p0);
        pA_p0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mmA0);
        mmC00 = _mm_add_ps(b0, mmC00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mmA0);
        mmC01 = _mm_add_ps(b1, mmC01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mmA0);
        mmC02 = _mm_add_ps(b2, mmC02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mmA0);
        mmC03 = _mm_add_ps(b3, mmC03);
        
        mmA1 = _mm_load_ps(pA_p1);
        pA_p1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mmA1);
        mmC10 = _mm_add_ps(b0, mmC10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mmA1);
        mmC11 = _mm_add_ps(b1, mmC11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mmA1);
        mmC12 = _mm_add_ps(b2, mmC12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mmA1);
        mmC13 = _mm_add_ps(b3, mmC13);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---     
      mmC00 = _mm_hadd_ps(mmC00, mmC01);
      mmC02 = _mm_hadd_ps(mmC02, mmC03);
      mmC00 = _mm_hadd_ps(mmC00, mmC02);
      // ---
      
      // SECOND ROW
      // --- horizontal ADD ---     
      mmC10 = _mm_hadd_ps(mmC10, mmC11);
      mmC12 = _mm_hadd_ps(mmC12, mmC13);
      mmC10 = _mm_hadd_ps(mmC10, mmC12);
      // ---
      
      _mm_store_ps(pC0, mmC00);
      _mm_store_ps(pC1, mmC10);
      
      pA_p0 -= K_p;
      pA_p1 = pA_p0 + K_p;
      
      pB_p0 += K_p*3;
      pB_p1 = pB_p0 + K_p;
      pB_p2 = pB_p1 + K_p;
      pB_p3 = pB_p2 + K_p;
      
      pC0 += 4;
      pC1 += 4;      
    }
    pA_p0 += K_p*2;
    pA_p1 += K_p*2;
    
    pB_p0 = B_p;
    pB_p1 = pB_p0 + K_p;
    pB_p2 = pB_p1 + K_p;
    pB_p3 = pB_p2 + K_p;

    pC0 += GBLAS_KERNEL_SIZE;
    pC1 = pC0 + GBLAS_KERNEL_SIZE;
  }
}

inline void KERNEL_p_sgemm_v2_r5_N(const float *A, const float *B, float *C, gblas_quantizer& Qa, gblas_quantizer& Qb)
{  
  //const float TIGHT_PACKING_A_MAX = 2.0f*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const float INV_EPS     =   sqrt(2.0f); //ceil(TIGHT_PACKING_A_MAX + DELTA);
  const float EPS         =   1.0f/INV_EPS;
  
  float A_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  float B_p[K_p*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128 acc0, acc1;
  
  const __m128 a_factor = _mm_set_ps(EPS, -1.0f, EPS, -1.0f);
  const __m128 b_factor = _mm_set_ps(INV_EPS, -1.0f, INV_EPS, -1.0f);
  
  // V.1
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    //acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, a_factor);
    
    acc1 = _mm_load_ps(&A[k + 4]);
    //acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&A_p[kh], acc0);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    //acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, b_factor);
    
    acc1 = _mm_load_ps(&B[k + 4]);
    //acc1 = Qb.quantize_sample(acc1); 
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    _mm_store_ps(&B_p[kh], acc0);
  }
  
  // V.2
//  const float* pA0 = A;
//  const float* pB0 = B;
//  float* pA1 = A_p;
//  float* pB1 = B_p;
//  
//  //for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=8, kh+=4 )
//  for ( int k = ((GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE) >> 3); k ; k-- )
//  {
//    /* packing A */
//    acc0 = _mm_load_ps(pA0);  pA0 += 4;
//    acc0 = Qa.quantize_sample(acc0);
//    acc0 = _mm_mul_ps(acc0, a_factor);
//    
//    acc1 = _mm_load_ps(pA0);  pA0 += 4;
//    acc1 = Qa.quantize_sample(acc1);
//    acc1 = _mm_mul_ps(acc1, a_factor);
//    
//    acc0 = _mm_hadd_ps(acc0, acc1);
//    
//    _mm_store_ps(pA1, acc0);  pA1 += 4;
//    
//    /* packing B */
//    acc0 = _mm_load_ps(pB0);  pB0 += 4;
//    acc0 = Qb.quantize_sample(acc0);
//    acc0 = _mm_mul_ps(acc0, b_factor);
//       
//    acc1 = _mm_load_ps(pB0);  pB0 += 4;
//    acc1 = Qb.quantize_sample(acc1); 
//    acc1 = _mm_mul_ps(acc1, b_factor);
//    
//    acc0 = _mm_hadd_ps(acc0, acc1);
//      
//    _mm_store_ps(pB1, acc0);  pB1 += 4;
//  }
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 mmA0, mmA1;
  
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;
  
  __m128 mmC00, mmC01, mmC02, mmC03, mmC10, mmC11, mmC12, mmC13; 
  
  const float* pA_p0 = A_p;
  const float* pA_p1 = pA_p0 + K_p;
  
  const float* pB_p0 = B_p;
  const float* pB_p1 = pB_p0 + K_p;
  const float* pB_p2 = pB_p1 + K_p;
  const float* pB_p3 = pB_p2 + K_p;
  
  float* pC0 = C;
  float* pC1 = pC0 + GBLAS_KERNEL_SIZE;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
      B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
      B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
      B3 = _mm_load_ps(pB_p3); pB_p3 += 4;
      
      mmA0 = _mm_load_ps(pA_p0);
      pA_p0 += 4;
      mmC00 = B0;
      mmC00 = _mm_mul_ps(mmC00, mmA0);
      mmC01 = B1;
      mmC01 = _mm_mul_ps(mmC01, mmA0);
      mmC02 = B2;
      mmC02 = _mm_mul_ps(mmC02, mmA0);
      mmC03 = B3;
      mmC03 = _mm_mul_ps(mmC03, mmA0);
      
      mmA1 = _mm_load_ps(pA_p1);
      pA_p1 += 4;
      mmC10 = B0;
      mmC10 = _mm_mul_ps(mmC10, mmA1);
      mmC11 = B1;
      mmC11 = _mm_mul_ps(mmC11, mmA1);
      mmC12 = B2;
      mmC12 = _mm_mul_ps(mmC12, mmA1);
      mmC13 = B3;
      mmC13 = _mm_mul_ps(mmC13, mmA1);
      
      for (int k = K_LOOPS-1; k; k--)  
      {
        B0 = _mm_load_ps(pB_p0); pB_p0 += 4;
        B1 = _mm_load_ps(pB_p1); pB_p1 += 4;
        B2 = _mm_load_ps(pB_p2); pB_p2 += 4;
        B3 = _mm_load_ps(pB_p3); pB_p3 += 4; 
        
        mmA0 = _mm_load_ps(pA_p0);
        pA_p0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mmA0);
        mmC00 = _mm_add_ps(b0, mmC00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mmA0);
        mmC01 = _mm_add_ps(b1, mmC01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mmA0);
        mmC02 = _mm_add_ps(b2, mmC02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mmA0);
        mmC03 = _mm_add_ps(b3, mmC03);
        
        mmA1 = _mm_load_ps(pA_p1);
        pA_p1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, mmA1);
        mmC10 = _mm_add_ps(b0, mmC10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, mmA1);
        mmC11 = _mm_add_ps(b1, mmC11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, mmA1);
        mmC12 = _mm_add_ps(b2, mmC12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, mmA1);
        mmC13 = _mm_add_ps(b3, mmC13);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---     
      mmC00 = _mm_hadd_ps(mmC00, mmC01);
      mmC02 = _mm_hadd_ps(mmC02, mmC03);
      mmC00 = _mm_hadd_ps(mmC00, mmC02);
      // ---
      
      // SECOND ROW
      // --- horizontal ADD ---     
      mmC10 = _mm_hadd_ps(mmC10, mmC11);
      mmC12 = _mm_hadd_ps(mmC12, mmC13);
      mmC10 = _mm_hadd_ps(mmC10, mmC12);
      // ---
      
      _mm_store_ps(pC0, mmC00);
      _mm_store_ps(pC1, mmC10);
      
      pA_p0 -= K_p;
      pA_p1 = pA_p0 + K_p;
      
      pB_p0 += K_p*3;
      pB_p1 = pB_p0 + K_p;
      pB_p2 = pB_p1 + K_p;
      pB_p3 = pB_p2 + K_p;
      
      pC0 += 4;
      pC1 += 4;
    }
    pA_p0 += K_p*2;
    pA_p1 += K_p*2;
    
    pB_p0 = B_p;
    pB_p1 = pB_p0 + K_p;
    pB_p2 = pB_p1 + K_p;
    pB_p3 = pB_p2 + K_p;
    
    pC0 += GBLAS_KERNEL_SIZE;
    pC1 = pC0 + GBLAS_KERNEL_SIZE;
  }
}

void KERNEL_p_sgemm_v2_r5_EC_v4(const int M, const int N,
                            const int K, const float alpha, const float *A,
                            const int lda, const float *B, const int ldb,
                            const float beta, float *C, const int ldc,
                            gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  // allocate memory for C
  float* C_positive = (float*)_mm_malloc(sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE, 16);
  float* C_negative = (float*)_mm_malloc(sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE, 16);
  
//#pragma omp parallel sections
  {
    /*
     * Positive section
     */
//#pragma omp section
    {
      KERNEL_p_sgemm_v2_r5_P(A, B, C_positive, Qa, Qb);
    }
    
    /*
     * Negative section
     */
//#pragma omp section
    {
      KERNEL_p_sgemm_v2_r5_N(A, B, C_negative, Qa, Qb);      
    }
  }
  
  /*
   * add and store into final C
   */
  const __m128 SCALE_FACTOR = _mm_set1_ps(0.5f);
  const __m128 BETA = _mm_set1_ps(beta);
  //const __m128 DE_Q_FACTOR =   _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  __m128 neg, pos;
  
//#pragma omp parallel for private(neg, pos)
  for (int ii = 0; ii < GBLAS_KERNEL_SIZE; ii++)
  {
    for (int jj = 0; jj < GBLAS_KERNEL_SIZE; jj+=4)
    {
      // load positive
      pos = _mm_load_ps(&C_positive[ii*GBLAS_KERNEL_SIZE + jj]);
      // load negative
      neg = _mm_load_ps(&C_negative[ii*GBLAS_KERNEL_SIZE + jj]);
      
      // unpacking by elimination
      pos = _mm_add_ps(pos, neg);
      pos = _mm_mul_ps(pos, SCALE_FACTOR);
      
      //pos = gblas_quantizer::dequantize_sample(pos, DE_Q_FACTOR);
      
      if ( beta != 0.0f )
      {
        neg = _mm_loadu_ps(&C[ii*ldc + jj]);
        neg = _mm_mul_ps(neg, BETA);
        pos  = _mm_add_ps(neg, pos);
      }
      
      // unaligned store into C
      _mm_storeu_ps(&C[ii*ldc + jj], pos);
    }
  }
  
  _mm_free(C_positive);
  _mm_free(C_negative);
}

inline void KERNEL_p4_sgemm_v2_r5_P(const float *A, const float *B, float *C, gblas_quantizer& Qa, gblas_quantizer& Qb, const float EPS, const float INV_EPS)
{
  const int KS_P4 = (GBLAS_KERNEL_SIZE >> 2);
  
  float *A_p = (float*)_mm_malloc(sizeof(float)*KS_P4*GBLAS_KERNEL_SIZE, 16);
  float *B_p = (float*)_mm_malloc(sizeof(float)*KS_P4*GBLAS_KERNEL_SIZE, 16);
  
  __m128 acc0, acc1, acc2; //, acc3;
  
  const __m128 a_factor = _mm_set_ps(EPS*EPS*EPS, EPS*EPS, EPS, 1.f);
  const __m128 b_factor = _mm_set_ps(INV_EPS*INV_EPS*INV_EPS, INV_EPS*INV_EPS, INV_EPS, 1.f);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=16, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, a_factor);
    acc1 = _mm_load_ps(&A[k + 4]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    acc1 = _mm_load_ps(&A[k + 8]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    acc2 = _mm_load_ps(&A[k + 12]);
    acc2 = Qa.quantize_sample(acc2);
    acc2 = _mm_mul_ps(acc2, a_factor);
    
    acc1 = _mm_hadd_ps(acc1, acc2);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    _mm_store_ps(&A_p[kh], acc0);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, b_factor);
    acc1 = _mm_load_ps(&B[k + 4]);
    acc1 = Qb.quantize_sample(acc1); 
    acc1 = _mm_mul_ps(acc1, b_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);

    acc1 = _mm_load_ps(&B[k + 8]);
    acc1 = Qb.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, b_factor);
    acc2 = _mm_load_ps(&B[k + 12]);
    acc2 = Qb.quantize_sample(acc2); 
    acc2 = _mm_mul_ps(acc2, b_factor);
    
    acc1 = _mm_hadd_ps(acc1, acc2);

    acc0 = _mm_hadd_ps(acc0, acc1);
    _mm_store_ps(&B_p[kh], acc0);
  }

  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 A0, A1;
  
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;
  
  __m128 C00, C01, C02, C03, C10, C11, C12, C13; 
  
  const float* pA0 = A_p;
  const float* pA1 = pA0 + KS_P4;
  
  const float* pB0 = B_p;
  const float* pB1 = pB0 + KS_P4;
  const float* pB2 = pB1 + KS_P4;
  const float* pB3 = pB2 + KS_P4;
  
  float* pC0 = C;
  float* pC1 = pC0 + GBLAS_KERNEL_SIZE;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      B0 = _mm_load_ps(pB0); pB0 += 4;
      B1 = _mm_load_ps(pB1); pB1 += 4;
      B2 = _mm_load_ps(pB2); pB2 += 4;
      B3 = _mm_load_ps(pB3); pB3 += 4;
      
      A0 = _mm_load_ps(pA0); pA0 += 4;
      C00 = B0;
      C00 = _mm_mul_ps(C00, A0);
      C01 = B1;
      C01 = _mm_mul_ps(C01, A0);
      C02 = B2;
      C02 = _mm_mul_ps(C02, A0);
      C03 = B3;
      C03 = _mm_mul_ps(C03, A0);
      
      A1 = _mm_load_ps(pA1); pA1 += 4;
      C10 = B0;
      C10 = _mm_mul_ps(C10, A1);
      C11 = B1;
      C11 = _mm_mul_ps(C11, A1);
      C12 = B2;
      C12 = _mm_mul_ps(C12, A1);
      C13 = B3;
      C13 = _mm_mul_ps(C13, A1);
      
      for (int k = (KS_P4 >> 2)-1; k; k--)  
      {
        B0 = _mm_load_ps(pB0); pB0 += 4;
        B1 = _mm_load_ps(pB1); pB1 += 4;
        B2 = _mm_load_ps(pB2); pB2 += 4;
        B3 = _mm_load_ps(pB3); pB3 += 4; 
        
        A0 = _mm_load_ps(pA0); pA0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A0);
        C00 = _mm_add_ps(b0, C00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A0);
        C01 = _mm_add_ps(b1, C01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A0);
        C02 = _mm_add_ps(b2, C02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A0);
        C03 = _mm_add_ps(b3, C03);
        
        A1 = _mm_load_ps(pA1); pA1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A1);
        C10 = _mm_add_ps(b0, C10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A1);
        C11 = _mm_add_ps(b1, C11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A1);
        C12 = _mm_add_ps(b2, C12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A1);
        C13 = _mm_add_ps(b3, C13);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---     
      C00 = _mm_hadd_ps(C00, C01);
      C02 = _mm_hadd_ps(C02, C03);
      C00 = _mm_hadd_ps(C00, C02);
      // ---
      
      // SECOND ROW
      // --- horizontal ADD ---     
      C10 = _mm_hadd_ps(C10, C11);
      C12 = _mm_hadd_ps(C12, C13);
      C10 = _mm_hadd_ps(C10, C12);
      // ---
      
      _mm_store_ps(pC0, C00);
      _mm_store_ps(pC1, C10);
      
      pA0 -= KS_P4;
      pA1 = pA0 + KS_P4;
      
      pB0 += KS_P4*3;
      pB1 = pB0 + KS_P4;
      pB2 = pB1 + KS_P4;
      pB3 = pB2 + KS_P4;
      
      pC0 += 4;
      pC1 += 4;      
    }
    pA0 += KS_P4*2;
    pA1 = pA0 + KS_P4;
    
    pB0 = B_p;
    pB1 = pB0 + KS_P4;
    pB2 = pB1 + KS_P4;
    pB3 = pB2 + KS_P4;
    
    pC0 += GBLAS_KERNEL_SIZE;
    pC1 = pC0 + GBLAS_KERNEL_SIZE;
  }
  
  _mm_free(A_p);
  _mm_free(B_p);
}

inline void KERNEL_p4_sgemm_v2_r5_N(const float *A, const float *B, float *C, gblas_quantizer& Qa, gblas_quantizer& Qb, const float EPS, const float INV_EPS)
{  
  const int KS_P4 = (GBLAS_KERNEL_SIZE >> 2);
  
  float *A_p = (float*)_mm_malloc(sizeof(float)*KS_P4*GBLAS_KERNEL_SIZE, 16);
  float *B_p = (float*)_mm_malloc(sizeof(float)*KS_P4*GBLAS_KERNEL_SIZE, 16);
  
  __m128 acc0, acc1, acc2; //, acc3;
  
  const __m128 a_factor = _mm_set_ps(-EPS*EPS*EPS, EPS*EPS, -EPS, 1.f);
  const __m128 b_factor = _mm_set_ps(-INV_EPS*INV_EPS*INV_EPS, INV_EPS*INV_EPS, -INV_EPS, 1.f);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=16, kh+=4 )
  {
    /* packing A */
    acc0 = _mm_load_ps(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, a_factor);
    acc1 = _mm_load_ps(&A[k + 4]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    acc1 = _mm_load_ps(&A[k + 8]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, a_factor);
    acc2 = _mm_load_ps(&A[k + 12]);
    acc2 = Qa.quantize_sample(acc2);
    acc2 = _mm_mul_ps(acc2, a_factor);
    
    acc1 = _mm_hadd_ps(acc1, acc2);
    
    acc0 = _mm_hadd_ps(acc0, acc1);
    _mm_store_ps(&A_p[kh], acc0);
    
    /* packing B */
    acc0 = _mm_load_ps(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_ps(acc0, b_factor);
    acc1 = _mm_load_ps(&B[k + 4]);
    acc1 = Qb.quantize_sample(acc1); 
    acc1 = _mm_mul_ps(acc1, b_factor);
    acc0 = _mm_hadd_ps(acc0, acc1);
    
    acc1 = _mm_load_ps(&B[k + 8]);
    acc1 = Qb.quantize_sample(acc1);
    acc1 = _mm_mul_ps(acc1, b_factor);
    acc2 = _mm_load_ps(&B[k + 12]);
    acc2 = Qb.quantize_sample(acc2); 
    acc2 = _mm_mul_ps(acc2, b_factor);
    acc1 = _mm_hadd_ps(acc1, acc2);
    
    acc0 = _mm_hadd_ps(acc0, acc1);    
    _mm_store_ps(&B_p[kh], acc0);
  }
  
  //print_matrix_matlab_notation(A_p, KERNEL_SIZE, K_p);
  
#ifdef DEBUG_PRINT
  cout << "inv_eps = " << INV_EPS << endl;
#endif
  
  __m128 A0, A1;
  
  __m128 B0, B1, B2, B3;
  __m128 b0, b1, b2, b3;
  
  __m128 C00, C01, C02, C03, C10, C11, C12, C13; 
  
  const float* pA0 = A_p;
  const float* pA1 = pA0 + KS_P4;
  
  const float* pB0 = B_p;
  const float* pB1 = pB0 + KS_P4;
  const float* pB2 = pB1 + KS_P4;
  const float* pB3 = pB2 + KS_P4;
  
  float* pC0 = C;
  float* pC1 = pC0 + GBLAS_KERNEL_SIZE;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 2); j; j--)  
    {
      B0 = _mm_load_ps(pB0); pB0 += 4;
      B1 = _mm_load_ps(pB1); pB1 += 4;
      B2 = _mm_load_ps(pB2); pB2 += 4;
      B3 = _mm_load_ps(pB3); pB3 += 4;
      
      A0 = _mm_load_ps(pA0); pA0 += 4;
      C00 = B0;
      C00 = _mm_mul_ps(C00, A0);
      C01 = B1;
      C01 = _mm_mul_ps(C01, A0);
      C02 = B2;
      C02 = _mm_mul_ps(C02, A0);
      C03 = B3;
      C03 = _mm_mul_ps(C03, A0);
      
      A1 = _mm_load_ps(pA1); pA1 += 4;
      C10 = B0;
      C10 = _mm_mul_ps(C10, A1);
      C11 = B1;
      C11 = _mm_mul_ps(C11, A1);
      C12 = B2;
      C12 = _mm_mul_ps(C12, A1);
      C13 = B3;
      C13 = _mm_mul_ps(C13, A1);
      
      for (int k = (KS_P4 >> 2)-1; k; k--)  
      {
        B0 = _mm_load_ps(pB0); pB0 += 4;
        B1 = _mm_load_ps(pB1); pB1 += 4;
        B2 = _mm_load_ps(pB2); pB2 += 4;
        B3 = _mm_load_ps(pB3); pB3 += 4; 
        
        A0 = _mm_load_ps(pA0); pA0 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A0);
        C00 = _mm_add_ps(b0, C00);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A0);
        C01 = _mm_add_ps(b1, C01);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A0);
        C02 = _mm_add_ps(b2, C02);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A0);
        C03 = _mm_add_ps(b3, C03);
        
        A1 = _mm_load_ps(pA1); pA1 += 4;
        b0 = B0;
        b0 = _mm_mul_ps(b0, A1);
        C10 = _mm_add_ps(b0, C10);
        b1 = B1;
        b1 = _mm_mul_ps(b1, A1);
        C11 = _mm_add_ps(b1, C11);
        b2 = B2;
        b2 = _mm_mul_ps(b2, A1);
        C12 = _mm_add_ps(b2, C12);
        b3 = B3;
        b3 = _mm_mul_ps(b3, A1);
        C13 = _mm_add_ps(b3, C13);
      }
      
      // FIRST ROW
      // --- horizontal ADD ---     
      C00 = _mm_hadd_ps(C00, C01);
      C02 = _mm_hadd_ps(C02, C03);
      C00 = _mm_hadd_ps(C00, C02);
      // ---
      
      // SECOND ROW
      // --- horizontal ADD ---     
      C10 = _mm_hadd_ps(C10, C11);
      C12 = _mm_hadd_ps(C12, C13);
      C10 = _mm_hadd_ps(C10, C12);
      // ---
      
      _mm_store_ps(pC0, C00);
      _mm_store_ps(pC1, C10);
      
      pA0 -= KS_P4;
      pA1 = pA0 + KS_P4;
      
      pB0 += KS_P4*3;
      pB1 = pB0 + KS_P4;
      pB2 = pB1 + KS_P4;
      pB3 = pB2 + KS_P4;
      
      pC0 += 4;
      pC1 += 4;      
    }
    pA0 += KS_P4*2;
    pA1 = pA0 + KS_P4;
    
    pB0 = B_p;
    pB1 = pB0 + KS_P4;
    pB2 = pB1 + KS_P4;
    pB3 = pB2 + KS_P4;
    
    pC0 += GBLAS_KERNEL_SIZE;
    pC1 = pC0 + GBLAS_KERNEL_SIZE;
  }
  
  _mm_free(A_p);
  _mm_free(B_p);
}

void KERNEL_p4_sgemm_v2_r5_EC_v4(const int M, const int N,
                                const int K, const float alpha, const float *A,
                                const int lda, const float *B, const int ldb,
                                const float beta, float *C, const int ldc,
                                gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  const float TIGHT_PACKING_A_MAX = 2.0f*GBLAS_KERNEL_SIZE*(float)(Qa.get_quantizer_step()*Qa.get_max_value()*Qb.get_quantizer_step()*Qb.get_max_value());
  const float INV_EPS     =   ceil(sqrt(TIGHT_PACKING_A_MAX) + DELTA);
  const float EPS         =   1.0f/INV_EPS;

  const float INV_EPS_UP  =   ceil(INV_EPS*INV_EPS);
  const float EPS_UP      =   1.0f/INV_EPS_UP;
  
  // allocate memory for C
  float* C_positive = (float*)_mm_malloc(sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE, 16);
  float* C_negative = (float*)_mm_malloc(sizeof(float)*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE, 16);  
    
//#pragma omp parallel sections
  {
    /*
     * Positive section
     */
//#pragma omp section
    {
      KERNEL_p4_sgemm_v2_r5_P(A, B, C_positive, Qa, Qb, EPS, INV_EPS);
    }
    
    /*
     * Negative section
     */
//#pragma omp section
    {
      KERNEL_p4_sgemm_v2_r5_N(A, B, C_negative, Qa, Qb, EPS, INV_EPS);      
    }
  }
  
  /*
   * add and store into final C
   */
  const __m128 SCALE_FACTOR = _mm_set1_ps(0.5f);
  const __m128 BETA = _mm_set1_ps(beta);
  const __m128 DE_Q_FACTOR = _mm_set1_ps(1.0f/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  __m128 neg, pos;
  
  //#pragma omp parallel for private(neg, pos)
  for (int ii = 0; ii < GBLAS_KERNEL_SIZE; ii++)
  {
    for (int jj = 0; jj < GBLAS_KERNEL_SIZE; jj+=4)
    {
      // load positive
      pos = _mm_load_ps(&C_positive[ii*GBLAS_KERNEL_SIZE + jj]);
      // load negative
      neg = _mm_load_ps(&C_negative[ii*GBLAS_KERNEL_SIZE + jj]);
      
      // unpacking by elimination
      pos = _mm_add_ps(pos, neg);
      pos = _mm_mul_ps(pos, SCALE_FACTOR);
      
      fast_unpack_tight_v2(pos, EPS_UP, INV_EPS_UP);
      pos = gblas_quantizer::dequantize_sample(pos, DE_Q_FACTOR);
      
      if ( beta != 0.0f )
      {
        neg = _mm_loadu_ps(&C[ii*ldc + jj]);
        neg = _mm_mul_ps(neg, BETA);
        pos  = _mm_add_ps(neg, pos);
      }
      
      // unaligned store into C
      _mm_storeu_ps(&C[ii*ldc + jj], pos);
    }
  }
  
  _mm_free(C_positive);
  _mm_free(C_negative);
}

