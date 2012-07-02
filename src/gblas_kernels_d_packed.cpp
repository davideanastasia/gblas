/*
 *  gblas_kernels_d_packed.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 31/08/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <cmath>

#include "gblas_kernels.h"
#include "gblas_matrix_utils.h"


#define K_P2            (GBLAS_KERNEL_SIZE >> 1)
#define M_P2            (GBLAS_KERNEL_SIZE >> 1)
#define K_P3            (GBLAS_KERNEL_SIZE/3)
#define K_P4            (GBLAS_KERNEL_SIZE >> 2)
#define K_LOOPS         (K_P2 >> 1)
#define K_LOOPS_P3      (K_P3 >> 1)
#define K_LOOPS_P4      (K_P4 >> 1)

void KERNEL_p_dgemm_v1_r3(const int M, const int N,
                          const int K, const double alpha, const double *A /* M x K */,
                          const int lda, const double *B /* K x N */, const int ldb,
                          const float beta, double *C /* M x N*/, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  __m128d cmp, disp;                              // unpacking!
  const __m128d __beta = _mm_set1_pd(beta);
  
  const double TIGHT_PACKING_A_MAX  = 2.0*(Qa.get_quantizer_step()*Qa.get_max_value())*(Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128d DE_Q_FACTOR          = _mm_set1_pd(1.0/(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const double INV_EPS              = ceil(2.0*TIGHT_PACKING_A_MAX + DELTA);
  const double EPS                  = 1.0/INV_EPS;
  
//  double *A_p = (double*)_mm_malloc(sizeof(double)*(KERNEL_SIZE >> 1)*KERNEL_SIZE, 16);
//  double *B_p = (double*)_mm_malloc(sizeof(double)*KERNEL_SIZE*KERNEL_SIZE, 16);

  static double A_p[(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  static double B_p[GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  const __m128d __eps      = _mm_set1_pd(EPS);
  const __m128d __inv_eps  = _mm_set1_pd(INV_EPS);
  
  for (int i = 0, ih = 0; i < GBLAS_KERNEL_SIZE; i+=2, ih++)
  {
    for (int k = 0; k < GBLAS_KERNEL_SIZE; k+=2)
    {
      // quantization and packing of A
      cmp  = _mm_load_pd(&A[(i+1)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qa.quantize_sample(cmp);
      cmp  = _mm_mul_pd(cmp, __eps);
      
      disp = _mm_load_pd(&A[i*GBLAS_KERNEL_SIZE + k]);
      disp = Qa.quantize_sample(disp);
      
      cmp  = _mm_add_pd(cmp, disp);
      _mm_store_pd(&A_p[ih*GBLAS_KERNEL_SIZE + k], cmp);
      
      // quantization of B
      cmp  = _mm_load_pd(&B[i*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[i*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_pd(&B[(i+1)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[(i+1)*GBLAS_KERNEL_SIZE + k], cmp);
    }
  }
  
#ifdef DEBUG_PRINT
  cout << "A_p = "; print_matrix_matlab_notation(A_p, (KERNEL_SIZE >> 1), KERNEL_SIZE);
  cout << "B_p = "; print_matrix_matlab_notation(B_p, KERNEL_SIZE, KERNEL_SIZE);
#endif
  
  __m128d A0, A1;
  __m128d B0, B1;
  __m128d b0, b1;
  
  __m128d P00, P01, P02;
  __m128d P10, P11, P12;
  
  const double* stB = B_p + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const double* stA = A_p + GBLAS_KERNEL_SIZE*(GBLAS_KERNEL_SIZE >> 1);
  
  const double* pB0 = B_p;
  const double* pB1 = B_p + GBLAS_KERNEL_SIZE;
  
  const double* pA_p0 = A_p;
  const double* pA_p1 = A_p + GBLAS_KERNEL_SIZE;
  
  double* pC0       = C;
  double* pC1       = C + ldc;
  double* pC2       = C + ldc*2;
  double* pC3       = C + ldc*3;
  
  //for (int i = (M_p >> 1); i; i--)  
  do
  {
    //for (int j = (KERNEL_SIZE >> 2); j ; j-- )
    do
    {
      B0 = _mm_load_pd(pB0);          pB0 += 2;
      B1 = _mm_load_pd(pB1);          pB1 += 2;
      
      A0 = _mm_load_pd(pA_p0);        pA_p0 += 2;
      P00 = B0; 
      P00 = _mm_mul_pd(P00, A0);
      P01 = B1;
      P01 = _mm_mul_pd(P01, A0);
      
      A1 = _mm_load_pd(pA_p1);        pA_p1 += 2;
      P10 = B0; 
      P10 = _mm_mul_pd(P10, A1);
      P11 = B1;
      P11 = _mm_mul_pd(P11, A1);
      
      for ( int k = ((GBLAS_KERNEL_SIZE >> 1)-1); k ; k-- )
      {
        B0 = _mm_load_pd(pB0);          pB0 += 2;
        B1 = _mm_load_pd(pB1);          pB1 += 2;
        
        A0 = _mm_load_pd(pA_p0);        pA_p0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, A0);
        P00 = _mm_add_pd(P00, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A0);
        P01 = _mm_add_pd(P01, b1);
        
        A1 = _mm_load_pd(pA_p1);        pA_p1 += 2;        
        b0 = B0;
        b0 = _mm_mul_pd(b0, A1);
        P10 = _mm_add_pd(P10, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A1);
        P11 = _mm_add_pd(P11, b1);
      }
      
      // --- horizontal ADD ---     
      P00 = _mm_hadd_pd(P00, P01);
      // ---
      
      // unpacking #1
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC0);                                         // &C[i*KERNEL_SIZE + j]
      P02 = _mm_mul_pd(P02, __beta);                                  // C[i*N + j] *= beta;
      P02 = _mm_add_pd(P02, 
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[i*N + j] += sample_i;
      _mm_storeu_pd(pC0, P02);                                         // &C[i*KERNEL_SIZE + j]
      pC0    += 2;
      
      P00 = _mm_mul_pd(_mm_sub_pd(P00, P01), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC1);
      P02 = _mm_mul_pd(P02, __beta);                                  // C[(i+1)*N + j] *= beta;
      P02 = _mm_add_pd(P02,
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC1, P02);                                         // &C[(i+1)*KERNEL_SIZE + j]
      pC1    += 2;
      
      // --- horizontal ADD ---     
      P10 = _mm_hadd_pd(P10, P11);
      // ---
      
      // unpacking #2
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC2);
      P12 = _mm_mul_pd(P12, __beta);                                  // C[i*N + j] *= beta;
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[i*N + j] += sample_i;
      _mm_storeu_pd(pC2, P12);                                         // &C[(i+2)*KERNEL_SIZE + j]
      pC2    += 2;
      
      P10 = _mm_mul_pd(_mm_sub_pd(P10, P11), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC3);
      P12 = _mm_mul_pd(P12, __beta);                                  // C[(i+1)*N + j] *= beta;
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC3, P12);
      pC3    += 2;
      
      pA_p0  -= GBLAS_KERNEL_SIZE;
      pA_p1  -= GBLAS_KERNEL_SIZE;
      
      pB0    += GBLAS_KERNEL_SIZE;
      pB1    += GBLAS_KERNEL_SIZE;
    }
    while ( pB0 != stB );
    
    pA_p0 += GBLAS_KERNEL_SIZE*2;
    pA_p1 += GBLAS_KERNEL_SIZE*2;
    
    pB0 = B_p;
    pB1 = B_p + GBLAS_KERNEL_SIZE;
    
    pC0 += (ldc*4 - GBLAS_KERNEL_SIZE);      //  next 4 rows
    pC1 += (ldc*4 - GBLAS_KERNEL_SIZE);      //  next 4 rows
    pC2 += (ldc*4 - GBLAS_KERNEL_SIZE);      //  next 4 rows
    pC3 += (ldc*4 - GBLAS_KERNEL_SIZE);      //  next 4 rows
    
  }
  while ( pA_p0 != stA );
  
  //_mm_free(A_p);
  //_mm_free(B_p);
}

void KERNEL_p3_dgemm_v1_r3(const int M, const int N,
                          const int K, const double alpha, const double *A /* M x K */,
                          const int lda, const double *B /* K x N */, const int ldb,
                          const float beta, double *C /* M x N*/, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb)
{  
  __m128d cmp, disp;                              // unpacking!
  const __m128d __beta = _mm_set1_pd(beta);
  
  const double TIGHT_PACKING_A_MAX  = 2.0*(Qa.get_quantizer_step()*Qa.get_max_value())*(Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128d DE_Q_FACTOR         = _mm_set1_pd(1.0/(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const double INV_EPS              = ceil(2.0*TIGHT_PACKING_A_MAX + DELTA);
  const double EPS                  = 1.0/INV_EPS;
  
//  double *A_p = (double*)_mm_malloc(sizeof(double)*(KERNEL_SIZE/3)*KERNEL_SIZE, 16);
//  double *B_p = (double*)_mm_malloc(sizeof(double)*KERNEL_SIZE*KERNEL_SIZE, 16);

  static double A_p[(GBLAS_KERNEL_SIZE/3)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  static double B_p[GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  const __m128d __eps       = _mm_set1_pd(EPS);
  const __m128d __eps_pow_2 = _mm_set1_pd(EPS*EPS);
  const __m128d __inv_eps   = _mm_set1_pd(INV_EPS);
  
  for (int i = 0, ih = 0; i < GBLAS_KERNEL_SIZE; i+=3, ih++)
  {
    for (int k = 0; k < GBLAS_KERNEL_SIZE; k+=2)
    {
      // quantization and packing of A
      disp = _mm_load_pd(&A[i*GBLAS_KERNEL_SIZE + k]);
      cmp = Qa.quantize_sample(disp);
      
      disp  = _mm_load_pd(&A[(i+1)*GBLAS_KERNEL_SIZE + k]);
      disp  = Qa.quantize_sample(disp);
      disp  = _mm_mul_pd(disp, __eps);
      cmp  = _mm_add_pd(cmp, disp);

      disp  = _mm_load_pd(&A[(i+2)*GBLAS_KERNEL_SIZE + k]);
      disp  = Qa.quantize_sample(disp);
      disp  = _mm_mul_pd(disp, __eps_pow_2);
      cmp  = _mm_add_pd(cmp, disp);
      
      _mm_store_pd(&A_p[ih*GBLAS_KERNEL_SIZE + k], cmp);
      
      // quantization of B
      cmp  = _mm_load_pd(&B[i*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[i*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_pd(&B[(i+1)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[(i+1)*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_pd(&B[(i+2)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[(i+2)*GBLAS_KERNEL_SIZE + k], cmp);
    }
  }
  
#ifdef DEBUG_PRINT
  cout << "A_p = "; print_matrix_matlab_notation(A_p, (GBLAS_KERNEL_SIZE/3), GBLAS_KERNEL_SIZE);
  cout << "B_p = "; print_matrix_matlab_notation(B_p, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE);
#endif
  
  __m128d A0, A1;
  __m128d B0, B1;
  __m128d b0, b1;
  
  __m128d P00, P01, P02;
  __m128d P10, P11, P12;
  
  const double* stB = B_p + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const double* stA = A_p + GBLAS_KERNEL_SIZE*(GBLAS_KERNEL_SIZE/3);
  
  const double* pB0 = B_p;
  const double* pB1 = B_p + GBLAS_KERNEL_SIZE;
  
  const double* pA_p0 = A_p;
  const double* pA_p1 = A_p + GBLAS_KERNEL_SIZE;
  
  double* pC0       = C;
  double* pC1       = C + ldc;
  double* pC2       = C + ldc*2;
  double* pC3       = C + ldc*3;
  double* pC4       = C + ldc*4;
  double* pC5       = C + ldc*5;
  
  
  //for (int i = (M_p >> 1); i; i--)  
  do
  {
    //for (int j = (KERNEL_SIZE >> 2); j ; j-- )
    do
    {
      B0 = _mm_load_pd(pB0);          pB0 += 2;
      B1 = _mm_load_pd(pB1);          pB1 += 2;
      
      A0 = _mm_load_pd(pA_p0);        pA_p0 += 2;
      P00 = B0; 
      P00 = _mm_mul_pd(P00, A0);
      P01 = B1;
      P01 = _mm_mul_pd(P01, A0);
      
      A1 = _mm_load_pd(pA_p1);        pA_p1 += 2;
      P10 = B0; 
      P10 = _mm_mul_pd(P10, A1);
      P11 = B1;
      P11 = _mm_mul_pd(P11, A1);
      
      for ( int k = ((GBLAS_KERNEL_SIZE >> 1)-1); k ; k-- )
      {
        B0 = _mm_load_pd(pB0);          pB0 += 2;
        B1 = _mm_load_pd(pB1);          pB1 += 2;
        
        A0 = _mm_load_pd(pA_p0);        pA_p0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, A0);
        P00 = _mm_add_pd(P00, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A0);
        P01 = _mm_add_pd(P01, b1);
        
        A1 = _mm_load_pd(pA_p1);        pA_p1 += 2;        
        b0 = B0;
        b0 = _mm_mul_pd(b0, A1);
        P10 = _mm_add_pd(P10, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A1);
        P11 = _mm_add_pd(P11, b1);
      }
      
      // --- horizontal ADD ---     
      P00 = _mm_hadd_pd(P00, P01);
      // ---
      
      // unpacking #1
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC0);                                         // &C[i*KERNEL_SIZE + j]
      P02 = _mm_mul_pd(P02, __beta);                                  // C[i*N + j] *= beta;
      P02 = _mm_add_pd(P02, 
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[i*N + j] += sample_i;
      _mm_storeu_pd(pC0, P02);                                         // &C[i*ldc + j];
      pC0    += 2;
      
      P00 = _mm_mul_pd(_mm_sub_pd(P00, P01), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC1);
      P02 = _mm_mul_pd(P02, __beta);
      P02 = _mm_add_pd(P02,
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC1, P02);                                         // &C[(i+1)*ldc + j];
      pC1    += 2;
      
      P00 = _mm_mul_pd(_mm_sub_pd(P00, P01), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC2);
      P02 = _mm_mul_pd(P02, __beta);
      P02 = _mm_add_pd(P02,
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC2, P02);                                         // &C[(i+2)*ldc + j];
      pC2    += 2;
      
      
      
      // --- horizontal ADD ---     
      P10 = _mm_hadd_pd(P10, P11);
      // ---
      // unpacking #2
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC3);
      P12 = _mm_mul_pd(P12, __beta);
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[i*N + j] += sample_i;
      _mm_storeu_pd(pC3, P12);                                         // &C[(i+3)*ldc + j];
      pC3    += 2;
      
      P10 = _mm_mul_pd(_mm_sub_pd(P10, P11), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      
      //mm_p11 = mm_full_round_v2(mm_p10);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC4);
      P12 = _mm_mul_pd(P12, __beta);
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC4, P12);                                         // &C[(i+4)*ldc + j];
      pC4    += 2;
      
      P10 = _mm_mul_pd(_mm_sub_pd(P10, P11), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC5);
      P12 = _mm_mul_pd(P12, __beta);                                  
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC5, P12);                                         // &C[(i+5)*ldc + j];
      pC5    += 2;
      
      pA_p0  -= GBLAS_KERNEL_SIZE;
      pA_p1  -= GBLAS_KERNEL_SIZE;
      
      pB0    += GBLAS_KERNEL_SIZE;
      pB1    += GBLAS_KERNEL_SIZE;
    }
    while ( pB0 != stB );
    
    pA_p0 += GBLAS_KERNEL_SIZE*2;
    pA_p1 += GBLAS_KERNEL_SIZE*2;
    
    pB0 = B_p;
    pB1 = B_p + GBLAS_KERNEL_SIZE;
    
    pC0 += (ldc*6 - GBLAS_KERNEL_SIZE);      //  next 6 rows
    pC1 += (ldc*6 - GBLAS_KERNEL_SIZE);      //  next 6 rows
    pC2 += (ldc*6 - GBLAS_KERNEL_SIZE);      //  next 6 rows
    pC3 += (ldc*6 - GBLAS_KERNEL_SIZE);      //  next 6 rows
    pC4 += (ldc*6 - GBLAS_KERNEL_SIZE);      //  next 6 rows
    pC5 += (ldc*6 - GBLAS_KERNEL_SIZE);      //  next 6 rows
    
  }
  while ( pA_p0 != stA );
  
  //_mm_free(A_p);
  //_mm_free(B_p);
}

void KERNEL_p4_dgemm_v1_r3(const int M, const int N,
                           const int K, const double alpha, const double *A /* M x K */,
                           const int lda, const double *B /* K x N */, const int ldb,
                           const float beta, double *C /* M x N*/, const int ldc,
                           gblas_quantizer& Qa, gblas_quantizer& Qb)
{
  __m128d cmp, disp;                              // unpacking!
  const __m128d __beta = _mm_set1_pd(beta);
  
  const double TIGHT_PACKING_A_MAX  = 2.0*(Qa.get_quantizer_step()*Qa.get_max_value())*(Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128d DE_Q_FACTOR         = _mm_set1_pd(1.0/(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const double INV_EPS              = ceil(2.0*TIGHT_PACKING_A_MAX + DELTA);
  const double EPS                  = 1.0/INV_EPS;
  
//  double *A_p = (double*)_mm_malloc(sizeof(double)*(KERNEL_SIZE/4)*KERNEL_SIZE, 16);
//  double *B_p = (double*)_mm_malloc(sizeof(double)*KERNEL_SIZE*KERNEL_SIZE, 16);

  static double A_p[(GBLAS_KERNEL_SIZE >> 2)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  static double B_p[GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  const __m128d __eps       = _mm_set1_pd(EPS);
  const __m128d __eps_pow_2 = _mm_set1_pd(EPS*EPS);
  const __m128d __eps_pow_3 = _mm_set1_pd(EPS*EPS*EPS);
  const __m128d __inv_eps   = _mm_set1_pd(INV_EPS);
  
  for (int i = 0, ih = 0; i < GBLAS_KERNEL_SIZE; i+=4, ih++)
  {
    for (int k = 0; k < GBLAS_KERNEL_SIZE; k+=2)
    {
      // quantization and packing of A
      disp = _mm_load_pd(&A[i*GBLAS_KERNEL_SIZE + k]);
      cmp = Qa.quantize_sample(disp);
      
      disp  = _mm_load_pd(&A[(i+1)*GBLAS_KERNEL_SIZE + k]);
      disp  = Qa.quantize_sample(disp);
      disp  = _mm_mul_pd(disp, __eps);
      cmp  = _mm_add_pd(cmp, disp);
      
      disp  = _mm_load_pd(&A[(i+2)*GBLAS_KERNEL_SIZE + k]);
      disp  = Qa.quantize_sample(disp);
      disp  = _mm_mul_pd(disp, __eps_pow_2);
      cmp  = _mm_add_pd(cmp, disp);

      disp  = _mm_load_pd(&A[(i+3)*GBLAS_KERNEL_SIZE + k]);
      disp  = Qa.quantize_sample(disp);
      disp  = _mm_mul_pd(disp, __eps_pow_3);
      cmp  = _mm_add_pd(cmp, disp);
      
      _mm_store_pd(&A_p[ih*GBLAS_KERNEL_SIZE + k], cmp);
      
      // quantization of B
      cmp  = _mm_load_pd(&B[i*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[i*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_pd(&B[(i+1)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[(i+1)*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_pd(&B[(i+2)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[(i+2)*GBLAS_KERNEL_SIZE + k], cmp);
      
      cmp  = _mm_load_pd(&B[(i+3)*GBLAS_KERNEL_SIZE + k]);
      cmp  = Qb.quantize_sample(cmp);
      _mm_store_pd(&B_p[(i+3)*GBLAS_KERNEL_SIZE + k], cmp);
    }
  }
  
#ifdef DEBUG_PRINT
  cout << "A_p = "; print_matrix_matlab_notation(A_p, (GBLAS_KERNEL_SIZE/4), GBLAS_KERNEL_SIZE);
  cout << "B_p = "; print_matrix_matlab_notation(B_p, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE);
#endif
  
  __m128d A0, A1;
  __m128d B0, B1;
  __m128d b0, b1;
  
  __m128d P00, P01, P02;
  __m128d P10, P11, P12;
  
  const double* stB = B_p + GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const double* stA = A_p + GBLAS_KERNEL_SIZE*(GBLAS_KERNEL_SIZE/4);
  
  const double* pB0 = B_p;
  const double* pB1 = B_p + GBLAS_KERNEL_SIZE;
  
  const double* pA_p0 = A_p;
  const double* pA_p1 = A_p + GBLAS_KERNEL_SIZE;
  
  double* pC0       = C;
  double* pC1       = C + ldc;
  double* pC2       = C + ldc*2;
  double* pC3       = C + ldc*3;
  double* pC4       = C + ldc*4;
  double* pC5       = C + ldc*5;
  double* pC6       = C + ldc*6;
  double* pC7       = C + ldc*7;  
  
  //for (int i = (M_p >> 1); i; i--)  
  do
  {
    //for (int j = (KERNEL_SIZE >> 2); j ; j-- )
    do
    {
      B0 = _mm_load_pd(pB0);          pB0 += 2;
      B1 = _mm_load_pd(pB1);          pB1 += 2;
      
      A0 = _mm_load_pd(pA_p0);        pA_p0 += 2;
      P00 = B0; 
      P00 = _mm_mul_pd(P00, A0);
      P01 = B1;
      P01 = _mm_mul_pd(P01, A0);
      
      A1 = _mm_load_pd(pA_p1);        pA_p1 += 2;
      P10 = B0; 
      P10 = _mm_mul_pd(P10, A1);
      P11 = B1;
      P11 = _mm_mul_pd(P11, A1);
      
      for ( int k = ((GBLAS_KERNEL_SIZE >> 1)-1); k ; k-- )
      {
        B0 = _mm_load_pd(pB0);          pB0 += 2;
        B1 = _mm_load_pd(pB1);          pB1 += 2;
        
        A0 = _mm_load_pd(pA_p0);        pA_p0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, A0);
        P00 = _mm_add_pd(P00, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A0);
        P01 = _mm_add_pd(P01, b1);
        
        A1 = _mm_load_pd(pA_p1);        pA_p1 += 2;        
        b0 = B0;
        b0 = _mm_mul_pd(b0, A1);
        P10 = _mm_add_pd(P10, b0);
        b1 = B1;
        b1 = _mm_mul_pd(b1, A1);
        P11 = _mm_add_pd(P11, b1);
      }
      
      // --- horizontal ADD ---     
      P00 = _mm_hadd_pd(P00, P01);
      // ---
      
      // unpacking #1
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC0);                                         // &C[i*KERNEL_SIZE + j]
      P02 = _mm_mul_pd(P02, __beta);                                  // C[i*N + j] *= beta;
      P02 = _mm_add_pd(P02, 
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[i*N + j] += sample_i;
      _mm_storeu_pd(pC0, P02);                                         // &C[i*ldc + j];
      pC0    += 2;
      
      P00 = _mm_mul_pd(_mm_sub_pd(P00, P01), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC1);
      P02 = _mm_mul_pd(P02, __beta);
      P02 = _mm_add_pd(P02,
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC1, P02);                                         // &C[(i+1)*ldc + j];
      pC1    += 2;
      
      P00 = _mm_mul_pd(_mm_sub_pd(P00, P01), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC2);
      P02 = _mm_mul_pd(P02, __beta);
      P02 = _mm_add_pd(P02,
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC2, P02);                                         // &C[(i+2)*ldc + j];
      pC2    += 2;
      
      P00 = _mm_mul_pd(_mm_sub_pd(P00, P01), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p01 = mm_full_round_v2(mm_p00);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P00,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P01      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P00, disp)));
      
      P02 = _mm_loadu_pd(pC3);
      P02 = _mm_mul_pd(P02, __beta);
      P02 = _mm_add_pd(P02,
                       gblas_quantizer::dequantize_sample(P01, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC3, P02);                                         // &C[(i+2)*ldc + j];
      pC3    += 2;
      
      // --- horizontal ADD ---     
      P10 = _mm_hadd_pd(P10, P11);
      // ---
      // unpacking #2
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);      
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC4);
      P12 = _mm_mul_pd(P12, __beta);
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[i*N + j] += sample_i;
      _mm_storeu_pd(pC4, P12);                                         // &C[(i+3)*ldc + j];
      pC4    += 2;
      
      P10 = _mm_mul_pd(_mm_sub_pd(P10, P11), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p11 = mm_full_round_v2(mm_p10);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC5);
      P12 = _mm_mul_pd(P12, __beta);
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC5, P12);                                         // &C[(i+4)*ldc + j];
      pC5    += 2;
      
      P10 = _mm_mul_pd(_mm_sub_pd(P10, P11), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC6);
      P12 = _mm_mul_pd(P12, __beta);                                  
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC6, P12);                                         // &C[(i+5)*ldc + j];
      pC6    += 2;

      P10 = _mm_mul_pd(_mm_sub_pd(P10, P11), __inv_eps);              //sample_d = (sample_d - sample_i)/eps;
      //mm_p11 = mm_full_round_v2(mm_p10);                            //sample_i = full_round(sample_d);
      cmp      = _mm_cmpgt_pd(P10,  _MM_ZERO_D);
      disp     = _mm_and_pd(cmp,    _MM_MASK_ONE_D);
      disp     = _mm_sub_pd(disp,   _MM_ZERO_DOT_FIVE_D);
      P11      = _mm_cvtepi32_pd(_mm_cvttpd_epi32(_mm_add_pd(P10, disp)));
      
      P12 = _mm_loadu_pd(pC7);
      P12 = _mm_mul_pd(P12, __beta);                                  
      P12 = _mm_add_pd(P12,
                       gblas_quantizer::dequantize_sample(P11, DE_Q_FACTOR));
      // C[(i+1)*N + j] += sample_i;
      _mm_storeu_pd(pC7, P12);                                         // &C[(i+5)*ldc + j];
      pC7    += 2;
      
      pA_p0  -= GBLAS_KERNEL_SIZE;
      pA_p1  -= GBLAS_KERNEL_SIZE;
      
      pB0    += GBLAS_KERNEL_SIZE;
      pB1    += GBLAS_KERNEL_SIZE;
    }
    while ( pB0 != stB );
    
    pA_p0 += GBLAS_KERNEL_SIZE*2;
    pA_p1 += GBLAS_KERNEL_SIZE*2;
    
    pB0 = B_p;
    pB1 = B_p + GBLAS_KERNEL_SIZE;
    
    pC0 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC1 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC2 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC3 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC4 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC5 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC6 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows
    pC7 += (ldc*8 - GBLAS_KERNEL_SIZE);      //  next 8 rows    
  }
  while ( pA_p0 != stA );
  
  //_mm_free(A_p);
  //_mm_free(B_p);
}

void KERNEL_p_dgemm_v2_r4(const int M, const int N,
                          const int K, const double alpha, const double *A,
                          const int lda, const double *B, const int ldb,
                          const double beta, double *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb)
{ 
  const __m128d __beta = _mm_set1_pd(beta);
  
  const double TIGHT_PACKING_A_MAX  = 2.0*(Qa.get_quantizer_step()*Qa.get_max_value())*(Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128d DE_Q_FACTOR         = _mm_set1_pd(1.0/(float)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const double INV_EPS              = ceil(2.0*TIGHT_PACKING_A_MAX + DELTA);
  const double EPS                  = 1.0/INV_EPS;
  
  static double A_p[(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  static double B_p[(GBLAS_KERNEL_SIZE >> 1)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  //double *A_p = (double*)_mm_malloc(sizeof(double)*(KERNEL_SIZE >> 1)*KERNEL_SIZE, 16);
  //double *B_p = (double*)_mm_malloc(sizeof(double)*(KERNEL_SIZE >> 1)*KERNEL_SIZE, 16);
  
  __m128d acc0, acc1;
  
  const __m128d a_factor_1 = _mm_set_pd(EPS, 1.0);
  const __m128d a_factor_2 = _mm_set_pd(-EPS, 1.0);
  const __m128d b_factor_1 = _mm_set_pd(INV_EPS, 1.0);
  const __m128d b_factor_2 = _mm_set_pd(-INV_EPS, 1.0);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=4, kh+=2 )
  {
    //A_p[kh]     =   A_p[k] + eps     * A_p[k+1];
    //A_p[kh + 1] =   A_p[k + 2] - eps     * A_p[k+3];
    /* packing A */
    acc0 = _mm_load_pd(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_pd(acc0, a_factor_1);
    
    acc1 = _mm_load_pd(&A[k + 2]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_pd(acc1, a_factor_2);
    
    acc0 = _mm_hadd_pd(acc0, acc1);
    
    _mm_store_pd(&A_p[kh], acc0);
    
    //B_p[kh]     =   B_p[k] + inv_eps * B_p[k+1];
    //B_p[kh + 1] =   B_p[k + 2] - inv_eps * B_p[k+3];
    /* packing B */
    acc0 = _mm_load_pd(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_pd(acc0, b_factor_1);
    
    acc1 = _mm_load_pd(&B[k + 2]);
    acc1 = Qb.quantize_sample(acc1);    
    acc1 = _mm_mul_pd(acc1, b_factor_2);
    
    acc0 = _mm_hadd_pd(acc0, acc1);
    
    _mm_store_pd(&B_p[kh], acc0);
  }
  
  __m128d mmA0, mmA1;
  
  __m128d B0, B1; //, B2, B3;
  __m128d b0, b1; //, b2, b3;
  
  __m128d mmC00, mmC01; //, mmC02, mmC03;
  __m128d mmC10, mmC11; //, mmC12, mmC13; 
  
  const double* pA_p0 = A_p;
  const double* pA_p1 = A_p + K_P2;
  
  const double* pB_p0 = B_p;
  const double* pB_p1 = B_p + K_P2;
  
  //  const double* stB = B + KERNEL_SIZE*K_P2;
  //  const double* stA = A + KERNEL_SIZE*K_P2;
  
  double* pC0 = C;
  double* pC1 = C + ldc;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
    //do
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 1); j; j--)
      //do
    {     
      B0 = _mm_load_pd(pB_p0); pB_p0 += 2;
      B1 = _mm_load_pd(pB_p1); pB_p1 += 2;
      
      mmA0 = _mm_load_pd(pA_p0);
      pA_p0 += 2;
      mmC00 = B0;
      mmC00 = _mm_mul_pd(mmC00, mmA0);
      mmC01 = B1;
      mmC01 = _mm_mul_pd(mmC01, mmA0);
      
      mmA1 = _mm_load_pd(pA_p1);
      pA_p1 += 2;
      mmC10 = B0;
      mmC10 = _mm_mul_pd(mmC10, mmA1);
      mmC11 = B1;
      mmC11 = _mm_mul_pd(mmC11, mmA1);
      
      for (int k = (K_LOOPS-1); k; k--)  
      {
        B0 = _mm_load_pd(pB_p0); pB_p0 += 2;
        B1 = _mm_load_pd(pB_p1); pB_p1 += 2;
        
        mmA0 = _mm_load_pd(pA_p0);
        pA_p0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, mmA0);
        mmC00 = _mm_add_pd(b0, mmC00);
        b1 = B1;
        b1 = _mm_mul_pd(b1, mmA0);
        mmC01 = _mm_add_pd(b1, mmC01);
        
        mmA1 = _mm_load_pd(pA_p1);
        pA_p1 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, mmA1);
        mmC10 = _mm_add_pd(b0, mmC10);
        b1 = B1;
        b1 = _mm_mul_pd(b1, mmA1);
        mmC11 = _mm_add_pd(b1, mmC11);
      }
      
      acc0 = _mm_loadu_pd(pC0);
      acc0 = _mm_mul_pd(acc0, __beta);
      // --- horizontal ADD ---     
      mmC00 = _mm_hadd_pd(mmC00, mmC01);
      // ---
      fast_unpack_tight_v2(mmC00, EPS, INV_EPS);
      mmC00 = gblas_quantizer::dequantize_sample(mmC00, DE_Q_FACTOR);
      
      acc0 = _mm_add_pd(acc0, mmC00);
      _mm_storeu_pd(pC0, acc0);
      
      acc1 = _mm_loadu_pd(pC1);
      acc1 = _mm_mul_pd(acc1, __beta);
      // --- horizontal ADD ---     
      mmC10 = _mm_hadd_pd(mmC10, mmC11);
      // ---
      fast_unpack_tight_v2(mmC10, EPS, INV_EPS);
      mmC10 = gblas_quantizer::dequantize_sample(mmC10, DE_Q_FACTOR);
      
      acc1 = _mm_add_pd(acc1, mmC10);
      _mm_storeu_pd(pC1, acc1);
      
      pA_p0 -= K_P2;
      pA_p1 -= K_P2;
      
      pB_p0 += K_P2;
      pB_p1 += K_P2;
      
      pC0 += 2;
      pC1 += 2;
    }
    //while ( pB_p0 != stB );
    
    pA_p0 += K_P2*2;
    pA_p1 += K_P2*2;
    
    pB_p0 = B_p;
    pB_p1 = B_p + K_P2;
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
    pC1 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
  }
  //while ( pA_p0 != stA );
  
  //_mm_free(A_p);
  //_mm_free(B_p);
}

// Derived from KERNEL_p3_dgemm_v2_r3, but implements quantization in the right way!
void KERNEL_p3_dgemm_v2_r4(const int M, const int N,
                          const int K, const double alpha, const double *A,
                          const int lda, const double *B, const int ldb,
                          const double beta, double *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb)
{ 
  const __m128d __beta              = _mm_set1_pd(beta);
  
  const double TIGHT_PACKING_A_MAX  = 2.0*(Qa.get_quantizer_step()*Qa.get_max_value())*(Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128d DE_Q_FACTOR         = _mm_set1_pd(1.0/(double)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const double INV_EPS              = ceil(2.0*TIGHT_PACKING_A_MAX + DELTA);
  const double EPS                  = 1.0/INV_EPS;
  
  static double A_p[(GBLAS_KERNEL_SIZE/3)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  static double B_p[(GBLAS_KERNEL_SIZE/3)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
     
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=3, kh++ )
  {
    /* packing A */
    A_p[kh]
    = Qa.quantize_sample(A[k])
    + EPS * Qa.quantize_sample(A[k+1])
    + (EPS*EPS) * Qa.quantize_sample(A[k+2]);
    
    /* packing B */ 
    B_p[kh]
    = Qb.quantize_sample(B[k])
    + INV_EPS * Qb.quantize_sample(B[k+1])
    + (INV_EPS*INV_EPS) * Qb.quantize_sample(B[k+2]);
  }
  
  __m128d acc0, acc1;
  __m128d mmA0, mmA1;
  
  __m128d B0, B1; //, B2, B3;
  __m128d b0, b1; //, b2, b3;
  
  __m128d mmC00, mmC01; //, mmC02, mmC03;
  __m128d mmC10, mmC11; //, mmC12, mmC13; 
  
  const double* pA_p0 = A_p;
  const double* pA_p1 = A_p + K_P3;
  
  const double* pB_p0 = B_p;
  const double* pB_p1 = B_p + K_P3;
  
  //  const double* stB = B + KERNEL_SIZE*K_P2;
  //  const double* stA = A + KERNEL_SIZE*K_P2;
  
  double* pC0 = C;
  double* pC1 = C + ldc;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
    //do
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 1); j; j--)
      //do
    {     
      B0 = _mm_load_pd(pB_p0); pB_p0 += 2;
      B1 = _mm_load_pd(pB_p1); pB_p1 += 2;
      
      mmA0 = _mm_load_pd(pA_p0);
      pA_p0 += 2;
      mmC00 = B0;
      mmC00 = _mm_mul_pd(mmC00, mmA0);
      mmC01 = B1;
      mmC01 = _mm_mul_pd(mmC01, mmA0);
      
      mmA1 = _mm_load_pd(pA_p1);
      pA_p1 += 2;
      mmC10 = B0;
      mmC10 = _mm_mul_pd(mmC10, mmA1);
      mmC11 = B1;
      mmC11 = _mm_mul_pd(mmC11, mmA1);
      
      for (int k = (K_LOOPS_P3-1); k; k--)  
      {
        B0 = _mm_load_pd(pB_p0); pB_p0 += 2;
        B1 = _mm_load_pd(pB_p1); pB_p1 += 2;
        
        mmA0 = _mm_load_pd(pA_p0);
        pA_p0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, mmA0);
        mmC00 = _mm_add_pd(b0, mmC00);
        b1 = B1;
        b1 = _mm_mul_pd(b1, mmA0);
        mmC01 = _mm_add_pd(b1, mmC01);
        
        mmA1 = _mm_load_pd(pA_p1);
        pA_p1 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, mmA1);
        mmC10 = _mm_add_pd(b0, mmC10);
        b1 = B1;
        b1 = _mm_mul_pd(b1, mmA1);
        mmC11 = _mm_add_pd(b1, mmC11);
      }
      
      // --- horizontal ADD ---     
      mmC00 = _mm_hadd_pd(mmC00, mmC01);
      // ---
      fast_unpack_tight_v4(mmC00, EPS, INV_EPS);
      mmC00 = gblas_quantizer::dequantize_sample(mmC00, DE_Q_FACTOR);

      acc0 = _mm_loadu_pd(pC0);
      acc0 = _mm_mul_pd(acc0, __beta);
      acc0 = _mm_add_pd(acc0, mmC00);
      _mm_storeu_pd(pC0, acc0);
      
      // --- horizontal ADD ---     
      mmC10 = _mm_hadd_pd(mmC10, mmC11);
      // ---
      fast_unpack_tight_v4(mmC10, EPS, INV_EPS);
      mmC10 = gblas_quantizer::dequantize_sample(mmC10, DE_Q_FACTOR);
      
      acc1 = _mm_loadu_pd(pC1);
      acc1 = _mm_mul_pd(acc1, __beta);
      acc1 = _mm_add_pd(acc1, mmC10);
      _mm_storeu_pd(pC1, acc1);
      
      pA_p0 -= K_P3;
      pA_p1 -= K_P3;
      
      pB_p0 += K_P3;
      pB_p1 += K_P3;
      
      pC0 += 2;
      pC1 += 2;
    }
    //while ( pB_p0 != stB );
    
    pA_p0 += K_P3*2;
    pA_p1 += K_P3*2;
    
    pB_p0 = B_p;
    pB_p1 = B_p + K_P3;
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
    pC1 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
  }
  //while ( pA_p0 != stA );
}

// Derived from KERNEL_p_dgemm_v2_r3, but implements quantization in the right way!
void KERNEL_p4_dgemm_v2_r4(const int M, const int N,
                           const int K, const double alpha, const double *A,
                           const int lda, const double *B, const int ldb,
                           const double beta, double *C, const int ldc,
                           gblas_quantizer& Qa, gblas_quantizer& Qb)
{ 
  const __m128d __beta = _mm_set1_pd(beta);
  
  const double TIGHT_PACKING_A_MAX  = 2.0*(Qa.get_quantizer_step()*Qa.get_max_value())*(Qb.get_quantizer_step()*Qb.get_max_value())*GBLAS_KERNEL_SIZE;
  const __m128d DE_Q_FACTOR         = _mm_set1_pd(1.0/(double)(Qa.get_quantizer_step()*Qb.get_quantizer_step()));
  const double INV_EPS              = ceil(2.0*TIGHT_PACKING_A_MAX + DELTA);
  const double EPS                  = 1.0/INV_EPS;
  
  static double A_p[(GBLAS_KERNEL_SIZE >> 2)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  static double B_p[(GBLAS_KERNEL_SIZE >> 2)*GBLAS_KERNEL_SIZE] __attribute__ ((aligned (16)));
  
  __m128d acc0, acc1;
  const __m128d A_eps0 = _mm_set_pd(EPS, 1.0);
  const __m128d A_eps1 = _mm_set_pd(EPS*EPS*EPS, EPS*EPS);

  const __m128d B_eps0 = _mm_set_pd(INV_EPS, 1.0);
  const __m128d B_eps1 = _mm_set_pd(INV_EPS*INV_EPS*INV_EPS, INV_EPS*INV_EPS);
  
  for ( int k = 0, kh = 0; k < GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE; k+=4, kh++ )
  {   
    /* packing A */
//    A_p[kh]
//    = Qa.quantize_sample(A[k])
//    + EPS * Qa.quantize_sample(A[k+1])
//    + (EPS*EPS) * Qa.quantize_sample(A[k+2])
//    + (EPS*EPS*EPS) * Qa.quantize_sample(A[k+3]);
    
    acc0 = _mm_load_pd(&A[k]);
    acc0 = Qa.quantize_sample(acc0);
    acc0 = _mm_mul_pd(acc0, A_eps0);
    
    acc1 = _mm_load_pd(&A[k+2]);
    acc1 = Qa.quantize_sample(acc1);
    acc1 = _mm_mul_pd(acc1, A_eps1);
    
    acc0 = _mm_hadd_pd(acc0, acc1);
    acc0 = _mm_hadd_pd(acc0, acc1);
    
    _mm_store_sd(&A_p[kh], acc0);
    
    /* packing B */ 
//    B_p[kh]
//    = Qb.quantize_sample(B[k])
//    + INV_EPS * Qb.quantize_sample(B[k+1])
//    + (INV_EPS*INV_EPS) * Qb.quantize_sample(B[k+2])
//    + (INV_EPS*INV_EPS*INV_EPS) * Qb.quantize_sample(B[k+3]);
    
    acc0 = _mm_load_pd(&B[k]);
    acc0 = Qb.quantize_sample(acc0);
    acc0 = _mm_mul_pd(acc0, B_eps0);
    
    acc1 = _mm_load_pd(&B[k+2]);
    acc1 = Qb.quantize_sample(acc1);
    acc1 = _mm_mul_pd(acc1, B_eps1);
    
    acc0 = _mm_hadd_pd(acc0, acc1);
    acc0 = _mm_hadd_pd(acc0, acc1);
    
    _mm_store_sd(&B_p[kh], acc0);
  }
  
//  cout << "A_p = " << endl; print_matrix_matlab_notation(A_p, KERNEL_SIZE, (KERNEL_SIZE >> 2));
//  cout << "B_p = " << endl; print_matrix_matlab_notation(B_p, KERNEL_SIZE, (KERNEL_SIZE >> 2));
  
  __m128d mmA0, mmA1;
  
  __m128d B0, B1; //, B2, B3;
  __m128d b0, b1; //, b2, b3;
  
  __m128d mmC00, mmC01; //, mmC02, mmC03;
  __m128d mmC10, mmC11; //, mmC12, mmC13; 
  
  const double* pA_p0 = A_p;
  const double* pA_p1 = A_p + K_P4;
  
  const double* pB_p0 = B_p;
  const double* pB_p1 = B_p + K_P4;
   
  double* pC0 = C;
  double* pC1 = C + ldc;
  
  for (int i = (GBLAS_KERNEL_SIZE >> 1); i; i--) // M // rows A
    //do
  {
    for (int j = (GBLAS_KERNEL_SIZE >> 1); j; j--)
      //do
    {     
      B0 = _mm_load_pd(pB_p0); pB_p0 += 2;
      B1 = _mm_load_pd(pB_p1); pB_p1 += 2;
      
      mmA0 = _mm_load_pd(pA_p0);
      pA_p0 += 2;
      mmC00 = B0;
      mmC00 = _mm_mul_pd(mmC00, mmA0);
      mmC01 = B1;
      mmC01 = _mm_mul_pd(mmC01, mmA0);
      
      mmA1 = _mm_load_pd(pA_p1);
      pA_p1 += 2;
      mmC10 = B0;
      mmC10 = _mm_mul_pd(mmC10, mmA1);
      mmC11 = B1;
      mmC11 = _mm_mul_pd(mmC11, mmA1);
      
      for (int k = (K_LOOPS_P4-1); k; k--)  
      {
        B0 = _mm_load_pd(pB_p0); pB_p0 += 2;
        B1 = _mm_load_pd(pB_p1); pB_p1 += 2;
        
        mmA0 = _mm_load_pd(pA_p0);
        pA_p0 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, mmA0);
        mmC00 = _mm_add_pd(b0, mmC00);
        b1 = B1;
        b1 = _mm_mul_pd(b1, mmA0);
        mmC01 = _mm_add_pd(b1, mmC01);
        
        mmA1 = _mm_load_pd(pA_p1);
        pA_p1 += 2;
        b0 = B0;
        b0 = _mm_mul_pd(b0, mmA1);
        mmC10 = _mm_add_pd(b0, mmC10);
        b1 = B1;
        b1 = _mm_mul_pd(b1, mmA1);
        mmC11 = _mm_add_pd(b1, mmC11);
      }
      
      // --- horizontal ADD ---     
      mmC00 = _mm_hadd_pd(mmC00, mmC01);
      // ---
      fast_unpack_tight_v4(mmC00, EPS, INV_EPS);
      mmC00 = gblas_quantizer::dequantize_sample(mmC00, DE_Q_FACTOR);

      acc0 = _mm_loadu_pd(pC0);
      acc0 = _mm_mul_pd(acc0, __beta);
      acc0 = _mm_add_pd(acc0, mmC00);
      _mm_storeu_pd(pC0, acc0);
      
      // --- horizontal ADD ---     
      mmC10 = _mm_hadd_pd(mmC10, mmC11);
      // ---
      fast_unpack_tight_v4(mmC10, EPS, INV_EPS);
      mmC10 = gblas_quantizer::dequantize_sample(mmC10, DE_Q_FACTOR);
      
      acc1 = _mm_loadu_pd(pC1);
      acc1 = _mm_mul_pd(acc1, __beta);
      acc1 = _mm_add_pd(acc1, mmC10);
      _mm_storeu_pd(pC1, acc1);
      
      pA_p0 -= K_P4;
      pA_p1 -= K_P4;
      
      pB_p0 += K_P4;
      pB_p1 += K_P4;
      
      pC0 += 2;
      pC1 += 2;
    }
    //while ( pB_p0 != stB );
    
    pA_p0 += K_P4*2;
    pA_p1 += K_P4*2;
    
    pB_p0 = B_p;
    pB_p1 = B_p + K_P4;
    
    pC0 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
    pC1 += (ldc*2 - GBLAS_KERNEL_SIZE);       //  next 2 rows
  }
  //while ( pA_p0 != stA );
}