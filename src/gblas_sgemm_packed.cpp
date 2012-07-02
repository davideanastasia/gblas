/*
 *  gblas_sgemm_packed.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 07/07/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 *  A[M*K]
 *  B[K*N]
 *  C[M*N]
 *
 */

#include <cmath>
#include <float.h>
#include <iostream>

#include "gblas_kernels.h"
#include "gblas_stat_model.h"
#include "gblas_matrix_utils.h"


int gblas_sgemm_snr(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const float alpha, const float *A /* M x K */,
              const int lda, const float *B /* K x N */, const int ldb,
              const float beta, float *C /* M x N*/, const int ldc,
              const double TARGET_SNR, const double VAR_A, const double VAR_B)
{
  // conversion to MSE
  double target_mse = (double(K)*VAR_A*VAR_B)/(pow(10.0, (TARGET_SNR/10.0)));
  
  return gblas_sgemm_mse(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, target_mse);
}

int gblas_sgemm_mse(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const float alpha, const float *A /* M x K */,
              const int lda, const float *B /* K x N */, const int ldb,
              const float beta, float *C /* M x N*/, const int ldc,
              const double TARGET_MSE
#ifdef EXTERNAL_A_QUANT
              , const double  A_QUANT
#endif
              )
{
  const int BLOCK_ELEMS = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  int num_accelerated_block = 0;  
 
  float* A_max  = (float*)_mm_malloc(sizeof(float)*K_blocks*M_blocks, 16);
  float* A_min  = (float*)_mm_malloc(sizeof(float)*K_blocks*M_blocks, 16);
  float* __A    = (float*)_mm_malloc(sizeof(float)*K*M, 16); 

  float* B_max  = (float*)_mm_malloc(sizeof(float)*K_blocks*N_blocks, 16);
  float* B_min  = (float*)_mm_malloc(sizeof(float)*K_blocks*N_blocks, 16);
  float* __B = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  
  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A, A_max, A_min);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A, A_max, A_min);
  //exit(-1);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B, B_max, B_min);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B, B_max, B_min);
  //exit(-1);
  
  float *p_A = __A;
  float *p_B = __B;
  float *p_C = C;
  
  // --- Best parameters selection ---
  gblas_quantizer Qa;
  gblas_quantizer Qb;
  
  float* max_a  = new float[K_blocks];
  float* max_b  = new float[K_blocks];
  float* Ca     = new float[K_blocks];
  float* Cb     = new float[K_blocks];
  double* P_e   = new double[K_blocks];
  int*   Ctrl   = new int[K_blocks];
  // ---------------------------------
  
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B
    {
      // ----
      double cumulative_P_e = 0.0;
      {
        for ( int c = 0; c < K_blocks; c++ )
        {
          max_a[c] = max(abs(A_max[ii*K/GBLAS_KERNEL_SIZE + c]), abs(A_min[ii*K/GBLAS_KERNEL_SIZE + c]));
          max_b[c] = max(abs(B_max[jj*K/GBLAS_KERNEL_SIZE + c]), abs(B_min[jj*K/GBLAS_KERNEL_SIZE + c]));
#ifdef EXTERNAL_A_QUANT
          P_e[c] = control_pack_packing_based_s_external_Aquant(max_a[c], max_b[c], Ca[c], Cb[c], Ctrl[c], A_QUANT);  // select best parameters for each block
#else
          P_e[c] = control_pack_packing_based_s(max_a[c], max_b[c], Ca[c], Cb[c], Ctrl[c]);  // select best parameters for each block
#endif
          
          cumulative_P_e += P_e[c];
        }
#ifndef EXTERNAL_A_QUANT
#ifdef ENABLE_PRUNING
        select_accelerated_blocks_by_quality_s(K_blocks, P_e, cumulative_P_e, Ctrl, TARGET_MSE);
#endif
#else
        std::cout << cumulative_P_e << ", " << Ca[0] << ", " << Cb[0] << ", ";
#endif
      }
      // ----
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      p_A = &__A[(ii*K_blocks)*BLOCK_ELEMS];
      p_B = &__B[(jj*K_blocks)*BLOCK_ELEMS];
      
      if ( Ctrl[0] > 1 )
      {        
        std::cout << 2 << " " << max_a[0] << " " << Ca[0] << " " << max_b[0] << " " << Cb[0] << std::endl;

        // pack 2
        Qa.set_min_value(0);
        Qb.set_min_value(0);
        
        Qa.set_max_value(max_a[0]);
        Qb.set_max_value(max_b[0]);
        
        Qa.set_q_step(Ca[0]);
        Qb.set_q_step(Cb[0]);
        
#ifdef FLOAT_SYM        
        KERNEL_p_sgemm_v2_r5(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc, Qa, Qb);
#else
        KERNEL_p_sgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc,  Qa, Qb);
#endif
        num_accelerated_block++;
      }
      else
      {
        std::cerr << 1 << " " << max_a[0] << " " << 1.0 << " " << max_b[0] << " " << 1.0 << std::endl;
        
        // standard
        KERNEL_std_sgemm_v6(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc);
      }
      
      for (int kk = 1; kk < K_blocks; kk++)
      {
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        if ( Ctrl[kk] > 1 )
        {
          //cerr << 2 << " " << max_a[kk] << " " << Ca[kk] << " " << max_b[kk] << " " << Cb[kk]; // << endl;
          
          // pack 2
          Qa.set_min_value(0);
          Qb.set_min_value(0);
          
          Qa.set_max_value(max_a[kk]);
          Qb.set_max_value(max_b[kk]);
          
          Qa.set_q_step(Ca[kk]);
          Qb.set_q_step(Cb[kk]);
          
#ifdef FLOAT_SYM
          KERNEL_p_sgemm_v2_r5(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc,  Qa, Qb);
#else
          KERNEL_p_sgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc,  Qa, Qb);
#endif
          num_accelerated_block++;
        }
        else
        {
          //cerr << 1 << " " << max_a[kk] << " " << 1.0 << " " << max_b[kk] << " " << 1.0; // << endl;
          
          // standard
          KERNEL_std_sgemm_v6(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc);
        }
      }
    }
  }
  
  _mm_free(__A);
  _mm_free(A_max);
  _mm_free(A_min);
  
  _mm_free(__B);  
  _mm_free(B_max);
  _mm_free(B_min);
  
  delete[] max_a;
  delete[] max_b;
  delete[] Ca;
  delete[] Cb;
  delete[] Ctrl;
  
  return num_accelerated_block;
}

int gblas_sgemm_mu(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const float alpha, const float *A /* M x K */,
              const int lda, const float *B /* K x N */, const int ldb,
              const float beta, float *C /* M x N*/, const int ldc,
              const double TARGET_PERC_ACC
#ifdef EXTERNAL_A_QUANT
              , const double  A_QUANT
#endif
              )
{
  const int BLOCK_ELEMS = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  int num_accelerated_block = 0;  
  
  float* A_max  = (float*)_mm_malloc(sizeof(float)*K_blocks*M_blocks, 16);
  float* A_min  = (float*)_mm_malloc(sizeof(float)*K_blocks*M_blocks, 16);
  float* __A    = (float*)_mm_malloc(sizeof(float)*K*M, 16); 
  
  float* B_max  = (float*)_mm_malloc(sizeof(float)*K_blocks*N_blocks, 16);
  float* B_min  = (float*)_mm_malloc(sizeof(float)*K_blocks*N_blocks, 16);
  float* __B = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  
  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A, A_max, A_min);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A, A_max, A_min);
  //exit(-1);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B, B_max, B_min);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B, B_max, B_min);
  //exit(-1);
  
  float *p_A = __A;
  float *p_B = __B;
  float *p_C = C;
  
#ifndef _OPENMP
  // --- Best parameters selection ---
  gblas_quantizer Qa;
  gblas_quantizer Qb;
  
  float* max_a  = new float[K_blocks];
  float* max_b  = new float[K_blocks];
  float* Ca     = new float[K_blocks];
  float* Cb     = new float[K_blocks];
  double* P_e   = new double[K_blocks];
  int*   Ctrl   = new int[K_blocks];
  // ---------------------------------
#endif
  
#pragma omp parallel for private(p_A, p_B, p_C)
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
    
#ifdef _OPENMP
    // --- Best parameters selection ---
    gblas_quantizer Qa;
    gblas_quantizer Qb;
    
    float* max_a  = new float[K_blocks];
    float* max_b  = new float[K_blocks];
    float* Ca     = new float[K_blocks];
    float* Cb     = new float[K_blocks];
    double* P_e   = new double[K_blocks];
    int*   Ctrl   = new int[K_blocks];
    // ---------------------------------
#endif
    
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B
    {
      // ----
      double cumulative_P_e = 0.0;
      {
        for ( int c = 0; c < K_blocks; c++ )
        {
          max_a[c] = max(abs(A_max[ii*K/GBLAS_KERNEL_SIZE + c]), abs(A_min[ii*K/GBLAS_KERNEL_SIZE + c]));
          max_b[c] = max(abs(B_max[jj*K/GBLAS_KERNEL_SIZE + c]), abs(B_min[jj*K/GBLAS_KERNEL_SIZE + c]));
#ifdef EXTERNAL_A_QUANT
          P_e[c] = control_pack_packing_based_s_external_Aquant(max_a[c], max_b[c], Ca[c], Cb[c], Ctrl[c], A_QUANT);  // select best parameters for each block
#else
          P_e[c] = control_pack_packing_based_s(max_a[c], max_b[c], Ca[c], Cb[c], Ctrl[c]);  // select best parameters for each block
#endif
          
          cumulative_P_e += P_e[c];
        }
#ifndef EXTERNAL_A_QUANT
#ifdef ENABLE_PRUNING

        select_accelerated_blocks_by_throughput_s(K_blocks, P_e, cumulative_P_e, Ctrl, (float)TARGET_PERC_ACC);

#endif
#else
        std::cout << cumulative_P_e << ", " << Ca[0] << ", " << Cb[0] << ", ";
#endif
      }
      // ----
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      p_A = &__A[(ii*K_blocks)*BLOCK_ELEMS];
      p_B = &__B[(jj*K_blocks)*BLOCK_ELEMS];
      
      if ( Ctrl[0] > 1 )
      {        
        // pack 2
        Qa.set_min_value(0);
        Qb.set_min_value(0);
        
        Qa.set_max_value(max_a[0]);
        Qb.set_max_value(max_b[0]);
        
        Qa.set_q_step(Ca[0]);
        Qb.set_q_step(Cb[0]);
        
#ifdef FLOAT_SYM        
        KERNEL_p_sgemm_v2_r5(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc, Qa, Qb);
#else
        KERNEL_p_sgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc,  Qa, Qb);
#endif
        num_accelerated_block++;
      }
      else
      {
        // standard
        if ( beta == 0.0f )
        {
          KERNEL_std_sgemm_v6_B0(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, p_C, ldc);
        }
        else if (beta == 1.0f)
        {
          KERNEL_std_sgemm_v6_B1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, p_C, ldc);
        }
        else
        {
          KERNEL_std_sgemm_v6_BX(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc);
        }
      }
      
      for (int kk = 1; kk < K_blocks; kk++)
      {
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        if ( Ctrl[kk] > 1 )
        {          
          // pack 2
          Qa.set_min_value(0);
          Qb.set_min_value(0);
          
          Qa.set_max_value(max_a[kk]);
          Qb.set_max_value(max_b[kk]);
          
          Qa.set_q_step(Ca[kk]);
          Qb.set_q_step(Cb[kk]);
          
#ifdef FLOAT_SYM
          KERNEL_p_sgemm_v2_r5(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc,  Qa, Qb);
#else
          KERNEL_p_sgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc,  Qa, Qb);
#endif
          num_accelerated_block++;
        }
        else
        {          
          // standard
          KERNEL_std_sgemm_v6_B1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, p_C, ldc);
        }
      }
    }
    
#ifdef _OPENMP
    delete[] max_a;
    delete[] max_b;
    delete[] Ca;
    delete[] Cb;
    delete[] Ctrl;
    delete[] P_e;
#endif
    
  }
  
  _mm_free(__A);
  _mm_free(A_max);
  _mm_free(A_min);
  
  _mm_free(__B);  
  _mm_free(B_max);
  _mm_free(B_min);
  
#ifndef _OPENMP  
  delete[] max_a;
  delete[] max_b;
  delete[] Ca;
  delete[] Cb;
  delete[] Ctrl;
  delete[] P_e;
#endif
  
  return num_accelerated_block;
}

#ifdef _OPENMP
#define OPENMP_ENABLE_SGEMM_MU_EC
#endif

int gblas_sgemm_mu_EC(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A /* M x K */,
                   const int lda, const float *B /* K x N */, const int ldb,
                   const float beta, float *C /* M x N*/, const int ldc,
                   const double TARGET_PERC_ACC
#ifdef EXTERNAL_A_QUANT
                   , const double  A_QUANT
#endif
                   )
{
  const int BLOCK_ELEMS = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  int num_accelerated_block = 0;  
  
  float* A_max  = (float*)_mm_malloc(sizeof(float)*K_blocks*M_blocks, 16);
  float* A_min  = (float*)_mm_malloc(sizeof(float)*K_blocks*M_blocks, 16);
  float* __A    = (float*)_mm_malloc(sizeof(float)*K*M, 16); 
  
  float* B_max  = (float*)_mm_malloc(sizeof(float)*K_blocks*N_blocks, 16);
  float* B_min  = (float*)_mm_malloc(sizeof(float)*K_blocks*N_blocks, 16);
  float* __B = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  
  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A, A_max, A_min);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A, A_max, A_min);
  //exit(-1);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B, B_max, B_min);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B, B_max, B_min);
  //exit(-1);
  
  float *p_A = __A;
  float *p_B = __B;
  float *p_C = C;

#ifndef OPENMP_ENABLE_SGEMM_MU_EC
  // --- Best parameters selection ---
  gblas_quantizer Qa;
  gblas_quantizer Qb;
  
  float* max_a  = new float[K_blocks];
  float* max_b  = new float[K_blocks];
  float* Ca     = new float[K_blocks];
  float* Cb     = new float[K_blocks];
  double* P_e   = new double[K_blocks];
  int*   Ctrl   = new int[K_blocks];
  // ---------------------------------
#endif
  
#ifdef OPENMP_ENABLE_SGEMM_MU_EC
  #pragma omp parallel for private (p_A, p_B, p_C) reduction(+:num_accelerated_block)
#endif
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
#ifdef OPENMP_ENABLE_SGEMM_MU_EC
    // --- Best parameters selection ---
    gblas_quantizer Qa;
    gblas_quantizer Qb;
    
    float* max_a  = new float[K_blocks];
    float* max_b  = new float[K_blocks];
    float* Ca     = new float[K_blocks];
    float* Cb     = new float[K_blocks];
    double* P_e   = new double[K_blocks];
    int*   Ctrl   = new int[K_blocks];
    // ---------------------------------
#endif
    
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B
    {
      // ----
      double cumulative_P_e = 0.0;
      {
        for ( int c = 0; c < K_blocks; c++ )
        {
          max_a[c] = max(abs(A_max[ii*K/GBLAS_KERNEL_SIZE + c]), abs(A_min[ii*K/GBLAS_KERNEL_SIZE + c]));
          max_b[c] = max(abs(B_max[jj*K/GBLAS_KERNEL_SIZE + c]), abs(B_min[jj*K/GBLAS_KERNEL_SIZE + c]));
#ifdef EXTERNAL_A_QUANT
          P_e[c] = control_pack_packing_based_s_external_Aquant(max_a[c], max_b[c], Ca[c], Cb[c], Ctrl[c], A_QUANT);  // select best parameters for each block
#else
          P_e[c] = control_pack_packing_based_s(max_a[c], max_b[c], Ca[c], Cb[c], Ctrl[c]);  // select best parameters for each block
#endif
          
          cumulative_P_e += P_e[c];
        }
#ifndef EXTERNAL_A_QUANT
#ifdef ENABLE_PRUNING
        
        select_accelerated_blocks_by_throughput_s(K_blocks, P_e, cumulative_P_e, Ctrl, (float)TARGET_PERC_ACC);
        
#endif
#else
        std::cout << cumulative_P_e << ", " << Ca[0] << ", " << Cb[0] << ", ";
#endif
      }
      // ----
      
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      p_A = &__A[(ii*K_blocks)*BLOCK_ELEMS];
      p_B = &__B[(jj*K_blocks)*BLOCK_ELEMS];
      
      if ( Ctrl[0] > 1 )
      {        
        // pack 2
        Qa.set_min_value(0);
        Qb.set_min_value(0);
        
        Qa.set_max_value(max_a[0]);
        Qb.set_max_value(max_b[0]);
        
        Qa.set_q_step(Ca[0]);
        Qb.set_q_step(Cb[0]);
               
        //KERNEL_p_sgemm_v2_r5_EC_v4
        KERNEL_p4_sgemm_v2_r5_EC_v4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc, Qa, Qb);

        num_accelerated_block++;
      }
      else
      {
        // standard
        KERNEL_std_sgemm_v6(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, beta, p_C, ldc);
      }
      
      for (int kk = 1; kk < K_blocks; kk++)
      {
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        if ( Ctrl[kk] > 1 )
        {          
          // pack 2
          Qa.set_min_value(0);
          Qb.set_min_value(0);
          
          Qa.set_max_value(max_a[kk]);
          Qb.set_max_value(max_b[kk]);
          
          Qa.set_q_step(Ca[kk]);
          Qb.set_q_step(Cb[kk]);
          
          //KERNEL_p_sgemm_v2_r5_EC_v4
          KERNEL_p4_sgemm_v2_r5_EC_v4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc,  Qa, Qb);

          num_accelerated_block++;
        }
        else
        {          
          // standard
          KERNEL_std_sgemm_v6(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0f), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, (1.0f), p_C, ldc);
        }
      }      
    }
    
#ifdef OPENMP_ENABLE_SGEMM_MU_EC
    delete[] max_a;
    delete[] max_b;
    delete[] Ca;
    delete[] Cb;
    delete[] Ctrl;
    delete[] P_e;
#endif
  }
  
  _mm_free(__A);
  _mm_free(A_max);
  _mm_free(A_min);
  
  _mm_free(__B);  
  _mm_free(B_max);
  _mm_free(B_min);
  
#ifndef OPENMP_ENABLE_SGEMM_MU_EC
  delete[] max_a;
  delete[] max_b;
  delete[] Ca;
  delete[] Cb;
  delete[] Ctrl;
  delete[] P_e;
#endif
  
  return num_accelerated_block;
}
