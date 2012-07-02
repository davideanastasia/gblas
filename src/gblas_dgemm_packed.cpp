/*
 *  gblas_dgemm_packed.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 30/06/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <float.h>
#include <cmath>
#include <iostream>

#include "gblas_kernels.h"
#include "gblas_stat_model.h"
#include "gblas_matrix_utils.h"

int gblas_dgemm_snr(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
             const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
             const int K, const double alpha, const double *A /* M x K */,
             const int lda, const double *B /* K x N */, const int ldb,
             const double beta, double *C /* M x N*/, const int ldc,
             const double TARGET_SNR, const double VAR_A, const double VAR_B)
{
  // conversion to MSE
  double target_mse = (double(K)*VAR_A*VAR_B)/(pow(10.0, (TARGET_SNR/10.0)));
  
  return gblas_dgemm_mse(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, target_mse);
}

int gblas_dgemm_mse(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
             const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
             const int K, const double alpha, const double *A /* M x K */,
             const int lda, const double *B /* K x N */, const int ldb,
             const double beta, double *C /* M x N*/, const int ldc,
             const double TARGET_MSE
#ifdef EXTERNAL_A_QUANT
             , const long   A_QUANT
#endif
             )
{  
  const int BLOCK_ELEMS  = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  int num_accelerated_block[NUM_PACKING_DOUBLE] = {0, 0, 0, 0};

  
  double* A_max = new double[(K/GBLAS_KERNEL_SIZE)*(M/GBLAS_KERNEL_SIZE)];
  double* A_min = new double[(K/GBLAS_KERNEL_SIZE)*(M/GBLAS_KERNEL_SIZE)];
  double* __A = (double*)_mm_malloc(sizeof(double)*K*M, 16); 
  
  double* B_max = new double[(K/GBLAS_KERNEL_SIZE)*(N/GBLAS_KERNEL_SIZE)];
  double* B_min = new double[(K/GBLAS_KERNEL_SIZE)*(N/GBLAS_KERNEL_SIZE)];
  double* __B = (double*)_mm_malloc(sizeof(double)*K*N, 16);
  
  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A, A_max, A_min);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A, A_max, A_min);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B, B_max, B_min);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B, B_max, B_min);
  //exit(-1);
  
  double *p_A = __A;
  double *p_B = __B;
  double *p_C = C;
  
  // --- Best parameters selection ---
  gblas_quantizer Qa;
  gblas_quantizer Qb;
  
  double* max_a  = new double[K_blocks];
  double* max_b  = new double[K_blocks];
  int*   Ctrl    = new int[K_blocks];
  double* Ca     = new double[K_blocks*NUM_PACKING_DOUBLE];  // 1, 2, 3 and 4 packings
  double* Cb     = new double[K_blocks*NUM_PACKING_DOUBLE];
  double* P_e    = new double[K_blocks*NUM_PACKING_DOUBLE];
  // ---------------------------------
  
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B 
    {
      // ----
      {
        for ( int c = 0; c < K_blocks; c++ )
        {
          max_a[c] = max(abs(A_max[ii*K/GBLAS_KERNEL_SIZE + c]), abs(A_min[ii*K/GBLAS_KERNEL_SIZE + c]));
          max_b[c] = max(abs(B_max[jj*K/GBLAS_KERNEL_SIZE + c]), abs(B_min[jj*K/GBLAS_KERNEL_SIZE + c]));
#ifdef EXTERNAL_A_QUANT
          control_pack_packing_based_d_external_Aquant(max_a[c], max_b[c], &Ca[c*NUM_PACKING_DOUBLE], &Cb[c*NUM_PACKING_DOUBLE], &P_e[c*NUM_PACKING_DOUBLE], Ctrl[c], A_QUANT);
#else
          control_pack_packing_based_d(max_a[c], max_b[c], &Ca[c*NUM_PACKING_DOUBLE], &Cb[c*NUM_PACKING_DOUBLE], &P_e[c*NUM_PACKING_DOUBLE], Ctrl[c]);  // select best parameters for each block
#endif
        }
        
#ifndef EXTERNAL_A_QUANT
#ifdef ENABLE_PRUNING
        //cout << "Packings (before pruning) = "; print_matrix_matlab_notation(Ctrl, 1, K_BLOCKS);
        
        select_accelerated_blocks_by_quality_d(K_blocks, P_e, Ctrl, TARGET_MSE, K);  
        
        //cout << "Packings (after pruning) = "; print_matrix_matlab_notation(Ctrl, 1, K_BLOCKS);
#endif
#else        
        std::cout << P_e[3] << ", " << Ca[3] << ", " << Cb[3] << ", ";   
#endif
      }
      // ----      
      
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      for (int kk = 0; kk < K_blocks; kk++)
      {   
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        num_accelerated_block[Ctrl[kk]-1]++;
        
        Qa.set_min_value(0);
        Qb.set_min_value(0);
        
        Qa.set_max_value(max_a[kk]);
        Qb.set_max_value(max_b[kk]);
        
        //cout << Ctrl[kk] << " " << max_a[kk] << " " << Ca[kk*NUM_PACKING_DOUBLE + Ctrl[kk] - 1] << " " << max_b[kk] << " " << Cb[kk*NUM_PACKING_DOUBLE + Ctrl[kk] - 1] << endl;
        
        switch ( Ctrl[kk] )
        {
          case 4:
          {
            Qa.set_q_step(Ca[kk*NUM_PACKING_DOUBLE + 3]);
            Qb.set_q_step(Cb[kk*NUM_PACKING_DOUBLE + 3]);
            
#ifdef DOUBLE_SYM
            KERNEL_p4_dgemm_v2_r4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#else
            KERNEL_p4_dgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#endif
          }
            break;
          case 3:
          {
            Qa.set_q_step(Ca[kk*NUM_PACKING_DOUBLE + 2]);
            Qb.set_q_step(Cb[kk*NUM_PACKING_DOUBLE + 2]);
            
#ifdef DOUBLE_SYM
            KERNEL_p3_dgemm_v2_r4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#else
            KERNEL_p3_dgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#endif
          }
            break;
          case 2:
          {
            Qa.set_q_step(Ca[kk*NUM_PACKING_DOUBLE + 1]);
            Qb.set_q_step(Cb[kk*NUM_PACKING_DOUBLE + 1]);
            
#ifdef DOUBLE_SYM
            KERNEL_p_dgemm_v2_r4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#else
            KERNEL_p_dgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#endif
          }
            break;
          case 1:
          default:
            KERNEL_std_dgemm_v1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc);
            break;
        }
      }
    }
  }
  
  _mm_free(__A);
  delete[] A_max;
  delete[] A_min;
  
  _mm_free(__B);  
  delete[] B_max;
  delete[] B_min;
  
  delete[] max_a;
  delete[] max_b;
  delete[] Ca;
  delete[] Cb;
  delete[] Ctrl;
  
  //cout << "(" << num_accelerated_block[0] << ", " << num_accelerated_block[1] << ", " << num_accelerated_block[2] << ", " << num_accelerated_block[3] << ")" << endl;
  
  return num_accelerated_block[1]+num_accelerated_block[2]+num_accelerated_block[3];
}

int gblas_dgemm_mu(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const double alpha, const double *A /* M x K */,
                   const int lda, const double *B /* K x N */, const int ldb,
                   const double beta, double *C /* M x N*/, const int ldc,
                   const double TARGET_PERC_ACC
#ifdef EXTERNAL_A_QUANT
                   , const long   A_QUANT
#endif
                   )
{  
  const int BLOCK_ELEMS  = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  const int M_blocks    = M/GBLAS_KERNEL_SIZE;
  const int N_blocks    = N/GBLAS_KERNEL_SIZE;
  const int K_blocks    = K/GBLAS_KERNEL_SIZE;
  
  int num_accelerated_block[NUM_PACKING_DOUBLE] = {0, 0, 0, 0};
  
  
  double* A_max = new double[(K/GBLAS_KERNEL_SIZE)*(M/GBLAS_KERNEL_SIZE)];
  double* A_min = new double[(K/GBLAS_KERNEL_SIZE)*(M/GBLAS_KERNEL_SIZE)];
  double* __A = (double*)_mm_malloc(sizeof(double)*K*M, 16); 
  
  double* B_max = new double[(K/GBLAS_KERNEL_SIZE)*(N/GBLAS_KERNEL_SIZE)];
  double* B_min = new double[(K/GBLAS_KERNEL_SIZE)*(N/GBLAS_KERNEL_SIZE)];
  double* __B = (double*)_mm_malloc(sizeof(double)*K*N, 16);
  
  if ( TransA == CblasNoTrans )
    row_to_block_major(CblasNoTrans, M, K, lda, A, alpha, __A, A_max, A_min);
  else 
    row_to_block_major(CblasTrans, K, M, lda, A, alpha, __A, A_max, A_min);
  
  if ( TransB == CblasNoTrans )
    row_to_block_major(CblasTrans, K, N, ldb, B, (1.0f), __B, B_max, B_min);
  else 
    row_to_block_major(CblasNoTrans, N, K, ldb, B, (1.0f), __B, B_max, B_min);
  //exit(-1);
  
  double *p_A = __A;
  double *p_B = __B;
  double *p_C = C;
  
  // --- Best parameters selection ---
  gblas_quantizer Qa;
  gblas_quantizer Qb;
  
  double* max_a  = new double[K_blocks];
  double* max_b  = new double[K_blocks];
  int*   Ctrl    = new int[K_blocks];
  double* Ca     = new double[K_blocks*NUM_PACKING_DOUBLE];  // 1, 2, 3 and 4 packings
  double* Cb     = new double[K_blocks*NUM_PACKING_DOUBLE];
  double* P_e    = new double[K_blocks*NUM_PACKING_DOUBLE];
  // ---------------------------------
  
  for (int ii = 0; ii < M_blocks; ii++)  // blocks of A
  {
    for (int jj = 0; jj < N_blocks; jj++)  // blocks of B 
    {
      // ----
      {
        for ( int c = 0; c < K_blocks; c++ )
        {
          max_a[c] = max(abs(A_max[ii*K/GBLAS_KERNEL_SIZE + c]), abs(A_min[ii*K/GBLAS_KERNEL_SIZE + c]));
          max_b[c] = max(abs(B_max[jj*K/GBLAS_KERNEL_SIZE + c]), abs(B_min[jj*K/GBLAS_KERNEL_SIZE + c]));
#ifdef EXTERNAL_A_QUANT
          control_pack_packing_based_d_external_Aquant(max_a[c], max_b[c], &Ca[c*NUM_PACKING_DOUBLE], &Cb[c*NUM_PACKING_DOUBLE], &P_e[c*NUM_PACKING_DOUBLE], Ctrl[c], A_QUANT);
#else
          control_pack_packing_based_d(max_a[c], max_b[c], &Ca[c*NUM_PACKING_DOUBLE], &Cb[c*NUM_PACKING_DOUBLE], &P_e[c*NUM_PACKING_DOUBLE], Ctrl[c]);  // select best parameters for each block
#endif
        }
        
#ifndef EXTERNAL_A_QUANT
#ifdef ENABLE_PRUNING
        //cout << "Packings (before pruning) = "; print_matrix_matlab_notation(Ctrl, 1, K_BLOCKS);

        select_accelerated_blocks_by_throughput_d(K_blocks, P_e, Ctrl, TARGET_PERC_ACC);
        
        //cout << "Packings (after pruning) = "; print_matrix_matlab_notation(Ctrl, 1, K_BLOCKS);
#endif
#else        
        std::cout << P_e[3] << ", " << Ca[3] << ", " << Cb[3] << ", ";   
#endif
      }
      // ----      
      
      p_C = &C[ii*GBLAS_KERNEL_SIZE*ldc + jj*GBLAS_KERNEL_SIZE];
      
      for (int kk = 0; kk < K_blocks; kk++)
      {   
        p_A = &__A[(ii*K_blocks + kk)*BLOCK_ELEMS];
        p_B = &__B[(jj*K_blocks + kk)*BLOCK_ELEMS];
        
        num_accelerated_block[Ctrl[kk]-1]++;
        
        Qa.set_min_value(0);
        Qb.set_min_value(0);
        
        Qa.set_max_value(max_a[kk]);
        Qb.set_max_value(max_b[kk]);
        
        //cout << Ctrl[kk] << " " << max_a[kk] << " " << Ca[kk*NUM_PACKING_DOUBLE + Ctrl[kk] - 1] << " " << max_b[kk] << " " << Cb[kk*NUM_PACKING_DOUBLE + Ctrl[kk] - 1] << endl;
        
        switch ( Ctrl[kk] )
        {
          case 4:
          {
            Qa.set_q_step(Ca[kk*NUM_PACKING_DOUBLE + 3]);
            Qb.set_q_step(Cb[kk*NUM_PACKING_DOUBLE + 3]);
            
#ifdef DOUBLE_SYM
            KERNEL_p4_dgemm_v2_r4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#else
            KERNEL_p4_dgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#endif
          }
            break;
          case 3:
          {
            Qa.set_q_step(Ca[kk*NUM_PACKING_DOUBLE + 2]);
            Qb.set_q_step(Cb[kk*NUM_PACKING_DOUBLE + 2]);
            
#ifdef DOUBLE_SYM
            KERNEL_p3_dgemm_v2_r4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#else
            KERNEL_p3_dgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#endif
          }
            break;
          case 2:
          {
            Qa.set_q_step(Ca[kk*NUM_PACKING_DOUBLE + 1]);
            Qb.set_q_step(Cb[kk*NUM_PACKING_DOUBLE + 1]);
            
#ifdef DOUBLE_SYM
            KERNEL_p_dgemm_v2_r4(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#else
            KERNEL_p_dgemm_v1_r3(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc,  Qa, Qb);
#endif
          }
            break;
          case 1:
          default:
            KERNEL_std_dgemm_v1(GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE, (1.0), p_A, GBLAS_KERNEL_SIZE, p_B, GBLAS_KERNEL_SIZE, ((kk==0)?beta:1.0), p_C, ldc);
            break;
        }
      }
    }
  }
  
  _mm_free(__A);
  delete[] A_max;
  delete[] A_min;
  
  _mm_free(__B);  
  delete[] B_max;
  delete[] B_min;
  
  delete[] max_a;
  delete[] max_b;
  delete[] Ca;
  delete[] Cb;
  delete[] Ctrl;
  
  //cout << "(" << num_accelerated_block[0] << ", " << num_accelerated_block[1] << ", " << num_accelerated_block[2] << ", " << num_accelerated_block[3] << ")" << endl;
  
  return num_accelerated_block[1]+num_accelerated_block[2]+num_accelerated_block[3];
}

