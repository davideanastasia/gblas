/*
 *  num_repr_sgemm.cpp
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mm_malloc.h>
#include <iostream>
#include <fstream>

#include "gblas.h"

#include "msec_timer.h"
#include "matrix_utils.h"
#include "high_priority_process.h"

int main(int argc, char *argv[])
{
  start_high_priority();
  
  srand((float)time(NULL));
  cout << "Model Precision Test (SGEMM)" << endl << endl;
  
  const int NUM_ITER = 1;
  
#ifndef EXTERNAL_A_QUANT 
  cout << "EXTERNAL_A_QUANT Macro is not currectly enable, so test_sgemm_std_deviation_unpacking_calculation_v3() cannot run correctly" << endl;  
  exit(0);
#endif
  
  const float MAX_VALUE = 128.0f;
  const double VAR_A = 4.0*double(MAX_VALUE*MAX_VALUE)/12.0;
  const double VAR_B = VAR_A;     // to be changed if the input is different!
  
  float alpha = 1.0f;
  float beta  = 0.0f;
  
  int NB = GBLAS_KERNEL_SIZE;
  int N = NB;
  int M = NB;
  int K = NB;
  
  float *A = (float*)_mm_malloc(sizeof(float)*M*K, 16);
  float *B = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  float *C = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  float *C_final = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  
  double snr = 0.0;
  double mse = 0.0;
  
  // ----------------------------------------------------
  cout << "M, N, K, Max Input, Rmax, ";
  for (int e = 0; e < NUM_ITER; e++)
  {
    cout << "MSE (model) #" << e << ", ";
    cout << "Ca (model) #" << e << ", ";
    cout << "Cb (model) #" << e << ", ";
  }
  cout << "MSE (measured), SNR (measured)" << endl;
  // ----------------------------------------------------
  
  for (double Aquant = (floor(5000/GBLAS_KERNEL_SIZE))*GBLAS_KERNEL_SIZE; Aquant <= 500000; Aquant += GBLAS_KERNEL_SIZE)
  {
    cout << M << ", " << N << ", " << K << ", " << MAX_VALUE << ", " << Aquant << ", ";
    mse = 0.0;
    for ( int e = 0; e < NUM_ITER; e++ )
    {
      set_matrix_2_value(C, M, N, 0.0f);
      set_matrix_2_value(C_final, M, N, 0.0f);
      
      set_matrix_random_values(A, M, K, MAX_VALUE);
      set_matrix_random_values(B, K, N, MAX_VALUE);
      
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A /* M x K */, K, B /* K x N */, N, beta, C_final /* M x N*/, N);
#ifdef EXTERNAL_A_QUANT 
      gblas_sgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A /* M x K */, K, B /* K x N */, N, beta, C /* M x N*/, N, 100.0, Aquant);
#endif      
      mse += calculate_mse(C_final, C, M, N);
    }
    
    mse = mse/NUM_ITER;
    snr = 10*log10((double(K)*VAR_A*VAR_B)/mse);
    
    cout << mse << ", " << snr << endl << flush;
  }
  
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_final);
  
  exit_high_priority();
}