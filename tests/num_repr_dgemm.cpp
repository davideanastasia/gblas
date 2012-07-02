/*
 *  num_repr_dgemm.cpp
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
  
  srand ( (float)time(NULL) );
  cout << "Numerical Representation Noise Test (DGEMM)" << endl << endl;
  
#ifndef FIXED_COMPOUNDERS
  cout << "FIXED_COMPOUNDERS is unset. Please, set FIXED_COMPOUNDERS and recompile GBLAS" << endl;
  exit(0);
#endif
  
  const int NUM_ITER = 5;
  
  double alpha = 1.0f;
  double beta  = 0.0f;
  
  int NB = GBLAS_KERNEL_SIZE;
  int N = NB;
  int M = NB;
  int K = NB;
  
  double *A = (double*)_mm_malloc(sizeof(double)*M*K, 16);
  double *B = (double*)_mm_malloc(sizeof(double)*K*N, 16);
  double *C = (double*)_mm_malloc(sizeof(double)*M*N, 16);
  double *C_final = (double*)_mm_malloc(sizeof(double)*M*N, 16);
  double *C_diff = (double*)_mm_malloc(sizeof(double)*M*N*NUM_ITER, 16); 
  
  double mean, var;
  double MAX_VALUE_A, MAX_VALUE_B;
  
  MAX_VALUE_A = 4.0;
  
  //for (double Aquant = (floor(50000/GBLAS_KERNEL_SIZE))*GBLAS_KERNEL_SIZE; Aquant <= 500000; Aquant += GBLAS_KERNEL_SIZE)
  for (double Aquant = (floor(5000/GBLAS_KERNEL_SIZE))*GBLAS_KERNEL_SIZE; Aquant <= 500000; Aquant += GBLAS_KERNEL_SIZE)  // double, sym, P4
  {   
    MAX_VALUE_B = floor((Aquant/(2*MAX_VALUE_A*GBLAS_KERNEL_SIZE)) + 0.5);
    
    cout << M << ", " << N << ", " << K << ", ";
    cout << MAX_VALUE_A << ", " << MAX_VALUE_B << ", " << Aquant << ", " << 2*K*MAX_VALUE_A*MAX_VALUE_B;
    for ( int e = 0; e < NUM_ITER; e++ )
    {
      set_matrix_2_value(C, M, N, 0.0);
      set_matrix_2_value(C_final, M, N, 0.0);
      
      set_matrix_random_integers_values(A, M, K, MAX_VALUE_A);
      set_matrix_random_integers_values(B, K, N, MAX_VALUE_B);
      
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C_final, N);
      
      gblas_dgemm_mse(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
      
      matrix_difference(C_final, C, M, N, &C_diff[M*N*e]);
    }
    
    calculate_var_and_mean(C_diff, M*N*NUM_ITER, 1, mean, var);
    cout << ", " <<  mean << ", " << var;
    cout << endl;
  }
  
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_final);
  _mm_free(C_diff);
  
  exit_high_priority();
}