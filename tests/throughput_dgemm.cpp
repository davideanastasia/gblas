/*
 *  throughput_dgemm.cpp
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
#include "high_priority_process.h"
#include "msec_timer.h"
#include "matrix_utils.h"
#include "high_priority_process.h"

#define NUM_SNR_TEST_POINT  10
double SNR_TEST_POINTS[NUM_SNR_TEST_POINT] = {5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 80.0};

int main(int argc, char *argv[])
{
  start_high_priority();
  
  const int NUM_RUNS = 4;
  
#ifdef EXTERNAL_A_QUANT
  cout << "EXTERNAL_A_QUANT is set: no way!" << endl;
  exit(0);  
#endif
  
#ifdef FIXED_COMPOUNDERS
  cout << "FIXED_COMPOUNDERS is set: no way!" << endl;
  exit(0);  
#endif
  
#ifndef ENABLE_PRUNING
  cout << "ENABLE_PRUNING is set: no way!" << endl;
  exit(0);  
#endif  
  
  srand( (int)(time(NULL)) ); // fixed seed! // 0.0f
  
  cout << "TEST THROUGHPUT DGEMM" << endl << endl;
  
  int block_accerated;
  
  double alpha = 1.0f;
  double beta  = 0.0f;
  
  const double MAX_VALUE = 128;
  const double VAR_A = 4.0*double(128*128)/12.0;
  const double VAR_B = VAR_A;     // to be changed if the input is different!
  
  msec_timer atlas_t;
  msec_timer bm_format_w_kernel_t;
  msec_timer bm_format_w_kernel_w_packing_t;
  
  int N = GBLAS_KERNEL_SIZE;
  int M = GBLAS_KERNEL_SIZE;
  int K = GBLAS_KERNEL_SIZE;
  
  int init_NB = GBLAS_KERNEL_SIZE; //(300/GBLAS_KERNEL_SIZE)*GBLAS_KERNEL_SIZE;
  int end_NB = ((4000/GBLAS_KERNEL_SIZE)+1)*GBLAS_KERNEL_SIZE;
  
  // what is it going to be the biggest size?
  double *A        = (double*)_mm_malloc(sizeof(double)*end_NB*end_NB, 16);
  double *B        = (double*)_mm_malloc(sizeof(double)*end_NB*end_NB, 16);
  double *C        = (double*)_mm_malloc(sizeof(double)*end_NB*end_NB, 16);
  double *C_final  = (double*)_mm_malloc(sizeof(double)*end_NB*end_NB, 16);
  
  double Gflops = 0.0; // temp var
  
  cout << "M, N, K, t (plain), Mflops (plain), Perc Peak Perf (plain), t (GOTO), Mflops (GOTO), Perc Peak Perf (GOTO), SNR (ask), SNR(retr), Num Acc Blk, t (GBLAS), Mflops (GBLAS), Perc Peak Perf (GBLAS)" << endl;
  for (int NB = init_NB; NB <= end_NB; NB += GBLAS_KERNEL_SIZE)
  {
    atlas_t.reset();
    bm_format_w_kernel_t.reset();
    
    N = NB;
    M = NB;
    K = NB;
    
    double scale_factor = 2*K; //2*K*M*N;
    double peak_performance_d = 2660*4;
    
    // ---- RANDOM ----
    set_matrix_2_value(C, M, N, 0.0);
    set_matrix_2_value(C_final, M, N, 0.0);
    
    //set_matrix_random_values_by_block(A, M, K, GBLAS_KERNEL_SIZE);
    //set_matrix_random_values_by_block(B, K, N, GBLAS_KERNEL_SIZE);
    
    set_matrix_random_values(A, M, K, MAX_VALUE);
    set_matrix_random_values(B, K, N, MAX_VALUE);
    
    cout << M << ", " << N << ", " << K;
    
    for (int n_r = 0; n_r < NUM_RUNS; n_r++)
    {
      bm_format_w_kernel_t.start();
      gblas_dgemm_plain(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A /* M x K */, K, B /* K x N */, N, beta, C_final /* M x N*/, N);
      bm_format_w_kernel_t.stop_and_update();
      
      atlas_t.start();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A /* M x K */, K, B /* K x N */, N, beta, C_final /* M x N*/, N);
      atlas_t.stop_and_update();
    }
    
    cout << ", " << bm_format_w_kernel_t.get_time()/NUM_RUNS;
    Gflops = (double)convert_to_gigaflops(bm_format_w_kernel_t.get_time()/NUM_RUNS, scale_factor)*M*N;
    cout << ", " << Gflops;
    cout << ", " << Gflops/peak_performance_d*100.0;
    cout << ", " << atlas_t.get_time()/NUM_RUNS;
    
    Gflops = (double)convert_to_gigaflops(atlas_t.get_time()/NUM_RUNS, scale_factor)*M*N;
    cout << ", " << Gflops;
    cout << ", " << Gflops/peak_performance_d*100.0;
    cout << flush;
    
    for (int snr_p = 0; snr_p < NUM_SNR_TEST_POINT; snr_p ++)
    {
      double curr_snr = 0.0;
      bm_format_w_kernel_w_packing_t.reset();
      
      for (int n_r = 0; n_r < NUM_RUNS; n_r++)
      {
        bm_format_w_kernel_w_packing_t.start();
        block_accerated = gblas_dgemm_snr(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A /* M x K */, K, B /* K x N */, N, beta, C /* M x N*/, N, SNR_TEST_POINTS[snr_p], VAR_A, VAR_B);
        bm_format_w_kernel_w_packing_t.stop_and_update();
        
        curr_snr += calculate_snr(C_final, C, M, N, K, VAR_A, VAR_B);
      }
      
      cout << ", " << SNR_TEST_POINTS[snr_p];      
      cout << ", " << curr_snr/NUM_RUNS;
      cout << ", " << block_accerated << "/" << ((M/GBLAS_KERNEL_SIZE)*(N/GBLAS_KERNEL_SIZE)*(K/GBLAS_KERNEL_SIZE));
      cout << ", " << bm_format_w_kernel_w_packing_t.get_time()/NUM_RUNS;
      Gflops = (double)convert_to_gigaflops(bm_format_w_kernel_w_packing_t.get_time()/NUM_RUNS, scale_factor)*M*N;
      cout << ", " << Gflops;
      cout << ", " << Gflops/peak_performance_d*100.0;
      cout << flush;
    }
    cout << endl << flush;
  }
  
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_final);
  
  exit_high_priority();
}