/*
 *  perf_tests.cpp
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xmmintrin.h>

#include "gblas.h"

#include "perf_tests.h"
#include "msec_timer.h"
#include "matrix_utils.h"
#include "high_priority_process.h"

#define     MAX_VALUE_TEST                (22)
#define     I_NEED_THIS_NUMBER            (MAX_VALUE_TEST)

using namespace std;


void test_dgemm_companders()
{ 
//#ifdef EXTERNAL_A_QUANT
//  cout << "EXTERNAL_A_QUANT is defined! Recompile the project!" << endl;
//  exit(1);
//#endif
//  
//#ifdef FIXED_COMPOUNDERS
//  cout << "FIXED_COMPOUNDERS is defined! Recompile the project!" << endl;
//  exit(1);
//#endif
//  
//#ifndef ENABLE_PRUNING
//  cout << "ENABLE_PRUNING is NOT defined! Recompile the project!" << endl;
//  exit(1);
//#endif
//  
//  const double ALPHA = 1.0;
//  const double BETA  = 0.0; 
//  
//  //cout << "test_dgemm_companders()" << endl << endl;
//  
//  double temp = 0.0;
//  // open file
//  ifstream indata;
//  indata.open("gblas_data.txt");
//  if ( !indata.is_open() )
//  {
//    cerr << "Error: file could not be opened" << endl;
//    exit(1);
//  }
//  
//  // read quality
//  indata >> temp;
//  const double Q_DB = temp;
//  
//  // read matrix sizes
//  //int NB = GBLAS_KERNEL_SIZE*2;
//
//  indata >> temp;
//  int N = (int)(temp);  
//  
//  indata >> temp;
//  int M = (int)(temp);
//  
//  indata >> temp;
//  int K = (int)(temp);
//  
//  double *A       = (double*)_mm_malloc(sizeof(double)*M*K, 16);
//  double *B       = (double*)_mm_malloc(sizeof(double)*K*N, 16);
//  double *C       = (double*)_mm_malloc(sizeof(double)*M*N, 16);
//  
//  //srand((float)time(NULL));
//  //for (int idx = 0; idx < M*N; idx++) { C[idx] = 0.0; } // one; one *= -1; } 
//  //fill_matrix_random_values(A, M, K, MAX_VALUE_TEST, (float)time(NULL));
//  //fill_matrix_random_values(B, K, N, MAX_VALUE_TEST, (float)time(NULL));
//  
//  int sample = 0;
//  while ( !indata.eof() )
//  {
//    indata >> temp;
//    if ( sample <= (N*K-1) )
//    {
//      // fill A
//      A[sample] = temp;
//    }
//    else
//    {
//      // fill B
//      B[sample - N*K] = temp;
//    }
//    sample++;
//  }
//  indata.close();
//  
//  dgemm_v6(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A /* M x K */, K, B /* K x N */, N, BETA, C /* M x N*/, N, Q_DB);
//  //cout << endl << "C = " << endl; print_matrix_matlab_notation(C, NB, NB);
//  
//  _mm_free(A);
//  _mm_free(B);
//  _mm_free(C);
}

void test_sgemm_companders(float dB)
{ 
#ifdef EXTERNAL_A_QUANT
  cout << "EXTERNAL_A_QUANT is defined! Recompile the project!" << endl;
  exit(1);
#endif
  
#ifdef FIXED_COMPOUNDERS
  cout << "FIXED_COMPOUNDERS is defined! Recompile the project!" << endl;
  exit(1);
#endif
  
#ifndef ENABLE_PRUNING
  cout << "ENABLE_PRUNING is NOT defined! Recompile the project!" << endl;
  exit(1);
#endif
  
  const float ALPHA = 1.0f;
  const float BETA  = 0.0f; 
  
  //cout << "test_dgemm_companders()" << endl << endl;
  
  // open file
  FILE* in_file;
  in_file = fopen("gblas_data.txt", "rb");
  
  if ( in_file == NULL )
  {
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }

  // read quality
  const float Q_DB = dB; // STATIC VALUE! TO BE CHANGED
  
  int N = 288;
  int M = 288;
  int K = 288;
  
  float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
  float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  
  fread(A, sizeof(float), M*K, in_file);
  fread(B, sizeof(float), N*K, in_file);

  fclose(in_file);
  
  float max_f, min_f;
  get_matrix_min_max(A, M, K, max_f, min_f);
  max_f = 128; //max(abs(max_f),abs(min_f));
  const double VAR_A = 4.0*double(max_f*max_f)/12.0;
  
  get_matrix_min_max(B, K, N, max_f, min_f);
  max_f = 128; //max(abs(max_f),abs(min_f));
  const double VAR_B = 4.0*double(max_f*max_f)/12.0;
  
  gblas_sgemm_snr(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A /* M x K */, K, B /* K x N */, N, BETA, C /* M x N*/, N, Q_DB, VAR_A, VAR_B);

  print_matrix_plain_notation(C, M, N);
  
  FILE* out_file;
  out_file = fopen("result.bin", "wb");
  fwrite(C, sizeof(float), M*N, out_file);
  fclose(out_file);
  
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
}

//
//void test_sgemm_companders()
//{ 
//#ifdef EXTERNAL_A_QUANT
//  cout << "EXTERNAL_A_QUANT is defined! Recompile the project!" << endl;
//  exit(1);
//#endif
//  
//#ifdef FIXED_COMPOUNDERS
//  cout << "FIXED_COMPOUNDERS is defined! Recompile the project!" << endl;
//  exit(1);
//#endif
//  
//#ifndef ENABLE_PRUNING
//  cout << "ENABLE_PRUNING is NOT defined! Recompile the project!" << endl;
//  exit(1);
//#endif
//  
//  const float ALPHA = 1.0;
//  const float BETA  = 0.0; 
//  
//  //cout << "test_dgemm_companders()" << endl << endl;
//  
//  float temp = 0.0;
//  // open file
//  ifstream indata;
//  indata.open("gblas_data.txt");
//  if ( !indata.is_open() )
//  {
//    cerr << "Error: file could not be opened" << endl;
//    exit(1);
//  }
//  
//  // read quality
//  indata >> temp;
//  const float Q_DB = temp;
//  
//  // read matrix sizes
//  //int NB = GBLAS_KERNEL_SIZE*2;
//  
//  
//  indata >> temp;
//  int N = (int)(temp);  
//  
//  indata >> temp;
//  int M = (int)(temp);
//  
//  indata >> temp;
//  int K = (int)(temp);
//  
//  float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
//  float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
//  float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);
//  
//  //srand((float)time(NULL));
//  //for (int idx = 0; idx < M*N; idx++) { C[idx] = 0.0; } // one; one *= -1; } 
//  //fill_matrix_random_values(A, M, K, MAX_VALUE_TEST, (float)time(NULL));
//  //fill_matrix_random_values(B, K, N, MAX_VALUE_TEST, (float)time(NULL));
//  
//  int sample = 0;
//  while ( !indata.eof() )
//  {
//    indata >> temp;
//    if ( sample <= (N*K-1) )
//    {
//      // fill A
//      A[sample] = temp;
//    }
//    else
//    {
//      // fill B
//      B[sample - N*K] = temp;
//    }
//    sample++;
//  }
//  indata.close();
//  
//  dgemm_v6(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A /* M x K */, K, B /* K x N */, N, BETA, C /* M x N*/, N, Q_DB);
//  //cout << endl << "C = " << endl; print_matrix_matlab_notation(C, NB, NB);
//  
//  _mm_free(A);
//  _mm_free(B);
//  _mm_free(C);
//}
//
//



void check_sGEMM_correctness_against_Matlab()
{
  srand(0.0f);
  
  int M = 288; //GBLAS_KERNEL_SIZE;
  int N = 288; //GBLAS_KERNEL_SIZE;
  int K = 288*2; //GBLAS_KERNEL_SIZE*2;
  
  float alpha = 1.0f;
  float beta = 1.0f;

  float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
  float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);

  set_matrix_random_integers_values(A, M, K, 16.0f);
  set_matrix_random_integers_values(B, K, N, 16.0f);

  
  write_matrix_to_file(A, M, K, "A.dat");
  write_matrix_to_file(B, K, N, "B.dat");
  
  gblas_sgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N, 0.0);
  //std_sgemm_v4(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
  //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
  
  write_matrix_to_file(C, M, N, "C.dat");
}


void check_sGEMM_correctness_against_Goto()
{
  srand(0.0f);

  const float MAX_VAL = 8.0f;
  
  const int ADD_K = 2;
  const int ADD_N = 1;
  const int ADD_M = 5;
  
  int M = GBLAS_KERNEL_SIZE*4;
  int N = GBLAS_KERNEL_SIZE*2;
  int K = GBLAS_KERNEL_SIZE*2;
  
  float alpha = 1.0f;
  float beta = 3.0f;
  
  float *A       = (float*)_mm_malloc(sizeof(float)*(M + ADD_M)*(K + ADD_K), 16);
  float *B       = (float*)_mm_malloc(sizeof(float)*(K + ADD_K)*(N + ADD_N), 16);
  float *C       = (float*)_mm_malloc(sizeof(float)*(M + ADD_M)*(N + ADD_N), 16);
  float *C_goto  = (float*)_mm_malloc(sizeof(float)*(M + ADD_M)*(N + ADD_N), 16);
  
  set_matrix_random_values(A, (M + ADD_M), (K + ADD_K), MAX_VAL);
  set_matrix_random_values(B, (K + ADD_K), (N + ADD_N), MAX_VAL);
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0f);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0f);
   
  //sgemm_v14(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C, (N + ADD_N), 0);    // 0% acceleration
  gblas_sgemm_plain(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C, (N + ADD_N));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C_goto, (N + ADD_N));
  
  printf("CblasNoTrans vs. CblasNoTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  //print_matrix_difference_matlab_notation(C, C_goto, (M+ADD_M), (N+ADD_N));
  
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0f);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0f);
  
  //sgemm_v14(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, (M + ADD_M), B, (N + ADD_N), beta, C, (N + ADD_N), 0);
  gblas_sgemm_plain(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, (M + ADD_M), B, (N + ADD_N), beta, C, (N + ADD_N));
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, (M + ADD_M), B, (N + ADD_N), beta, C_goto, (N + ADD_N));
  
  printf("CblasTrans vs. CblasNoTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  //print_matrix_difference_matlab_notation(C, C_goto, (M+ADD_M), (N+ADD_N));
  //print_matrix_matlab_notation(C, (M + ADD_M), (N + ADD_N));
  //print_matrix_matlab_notation(C_goto, (M + ADD_M), (N + ADD_N));
  
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0f);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0f);
  
  //sgemm_v14(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, (K + ADD_K), B, (K + ADD_K), beta, C, (N + ADD_N), 0);
  gblas_sgemm_plain(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, (K + ADD_K), B, (K + ADD_K), beta, C, (N + ADD_N));
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, (K + ADD_K), B, (K + ADD_K), beta, C_goto, (N + ADD_N));
  
  printf("CblasNoTrans vs. CblasTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  //print_matrix_difference_matlab_notation(C, C_goto, (M+ADD_M), (N+ADD_N));
  
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0f);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0f);
  
  //sgemm_v14(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, (M + ADD_M), B, (K + ADD_K), beta, C, (N + ADD_N), 0);
  gblas_sgemm_plain(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, (M + ADD_M), B, (K + ADD_K), beta, C, (N + ADD_N));
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, (M + ADD_M), B, (K + ADD_K), beta, C_goto, (N + ADD_N));
  
  printf("CblasTrans vs. CblasTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  //print_matrix_difference_matlab_notation(C, C_goto, (M+ADD_M), (N+ADD_N));
}

void check_dGEMM_correctness_against_Goto()
{
  printf("Check DGEMM vs. Goto\n");
  
  srand(0.0f);
  
  const double MAX_VAL = 8.0;
  
  const int ADD_K = 3;
  const int ADD_N = 1;
  const int ADD_M = 5;
  
  int M = GBLAS_KERNEL_SIZE*4;
  int N = GBLAS_KERNEL_SIZE*2;
  int K = GBLAS_KERNEL_SIZE*7;
  
  double alpha = 0.4f;
  double beta = 3.0f;
  
  double *A       = (double*)_mm_malloc(sizeof(double)*(M + ADD_M)*(K + ADD_K), 16);
  double *B       = (double*)_mm_malloc(sizeof(double)*(K + ADD_K)*(N + ADD_N), 16);
  double *C       = (double*)_mm_malloc(sizeof(double)*(M + ADD_M)*(N + ADD_N), 16);
  double *C_goto  = (double*)_mm_malloc(sizeof(double)*(M + ADD_M)*(N + ADD_N), 16);
  
  set_matrix_random_integers_values(A, (M + ADD_M), (K + ADD_K), MAX_VAL);
  set_matrix_random_integers_values(B, (K + ADD_K), (N + ADD_N), MAX_VAL);
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0);
  
  gblas_dgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C, (N + ADD_N), 0);    // 0% acceleration
  //std_dgemm_v7(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C, (N + ADD_N));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C_goto, (N + ADD_N));
  
  printf("CblasNoTrans vs. CblasNoTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0);
  
  gblas_dgemm_mu(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, (M + ADD_M), B, (N + ADD_N), beta, C, (N + ADD_N), 0);
  //std_dgemm_v7(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, (M + ADD_M), B, (N + ADD_N), beta, C, (N + ADD_N));
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, (M + ADD_M), B, (N + ADD_N), beta, C_goto, (N + ADD_N));
  
  printf("CblasTrans vs. CblasNoTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  //print_matrix_difference_matlab_notation(C, C_goto, (M+ADD_M), (N+ADD_N));
  //print_matrix_matlab_notation(C, (M + ADD_M), (N + ADD_N));
  //print_matrix_matlab_notation(C_goto, (M + ADD_M), (N + ADD_N));
  
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0);
  
  gblas_dgemm_mu(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, (K + ADD_K), B, (K + ADD_K), beta, C, (N + ADD_N), 0);
  //std_dgemm_v7(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, (K + ADD_K), B, (K + ADD_K), beta, C, (N + ADD_N));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, (K + ADD_K), B, (K + ADD_K), beta, C_goto, (N + ADD_N));
  
  printf("CblasNoTrans vs. CblasTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
  
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0);
  
  gblas_dgemm_mu(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, (M + ADD_M), B, (K + ADD_K), beta, C, (N + ADD_N), 0);
  //std_dgemm_v7(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, (M + ADD_M), B, (K + ADD_K), beta, C, (N + ADD_N));
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A, (M + ADD_M), B, (K + ADD_K), beta, C_goto, (N + ADD_N));
  
  printf("CblasTrans vs. CblasTrans: ");
  if ( compare_matrixes(C_goto, C, (M + ADD_M), (N + ADD_N)) )
  {
    printf("OK\n");
  } else {
    printf("Oh, oh, it doesn't work!!\n");
  }
}

void check_dGEMM()
{
  printf("Check DGEMM vs. Goto\n");
  
  srand(0.0f);
  
  const int ADD_K = 0;
  const int ADD_N = 0;
  const int ADD_M = 0;
  
  int M = GBLAS_KERNEL_SIZE;
  int N = GBLAS_KERNEL_SIZE;
  int K = GBLAS_KERNEL_SIZE*4;
  
  double alpha = 0.4f;
  double beta = 3.0f;
  
  double *A       = (double*)_mm_malloc(sizeof(double)*(M + ADD_M)*(K + ADD_K), 16);
  double *B       = (double*)_mm_malloc(sizeof(double)*(K + ADD_K)*(N + ADD_N), 16);
  double *C       = (double*)_mm_malloc(sizeof(double)*(M + ADD_M)*(N + ADD_N), 16);
  double *C_goto  = (double*)_mm_malloc(sizeof(double)*(M + ADD_M)*(N + ADD_N), 16);
  
  set_matrix_random_values_by_block(A, (M + ADD_M), (K + ADD_K), GBLAS_KERNEL_SIZE);
  set_matrix_random_values_by_block(B, (K + ADD_K), (N + ADD_N), GBLAS_KERNEL_SIZE);
  set_matrix_2_value(C, (M + ADD_M), (N + ADD_N), 1.0);
  set_matrix_2_value(C_goto, (M + ADD_M), (N + ADD_N), 1.0);
  
  gblas_dgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, (K + ADD_K), B, (N + ADD_N), beta, C, (N + ADD_N), 0);    // 0% acceleration
}

void exp_paper_mse_vs_gflops_sgemm()
{
  cout << "exp_paper_mse_vs_gflops_sgemm()" << endl;
 
  cout << "Kernel Size = " << GBLAS_KERNEL_SIZE << endl;
  cout << "Sym (float)? = ";
#ifdef FLOAT_SYM
  cout << "yes";
#else
  cout << "no";
#endif
  cout << endl;
  
  srand ( (float)time(NULL) );
  
  msec_timer goto_t;
  msec_timer gblas_pack_t;
  
  const int M = GBLAS_KERNEL_SIZE;
  const int N = M;
  const int K = GBLAS_KERNEL_SIZE*100;
  const double scale_factor = 2*K;
  
  float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
  float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  float *C_cblas = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  
  const float INPUT_RANGE = 128.0f;
  const double VAR_A = 4.0*double(INPUT_RANGE*INPUT_RANGE)/12.0;
  const double VAR_B = VAR_A;     // to be changed if the input is different!
  
  const int NUM_RUNS = 5;
//  const double MSE_START  = 100;
//  const double MSE_STOP   = 100000;

  const double SNR_START  = 10;
  const double SNR_STOP   = 80;
  
  cout << "A(" << M << "," << K << ") x B(" << K << "," << N << ")" << endl;
  cout << "Input Range = (" << -INPUT_RANGE << "," << INPUT_RANGE << ")" << endl;
  
  const float ALPHA = 1.0f;
  const float BETA  = 0.0f;
  
  for (double Curr_SNR = SNR_START; Curr_SNR <= SNR_STOP; Curr_SNR++)
  //for (int snr_p = 0; snr_p < NUM_SNR_TEST_POINT; snr_p ++)
  {
    goto_t.reset();
    gblas_pack_t.reset();
    double acc_snr = 0.0;
    int acc_blk = 0;
    
    for (int n_r = 0; n_r < NUM_RUNS; n_r++)
    {
      set_matrix_random_values(A, M, K, INPUT_RANGE);
      set_matrix_random_values(B, K, N, INPUT_RANGE);
      
      goto_t.start();
      acc_blk += gblas_sgemm_snr(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C, N, Curr_SNR, VAR_A, VAR_B);
      goto_t.stop_and_update();
      
      gblas_pack_t.start();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_cblas, N);
      gblas_pack_t.stop_and_update();
      
      acc_snr += calculate_snr(C_cblas, C, M, N, K, VAR_A, VAR_B);
    }
    
    cout << acc_snr/NUM_RUNS << ", ";
    cout << Curr_SNR << ", ";
    cout << ((double)acc_blk)/NUM_RUNS << ", ";
    cout << (double)convert_to_gigaflops(gblas_pack_t.get_time()/NUM_RUNS, scale_factor)*M*N << ", ";
    cout << (double)convert_to_gigaflops(goto_t.get_time()/NUM_RUNS, scale_factor)*M*N;
    cout << endl;
  }
  
  _mm_free(A);
  _mm_free(B);  
  _mm_free(C);  
  _mm_free(C_cblas);  
}

void exp_paper_mse_vs_gflops_dgemm()
{ 
  cout << "exp_paper_mse_vs_gflops_dgemm()" << endl;
  
  cout << "Kernel Size = " << GBLAS_KERNEL_SIZE << endl;
  cout << "Sym (double)? = ";
#ifdef DOUBLE_SYM
  cout << "yes";
#else
  cout << "no";
#endif
  cout << endl;
  
  msec_timer goto_t;
  msec_timer gblas_pack_t;
  
  const int M = GBLAS_KERNEL_SIZE;
  const int N = M;
  const int K = GBLAS_KERNEL_SIZE*100;
  const double scale_factor = 2*K;
  
  double *A       = (double*)_mm_malloc(sizeof(double)*M*K, 16);
  double *B       = (double*)_mm_malloc(sizeof(double)*K*N, 16);
  double *C       = (double*)_mm_malloc(sizeof(double)*M*N, 16);
  double *C_cblas = (double*)_mm_malloc(sizeof(double)*M*N, 16);
    
  const double INPUT_RANGE = 128.0;
  const double VAR_A = 4.0*double(INPUT_RANGE*INPUT_RANGE)/12.0;
  const double VAR_B = VAR_A;     // to be changed if the input is different!
  
  const int NUM_RUNS = 5;
  const double SNR_START  = 10;
  const double SNR_STOP   = 100;
  
  cout << "A(" << M << "," << K << ") x B(" << K << "," << N << ")" << endl;
  cout << "Input Range = (" << -INPUT_RANGE << "," << INPUT_RANGE << ")" << endl;
  
  const double ALPHA = 1.0f;
  const double BETA  = 0.0f;
  
  for (double Curr_SNR = SNR_START; Curr_SNR <= SNR_STOP; Curr_SNR++)
  //for (int mse = MSE_START; mse <= MSE_STOP; mse += 500)
  {
    goto_t.reset();
    gblas_pack_t.reset();
    double acc_snr = 0.0;
    int acc_blk = 0;
    
    for (int n_r = 0; n_r < NUM_RUNS; n_r++)
    {
      set_matrix_random_values(A, M, K, INPUT_RANGE);
      set_matrix_random_values(B, K, N, INPUT_RANGE);
      
      goto_t.start();
      acc_blk += gblas_dgemm_snr(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C, N, Curr_SNR, VAR_A, VAR_B);
      goto_t.stop_and_update();
      
      gblas_pack_t.start();
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_cblas, N);
      gblas_pack_t.stop_and_update();
      
      acc_snr += calculate_snr(C_cblas, C, M, N, K, VAR_A, VAR_B);
    }
    cout << acc_snr/NUM_RUNS << ", ";
    cout << Curr_SNR << ", ";
    cout << ((double)acc_blk)/NUM_RUNS;
    cout << ", " << (double)convert_to_gigaflops(gblas_pack_t.get_time()/NUM_RUNS, scale_factor)*M*N << ", " << (double)convert_to_gigaflops(goto_t.get_time()/NUM_RUNS, scale_factor)*M*N;
    cout << endl;
    

  }
  
  _mm_free(A);
  _mm_free(B);  
  _mm_free(C);  
  _mm_free(C_cblas);  
}

void exp_paper_accel_vs_gflops_sgemm()
{  
  msec_timer goto_t;
  msec_timer gblas_pack_t;
  
  const int FACTOR = 4032/GBLAS_KERNEL_SIZE;
  const int M = GBLAS_KERNEL_SIZE*FACTOR; 
  const int N = M;
  const int K = M;
  
  float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
  float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  float *C_cblas = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  
  if ( A == NULL || B == NULL || C == NULL || C_cblas == NULL )
  {
    std::cerr << "Memory allocation problem" << std::endl;
    exit(1);
  }
  
  const double scale_factor = 2*K;
  
  const float ALPHA = 1.0f;
  const float BETA  = 0.0f;
  
  int acc_blk;
  
  for (int acc = 0; acc <= 100; acc += 10)
  {
    set_matrix_random_integers_values(A, M, K, 16.0f);
    set_matrix_random_integers_values(B, K, N, 16.0f);
    
    goto_t.start();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_cblas, N);    
    goto_t.stop_and_update();
    
    gblas_pack_t.start();
    acc_blk = gblas_sgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C, N, acc);
    gblas_pack_t.stop_and_update();
    
    cout << M << ", " << N << ", " << K << ", ";
    cout << calculate_mse(C_cblas, C, M, N) << ", " << acc << ", " << acc_blk << "/" << (FACTOR*FACTOR*FACTOR) << ", ";
    cout << (double)convert_to_gigaflops(gblas_pack_t.get_time(), scale_factor)*M*N << ", ";
    cout << (double)convert_to_gigaflops(goto_t.get_time(), scale_factor)*M*N;
    cout << endl;
    
    goto_t.reset();
    gblas_pack_t.reset();
  }
  
  _mm_free(A);
  _mm_free(B);  
  _mm_free(C);  
  _mm_free(C_cblas);  
}

void exp_paper_accel_vs_gflops_sgemm_with_EC()
{  
  msec_timer goto_t;
  msec_timer gblas_pack_t;
  msec_timer gblas_pack_t_EC;
  
  const int FACTOR = 9; //GBLAS_KERNEL_SIZE*4/GBLAS_KERNEL_SIZE;
  const int M = GBLAS_KERNEL_SIZE*FACTOR; 
  const int N = M;
  const int K = M;
  
  float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
  float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
  float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  float *C_EC    = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  float *C_cblas = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  
  //float *C_zero = (float*)_mm_malloc(sizeof(float)*M*N, 16);
  //set_matrix_2_value(C_zero, M, N, 0.0f); 
  
  if ( A == NULL || B == NULL || C == NULL || C_cblas == NULL )
  {
    std::cerr << "Memory allocation problem" << std::endl;
    exit(1);
  }
  
  const double scale_factor = 2*K;
  
  const float ALPHA = 1.0f;
  const float BETA  = 0.0f;
  
  set_matrix_random_integers_values(A, M, K, 16.0f);
  set_matrix_random_integers_values(B, K, N, 16.0f);
  
  int acc_blk = 0, acc_blk_EC = 0;
  std::cout << "M, N, K, Acc (%), MSE(GBLAS), MSE(GBLAS w/EC), Blk Acc, Blk Acc w/EC, Gflops (GOTO), Gflops (GBLAS), Gflops (GBLAS w/ EC)" << std::endl;
  for (int acc = 0; acc <= 100; acc += 10)
  {
    goto_t.start();
    gblas_sgemm_plain(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_cblas, N);
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_cblas, N);    
    goto_t.stop_and_update();
 
    gblas_pack_t.start();
    acc_blk = gblas_sgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C, N, acc);
    gblas_pack_t.stop_and_update();

    gblas_pack_t_EC.start();
    acc_blk_EC = gblas_sgemm_mu_EC(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_EC, N, acc);
    gblas_pack_t_EC.stop_and_update();

//    cout << "C = " << endl;
//    print_matrix_matlab_notation(C, M, N);
//    
//    cout << "C_EC = " << endl;
//    print_matrix_matlab_notation(C_EC, M, N);
    
    //print_matrix_difference_matlab_notation(C_cblas, C_EC, M, N);
    
    cout << M << ", " << N << ", " << K << ", " << acc << ", ";
    cout << calculate_mse(C_cblas, C, M, N) << ", ";
    //cout << calculate_mse(C_zero, C_EC, M, N) << ", ";
    cout << calculate_mse(C_cblas, C_EC, M, N) << ", ";
    cout << acc_blk << "/" << (FACTOR*FACTOR*FACTOR) << ", ";
    cout << acc_blk_EC << "/" << (FACTOR*FACTOR*FACTOR) << ", ";
    cout << (double)convert_to_gigaflops(goto_t.get_time(), scale_factor)*M*N << ", ";    
    cout << (double)convert_to_gigaflops(gblas_pack_t.get_time(), scale_factor)*M*N << ", ";
    cout << (double)convert_to_gigaflops(gblas_pack_t_EC.get_time(), scale_factor)*M*N;
    cout << endl;
    
    goto_t.reset();
    gblas_pack_t.reset();
    gblas_pack_t_EC.reset();    
  }
  
  _mm_free(A);
  _mm_free(B);  
  _mm_free(C);  
  _mm_free(C_cblas);  
}


void exp_paper_accel_vs_gflops_dgemm()
{
  msec_timer goto_t;
  msec_timer gblas_pack_t;

  const int FACTOR = 4032/GBLAS_KERNEL_SIZE;
  const int M = GBLAS_KERNEL_SIZE*FACTOR;
  const int N = M;
  const int K = M;

  double *A       = (double*)_mm_malloc(sizeof(double)*M*K, 16);
  double *B       = (double*)_mm_malloc(sizeof(double)*K*N, 16);
  double *C       = (double*)_mm_malloc(sizeof(double)*M*N, 16);
  double *C_cblas = (double*)_mm_malloc(sizeof(double)*M*N, 16);

  if ( A == NULL || B == NULL || C == NULL || C_cblas == NULL )
  {
    std::cerr << "Memory allocation problem" << std::endl;
    exit(1);
  }

  const double scale_factor = 2*K;

  const float ALPHA = 1.0f;
  const float BETA  = 0.0f;

  int acc_blk;

  for (int acc = 0; acc <= 100; acc += 10)
  {
    set_matrix_random_values(A, M, K, 1024.0);
    set_matrix_random_values(B, K, N, 1024.0);

    goto_t.start();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C_cblas, N);
    goto_t.stop_and_update();

    gblas_pack_t.start();
    acc_blk = gblas_dgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, K, B, N, BETA, C, N, acc);
    gblas_pack_t.stop_and_update();

    cout << M << ", " << N << ", " << K << ", ";
    cout << calculate_mse(C_cblas, C, M, N) << ", " << acc << ", " << acc_blk << "/" << (FACTOR*FACTOR*FACTOR) << ", ";
    cout << (double)convert_to_gigaflops(gblas_pack_t.get_time(), scale_factor)*M*N << ", ";
    cout << (double)convert_to_gigaflops(goto_t.get_time(), scale_factor)*M*N;
    cout << endl;

    goto_t.reset();
    gblas_pack_t.reset();
  }

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_cblas);
}


#include "gblas_sse_utils.h"

void test_unpack_complete_tight_v1()
{
  float inv_eps = 61.0f + 50.0f;
  float eps = 1.0f/(inv_eps);
  
  float p1 = float(31)*inv_eps + float(22) + float(9)*eps;
  float p2 = float(23)*inv_eps - float(25) + float(13)*eps;
  float p3 = float(34)*inv_eps + float(23) - float(11)*eps;
  float p4 = float(13)*inv_eps - float(19) + float(12)*eps;
  
  __m128 i = _mm_set_ps(p4, p3, p2, p1);
  
  __m128 o1, o2, o3;
  
  //unpack_complete_tight_v1(i, eps, inv_eps, o1, o2, o3);
  //fast_unpack_tight_v2(i, eps, inv_eps);
  {
    const __m128 EPS = _mm_set1_ps(eps);
    const __m128 INV_EPS = _mm_set1_ps(inv_eps);
    __m128 cmp;
                                                // i = x1*eps^-1 + x2 + x3*eps
    o1   = _mm_mul_ps(i, EPS);                  // o1 = x1 + x2*eps + x3*eps^2
    
    cmp = _mm_cmpgt_ps(o1, _MM_ZERO_S);           
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o1   = _mm_add_ps(o1, cmp);                     
    
    o1   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o1));   // o1 = round(o1) = x1
    
    o2   = _mm_mul_ps(o1, INV_EPS);                 // o2 = x1*eps^-1
    i   = _mm_sub_ps(i, o2);                        // i = i - o2 = x2 + x3*eps
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o2   = _mm_add_ps(i, cmp);                      // o2 = i +- 0.5
    
    o2   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o2));   // o2 = round(o2) = x2
    
    o3   = _mm_sub_ps(i, o2);                       // o3 = i - o2 = x3*eps
    o3   = _mm_mul_ps(o3, INV_EPS);                 // o3 = o3*eps^-1 = x3~ // SOMETHING WRONG HERE!
    
    cmp = _mm_cmpgt_ps(o3, _MM_ZERO_S);
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o3   = _mm_add_ps(o3, cmp);
    
    o3   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o3));   // o3 = round(o3) = x3
  }
  
  
  float x[12] __attribute__ ((aligned (16)));
  
  _mm_store_ps(&x[0], o1);
  _mm_store_ps(&x[4], o2);
  _mm_store_ps(&x[8], o3);
  
  //_mm_store_ps(&x[0], i);
  //_mm_store_ps(&x[4], i);
  //_mm_store_ps(&x[8], i);
  
  for (int idx = 0; idx < 12; idx++) cout << x[idx] << " ";
  cout << endl;
}


void test_unpack_complete_tight_v2()
{
  float inv_eps = 2000.0f + 50.0f;
  float eps = 1.0f/(inv_eps);
  
  float p1 = float(31) + float(22)*eps + float(9)*eps*eps;
  float p2 = float(23) - float(25)*eps + float(13)*eps*eps;
  float p3 = float(34) + float(23)*eps - float(11)*eps*eps;
  float p4 = float(13) - float(19)*eps + float(12)*eps*eps;
  
  __m128 i = _mm_set_ps(p4, p3, p2, p1);
  
  __m128 o1, o2, o3;
  
  //unpack_complete_tight_v1(i, eps, inv_eps, o1, o2, o3);
  //fast_unpack_tight_v2(i, eps, inv_eps);
  {
    //const __m128 EPS = _mm_set1_ps(eps);
    const __m128 INV_EPS = _mm_set1_ps(inv_eps);
    __m128 cmp;
    // i = x1 + x2*eps + x3*eps^2
    // o1   = i ; //_mm_mul_ps(i, EPS);                  // o1 = x1 + x2*eps + x3*eps^2
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);           
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o1  = _mm_add_ps(i, cmp);                     
    
    o1  = _mm_cvtepi32_ps(_mm_cvttps_epi32(o1));     // o1 = round(o1) = x1
    
    o2  = _mm_sub_ps(i, o1);                        // o2 = i - o2 = x2*eps + x3*eps^2
    i  = _mm_mul_ps(o2, INV_EPS);                   // i = i*eps^-1 = x2 + x3*eps
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o2  = _mm_add_ps(i, cmp);                      // o2 = i +- 0.5
    
    o2  = _mm_cvtepi32_ps(_mm_cvttps_epi32(o2));   // o2 = round(i) = x2
    
    o3   = _mm_sub_ps(i, o2);                       // o3 = i - o2 = x3*eps
    i   = _mm_mul_ps(o3, INV_EPS);                   // o3 = o3*eps^-1 = x3~ // SOMETHING WRONG HERE!
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o3   = _mm_add_ps(i, cmp);
    
    o3   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o3));   // o3 = round(o3) = x3
  }
  
  
  float x[12] __attribute__ ((aligned (16)));
  
  _mm_store_ps(&x[0], o1);
  _mm_store_ps(&x[4], o2);
  _mm_store_ps(&x[8], o3);
  
  //_mm_store_ps(&x[0], i);
  //_mm_store_ps(&x[4], i);
  //_mm_store_ps(&x[8], i);
  
  for (int idx = 0; idx < 12; idx++) cout << x[idx] << " ";
  cout << endl;
}

void test_unpack_complete_tight_v3()
{
  float inv_eps = 2000.0f + 50.0f;
  float eps = 1.0f/(inv_eps);
  
  float p1 = float(31)*eps*eps*eps + float(22)*eps + float(9)*eps*eps;
  float p2 = float(23)*eps*eps*eps - float(25)*eps + float(13)*eps*eps;
  float p3 = float(34)*eps*eps*eps + float(23)*eps - float(11)*eps*eps;
  float p4 = float(13)*eps*eps*eps - float(19)*eps + float(12)*eps*eps;
  
  __m128 i = _mm_set_ps(p4, p3, p2, p1);
  
  __m128 o1, o2, o3;
  
  //unpack_complete_tight_v1(i, eps, inv_eps, o1, o2, o3);
  //fast_unpack_tight_v2(i, eps, inv_eps);
  {
    //const __m128 EPS = _mm_set1_ps(eps);
    const __m128 INV_EPS = _mm_set1_ps(inv_eps);
    __m128 cmp;
    // i = x1 + x2*eps + x3*eps^2
    i = _mm_mul_ps(i, INV_EPS);                  // o1 = x1 + x2*eps + x3*eps^2
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);           
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o1  = _mm_add_ps(i, cmp);                     
    
    o1  = _mm_cvtepi32_ps(_mm_cvttps_epi32(o1));     // o1 = round(o1) = x1
    
    o2  = _mm_sub_ps(i, o1);                        // o2 = i - o2 = x2*eps + x3*eps^2
    i  = _mm_mul_ps(o2, INV_EPS);                   // i = i*eps^-1 = x2 + x3*eps
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o2  = _mm_add_ps(i, cmp);                      // o2 = i +- 0.5
    
    o2  = _mm_cvtepi32_ps(_mm_cvttps_epi32(o2));   // o2 = round(i) = x2
    
    o3   = _mm_sub_ps(i, o2);                       // o3 = i - o2 = x3*eps
    i   = _mm_mul_ps(o3, INV_EPS);                   // o3 = o3*eps^-1 = x3~ // SOMETHING WRONG HERE!
    
    cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);
    cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
    cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
    o3   = _mm_add_ps(i, cmp);
    
    o3   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o3));   // o3 = round(o3) = x3
  }
  
  
  float x[12] __attribute__ ((aligned (16)));
  
  _mm_store_ps(&x[0], o1);
  _mm_store_ps(&x[4], o2);
  _mm_store_ps(&x[8], o3);
  
  //_mm_store_ps(&x[0], i);
  //_mm_store_ps(&x[4], i);
  //_mm_store_ps(&x[8], i);
  
  for (int idx = 0; idx < 12; idx++) cout << x[idx] << " ";
  cout << endl;
}

void test_pack_unpack_v1()
{
  float inv_eps = 33.0f + 0.0f;
  float eps = 1.0f/(inv_eps);
  
  float p1 = float(21) + float(30)*eps + float(9)*eps*eps;
  float p2 = - float(21) + float(30)*eps - float(9)*eps*eps;
  
  __m128 pd = _mm_set1_ps(p1);
  __m128 pu = _mm_set1_ps(p2);

  __m128 d = _mm_add_ps(pd, pu);
   d = _mm_mul_ps(d, _mm_set1_ps(inv_eps/2.0f));
  
  
  float x[4] __attribute__ ((aligned (16)));
  
  _mm_store_ps(&x[0], d);
  
  //_mm_store_ps(&x[0], i);
  //_mm_store_ps(&x[4], i);
  //_mm_store_ps(&x[8], i);
  
  for (int idx = 0; idx < 4; idx++) cout << x[idx] << " ";
  cout << endl;
}

