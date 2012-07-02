/*
 *  matrix_utils_t.h
 *  Template functions for matrix_utils.h
 *  gblas
 *
 *  Created by Davide Anastasia on 07/07/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010 University College London. All rights reserved.
 *
 */

#ifndef __MATRIX_UTILS_TEMPLATE_H__
#define __MATRIX_UTILS_TEMPLATE_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#include "matrix_utils.h"

using namespace std;

template <class T>
void reset_matrix(T* mat_computed, const T* mat_original, int rows, int cols)
{
  for (int i=0; i < rows; i++)
  {
    for (int j=0; j < cols; j++)
    {
      mat_computed[i*cols + j] = mat_original[i*cols + j];  // reset
    }
  }
}

template <class T>
bool compare_matrixes(const T* corr, const T* computed, int rows, int cols)
{ 
  return (!memcmp(corr, computed, sizeof(T)*rows*cols));
}

template <class T>
void print_matrix_matlab_notation(const T* mat, int rows, int cols)
{
  cout << "[";
  for (int i=0; i < rows; i++)
  {
    for (int j=0; j < cols; j++)
    {
      cout << " " ; 
      cout << setw(5);
      cout << mat[i*cols + j];
      
      if ( j == (cols-1) )
      {
        if ( i == (rows-1) )
        {
          cout << "]";
          cout << endl;
        }
        else
        {
          cout << ";";
          cout << endl;
        }
      }
    }
  }
}

template <class T>
void print_matrix_plain_notation(const T* mat, const int ROWS, const int COLS)
{
  for (int ii=0; ii < ROWS*COLS; ii++)
  {
    cout << mat[ii] << " ";
  }
}

template <class T>
void print_matrix_csv_notation(const T* mat, const int ROWS, const int COLS)
{
  for (int ii=0; ii < ROWS*COLS-1; ii++)
  {
    cout << mat[ii] << ", ";
  }
  cout << mat[ROWS*COLS-1];
}

template <class T>
void print_matrix_values_to_file(const T* mat, int rows, int cols, std::ofstream& stream)
{
  const int M_ELEMS = rows*cols;
  
  for (int i = 0; i < M_ELEMS; i++)
  {
    stream << mat[i] << " ";
  }
  stream << endl;
}

template <class T>
void print_matrix_matlab_notation_i(const T* mat, int rows, int cols)
{
  cout << "[";
  for (int i=0; i < rows; i++)
  {
    for (int j=0; j < cols; j++)
    {
      cout << " " << (int)(mat[i*cols + j]);
      if ( j == (cols-1) ) {
        if ( i == (rows-1) ) {
          cout << "]" << endl;
        } else {
          cout << ";" << endl;
        }
      }
    }
  }
}

template <class T>
void print_matrix_difference_matlab_notation(const T* mat1, const T* mat2, int rows, int cols)
{
  cout << "[";
  for (int i=0; i < rows; i++)
  {
    for (int j=0; j < cols; j++)
    {
      cout << " " << (mat1[i*cols + j] - mat2[i*cols + j]);
      if ( j == (cols-1) ) {
        if ( i == (rows-1) ) {
          cout << "]" << endl;
        } else {
          cout << ";" << endl;
        }
      }
    }
  }
}

template <class T>
void matrix_difference(const T* mat1, const T* mat2, int rows, int cols, T* mat_out)
{
  const int N_ELEMS = rows*cols;

  for (int idx = 0; idx < N_ELEMS; idx++)
  {
    mat_out[idx] = (mat1[idx] - mat2[idx]);
  }
}

template <class T>
void matrix_difference_v2(const T* mat1, const T* mat2, int rows, int cols, T* mat_out, T& avg, T& max_abs)
{
  avg = 0.0f; max_abs = 0.0f;
  const int elems = rows*cols;
  
  for (int idx = 0; idx < elems; idx++)
  {
    mat_out[idx] = (mat1[idx] - mat2[idx]);
    avg += (mat_out[idx]*mat_out[idx]);
    
    if ( abs(mat_out[idx]) > max_abs ) max_abs = abs(mat_out[idx]);
  }
  
  avg /= elems;
}

template <class T>
int count_num_no_zero_elem(const T* mat_in, int rows, int cols)
{
  int no_zero_elems = 0;
  const int elems = rows*cols;
  for (int idx = 0; idx < elems; idx++)
  {
    if ( mat_in[idx] != 0 ) no_zero_elems++;
  }
  return no_zero_elems;
}

//TODO: double check of this function!
#define NUM_MAX_VALUES  (19)
const int MAX_VALUES[NUM_MAX_VALUES] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

template <class T>
void set_matrix_random_values_by_block(T* mat_in, int rows, int cols, int blk_size, float seed = 0)
{
  //srand(seed);
  int max_value = 0;
  
  for (int i = 0; i < rows; i += blk_size)
  {
    for (int j = 0; j < cols; j += blk_size)
    {
      max_value = MAX_VALUES[(rand()%NUM_MAX_VALUES)];        
      
      for (int i2 = 0; i2 < blk_size; i2++)
      {
        for (int j2 = 0; j2 < blk_size; j2++)
        {
          mat_in[(i+i2)*cols + j + j2] = ((float)rand()/RAND_MAX)*2*(max_value+1) - (max_value+1);
        }
      }
    }
  }  
}

template <class T>
void set_matrix_random_values(T* mat_in, int rows, int cols, const T V_MAX, float seed = 0)
{
  //srand(seed);
  
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      mat_in[i*cols + j] = (T)(((float)rand()/RAND_MAX)*2*(V_MAX+1) - (V_MAX+1));
    }
  }  
}

template <class T>
void set_matrix_random_integers_values(T* mat_in, const int ROWS, const int COLS, const T V_MAX, float seed = 0)
{
  //srand(seed);
  
  long V_MAX_I = floor(abs(V_MAX)+0.5);
  if ( V_MAX_I == 0 ) V_MAX_I = 1;
  
  for (int ix = 0; ix < (ROWS*COLS); ix++)
  {
    mat_in[ix] = (T)(((rand()%(2*V_MAX_I+1))) - V_MAX_I);
  }  
}

template <class Type>
void set_matrix_2_value(Type* mat_in, const int ROWS, const int COLS, const Type VALUE)
{ 
  for (int ix = 0; ix < ROWS*COLS; ix++)
  {
    mat_in[ix] = VALUE;
  }  
}


template <class T>
float calculate_psnr(const T* m_orig, const T* m_calc, int rows, int cols)
{
  T noise = 0.0f;
  T signal = 0.0f;
  const int elems = rows*cols;
  
  for (int idx = 0; idx < elems; idx++)
  {
    noise  += (m_orig[idx] - m_calc[idx])*(m_orig[idx] - m_calc[idx]);
    signal += m_orig[idx]*m_orig[idx];
  }
  
  return (10.0f*(float)log10(signal/noise));
}

template <class T>
double calculate_mse(const T* m_orig, const T* m_calc, int rows, int cols)
{
  double noise = 0.0;
  //T signal = 0.0f;
  const int elems = rows*cols;
  
#pragma omp parallel for reduction(+:noise)
  for (int idx = 0; idx < elems; idx++)
  {
    noise  += (double)(m_orig[idx] - m_calc[idx])*(m_orig[idx] - m_calc[idx]);
  }
  noise /= elems;
  
  return noise;
}

//template <class T>
//double calculate_snr(const T* m_orig, const T* m_calc, const int rows, const int cols, const int K, const T max_value_A, const T max_value_B)
//{
//  double noise = 0.0;
//  const int ELEMS = rows*cols;
//  
//  for (int idx = 0; idx < ELEMS; idx++)
//  {
//    noise += (double)(m_orig[idx] - m_calc[idx])*(m_orig[idx] - m_calc[idx]);
//  }
//  noise /= ELEMS;
//  
//  double std_A = 2.0*max_value_A/sqrt(12.0);
//  double std_B = 2.0*max_value_B/sqrt(12.0);
//  
//  double signal_power = (double)(K)*(std_A*std_A*std_B*std_B);
//  
//  //signal = ((T)K)*powf((2*max_value/sqrt(12.0f)), 4.0f);
//  
//  return (10.0*log10(signal_power/noise));
//}

template <class T>
double calculate_snr(const T* m_orig, const T* m_calc, const int rows, const int cols, const int K, const double VAR_A, const double VAR_B)
{
  double noise = calculate_mse(m_orig, m_calc, rows, cols);
  
  double signal_power = (double(K)*VAR_A*VAR_B);
  
  return (10.0*log10(signal_power/noise));
}

template <class Matrix_Type>
void write_matrix_to_file(Matrix_Type* A, const int ROWS, const int COLS, const char* fname)
{
  FILE* fout = fopen(fname, "wb"); // open write-binary
  fwrite(A, sizeof(Matrix_Type), ROWS*COLS, fout);
  fclose(fout);
}

template <class T>
void calculate_var_and_mean(const T* m_in, const int rows, const int cols, double &mean, double &stddev)
{
  // mean!
  const int N_ELEMS = rows*cols;
  double tmp = 0.0;
  
  for (int idx = 0; idx < N_ELEMS; idx++)
  {
    tmp += (double(m_in[idx])/N_ELEMS); // I divide during the interation to avoid precision errors
  }
  
  mean = tmp;
  
  // std dev
  tmp = 0.0;
  for (int idx = 0; idx < N_ELEMS; idx++)
  {
    tmp += pow((double(m_in[idx]) - mean), 2.0);
  }
  
  stddev = tmp/N_ELEMS;
}

#endif
