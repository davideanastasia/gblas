/*
 *  matrix_utils.h
 *  xblas
 *
 *  Created by Davide Anastasia on 30/06/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010 University College London. All rights reserved.
 *
 */

#ifndef __MATRIX_UTILS_H__
#define __MATRIX_UTILS_H__

#include <float.h>

#include "matrix_utils_t.h"

//int row_col_2_lin(int row, int col, int n_col);
inline int row_col_2_lin(int row, int col, int n_col)
{
  return (row*n_col + col);
}

//bool compare_matrixes(const double* mat1, double* mat2, int rows, int cols);
bool compare_matrixes_and_reset(double* mat_computed, double* mat_original, int rows, int cols);

//void print_matrix_matlab_notation(const double* mat, int rows, int cols);
//void reset_matrix(double* mat_computed, double* mat_original, int rows, int cols);

//void swap_pointers(void * p_one, void * p_two);

float matrix_difference(const float* mat1, const float* mat2, int rows, int cols, float* mat_out);

template <class T>
inline void get_matrix_min_max(const T* mat, int rows, int cols, T& max_f, T& min_f)
{
  T _max = -FLT_MAX;
  T _min = FLT_MAX;
  for (int idx=0; idx < rows*cols; idx++)
  {
    _max = max(_max, mat[idx]);
    _min = min(_min, mat[idx]);
  }
  max_f = _max;
  min_f = _min;
}

#endif