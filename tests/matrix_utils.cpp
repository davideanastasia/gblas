/*
 *  matrix_utils.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 30/06/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010 University College London. All rights reserved.
 *
 */

#include "matrix_utils.h"
//#include "sse_utils.h"

using namespace std;

bool compare_matrixes_and_reset(double* mat_computed, double* mat_original, int rows, int cols)
{
  int idx = 0;
  bool test = true;
  for (int i=0; i < rows; i++)
  {
    for (int j=0; j < cols; j++)
    {
      idx = row_col_2_lin(i, j, cols);
      if ( mat_computed[idx] != mat_original[idx] ) test = false;
      mat_computed[idx] = mat_original[idx];  // reset
    }
  }
  
  return test;
}

float matrix_difference(const float* mat1, const float* mat2, int rows, int cols, float* mat_out)
{
  const int ELEMS = rows*cols;
  float max_abs = 0.0f;
  
  for (int idx = 0; idx < ELEMS; idx++)
  {
    mat_out[idx] = mat1[idx] - mat2[idx];
    
    max_abs = max(max_abs, abs(mat_out[idx]));
  }
                  
  return max_abs;
}  

