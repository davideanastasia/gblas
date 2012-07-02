/*
 *  gblas_matrix_utils.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 30/06/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <cmath>
#include <cstdio>
#include <float.h>

#include "gblas_matrix_utils.h"
#include "gblas_sse_utils.h"

using namespace std;

void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const float* m_in, const float alpha, float* m_out, float* m_max, float* m_min)
{
  //print_matrix_matlab_notation(m_in, rows, ld);
  
  const int _ROWS_ = rows - (rows%GBLAS_KERNEL_SIZE);
  const int _COLS_ = cols - (cols%GBLAS_KERNEL_SIZE);

  int a_idx = 0;
  __m128 _max, _min;
  
  if ( Trans == CblasNoTrans )
  {
    const float* from;
    float* to = m_out;
    
    __m128 T0;
    
    if ( alpha == 1.0f )
    {
      printf("A \n");

      for (int i=0, idx = 0; i < _ROWS_; i+=GBLAS_KERNEL_SIZE) // , idx = 0
      {
        for (int k = 0; k < _COLS_; k+=GBLAS_KERNEL_SIZE)
        {
          // reset counter
          _max = _mm_set1_ps(FLT_MIN);
          _min = _mm_set1_ps(FLT_MAX);
          for (int r = 0; r<GBLAS_KERNEL_SIZE; r++)
          {
            from = &m_in[(i+r)*ld + k];
            
            for (int c = 0; c<GBLAS_KERNEL_SIZE; c+=4, idx+=4) // , idx+=4
            {
              T0 = _mm_loadu_ps(from);
              // Min/Max
              _max = _mm_max_ps(_max, T0);
              _min = _mm_min_ps(_min, T0);
              // END: Min/Max
              
              _mm_store_ps(to, T0);
              
              to += 4;
              from += 4;
            }
          }
          //save max value!
          horiz_max(_max, m_max[a_idx]);
          horiz_min(_min, m_min[a_idx]);
          
          a_idx++;
        }
      }
      
    }
    else
    {     
      printf("B \n");
      
      const __m128 ALPHA = _mm_set1_ps(alpha);
      
      for (int i=0; i < _ROWS_; i+=GBLAS_KERNEL_SIZE) // , idx = 0
      {
        for (int k = 0; k < _COLS_; k+=GBLAS_KERNEL_SIZE)
        {
          // reset counter
          _max = _mm_set1_ps(FLT_MIN); _min = _mm_set1_ps(FLT_MAX);
          
          for (int r = 0; r<GBLAS_KERNEL_SIZE; r++)
          {
            from = &m_in[(i+r)*ld + k];
            
            for (int c = 0; c<GBLAS_KERNEL_SIZE; c+=4) // , idx+=4
            {
              T0 = _mm_loadu_ps(from);
              T0 = _mm_mul_ps(T0, ALPHA);
              // Min/Max
              _max = _mm_max_ps(_max, T0);
              _min = _mm_min_ps(_min, T0);
              // END: Min/Max
              _mm_store_ps(to, T0); // &m_out[idx]
              
              to += 4;
              from += 4;
            }
          }
          //save max value!
          horiz_max(_max, m_max[a_idx]);
          horiz_min(_min, m_min[a_idx]);
          
          a_idx++;
        }
      }
      
    }
  }
  else // CblasTrans
  {
    __m128 B0, B1, B2, B3;
    
    const float *from_r0, *from_r1, *from_r2, *from_r3;
    float *to_r0, *to_r1, *to_r2, *to_r3;
    
    if ( alpha == 1.0f )
    {
      printf("C \n");
      
      a_idx = 0;
      for (int cc = 0; cc < _COLS_; cc += GBLAS_KERNEL_SIZE)
      { 
        for (int rr = 0; rr < _ROWS_; rr += GBLAS_KERNEL_SIZE)
        {
          // reset min/max
          _max = _mm_set1_ps(-1000.f); //FLT_MAX);
          _min = _mm_set1_ps(FLT_MAX);
          
          for (int r=0; r<GBLAS_KERNEL_SIZE; r+=4)
          {
            // update pointers
            from_r0 = &m_in[(rr + r)*ld   + cc]; // pivot
            from_r1 = from_r0 + ld;
            from_r2 = from_r1 + ld;
            from_r3 = from_r2 + ld;
            
            to_r0 = &m_out[rows*cc + rr*GBLAS_KERNEL_SIZE + r]; // pivot
            to_r1 = to_r0 + GBLAS_KERNEL_SIZE;
            to_r2 = to_r1 + GBLAS_KERNEL_SIZE;
            to_r3 = to_r2 + GBLAS_KERNEL_SIZE;
            
            for (int c=0; c<GBLAS_KERNEL_SIZE; c+=4)
            {         
              B0 = _mm_loadu_ps(from_r0);
              B1 = _mm_loadu_ps(from_r1);
              B2 = _mm_loadu_ps(from_r2);
              B3 = _mm_loadu_ps(from_r3);
              
              _MM_TRANSPOSE4_PS(B0, B1, B2, B3);
              
              // Min/Max
              _max = _mm_max_ps(_max, B0);
              _max = _mm_max_ps(_max, B1);
              _max = _mm_max_ps(_max, B2);
              _max = _mm_max_ps(_max, B3);
              
              _min = _mm_min_ps(_min, B0);
              _min = _mm_min_ps(_min, B1);
              _min = _mm_min_ps(_min, B2);
              _min = _mm_min_ps(_min, B3);
              
              // END: Min/Max
              _mm_store_ps(to_r0, B0);
              _mm_store_ps(to_r1, B1);
              _mm_store_ps(to_r2, B2);
              _mm_store_ps(to_r3, B3);
              
              // update pointers
              from_r0 += 4;
              from_r1 += 4;
              from_r2 += 4;
              from_r3 += 4;
              
              to_r0 += 4*GBLAS_KERNEL_SIZE;
              to_r1 = to_r0 + GBLAS_KERNEL_SIZE;
              to_r2 = to_r1 + GBLAS_KERNEL_SIZE;
              to_r3 = to_r2 + GBLAS_KERNEL_SIZE;
            }
          }
          //save max value!
          horiz_max(_max, m_max[a_idx]);
          horiz_min(_min, m_min[a_idx]);
          
          a_idx++;
        }
      }
      
    }
    else
    {
      printf("D \n");

      const __m128 ALPHA = _mm_set1_ps(alpha);
      
      a_idx = 0;
      for (int cc = 0; cc < _COLS_; cc += GBLAS_KERNEL_SIZE)
      { 
        for (int rr = 0; rr < _ROWS_; rr += GBLAS_KERNEL_SIZE)
        {
          // reset min/max
          _max = _mm_set1_ps(FLT_MIN); _min = _mm_set1_ps(FLT_MAX);
          
          for (int r=0; r<GBLAS_KERNEL_SIZE; r+=4)
          {
            // update pointers
            from_r0 = &m_in[(rr + r)*ld   + cc]; // pivot
            from_r1 = from_r0 + ld;
            from_r2 = from_r1 + ld;
            from_r3 = from_r2 + ld;
            
            to_r0 = &m_out[rows*cc + rr*GBLAS_KERNEL_SIZE + r]; // pivot
            to_r1 = to_r0 + GBLAS_KERNEL_SIZE;
            to_r2 = to_r1 + GBLAS_KERNEL_SIZE;
            to_r3 = to_r2 + GBLAS_KERNEL_SIZE;
            
            for (int c=0; c<GBLAS_KERNEL_SIZE; c+=4)
            {         
              B0 = _mm_loadu_ps(from_r0);
              B1 = _mm_loadu_ps(from_r1);
              B2 = _mm_loadu_ps(from_r2);
              B3 = _mm_loadu_ps(from_r3);
              
              _MM_TRANSPOSE4_PS(B0, B1, B2, B3);
              
              // multiply by alpha
              B0 = _mm_mul_ps(B0, ALPHA);
              B1 = _mm_mul_ps(B1, ALPHA);
              B2 = _mm_mul_ps(B2, ALPHA);
              B3 = _mm_mul_ps(B3, ALPHA);
              
              // Min/Max
              _max = _mm_max_ps(_max, B0);
              _max = _mm_max_ps(_max, B1);
              _max = _mm_max_ps(_max, B2);
              _max = _mm_max_ps(_max, B3);
              
              _min = _mm_min_ps(_min, B0);
              _min = _mm_min_ps(_min, B1);
              _min = _mm_min_ps(_min, B2);
              _min = _mm_min_ps(_min, B3);
              
              // END: Min/Max
              _mm_store_ps(to_r0, B0);
              _mm_store_ps(to_r1, B1);
              _mm_store_ps(to_r2, B2);
              _mm_store_ps(to_r3, B3);
              
              // update pointers
              from_r0 += 4;
              from_r1 += 4;
              from_r2 += 4;
              from_r3 += 4;
              
              to_r0 += 4*GBLAS_KERNEL_SIZE;
              to_r1 = to_r0 + GBLAS_KERNEL_SIZE;
              to_r2 = to_r1 + GBLAS_KERNEL_SIZE;
              to_r3 = to_r2 + GBLAS_KERNEL_SIZE;
            }
          }
          //save max value!
          horiz_max(_max, m_max[a_idx]);
          horiz_min(_min, m_min[a_idx]);
          
          a_idx++;
        }
      }
    }
  }

  //print_matrix_matlab_notation(m_out, rows*((int)(cols/GBLAS_KERNEL_SIZE)), GBLAS_KERNEL_SIZE);  
}

void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const float* m_in, const float alpha, float* m_out)
{
  //print_matrix_matlab_notation(m_in, rows, ld);
    
  const int _ROWS_ = rows - (rows%GBLAS_KERNEL_SIZE);
  const int _COLS_ = cols - (cols%GBLAS_KERNEL_SIZE);
  
  if ( Trans == CblasNoTrans )
  {
    if ( alpha != 1.0f )
    {      
      for (int rr=0, idx = 0; rr < _ROWS_; rr += GBLAS_KERNEL_SIZE) // , idx = 0
      {
        for (int cc = 0; cc < _COLS_; cc += GBLAS_KERNEL_SIZE)
        {
          for (int r = 0; r<GBLAS_KERNEL_SIZE; r++)
          {
            for (int c = 0; c<GBLAS_KERNEL_SIZE; c++, idx++) // , idx+=4
            {
              m_out[idx] = alpha * m_in[(rr + r)*ld + cc + c];
            }
          }
        }
      }
    }
    else
    {     
      for (int rr = 0, idx = 0; rr < _ROWS_; rr += GBLAS_KERNEL_SIZE) // , idx = 0
      {
        for (int cc = 0; cc < _COLS_; cc += GBLAS_KERNEL_SIZE)
        {
          for (int r = 0; r<GBLAS_KERNEL_SIZE; r++)
          {
            for (int c = 0; c<GBLAS_KERNEL_SIZE; c++, idx++) // , idx+=4
            {
              m_out[idx] = m_in[(rr + r)*ld + cc + c];
            }
          }
        }
      }
    }
  }
  else // CblasTrans
  {
    // TODO: improve this code because reads matrix jumping between lines
    if ( alpha != 1.0f )
    {
      for (int cc = 0, idx = 0; cc < _COLS_; cc += GBLAS_KERNEL_SIZE)
      {    
        for (int rr = 0; rr < _ROWS_; rr += GBLAS_KERNEL_SIZE)
        {
          for (int c = 0; c < GBLAS_KERNEL_SIZE; c++)
          {   
            for (int r = 0; r < GBLAS_KERNEL_SIZE; r++, idx++)
            {
              m_out[idx] = alpha * m_in[(rr + r)*ld + cc + c];
            }
          }
        }
      }
    }
    else
    {
      for (int cc = 0, idx = 0; cc < _COLS_; cc += GBLAS_KERNEL_SIZE)
      {    
        for (int rr = 0; rr < _ROWS_; rr += GBLAS_KERNEL_SIZE)
        {
          for (int c = 0; c < GBLAS_KERNEL_SIZE; c++)
          {   
            for (int r = 0; r < GBLAS_KERNEL_SIZE; r++, idx++)
            {
              m_out[idx] = m_in[(rr + r)*ld + cc + c];
            }
          }
        }
      }
    }
  }
  
  //print_matrix_matlab_notation(m_out, rows*((int)(cols/GBLAS_KERNEL_SIZE)), GBLAS_KERNEL_SIZE);  
}

void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const double* m_in, const double alpha, double* m_out, double* m_max, double* m_min)
{
  //print_matrix_matlab_notation(m_in, rows, cols);

  if ( alpha != 1.0f )
  {
    // alpha is different than 1.0f, so a multiply is needed
    if ( Trans == CblasTrans )
    {
      double _max, _min;
      double temp;
      int b_idx = 0;
      int idx = 0;
      for (int n = 0; n < cols; n+=GBLAS_KERNEL_SIZE)
      {    
        for (int k = 0; k < rows; k+=GBLAS_KERNEL_SIZE)
        {
          _max = DBL_MIN;
          _min = DBL_MAX;
          for (int c = 0; c < GBLAS_KERNEL_SIZE; c++)
          {   
            for (int r = 0; r < GBLAS_KERNEL_SIZE; r++, idx++)
            {         
              temp = m_in[(k + r)*ld + n + c];
              
              temp = temp * alpha;
              
              _max = max(_max, temp);
              _min = min(_min, temp);
              
              m_out[idx] = temp;
            }
          }
          
          //save max value!
          m_max[b_idx] = _max;
          m_min[b_idx] = _min;
          
          b_idx++;
        }
      }
      
    }
    else
    {
      const __m128d ALPHA = _mm_set1_pd(alpha);
      __m128d _max, _min;
      __m128d temp;
      int a_idx = 0;
      
      for (int i=0, idx = 0; i < rows; i+=GBLAS_KERNEL_SIZE) // , idx = 0
      {
        for (int k=0; k < cols; k+=GBLAS_KERNEL_SIZE)
        {
          // reset counter
          _max = _mm_set1_pd(DBL_MIN);
          _min = _mm_set1_pd(DBL_MAX);
          for (int r=0; r<GBLAS_KERNEL_SIZE; r++)
          {
            for (int c=0; c<GBLAS_KERNEL_SIZE; c+=2, idx+=2)
            {
              temp = _mm_loadu_pd(&m_in[(i+r)*ld + k + c]);
              temp = _mm_mul_pd(temp, ALPHA);
              
              _mm_store_pd(&m_out[idx], temp);
              
              _max = _mm_max_pd(_max, temp);
              _min = _mm_min_pd(_min, temp);
            }
          }
          //save max value!
          horiz_max(_max, m_max[a_idx]);
          horiz_min(_min, m_min[a_idx]);
          
          a_idx++;
        }
      }
    }
  } else {
    // alpha is 1.0f, so no multiply is needed
    if ( Trans == CblasTrans )
    {
      double _max, _min;
      double temp;
      int b_idx = 0;
      int idx = 0;
      for (int n = 0; n < cols; n+=GBLAS_KERNEL_SIZE)
      {    
        for (int k = 0; k < rows; k+=GBLAS_KERNEL_SIZE)
        {
          _max = DBL_MIN;
          _min = DBL_MAX;
          for (int c = 0; c < GBLAS_KERNEL_SIZE; c++)
          {   
            for (int r = 0; r < GBLAS_KERNEL_SIZE; r++, idx++)
            {         
              temp = m_in[(k + r)*ld + n + c];
              
              _max = max(_max, temp);
              _min = min(_min, temp);
              
              m_out[idx] = temp;
            }
          }
          
          //save max value!
          m_max[b_idx] = _max;
          m_min[b_idx] = _min;
          
          b_idx++;
        }
      }   
    }
    else
    {
      __m128d _max, _min;
      __m128d temp;
      int a_idx = 0;
      
      for (int i=0, idx = 0; i < rows; i+=GBLAS_KERNEL_SIZE) // , idx = 0
      {
        for (int k=0; k < cols; k+=GBLAS_KERNEL_SIZE)
        {
          // reset counter
          _max = _mm_set1_pd(DBL_MIN);
          _min = _mm_set1_pd(DBL_MAX);
          for (int r=0; r<GBLAS_KERNEL_SIZE; r++)
          {
            for (int c=0; c<GBLAS_KERNEL_SIZE; c+=2, idx+=2)
            {
              temp = _mm_loadu_pd(&m_in[(i+r)*ld + k + c]);
              
              _mm_store_pd(&m_out[idx], temp);
              
              _max = _mm_max_pd(_max, temp);
              _min = _mm_min_pd(_min, temp);
            }
          }
          //save max value!
          horiz_max(_max, m_max[a_idx]);
          horiz_min(_min, m_min[a_idx]);
          
          a_idx++;
        }
      }
    } 
  }
  //print_matrix_matlab_notation(m_out, rows*cols/GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE);
}

void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const double* m_in, const double alpha, double* m_out)
{
  //print_matrix_matlab_notation(m_in, rows, cols);
  
  const int _ROWS_ = rows - (rows%GBLAS_KERNEL_SIZE);
  const int _COLS_ = cols - (cols%GBLAS_KERNEL_SIZE);
  
  if ( Trans == CblasTrans )
  {
    if ( alpha != 1.0f )
    {
      for (int n = 0, idx = 0; n < _COLS_; n+=GBLAS_KERNEL_SIZE)
      {    
        for (int k = 0; k < _ROWS_; k+=GBLAS_KERNEL_SIZE)
        {
          for (int c = 0; c < GBLAS_KERNEL_SIZE; c++)
          {   
            for (int r = 0; r < GBLAS_KERNEL_SIZE; r++, idx++)
            {
              m_out[idx] = alpha * m_in[(k + r)*ld + n + c];
            }
          }
        }
      }
    }
    else
    {      
      for (int n = 0, idx = 0; n < _COLS_; n+=GBLAS_KERNEL_SIZE)
      {    
        for (int k = 0; k < _ROWS_; k+=GBLAS_KERNEL_SIZE)
        {
          for (int c = 0; c < GBLAS_KERNEL_SIZE; c++)
          {   
            for (int r = 0; r < GBLAS_KERNEL_SIZE; r++, idx++)
            {
              m_out[idx] = m_in[(k + r)*ld + n + c];
            }
          }
        }
      }
    }    
  }
  else
  {
    if ( alpha != 1.0f )
    {
      const __m128d ALPHA = _mm_set1_pd(alpha);
      for (int i=0, idx = 0; i < _ROWS_; i+=GBLAS_KERNEL_SIZE)
      {
        for (int k=0; k < _COLS_; k+=GBLAS_KERNEL_SIZE)
        {
          for (int r=0; r<GBLAS_KERNEL_SIZE; r++)
          {
            for (int c=0; c<GBLAS_KERNEL_SIZE; c+=2, idx+=2)
            {
              _mm_store_pd(&m_out[idx],
                           _mm_mul_pd(_mm_loadu_pd(&m_in[(i+r)*ld + k + c]),
                                      ALPHA));
            }
          }
        }
      }
    }
    else
    {
      for (int i=0, idx = 0; i < _ROWS_; i+=GBLAS_KERNEL_SIZE)
      {
        for (int k=0; k < _COLS_; k+=GBLAS_KERNEL_SIZE)
        {
          for (int r=0; r<GBLAS_KERNEL_SIZE; r++)
          {
            for (int c=0; c<GBLAS_KERNEL_SIZE; c+=2, idx+=2)
            {
              _mm_store_pd(&m_out[idx],
                           _mm_loadu_pd(&m_in[(i+r)*ld + k + c]));
            }
          }
        }
      }
    }
  }
  
  //print_matrix_matlab_notation(m_out, rows*cols/GBLAS_KERNEL_SIZE, GBLAS_KERNEL_SIZE);
}
