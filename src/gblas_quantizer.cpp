/*
 *  gblas_quantizer.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 17/09/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <iostream>

#include "gblas_quantizer.h"

gblas_quantizer::gblas_quantizer():
max_value(255),
//max_value_s(_mm_set1_ps(255.0f)),
min_value(0),
//min_value_s(_mm_set1_ps(0.0f)),
//quantization_levels(255),
q_step(1.0),
q_step_s(_mm_set1_ps(1.0f)),
q_step_d(_mm_set1_pd(1.0))
{
  //min_value = 0;
  //max_value = 255;
  //quantization_levels = 255;
  //q_step    = 1.0;
  
  //inv_q_step_s  = _mm_set1_ps(1.0f);
  //inv_q_step_d  = _mm_set1_pd(1.0);
}

gblas_quantizer::~gblas_quantizer()
{

}

double gblas_quantizer::get_max_value()
{
  return max_value;
}

double gblas_quantizer::get_min_value()
{
  return min_value;
}

void gblas_quantizer::quantization(const double* input, double* output, int rows, int cols)
{
  std::cerr << "Deprecated method: gblas_quantizer::quantization()" << std::endl;
  exit(0);
  
  //  for (int i=0; i < rows; i++)
  //  {
  //    for (int j=0; j < cols; j++)
  //    {
  //      output[i*cols + j] = quantize_sample(&input[i*cols + j]);
  //    }
  //  }
  
  //  for (int i=0; i < rows*cols; i++)
  //  {
  //    output[i] = (int)(input[i]/gblas_status.q_step + ZERO_DOT_FIVE); //quantize_sample(&input[i]);
  //  }
  __m128d curr;
  __m128d inv_q_step  = _mm_div_pd(_mm_set1_pd(1.0), _mm_set1_pd(q_step));
  const double* in_p  = input;
  double* out_p = output;
  
  for (int i=((rows*cols) >> 1); i > 0; i--)
  {
    curr = _mm_load_pd(in_p); in_p += 2;
    curr = _mm_mul_pd(curr, inv_q_step);    
    curr = _mm_add_pd(curr, _MM_ZERO_DOT_FIVE_D);
    curr = _mm_cvtepi32_pd(_mm_cvttpd_epi32(curr));
    _mm_store_pd(out_p, curr);  out_p += 2;
  }
}

void gblas_quantizer::quantization(const float* input, float* output, int rows, int cols)
{
  std::cerr << "Deprecated method: gblas_quantizer::quantization()" << std::endl;
  exit(0);
  
//  __m128 curr;
//  __m128 inv_q_step  = _mm_div_ps(_mm_set1_ps(1.0f), _mm_set1_ps((float)q_step));
//  const float* in_p  = input;
//  float* out_p = output;
//  
//  for (int i=((rows*cols) >> 2); i > 0; i--)
//  {
//    curr = _mm_load_ps(in_p); in_p += 4;
//    curr = _mm_mul_ps(curr, inv_q_step);    
//    curr = _mm_add_ps(curr, _MM_ZERO_DOT_FIVE_S);
//    curr = _mm_cvtepi32_ps(_mm_cvttps_epi32(curr));
//    _mm_store_ps(out_p, curr);  out_p += 4;
//  }
}

void gblas_quantizer::dequantization(const double* input, double* output, int rows, int cols, double q_factor)
{
  std::cerr << "Deprecated method: gblas_quantizer::quantization()" << std::endl;
  exit(0);
  
//  int i, j;
//  for (i=0; i < rows; i++)
//  {
//    for (j=0; j < cols; j++)
//    {
//      int idx = row_col_2_lin(i, j, cols);
//      
//      // quantization
//      output[idx] = dequantize_sample(&input[idx], q_factor);
//      //output[idx] = (int)(input[idx]*q_factor);
//    }
//  }
}

void gblas_quantizer::set_quantization_levels(int levels)
{
  std::cerr << "Deprecated method: gblas_quantizer::quantization()" << std::endl;
  exit(0);
//  quantization_levels = levels;
//  update_quantizer();
}

int gblas_quantizer::get_quantization_levels()
{
  std::cerr << "Deprecated method: gblas_quantizer::quantization()" << std::endl;
  exit(0);
  
//  return quantization_levels;
}

void gblas_quantizer::update_quantizer()
{
  //double t_q_step;
  //  int spread = (max_value-min_value);
  //if (max_value == min_value) min_value = min_value - 10;
  //  {
  //    q_step = max_value;
  //  }
  //  else
  //  {
  //t_q_step = ((double)(max_value-min_value))/quantization_levels;
  //  }
  //set_q_step(t_q_step);
}

void gblas_quantizer::set_max_value(double max_v)
{
  max_value = max_v;

  //update_quantizer();
}

void gblas_quantizer::set_min_value(double min_v)
{
  min_value = min_v;

  //update_quantizer();
}

double gblas_quantizer::get_quantizer_step()
{
  return q_step;
}
