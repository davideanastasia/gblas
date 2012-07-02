/*
 *  gblas_quantizer.h
 *  gblas
 *
 *  Created by Davide Anastasia on 17/09/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __GBLAS_QUANTIZER_H__
#define __GBLAS_QUANTIZER_H__

#include "gblas_sse_utils.h"

//#define     NOISE_FLOOR_DOUBLE            (3.1193e-011)     // (2^-34.9) //2.91038e-011 (2^-35); // double
//#define     NOISE_FLOOR_FLOAT             (1.13576e-07f)    // float
#define     DELTA                         (50)

class gblas_quantizer
{
private:
  double       max_value;
  //__m128    max_value_s;
  
  double       min_value;
  //__m128    min_value_s;

  double    q_step;
  __m128    q_step_s;
  __m128d   q_step_d;
  
public:
  gblas_quantizer();
  ~gblas_quantizer();
  
  void set_quantization_levels(int levels);
  int get_quantization_levels();
  void set_max_value(double max_v);
  void set_min_value(double min_v);
  void update_quantizer();
  
  double get_quantizer_step();
  
  inline int quantize_sample(const float input)
  {
    if ( input == 0.0f ) return 0;
    if ( input > 0.0f )  return (int)(input*q_step + ZERO_DOT_FIVE);
    return (int)(input*q_step - ZERO_DOT_FIVE);
  }
  
  inline int quantize_sample(const double input)
  {
    if ( input == 0.0 ) return 0;
    if ( input > 0.0 )  return (int)(input*q_step + ZERO_DOT_FIVE);
    return (int)(input*q_step - ZERO_DOT_FIVE);
  }
  
  inline __m128 quantize_sample(__m128 input)
  {
    __m128 rd;
    rd = _mm_cmpgt_ps(input, _MM_ZERO_S);
    rd = _mm_and_ps(rd, _MM_MASK_ONE_S);
    rd = _mm_sub_ps(rd, _MM_ZERO_DOT_FIVE_S);
    
    input = _mm_mul_ps(input, q_step_s);
    input = _mm_add_ps(input, rd);
    
    return _mm_cvtepi32_ps(_mm_cvttps_epi32(input));
  }

  inline __m128d quantize_sample(__m128d input)
  {
    __m128d rd;
    rd = _mm_cmpgt_pd(input, _MM_ZERO_D);
    rd = _mm_and_pd(rd, _MM_MASK_ONE_D);
    rd = _mm_sub_pd(rd, _MM_ZERO_DOT_FIVE_D);
    
    input = _mm_mul_pd(input, q_step_d);
    input = _mm_add_pd(input, rd);
    
    return _mm_cvtepi32_pd(_mm_cvttpd_epi32(input));
  }
  
  
  static int dequantize_sample(double* input, double q_factor)
  {
    return (int)((*input)*q_factor);
  }
  
  static __m128d dequantize_sample(const __m128d& input, const __m128d& q_factor)
  {
    return _mm_mul_pd(input, q_factor);
  }
  
  static __m128 dequantize_sample(const __m128& input, const __m128& q_factor)
  {
    return _mm_mul_ps(input, q_factor);
  }
  
  void quantization(const double* input, double* output, int rows, int cols);
  void quantization(const float* input, float* output, int rows, int cols);
  void dequantization(const double* input, double* output, int rows, int cols, double q_factor);
  
  double get_max_value();
  double get_min_value();
  
  void set_q_step(double step)
  {
    q_step = step;
    
    q_step_s = _mm_set1_ps((float)q_step);
    q_step_d = _mm_set1_pd(q_step);
  }
};

#endif