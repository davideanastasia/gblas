/*
 *  gblas_stat_model.cpp
 *  gblas
 *
 *  Created by Davide Anastasia on 15/12/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <iostream>

#include "gblas_stat_model.h"

//#define CTRL_CSV

inline int GET_IDX(int* x, int i)
{
  return (x[i] - 1);
}

// check why there is a systematic difference of 0.8dB between the model and the actual value
double control_pack_packing_based_s(float max_a, float max_b, float& Ca, float& Cb, int& num_pack)
{
  float std_A = (2.0f*(float)max_a)/SQRT_12;
  float std_B = (2.0f*(float)max_b)/SQRT_12;
  
  //float max_std_A = (2.0f*(float)MAX_VALUE_TEST)/SQRT_12; //max_a
  //float max_std_B = (2.0f*(float)MAX_VALUE_TEST)/SQRT_12; //max_b
    
  //float formula_R_power = (float)(KERNEL_SIZE)*(max_std_A*max_std_A*max_std_B*max_std_B);
  //float formula_R_power = (float)(GBLAS_KERNEL_SIZE)*(max_std_A*max_std_A*max_std_B*max_std_B);
  
  //float X1 = (float(A_RANGE_FLOAT)*INV_KERNEL_SIZE)/(max_a*max_b);
  
  num_pack = DEFAULT_FLOAT_PACKING;
  
#ifdef FIXED_COMPOUNDERS
  Ca = Cb = 1.0f;
#else
  
#ifdef FLOAT_SYM
  const float A_RANGE_S = (float)(A_RANGE_FLOAT_P2_SYM);
  const float NUM_ERR_VAR_S = (float)NUM_REPR_ERR_FLOAT_P2_SYM;
#else
  const float A_RANGE_S = (float)(A_RANGE_FLOAT_P2_ASYM);
  const float NUM_ERR_VAR_S = (float)NUM_REPR_ERR_FLOAT_P2_ASYM;
#endif
  
    Ca = 1.0f/max_a * (float)sqrt(A_RANGE_S/(2.0f*GBLAS_KERNEL_SIZE));
    Cb = 1.0f/max_b * (float)sqrt(A_RANGE_S/(2.0f*GBLAS_KERNEL_SIZE));

  //Ca = Cb = 1.0f; //.4f;
#endif

  
  //float P_e = KERNEL_SIZE*((1+24*X1*sigma_a*sigma_b)/(144*X1*X1));
  
  //static const float N_POW3 = GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE*GBLAS_KERNEL_SIZE;
  //static const float A_RANGE_FLOAT_POW2 = (A_RANGE_FLOAT*A_RANGE_FLOAT);
  
  //float P_e_t = (N_POW3 * (max_a*max_a*max_b*max_b) * INV_144)/A_RANGE_FLOAT_POW2;
  
  //float X2 = (sigma_a*sigma_a*INV_12)/(X1*X1);
  //float X3 = (sigma_b*sigma_b)*INV_12;
  //float X4 = ((1.0f/(X1*X1*144)) - ((float)P_e*(float)INV_KERNEL_SIZE));
  
  float std_qnoise_A = (1.0f/(Ca*SQRT_12));
  float std_qnoise_B = (1.0f/(Cb*SQRT_12));
  
#ifndef FIXED_COMPOUNDERS
  float formula_noise_power = (float)(GBLAS_KERNEL_SIZE)*((std_qnoise_A*std_qnoise_A*std_qnoise_B*std_qnoise_B) + (std_B*std_B*std_qnoise_A*std_qnoise_A) + (std_A*std_A*std_qnoise_B*std_qnoise_B)) + ((1.0f/(Ca*Ca*Cb*Cb))*NUM_ERR_VAR_S);
#else
  float formula_noise_power = (float)(GBLAS_KERNEL_SIZE)*((std_qnoise_A*std_qnoise_A*std_qnoise_B*std_qnoise_B) + (std_B*std_B*std_qnoise_A*std_qnoise_A) + (std_A*std_A*std_qnoise_B*std_qnoise_B));
#endif
  
  //float formula_PSNR = 10*log10(formula_R_power/P_e);
  //float diff_formula_power_percent = abs(100*((ERROR_POWER - formula_power)/ERROR_POWER));
  
#ifdef CTRL_CSV   
  cout << P_e << ", " << formula_R_power << ", " << std_A << ", " << std_B << ", " << std_qnoise_A << ", " << std_qnoise_B << ", " << formula_noise_power << ", " << formula_PSNR << ", " << Ca << ", " << Cb << ", ";
#endif
  
  //cout << formula_noise_power << endl;
  
  return formula_noise_power;
}

#ifdef EXTERNAL_A_QUANT
float control_pack_packing_based_s_external_Aquant(float max_a, float max_b, float& Ca, float& Cb, int& num_pack, long Aquant)
{
  float std_A = (2.0f*(float)max_a)/SQRT_12;
  float std_B = (2.0f*(float)max_b)/SQRT_12;
  
  num_pack = DEFAULT_FLOAT_PACKING;
  
#ifdef FLOAT_SYM
  //const float NUM_ERR_VAR_S = (float)NUM_REPR_ERR_FLOAT_P2_SYM;
#else
  //const float NUM_ERR_VAR_S = (float)NUM_REPR_ERR_FLOAT_P2_ASYM;
#endif
  
  const float A_QUANT_S = (float)(Aquant);
  
  Ca = 1.0f/max_a * (float)sqrt((float)(A_QUANT_S)/(2.0f*GBLAS_KERNEL_SIZE));
  Cb = 1.0f/max_b * (float)sqrt((float)(A_QUANT_S)/(2.0f*GBLAS_KERNEL_SIZE));
  
  
  float std_qnoise_A = (1.0f/(Ca*SQRT_12));
  float std_qnoise_B = (1.0f/(Cb*SQRT_12));
  
  float formula_noise_power = (float)(GBLAS_KERNEL_SIZE)*((std_qnoise_A*std_qnoise_A*std_qnoise_B*std_qnoise_B) + (std_B*std_B*std_qnoise_A*std_qnoise_A) + (std_A*std_A*std_qnoise_B*std_qnoise_B));
  // + ((1.0f/(Ca*Ca*Cb*Cb))*NUM_ERR_VAR_S);
  
#ifdef CTRL_CSV   
  cout << P_e << ", " << formula_R_power << ", " << std_A << ", " << std_B << ", " << std_qnoise_A << ", " << std_qnoise_B << ", " << formula_noise_power << ", " << formula_PSNR << ", " << Ca << ", " << Cb << ", ";
#endif
  
  return formula_noise_power;
}
#endif

void select_accelerated_blocks_by_quality_s(int num_blocks, double* P_e, double& cum_P_e, int* Ctrl, const double mse_user)
{
//  const float max_std_A = (2.0f*(float)MAX_VALUE_TEST)/SQRT_12;
//  const float max_std_B = (2.0f*(float)MAX_VALUE_TEST)/SQRT_12;
//  
//  const float formula_R_power = (float)(GBLAS_KERNEL_SIZE)*(max_std_A*max_std_A*max_std_B*max_std_B);
//  const float __P_e = formula_R_power / (powf(10.0f, (psnr_user/10.0f)));
  
  //cout << cum_P_e;
  
  int iter = num_blocks;
  while (iter > 0 && cum_P_e > mse_user)
  {
    // select_max
    int idx_max = 0;
    for (int idx = 0; idx < num_blocks; idx++)
    {
      if ( P_e[idx] > P_e[idx_max] ) idx_max = idx;
    }
    
    cum_P_e         -=  P_e[idx_max];   // subtract from cum_P_e
    Ctrl[idx_max]   =   1;              // decrease packing
    P_e[idx_max]    =   0.0f;
    
    iter--;
    
    //cout << " " << cum_P_e << " " ;
  }
  
  //cout << cum_P_e << ", ";
  //cout << endl;
}

void select_accelerated_blocks_by_throughput_s(int num_blocks, double* P_e, double& cum_P_e, int* Ctrl, double perc_acc)
{
  if ( perc_acc >= 100.0 )
  {
    //perc_acc = 100.0;
    return;
  }
  if ( perc_acc <= 0.0 )
  {
    //perc_acc = 0.0;
    cum_P_e = 0.0;
    for (int idx = 0; idx < num_blocks; idx++) Ctrl[idx] = 1; 
    return;
  }
  
  int iter = (int)ceil(((float)num_blocks*(100.0-perc_acc))/100.0);
  while ( iter > 0 )
  {
    // select_max
    int idx_max = 0;
    for (int idx = 0; idx < num_blocks; idx++)
    {
      if ( P_e[idx] > P_e[idx_max] ) idx_max = idx;
    }
    
    cum_P_e         -=  P_e[idx_max];   // subtract from cum_P_e    
    Ctrl[idx_max]   =   1;        // decrease packing
    P_e[idx_max]    =   0.0;
    
    iter--;
  }
}

void control_pack_packing_based_d(double max_a, double max_b, double* Ca, double* Cb, double* P_e, int& num_pack)
{
  double num_err[NUM_PACKING_DOUBLE];
  
  double std_A = (2.0*max_a)/SQRT_12;
  double std_B = (2.0*max_b)/SQRT_12;
  
  //double max_std_A = (2.0*(double)MAX_VALUE_TEST)/SQRT_12; //max_a
  //double max_std_B = (2.0*(double)MAX_VALUE_TEST)/SQRT_12; //max_b
  
  //double formula_R_power = (double)(KERNEL_SIZE)*(max_std_A*max_std_A*max_std_B*max_std_B);
  num_pack = DEFAULT_DOUBLE_PACKING;
  
  Ca[0] = 1.0;
  Cb[0] = 1.0;
  
#ifdef FIXED_COMPOUNDERS
  Ca[1] = Cb[1] = Ca[2] = Cb[2] = Ca[3] = Cb[3] = 1.0;
#else
  
#ifdef DOUBLE_SYM
  const double A_RANGE_D_P2 = (double)(A_RANGE_DOUBLE_P2_SYM);
  const double A_RANGE_D_P3 = (double)(A_RANGE_DOUBLE_P3_SYM);
  const double A_RANGE_D_P4 = (double)(A_RANGE_DOUBLE_P4_SYM);
#else
  const double A_RANGE_D_P2 = (double)(A_RANGE_DOUBLE_P2_ASYM);
  const double A_RANGE_D_P3 = (double)(A_RANGE_DOUBLE_P3_ASYM);
  const double A_RANGE_D_P4 = (double)(A_RANGE_DOUBLE_P4_ASYM);
#endif
  
  Ca[1] = 1.0/max_a * sqrt(A_RANGE_D_P2/(2.0*GBLAS_KERNEL_SIZE));
  Cb[1] = 1.0/max_b * sqrt(A_RANGE_D_P2/(2.0*GBLAS_KERNEL_SIZE));
  
  Ca[2] = 1.0/max_a * sqrt(A_RANGE_D_P3/(2.0*GBLAS_KERNEL_SIZE));
  Cb[2] = 1.0/max_b * sqrt(A_RANGE_D_P3/(2.0*GBLAS_KERNEL_SIZE));
  
  Ca[3] = 1.0/max_a * sqrt(A_RANGE_D_P4/(2.0*GBLAS_KERNEL_SIZE));
  Cb[3] = 1.0/max_b * sqrt(A_RANGE_D_P4/(2.0*GBLAS_KERNEL_SIZE));
#endif
  
  P_e[0] = 0.0;
  num_err[0] = 0.0;
  
#ifdef DOUBLE_SYM
  num_err[1] = (1.0/(Ca[1]*Ca[1]*Cb[1]*Cb[1]))*(double)(NUM_REPR_ERR_DOUBLE_P2_SYM);
  num_err[2] = (1.0/(Ca[2]*Ca[2]*Cb[2]*Cb[2]))*(double)(NUM_REPR_ERR_DOUBLE_P3_SYM);
  num_err[3] = (1.0/(Ca[3]*Ca[3]*Cb[3]*Cb[3]))*(double)(NUM_REPR_ERR_DOUBLE_P4_SYM);
#else
  num_err[1] = (1.0/(Ca[1]*Ca[1]*Cb[1]*Cb[1]))*(double)(NUM_REPR_ERR_DOUBLE_P2_ASYM);
  num_err[2] = (1.0/(Ca[2]*Ca[2]*Cb[2]*Cb[2]))*(double)(NUM_REPR_ERR_DOUBLE_P3_ASYM);
  num_err[3] = (1.0/(Ca[3]*Ca[3]*Cb[3]*Cb[3]))*(double)(NUM_REPR_ERR_DOUBLE_P4_ASYM);  
#endif
  
  for (int IDX = 1; IDX < NUM_PACKING_DOUBLE; IDX++)
  {
    double std_qnoise_A = (1.0/(Ca[IDX]*SQRT_12));
    double std_qnoise_B = (1.0/(Cb[IDX]*SQRT_12));
    
    P_e[IDX] = (double)(GBLAS_KERNEL_SIZE)*((std_qnoise_A*std_qnoise_A*std_qnoise_B*std_qnoise_B) + (std_B*std_B*std_qnoise_A*std_qnoise_A) + (std_A*std_A*std_qnoise_B*std_qnoise_B)) + num_err[IDX];
    
    //cout << "P_e[" << IDX << "] = " << P_e[IDX] << " " << Ca[IDX] << " " << Cb[IDX] << endl;
  }
}

void select_accelerated_blocks_by_quality_d(int num_blocks, double* P_e, int* Ctrl, const double mse_user, int K)
{
//  const double max_std_A = (2.0*(double)MAX_VALUE_TEST)/SQRT_12;
//  const double max_std_B = (2.0*(double)MAX_VALUE_TEST)/SQRT_12;
//  
//  const double formula_R_power = (double)(K)*(max_std_A*max_std_A*max_std_B*max_std_B);
  
  //const double __P_e = mse_user;// formula_R_power / (pow(10.0, (psnr_user/10.0)));
  
  if ( mse_user == 0.0 )
  {
    for (int b = 0; b < num_blocks; b++) Ctrl[b] = 1;
    //std::cout << 0.0 << ", ";
    return;
  }
  
  // P_e[0], P_e[1], P_e[2], P_e[3]
  
  // calculates the cumulative error based on the current packing for each block  
  double cum_P_e = 0.0;
  for (int i = 0; i < num_blocks; i++)
  {
    cum_P_e += P_e[i*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, i)];
  }
  
  //cout << " " << cum_P_e << " " ;
  
  while ( cum_P_e >= mse_user )
  {
    // select starting point
    int idx = 0;
    int idx_max = 0;
    double P_e_max = 0.0;
    
    // find max
    for (idx = 0; idx < num_blocks; idx++)
    {
      if ( P_e[idx*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx)] > P_e_max )
      {
        idx_max = idx;
        P_e_max = P_e[idx*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx)];
      }
    }
    
    cum_P_e -= P_e_max;
    
    P_e[idx_max*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx_max)] = 0.0; // reset error!
    
#ifdef DOUBLE_SYM
    // skipping packing 3 because it is too slow!
    if ( Ctrl[idx_max] == 4 ) { Ctrl[idx_max] = 2; } else { Ctrl[idx_max]--; } // decrease packing
#else
    Ctrl[idx_max]--;
#endif
    
    cum_P_e += P_e[idx_max*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx_max)];
    
    //cout << " " << cum_P_e << " " ;
  }
  
  //cout << cum_P_e << ", ";
  //cout << endl;
}



void select_accelerated_blocks_by_throughput_d(int num_blocks, double* P_e, int* Ctrl, double perc_acc)
{
  // it shouldn't be necessary because it is already made by control_pack_packing_based_s/d
  // NOTE: memset works ONLY with char!!!
  // memset(Ctrl, DEFAULT_DOUBLE_PACKING, num_blocks*sizeof(int));
  
//  if ( perc_acc >= 100.0f )
//  {
//    //perc_acc = 100.0f;
//    return;
//  }
  //TODO: enable them!!
//  if ( perc_acc <= 0.0f )
//  {
//    //perc_acc = 0.0f;
//    for (int b = 0; b < num_blocks; ++b) Ctrl[b] = 1;
//    return;
//  }
  
#ifdef DOUBLE_SYM
  int iter = (int)ceil(((float)(num_blocks*(NUM_PACKING_DOUBLE-2))*(100.0f-perc_acc))/100.0f);
#else
  int iter = (int)ceil(((float)(num_blocks*(NUM_PACKING_DOUBLE-1))*(100.0f-perc_acc))/100.0f);
#endif
  
  // PRINT PRUNING -------------------
//  for (int o = 0; o < num_blocks; o++)
//  {
//    std::cout << "["<< P_e[o*4 + (Ctrl[o]-1)] << "] ";
//  }
//  std::cout << std::endl;
  // END PRINT PRUNING ---------------
  
  while ( iter > 0 )
  {
    // select starting point
    int idx = 0;
    int idx_max = 0;
    double P_e_max = 0.0;
    
    // find max
    for (idx = 0; idx < num_blocks; idx++)
    {
      if ( P_e[idx*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx)] > P_e_max )
      {
        idx_max = idx;
        P_e_max = P_e[idx*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx)];
      }
    }
    
    P_e[idx_max*NUM_PACKING_DOUBLE + GET_IDX(Ctrl, idx_max)] = 0.0; // reset error!
#ifdef DOUBLE_SYM
    // skipping packing 3 because it is too slow!
    if ( Ctrl[idx_max] == 4 ) { Ctrl[idx_max] = 2; } else { Ctrl[idx_max]--; } // decrease packing
#else
    Ctrl[idx_max]--;
#endif
    iter--;
    
    // PRINT PRUNING -------------------
//    for (int o = 0; o < num_blocks; o++)
//    {
//      std::cout << "["<< P_e[o*4 + (Ctrl[o]-1)] << "] ";
//    }
//    std::cout << std::endl;
    // END PRINT PRUNING ---------------
  }
}

#ifdef EXTERNAL_A_QUANT
void control_pack_packing_based_d_external_Aquant(double max_a, double max_b, double* Ca, double* Cb, double* P_e, int& num_pack, long Aquant)
{
  double std_A = (2.0*max_a)/SQRT_12;
  double std_B = (2.0*max_b)/SQRT_12;
  
  num_pack = DEFAULT_DOUBLE_PACKING;
  
  Ca[0] = 1.0;
  Cb[0] = 1.0;
  
  const double A_QUANT_D = (double)(Aquant);
    
  Ca[1] = 1.0/max_a * sqrt(A_QUANT_D/(2.0*GBLAS_KERNEL_SIZE));
  Cb[1] = 1.0/max_b * sqrt(A_QUANT_D/(2.0*GBLAS_KERNEL_SIZE));
  
  Ca[2] = 1.0/max_a * sqrt(A_QUANT_D/(2.0*GBLAS_KERNEL_SIZE));
  Cb[2] = 1.0/max_b * sqrt(A_QUANT_D/(2.0*GBLAS_KERNEL_SIZE));
  
  Ca[3] = 1.0/max_a * sqrt(A_QUANT_D/(2.0*GBLAS_KERNEL_SIZE));
  Cb[3] = 1.0/max_b * sqrt(A_QUANT_D/(2.0*GBLAS_KERNEL_SIZE));

  
  P_e[0] = 0.0;
  
  for (int IDX = 1; IDX < NUM_PACKING_DOUBLE; IDX++)
  {    
    double std_qnoise_A = (1.0/(Ca[IDX]*SQRT_12));
    double std_qnoise_B = (1.0/(Cb[IDX]*SQRT_12));
    
    P_e[IDX] = (double)(GBLAS_KERNEL_SIZE)*((std_qnoise_A*std_qnoise_A*std_qnoise_B*std_qnoise_B) + (std_B*std_B*std_qnoise_A*std_qnoise_A) + (std_A*std_A*std_qnoise_B*std_qnoise_B));
  }
}
#endif
