/*
 *  main.cpp
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <cstdlib>

#include "gblas.h"
#include "perf_tests.h"
#include "high_priority_process.h"

int main(int argc, char *argv[])
{
  start_high_priority();
   
  // CORRECTNESS
  //check_sGEMM_correctness_against_Goto();
  //check_dGEMM_correctness_against_Goto();
  //check_sGEMM_correctness_against_Matlab();

  
  // PERFORMANCE
  //test2_icassp2011_sgemm_NBxNB_block_major_format_v2_2DPCA();
  
  //exp_paper_accel_vs_gflops_sgemm();
  //exp_paper_accel_vs_gflops_dgemm();
  exp_paper_accel_vs_gflops_sgemm_with_EC();
  
  //test_unpack_complete_tight_v1();
  //test_unpack_complete_tight_v2();
  //test_unpack_complete_tight_v3();
  //test_pack_unpack_v1();
  
  exit_high_priority();
}
