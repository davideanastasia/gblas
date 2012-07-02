/*
 *  perf_tests.h
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __PERF_TESTS_H__
#define __PERF_TESTS_H__

//void test2_icassp2011_sgemm_NBxNB_block_major_format_v2_2DPCA();

void test_dgemm_companders();
void test_sgemm_companders(float db);

void check_sGEMM_correctness_against_Matlab();
void check_sGEMM_correctness_against_Goto();
void check_dGEMM_correctness_against_Goto();
void check_dGEMM();

void exp_paper_mse_vs_gflops_sgemm();
void exp_paper_mse_vs_gflops_dgemm();

void exp_paper_accel_vs_gflops_sgemm();
void exp_paper_accel_vs_gflops_sgemm_with_EC();
void exp_paper_accel_vs_gflops_dgemm();

// remove this rubbish afterwards
void test_unpack_complete_tight_v1();
void test_unpack_complete_tight_v2();
void test_unpack_complete_tight_v3();

void test_pack_unpack_v1();

#endif
