/*
 *  gblas_stat_model.h
 *  gblas
 *
 *  Created by Davide Anastasia on 15/12/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __GBLAS_STAT_MODEL_H__
#define __GBLAS_STAT_MODEL_H__

#include "gblas.h"

#ifndef GBLAS_KERNEL_SIZE
#error GBLAS_KERNEL_SIZE is not defined
#endif

#define     DEFAULT_FLOAT_PACKING         (2)
#define     DEFAULT_DOUBLE_PACKING        (2)
#define     NUM_PACKING_DOUBLE            (4)             // DO NOT CHANGE!


#if GBLAS_KERNEL_SIZE == 288
#define     A_RANGE_FLOAT_P2_SYM          (260000)          //(120000)          //(2*3500)
#define     A_RANGE_FLOAT_P2_ASYM         (133000)          //(100000)          //(2*1950)

#define     NUM_REPR_ERR_FLOAT_P2_SYM     (3400)            //(6850)
#define     NUM_REPR_ERR_FLOAT_P2_ASYM    (1570)            //(2416)

#endif

#if GBLAS_KERNEL_SIZE <= 144 //== 144
#define     A_RANGE_FLOAT_P2_SYM          (260000)          //(120000)          //(2*3500)
#define     A_RANGE_FLOAT_P2_ASYM         (135000)          //(100000)          //(2*1950)

#define     NUM_REPR_ERR_FLOAT_P2_SYM     (4260)            //(9050)
#define     NUM_REPR_ERR_FLOAT_P2_ASYM    (1910)            //(2810)
#endif

#if GBLAS_KERNEL_SIZE == 288
#define     A_RANGE_DOUBLE_P2_SYM         (2000000000)    //(2*83000000)
#define     A_RANGE_DOUBLE_P3_SYM         (2800000)       //(1650000)
#define     A_RANGE_DOUBLE_P4_SYM         (31000)

#define     NUM_REPR_ERR_DOUBLE_P4_SYM    (53)            //(55)  //(129)
#define     NUM_REPR_ERR_DOUBLE_P3_SYM    (15000)         //(17000)         //(34100)
#define     NUM_REPR_ERR_DOUBLE_P2_SYM    (400000)        // experimental value, otherwise zero!

#define     A_RANGE_DOUBLE_P2_ASYM        (2000000000)    //(2000000000)  //(2*4130000) //(46000000)
#define     A_RANGE_DOUBLE_P3_ASYM        (2450000)       //(1350000)     //(80000)       //(160000)
#define     A_RANGE_DOUBLE_P4_ASYM        (31000)         //(16000)       //(10500)

#define     NUM_REPR_ERR_DOUBLE_P4_ASYM   (195)           //(340)           //(190)           //(369)
#define     NUM_REPR_ERR_DOUBLE_P3_ASYM   (18825)         //(15500)         //(18750)         //(33000)
#define     NUM_REPR_ERR_DOUBLE_P2_ASYM   (150000)        // experimental value, otherwise zero!
#endif

#if GBLAS_KERNEL_SIZE <= 144 //== 144
#define     A_RANGE_DOUBLE_P2_SYM         (2000000000)    //(2*83000000)
#define     A_RANGE_DOUBLE_P3_SYM         (2800000)       //(1650000)
#define     A_RANGE_DOUBLE_P4_SYM         (30000)         //(19000)

#define     NUM_REPR_ERR_DOUBLE_P4_SYM    (57)            //(120)            //(153)
#define     NUM_REPR_ERR_DOUBLE_P3_SYM    (20900)         //(15500)         //(34300)
#define     NUM_REPR_ERR_DOUBLE_P2_SYM    (0)

#define     A_RANGE_DOUBLE_P2_ASYM        (2000000000)    //(2000000000)  //(2*4130000) //(46000000)
#define     A_RANGE_DOUBLE_P3_ASYM        (2350000)       //(1350000)     //(80000)       //(160000)
#define     A_RANGE_DOUBLE_P4_ASYM        (30500)         //(16000)       //(10500)

#define     NUM_REPR_ERR_DOUBLE_P4_ASYM   (205)           //(200)           //(415)
#define     NUM_REPR_ERR_DOUBLE_P3_ASYM   (15400)         //(16000)         //(27400)
#define     NUM_REPR_ERR_DOUBLE_P2_ASYM   (0)
#endif

// constants
#define     SQRT_12                       (3.464101615137754f)
#define     INV_SQRT_12                   (0.288675134594813f)
#define     INV_12                        (0.083333333333333f)
#define     INV_144                       (0.006944444444444f)


/* Packing control functions */

void select_accelerated_blocks_by_quality_s(int num_blocks, double* P_e, double& cum_P_e, int* Ctrl, const double mse_user);
void select_accelerated_blocks_by_throughput_s(int num_blocks, double* P_e, double& cum_P_e, int* Ctrl, double perc_acc);

void select_accelerated_blocks_by_quality_d(int num_blocks, double* P_e, int* Ctrl, double psnr_user, int K);
void select_accelerated_blocks_by_throughput_d(int num_blocks, double* P_e, int* Ctrl, double perc_acc);

double control_pack_packing_based_s(float max_a, float max_b, float& Ca, float& Cb, int& num_pack);
void control_pack_packing_based_d(double max_a, double max_b, double* Ca, double* Cb, double* P_e, int& num_pack);

#ifdef EXTERNAL_A_QUANT
void control_pack_packing_based_d_external_Aquant(double max_a, double max_b, double* Ca, double* Cb, double* P_e, int& num_pack, long Aquant);
float control_pack_packing_based_s_external_Aquant(float max_a, float max_b, float& Ca, float& Cb, int& num_pack, long Aquant);
#endif

#endif
