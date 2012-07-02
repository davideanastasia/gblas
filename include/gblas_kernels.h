/*
 *  gblas_kernels.h
 *  gblas
 *
 *  Created by Davide Anastasia on 28/07/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __GBLAS_KERNELS_H__
#define __GBLAS_KERNELS_H__

#include "gblas.h" // get the value of GBLAS_KERNEL_SIZE
#include "gblas_quantizer.h"

/*
 * Single-Precision Standard Kernel(s)
 */
void KERNEL_std_sgemm_v6(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         const float beta, float *C, const int ldc);

void KERNEL_std_sgemm_v6_BX(const int M, const int N,
                            const int K, const float alpha, const float *A,
                            const int lda, const float *B, const int ldb,
                            const float beta, float *C, const int ldc);

void KERNEL_std_sgemm_v6_B1(const int M, const int N,
                            const int K, const float alpha, const float *A,
                            const int lda, const float *B, const int ldb,
                            float *C, const int ldc);

void KERNEL_std_sgemm_v6_B0(const int M, const int N,
                            const int K, const float alpha, const float *A,
                            const int lda, const float *B, const int ldb,
                            float *C, const int ldc);

void KERNEL_std_sgemm_v6_double_elaboration(const int M, const int N,
                                            const int K, const float alpha, const float *A,
                                            const int lda, const float *B, const int ldb,
                                            const float beta, float *C, const int ldc);

void KERNEL_std_sgemm_v7(const int M, const int N,
                         const int K, const float alpha, const float *A,
                         const int lda, const float *B, const int ldb,
                         const float beta, float *C, const int ldc);

/*
 * Double-Precision Standard Kernel(s)
 */
void KERNEL_std_dgemm_v1(const int M, const int N,
                         const int K, const double alpha, const double *A,
                         const int lda, const double *B, const int ldb,
                         const double beta, double *C, const int ldc);

/*
 * Single-Precision Asymmetric Packing Kernels
 */

void KERNEL_p_sgemm_v1_r3(const int M, const int N,
                          const int K, const float alpha, const float *A,
                          const int lda, const float *B, const int ldb,
                          const float beta, float *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb);

/*
 * Single-Precision Symmetric Packing Kernels
 */

void KERNEL_p_sgemm_v2_r5(const int M, const int N,
                          const int K, const float alpha, const float *A,
                          const int lda, const float *B, const int ldb,
                          const float beta, float *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p_sgemm_v2_r5_EC(const int M, const int N,
                             const int K, const float alpha, const float *A,
                             const int lda, const float *B, const int ldb,
                             const float beta, float *C, const int ldc,
                             gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p_sgemm_v2_r5_EC_v2(const int M, const int N,
                                const int K, const float alpha, const float *A,
                                const int lda, const float *B, const int ldb,
                                const float beta, float *C, const int ldc,
                                gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p_sgemm_v2_r5_EC_v3(const int M, const int N,
                                const int K, const float alpha, const float *A,
                                const int lda, const float *B, const int ldb,
                                const float beta, float *C, const int ldc,
                                gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p_sgemm_v2_r5_EC_v3_MT(const int M, const int N,
                                   const int K, const float alpha, const float *A,
                                   const int lda, const float *B, const int ldb,
                                   const float beta, float *C, const int ldc,
                                   gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p_sgemm_v2_r5_EC_v4(const int M, const int N,
                                   const int K, const float alpha, const float *A,
                                   const int lda, const float *B, const int ldb,
                                   const float beta, float *C, const int ldc,
                                   gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p4_sgemm_v2_r5_EC_v4(const int M, const int N,
                                 const int K, const float alpha, const float *A,
                                 const int lda, const float *B, const int ldb,
                                 const float beta, float *C, const int ldc,
                                 gblas_quantizer& Qa, gblas_quantizer& Qb);

/*
 * Double-Precision Asymmetric Packing Kernels
 */

void KERNEL_p_dgemm_v1_r3(const int M, const int N,
                          const int K, const double alpha, const double *A /* M x K */,
                          const int lda, const double *B /* K x N */, const int ldb,
                          const float beta, double *C /* M x N*/, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p3_dgemm_v1_r3(const int M, const int N,
                           const int K, const double alpha, const double *A /* M x K */,
                           const int lda, const double *B /* K x N */, const int ldb,
                           const float beta, double *C /* M x N*/, const int ldc,
                           gblas_quantizer& Qa, gblas_quantizer& Qb);
void KERNEL_p4_dgemm_v1_r3(const int M, const int N,
                           const int K, const double alpha, const double *A /* M x K */,
                           const int lda, const double *B /* K x N */, const int ldb,
                           const float beta, double *C /* M x N*/, const int ldc,
                           gblas_quantizer& Qa, gblas_quantizer& Qb);

/*
 * Double-Precision Symmetric Packing Kernels
 */
void KERNEL_p_dgemm_v2_r4(const int M, const int N,
                          const int K, const double alpha, const double *A,
                          const int lda, const double *B, const int ldb,
                          const double beta, double *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p3_dgemm_v2_r4(const int M, const int N,
                          const int K, const double alpha, const double *A,
                          const int lda, const double *B, const int ldb,
                          const double beta, double *C, const int ldc,
                          gblas_quantizer& Qa, gblas_quantizer& Qb);

void KERNEL_p4_dgemm_v2_r4(const int M, const int N,
                           const int K, const double alpha, const double *A,
                           const int lda, const double *B, const int ldb,
                           const double beta, double *C, const int ldc,
                           gblas_quantizer& Qa, gblas_quantizer& Qb);

#endif