/*
 *  gblas_matrix_utils.h
 *  gblas
 *
 *  Created by Davide Anastasia on 30/06/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __GBLAS_MATRIX_UTILS_H__
#define __GBLAS_MATRIX_UTILS_H__

#include "gblas.h"
#include "gblas_sse_utils.h"

void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const float* m_in, const float alpha, float* m_out, float* m_max, float* m_min);
void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const float* m_in, const float alpha, float* m_out);
void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const double* m_in, const double alpha, double* m_out, double* m_max, double* m_min);
void row_to_block_major(const enum CBLAS_TRANSPOSE Trans, const int rows, const int cols, const int ld, const double* m_in, const double alpha, double* m_out);

#endif