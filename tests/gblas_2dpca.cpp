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
#include <cstdio>
#include <mm_malloc.h>

#include "gblas.h"
#include "matrix_utils.h"

int main(int argc, char *argv[])
{   
    int M = 288; //GBLAS_KERNEL_SIZE;
    int N = 288; //GBLAS_KERNEL_SIZE;
    int K = 288; //GBLAS_KERNEL_SIZE;

    float alpha = 1.0f;
    float beta = 0.0f;

    float *A       = (float*)_mm_malloc(sizeof(float)*M*K, 16);
    float *B       = (float*)_mm_malloc(sizeof(float)*K*N, 16);
    float *C       = (float*)_mm_malloc(sizeof(float)*M*N, 16);

    FILE* fin = fopen("gblas_input.dat", "rb");

    fread(A, sizeof(float), M*K, fin);
    fread(B, sizeof(float), K*N, fin);

    fclose(fin);

    float max_a, min_a, max_b, min_b;

// V.2 (variance and SNR!)
    if (argc < 2) exit(-1);

    char* end;
    float snr = strtof(argv[1], &end);

    get_matrix_min_max(A, M, K, max_a, min_a);
    get_matrix_min_max(B, K, N, max_b, min_b);

    max_a = max(abs(max_a), abs(min_a));
    max_b = max(abs(max_b), abs(min_b));

    float var_a = 2.0f*max_a/sqrtf(12.0f);
    float var_b = 2.0f*max_b/sqrtf(12.0f);

    printf("max(A) = %f, var(A) = %f, max(B) = %f, var(B) = %f, snr = %f \n", max_a, var_a, max_b, var_b, snr);

    gblas_sgemm_snr(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N, snr, var_a, var_b);

// V.1 (acceleration percentage)
    //gblas_sgemm_mu(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N, 100.0);

    FILE* fout = fopen("gblas_output.bin", "wb"); // open write-binary
    fwrite(C, sizeof(float), M*N, fout);
    fclose(fout);

   //print_matrix_matlab_notation(C, M, N);
}
