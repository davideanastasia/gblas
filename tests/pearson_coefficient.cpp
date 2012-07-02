/*
 *  num_repr_dgemm.cpp
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <mm_malloc.h>

using namespace std;

template <class T>
void calculate_stddev_and_mean(const T* m_in, const int rows, const int cols, double &mean, double &stddev)
{
    // mean!
    const int N_ELEMS = rows*cols;
    double tmp = 0.0;

    for (int idx = 0; idx < N_ELEMS; idx++)
    {
        tmp += (double(m_in[idx])/N_ELEMS); // I divide during the interation to avoid precision errors
    }

    mean = tmp;

    // std dev
    tmp = 0.0;
    for (int idx = 0; idx < N_ELEMS; idx++)
    {
        tmp += pow((double(m_in[idx]) - mean), 2.0);
    }

    stddev = tmp/N_ELEMS;
}

int main(int argc, char *argv[])
{
    const int NUM_ITER = 2;
    const int GBLAS_KERNEL_SIZE = 288;

    int NB = GBLAS_KERNEL_SIZE;
    int N = NB;
    int M = NB;
    //int K = NB;

    const int N_ELEMS = M*N*NUM_ITER;

    float *I1 = (float*)_mm_malloc(sizeof(float)*N_ELEMS, 16);
    double *I2 = (double*)_mm_malloc(sizeof(double)*N_ELEMS, 16);

    FILE* f1 = fopen("20110905_num_repr_test_sgemm.dat", "rb"); // open write-binary
    FILE* f2 = fopen("20110905_model_prec_test_dgemm.dat", "rb"); // open write-binary
    
    double mean1, stddev1, mean2, stddev2;

    double max_r = 0.0;

    for (double Aquant = (floor(5000/GBLAS_KERNEL_SIZE))*GBLAS_KERNEL_SIZE; Aquant <= 500000; Aquant += GBLAS_KERNEL_SIZE)  // double, sym, P4
    {
        // read
        fread(I1, sizeof(float), N_ELEMS, f1);
        fread(I2, sizeof(double), N_ELEMS, f2);

        // calculates mean
        calculate_stddev_and_mean(I1, N_ELEMS, 1, mean1, stddev1);
        calculate_stddev_and_mean(I2, N_ELEMS, 1, mean2, stddev2);

        double tmp = 0.0;
        for (int idx = 0; idx < N_ELEMS; ++idx)
        {
            tmp += ( (double(I1[idx]) - mean1)*(I2[idx] - mean2) );
        }

        double r = (tmp/N_ELEMS)/(sqrt(stddev1)*sqrt(stddev2)); // 288*288*2 = 168k

        if (abs(r) > max_r) max_r = abs(r);

        cout << Aquant;
        cout << ", " <<  mean1 << ", " << stddev1;
        cout << ", " <<  mean2 << ", " << stddev2;
        cout << ", " << r  << ", " << max_r;

        cout << endl;
    }

    fclose(f1);
    fclose(f2);

    _mm_free(I1);
    _mm_free(I2);
}
