#ifndef __GBLAS_H__
#define __GBLAS_H__

#ifdef __cplusplus
extern "C" {
#endif
#include "cblas.h"
#ifdef __cplusplus
}
#endif

#define     GBLAS_KERNEL_SIZE            (288) //(288) //(144)

#define     FLOAT_SYM
#define     DOUBLE_SYM


#define     ENABLE_PRUNING                
//#define     EXTERNAL_A_QUANT                // enable this to pass Aquant as parameter in dgemm_v6/v7 and sgemm_v13/sgemm_v14
//#define     FIXED_COMPOUNDERS             //1.0


/* --------------- XBLAS --------------- */
int gblas_dgemm_snr(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                    const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                    const int K, const double alpha, const double *A /* M x K */,
                    const int lda, const double *B /* K x N */, const int ldb,
                    const double beta, double *C /* M x N*/, const int ldc,
                    const double TARGET_SNR, const double VAR_A, const double VAR_B);

int gblas_dgemm_mse(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
             const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
             const int K, const double alpha, const double *A /* M x K */,
             const int lda, const double *B /* K x N */, const int ldb,
             const double beta, double *C /* M x N*/, const int ldc,
             const double TARGET_MSE = 100000
#ifdef EXTERNAL_A_QUANT
             , const long   A_QUANT = 30000
#endif
             );


int gblas_dgemm_mu(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                    const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                    const int K, const double alpha, const double *A /* M x K */,
                    const int lda, const double *B /* K x N */, const int ldb,
                    const double beta, double *C /* M x N*/, const int ldc,
                    const double TARGET_PERC_ACC = 100
#ifdef EXTERNAL_A_QUANT
                    , const long   A_QUANT = 30000
#endif
                    );


void gblas_dgemm_plain(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const double alpha, const double *A /* M x K */,
                  const int lda, const double *B /* K x N */, const int ldb,
                  const double beta, double *C /* M x N*/, const int ldc);

void gblas_sgemm_plain(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                  const int K, const float alpha, const float *A /* M x K */,
                  const int lda, const float *B /* K x N */, const int ldb,
                  const float beta, float *C /* M x N*/, const int ldc);

int gblas_sgemm_snr(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                    const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                    const int K, const float alpha, const float *A /* M x K */,
                    const int lda, const float *B /* K x N */, const int ldb,
                    const float beta, float *C /* M x N*/, const int ldc,
                    const double TARGET_SNR, const double VAR_A, const double VAR_B);

int gblas_sgemm_mse(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const float alpha, const float *A /* M x K */,
              const int lda, const float *B /* K x N */, const int ldb,
              const float beta, float *C /* M x N*/, const int ldc,
              const double TARGET_MSE = 100000
#ifdef EXTERNAL_A_QUANT
              , const double   A_QUANT = 200000000
#endif
              );

int gblas_sgemm_mu(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
              const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const float alpha, const float *A /* M x K */,
              const int lda, const float *B /* K x N */, const int ldb,
              const float beta, float *C /* M x N*/, const int ldc,
              const double TARGET_PERC_ACC = 100
#ifdef EXTERNAL_A_QUANT
              , const double   A_QUANT = 200000000
#endif
              );

int gblas_sgemm_mu_EC(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A /* M x K */,
                   const int lda, const float *B /* K x N */, const int ldb,
                   const float beta, float *C /* M x N*/, const int ldc,
                   const double TARGET_PERC_ACC = 100
#ifdef EXTERNAL_A_QUANT
                   , const double   A_QUANT = 200000000
#endif
                   );

#endif
