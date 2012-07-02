/*
 *  gblas_sse_utils.h
 *  gblas
 *
 *  Created by Davide Anastasia on 07/07/2010.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __GBLAS_SSE_UTILS_H__
#define __GBLAS_SSE_UTILS_H__

#define   ZERO_DOT_FIVE     (0.5)

//#if __ppc__ || __ppc7400__ || __ppc64__ || __ppc970__
//#include <ppc_intrinsics.h>
//#elif __i386__ || __x86_64__
#include <emmintrin.h>
#include <pmmintrin.h>
//#include <tmmintrin.h>
//#else
//#error unsupported architecture
//#endif

static const __m128d  _MM_ZERO_D          = _mm_setzero_pd();
static const __m128   _MM_ZERO_S          = _mm_setzero_ps();
static const __m128d  _MM_ZERO_DOT_FIVE_D = _mm_set1_pd(0.5);
static const __m128   _MM_ZERO_DOT_FIVE_S = _mm_set1_ps(0.5f);
static const __m128d  _MM_MASK_ONE_D      = _mm_set1_pd(0x00000001);
static const __m128   _MM_MASK_ONE_S      = _mm_set1_ps(0x00000001);

template <class T>
T max(T v1, T v2)
{
  return (v1 > v2)? v1 : v2;
}

template <class T>
T min(T v1, T v2)
{
  return (v1 < v2)? v1 : v2;
}

inline void fast_unpack_tight_v2(__m128& i, const float& eps, const float& inv_eps)
{  
  // v.1
  /*
  //i               = _mm_add_ps(i, _mm_set1_ps(NOISE_FLOOR_FLOAT));
  
  __m128        a = _mm_mul_ps(i, _mm_set1_ps(eps));
  
  __m128      cmp = _mm_cmplt_ps(a, _MM_ZERO_S);
  __m128      add = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  
                a = _mm_sub_ps(a, add);
  
  __m128      a_i = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
  __m128      i_i = i;//_mm_cvtepi32_ps(_mm_cvttps_epi32(i)); // got rid of the .nnn
  
  __m128      a_i_eps = _mm_mul_ps(a_i, _mm_set1_ps(inv_eps));
  
  __m128      mod = _mm_sub_ps(i_i, a_i_eps);
              mod = _mm_add_ps(mod, _MM_ZERO_DOT_FIVE_S);
  
                i = _mm_cvtepi32_ps(_mm_cvttps_epi32(mod));
   */
  
  // v.2
  __m128 cmp, a;
  
  a   = _mm_mul_ps(i, _mm_set1_ps(eps));
  
  cmp = _mm_cmpgt_ps(a, _MM_ZERO_S);
  cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
  a   = _mm_add_ps(a, cmp);
  
  a   = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
  
  a   = _mm_mul_ps(a, _mm_set1_ps(inv_eps));
  a   = _mm_sub_ps(i, a);
  
  cmp = _mm_cmpgt_ps(a, _MM_ZERO_S);
  cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
  a   = _mm_add_ps(a, cmp);
  
  i   = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));     // return
}

inline void unpack_complete_tight_v1(__m128 i, const float eps, const float inv_eps, __m128& o1, __m128& o2, __m128& o3)
{
  __m128 cmp;
                                                  // i = x1*eps^-1 + x2 + x3*eps
  o1   = _mm_mul_ps(i, _mm_set1_ps(eps));         // o1 = x1 + x2*eps + x3*eps^2
  
  cmp = _mm_cmpgt_ps(o1, _MM_ZERO_S);           
  cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
  o1   = _mm_add_ps(o1, cmp);                     
  
  o1   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o1));   // o1 = round(o1) = x1
  
  o2   = _mm_mul_ps(o1, _mm_set1_ps(inv_eps));    // o2 = x1*eps^-1
  i   = _mm_sub_ps(i, o2);                        // i = i - o2 = x2 + x3*eps
  
  cmp = _mm_cmpgt_ps(i, _MM_ZERO_S);
  cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
  o2   = _mm_add_ps(i, cmp);                      // o2 = i +- 0.5
  
  o2   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o2));   // o2 = round(o2) = x2
  
  o3   = _mm_sub_ps(i, o2);                       // o3 = i - o2 = x3*eps
  o3   = _mm_mul_ps(o3, _mm_set1_ps(inv_eps));    // o3 = o3*eps^-1 = x3~
  
  cmp = _mm_cmpgt_ps(o3, _MM_ZERO_S);
  cmp = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  cmp = _mm_sub_ps(cmp, _MM_ZERO_DOT_FIVE_S);
  o3   = _mm_add_ps(o3, cmp);
  
  o3   = _mm_cvtepi32_ps(_mm_cvttps_epi32(o3));   // o3 = round(o3) = x3
}

//inline void fast_unpack_tight_v3(__m128& i, const float& eps, const float& inv_eps)
//{  
//  __m128        a = _mm_mul_ps(i, _mm_set1_ps(eps));
//  
//  __m128      cmp = _mm_cmplt_ps(a, _MM_ZERO_S);
//  __m128      add = _mm_and_ps(cmp, _MM_MASK_ONE_S);
//  
//  a = _mm_sub_ps(a, add);
//  
//  __m128      a_i = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
//  __m128      i_i = i;
//  
//  __m128      a_i_eps = _mm_mul_ps(a_i, _mm_set1_ps(inv_eps));
//  
//  __m128      mod = _mm_sub_ps(i_i, a_i_eps);
//  mod = _mm_add_ps(mod, _MM_ZERO_DOT_FIVE_S);
//  
//  i = _mm_cvtepi32_ps(_mm_cvttps_epi32(mod));
//}

inline void fast_unpack_tight_v2(__m128d& i, const double eps, const double inv_eps)
{
  //i               = _mm_add_ps(i, _mm_set1_ps(NOISE_FLOOR_FLOAT));
  __m128d cmp, a;
  
  a   = _mm_mul_pd(i, _mm_set1_pd(eps));
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
  
  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps));
  a   = _mm_sub_pd(i, a);
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  i   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));     // return
} 

inline void fast_unpack_tight_v3(__m128d& i, const double eps, const double inv_eps)
{
  __m128d cmp, a;

  a   = _mm_mul_pd(i, _mm_set1_pd(eps*eps));
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
  
  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps*inv_eps));
  i   = _mm_sub_pd(i, a);
  
  a   = _mm_mul_pd(i, _mm_set1_pd(eps));
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
  
  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps));
  a   = _mm_sub_pd(i, a);
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  i   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));     // return
} 

//inline void fast_unpack_tight_v4(__m128d& i, const double eps, const double inv_eps)
//{
//  __m128d cmp, a;
//  a   = _mm_mul_pd(i, _mm_set1_pd(eps*eps*eps));
//  
//  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
//  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
//  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
//  a   = _mm_add_pd(a, cmp);
//  
//  a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
//  
//  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps*inv_eps*inv_eps));
//  i   = _mm_sub_pd(i, a);
//  
//  a   = _mm_mul_pd(i, _mm_set1_pd(eps*eps));
//  
//  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
//  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
//  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
//  a   = _mm_add_pd(a, cmp);
//  
//  a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
//  
//  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps*inv_eps));
//  i   = _mm_sub_pd(i, a);
//  
//  a   = _mm_mul_pd(i, _mm_set1_pd(eps));
//  
//  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
//  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
//  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
//  a   = _mm_add_pd(a, cmp);
//  
//  a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
//  
//  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps));
//  a   = _mm_sub_pd(i, a);
//  
//  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
//  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
//  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
//  a   = _mm_add_pd(a, cmp);
//  
//  i   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));     // return
//} 

inline void fast_unpack_tight_v4(__m128d& i, const double eps, const double inv_eps)
{
  __m128d cmp, a;  
  a   = _mm_mul_pd(i, _mm_set1_pd(eps));
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  __m128d hi = _mm_unpackhi_pd(a, a);
  __m128d lo = _mm_unpacklo_pd(a, a);

  //a   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));
  hi = _mm_cvtsi64_sd(hi, _mm_cvttsd_si64(hi));
  lo = _mm_cvtsi64_sd(lo, _mm_cvttsd_si64(lo));  

  a   = _mm_shuffle_pd(lo, hi, _MM_SHUFFLE(3,2,1,0));
  
  a   = _mm_mul_pd(a, _mm_set1_pd(inv_eps));
  a   = _mm_sub_pd(i, a);
  
  cmp = _mm_cmpgt_pd(a, _MM_ZERO_D);
  cmp = _mm_and_pd(cmp, _MM_MASK_ONE_D);
  cmp = _mm_sub_pd(cmp, _MM_ZERO_DOT_FIVE_D);
  a   = _mm_add_pd(a, cmp);
  
  i   = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a));     // return
} 

//__m128  mm_abs(__m128 i);
//
//__m128d mm_full_round(__m128d i);
//__m128  mm_full_round(__m128  i);
//__m128  mm_full_round_v2(__m128 i);
//
//__m128  fast_unpack_loose(__m128 i, int bits);
//__m128  fast_unpack_tight(__m128 i, float eps, float inv_eps);
//__m128 horiz_add(__m128 tmm0, __m128 tmm1, __m128 tmm2, __m128 tmm3);
//__m128 horiz_add(__m128 tmm0);
//__m128d horiz_add(__m128d tmm0);

inline __m128d mm_full_round(__m128d i)
{
  return _mm_cvtepi32_pd(_mm_cvttpd_epi32(i));
}

inline __m128 mm_full_round(__m128 i)
{
  return _mm_cvtepi32_ps(_mm_cvttps_epi32(i));
}

inline __m128 mm_full_round_v2(__m128 i)
{
  __m128 cmp      = _mm_cmpgt_ps(i, _mm_set1_ps(0.0f));
  __m128 disp     = _mm_and_ps(cmp, _mm_set1_ps(0x00000001));
  disp            = _mm_sub_ps(disp, _mm_set1_ps(0.5f));
  
  return _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(i, disp)));
}

/*
 __m128 mm_full_round(__m128 i)
 {
 __m128 cmp      = _mm_cmplt_ps(i, _mm_set1_ps(0.0f));
 __m128 add      = _mm_and_ps(cmp, _mm_set1_ps(0x00000001));
 
 i               = _mm_sub_ps(i, add);
 
 return _mm_cvtepi32_ps(_mm_cvttps_epi32(i));
 }
 */

inline __m128 horiz_add(__m128& tmm0, __m128& tmm1, __m128& tmm2, __m128& tmm3)
{
  /*
   // V.1 (SSE2)
   __m128 xx1, xx2, xx3;
   
   xx1 = _mm_movelh_ps(tmm0, tmm1);                          // xx1  = A1 A2 B1 B2
   tmm1 = _mm_movehl_ps(tmm1, tmm0);                         // tmm1 = A3 A4 B3 B4
   xx1 = _mm_add_ps(xx1, tmm1);                              // xx1  = A1+A3 A2+A4 B1+B3 B2+B4
   
   xx2 = _mm_movelh_ps(tmm2, tmm3);                          // tmm2 = C1 C2 D1 D2
   tmm3 = _mm_movehl_ps(tmm3, tmm2);                         // tmm3 = C3 C4 D3 D4
   xx2 = _mm_add_ps(xx2, tmm3);                              // tmm3 = C1+C3 C2+C4 D1+D3 D2+D4
   
   xx3 = _mm_shuffle_ps(xx1, xx2, _MM_SHUFFLE(3,1,3,1));     // xx3  = A1+A3 B1+B3 C1+C3 D1+D3 // 0xDD
   xx1 = _mm_shuffle_ps(xx1, xx2, _MM_SHUFFLE(2,0,2,0));     // xx1  = A2+A4 B2+B4 C2+C4 D2+D4 // 0x88
   xx3 = _mm_add_ps(xx3, xx1);                               // xx3  = A1+A2+A3+A4 B1+B2+B3+B4
   //        C1+C2+C3+C4 D1+D2+D3+D4
   return xx3;
   */
  
  // This V.2 is slightly faster than the V.1 (SSE3)
  __m128 xx1, xx2, xx3;
  
  xx1 = _mm_hadd_ps(tmm0, tmm1);                              // xx1  = A1+A2 A3+A4 B1+B2 B3+B4
  xx2 = _mm_hadd_ps(tmm2, tmm3);                              // tmm3 = C1+C2 C3+C4 D1+D2 D3+D4
  
  xx3 = _mm_hadd_ps(xx1, xx2);                                // xx3  = A1+A2+A3+A4 B1+B2+B3+B4
  //        C1+C2+C3+C4 D1+D2+D3+D4
  return xx3; 
}

inline __m128 horiz_add(__m128 tmm0)
{
  /*
   // V.1 (SSE2)
   __m128 xx1, xx2, xx3;
   
   xx1 = _mm_movelh_ps(tmm0, tmm1);                          // xx1  = A1 A2 B1 B2
   tmm1 = _mm_movehl_ps(tmm1, tmm0);                         // tmm1 = A3 A4 B3 B4
   xx1 = _mm_add_ps(xx1, tmm1);                              // xx1  = A1+A3 A2+A4 B1+B3 B2+B4
   
   xx2 = _mm_movelh_ps(tmm2, tmm3);                          // tmm2 = C1 C2 D1 D2
   tmm3 = _mm_movehl_ps(tmm3, tmm2);                         // tmm3 = C3 C4 D3 D4
   xx2 = _mm_add_ps(xx2, tmm3);                              // tmm3 = C1+C3 C2+C4 D1+D3 D2+D4
   
   xx3 = _mm_shuffle_ps(xx1, xx2, _MM_SHUFFLE(3,1,3,1));     // xx3  = A1+A3 B1+B3 C1+C3 D1+D3 // 0xDD
   xx1 = _mm_shuffle_ps(xx1, xx2, _MM_SHUFFLE(2,0,2,0));     // xx1  = A2+A4 B2+B4 C2+C4 D2+D4 // 0x88
   xx3 = _mm_add_ps(xx3, xx1);                               // xx3  = A1+A2+A3+A4 B1+B2+B3+B4
   //        C1+C2+C3+C4 D1+D2+D3+D4
   return xx3;
   */
  
  // V.2
  __m128 xmm            = _mm_hadd_ps(tmm0, _MM_ZERO_S);
  xmm                   = _mm_hadd_ps(xmm, _MM_ZERO_S);
  return xmm; 
}

inline __m128d horiz_add(__m128d tmm0)
{
  return _mm_hadd_pd(tmm0, _MM_ZERO_D);
}

inline __m128 fast_unpack_loose(__m128 i, int bits)
{
  const unsigned int ui_mask = (1 << bits) - 1;
  
  __m128i i32 = _mm_cvttps_epi32(i); // conversion to int32 by truncation
  __m128i _mask = _mm_set_epi32(ui_mask, ui_mask, ui_mask, ui_mask);
  i32 = _mm_and_si128(i32, _mask);
  
  return _mm_cvtepi32_ps(i32); 
}

/*
 __m128 fast_unpack_tight(__m128 i, float eps, float inv_eps)
 {  
 //__m128 _f = _mm_mul_ps(i, _mm_set1_ps(-1.0f));
 //__m128 _i = _mm_and_ps(i, _f);  // sign out!
 
 // WORKS
 __m128 i_div_eps = _mm_mul_ps(i, _mm_set1_ps(eps));
 __m128 i_div_eps_floor = _mm_cvtepi32_ps(_mm_cvttps_epi32(i_div_eps));
 
 __m128 i_to_eps = _mm_mul_ps(i_div_eps_floor, _mm_set1_ps(inv_eps));
 __m128 mod = _mm_sub_ps(i, i_to_eps);
 
 return _mm_cvtepi32_ps(_mm_cvttps_epi32(mod));
 }
 */

inline __m128 fast_unpack_tight(__m128 i, float eps, float inv_eps)
{
  //i               = _mm_add_ps(i, _mm_set1_ps(NOISE_FLOOR_FLOAT));
  
  __m128  a       = _mm_mul_ps(i, _mm_set1_ps(eps));
  
  __m128 cmp      = _mm_cmplt_ps(a, _MM_ZERO_S);
  __m128 add      = _mm_and_ps(cmp, _MM_MASK_ONE_S);
  
  a               = _mm_sub_ps(a, add);
  
  __m128  a_i     = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
  __m128  i_i     = i;//_mm_cvtepi32_ps(_mm_cvttps_epi32(i)); // got rid of the .nnn
  
  __m128 a_i_eps  = _mm_mul_ps(a_i, _mm_set1_ps(inv_eps));
  
  __m128 mod      = _mm_sub_ps(i_i, a_i_eps);
  
  return _mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_add_ps(mod, _MM_ZERO_DOT_FIVE_S)));
}

inline __m128 mm_abs(__m128 i)
{
  __m128 i_m = _mm_mul_ps(i, _mm_set1_ps(-1.0f));
  return _mm_and_ps(i, i_m);
}

inline void horiz_max(__m128d mm0, double& output)
{
  __m128d A1_B1 = _mm_unpackhi_pd(mm0, _MM_ZERO_D);
  __m128d A0_B0 = _mm_unpacklo_pd(mm0, _MM_ZERO_D);
  
  A0_B0 = _mm_max_pd(A0_B0, A1_B1); 
  
  _mm_store_sd(&output, A0_B0);
}

inline void horiz_min(__m128d mm0, double& output)
{
  __m128d A1_B1 = _mm_unpackhi_pd(mm0, _MM_ZERO_D);
  __m128d A0_B0 = _mm_unpacklo_pd(mm0, _MM_ZERO_D);
  
  A0_B0 = _mm_min_pd(A0_B0, A1_B1); 
  
  _mm_store_sd(&output, A0_B0);
}

inline void horiz_max(__m128 mm0, float& output)
{
  __m128 unpack_hi = _mm_unpackhi_ps(mm0, _MM_ZERO_S);
  __m128 unpack_lo = _mm_unpacklo_ps(mm0, _MM_ZERO_S);
  
  __m128 max = _mm_max_ps(unpack_hi, unpack_lo);
  
  unpack_hi = _mm_unpackhi_ps(max, _MM_ZERO_S);
  unpack_lo = _mm_unpacklo_ps(max, _MM_ZERO_S);
  
  max = _mm_max_ps(unpack_hi, unpack_lo);
  
  _mm_store_ss(&output, max);
}

inline void horiz_min(__m128 mm0, float& output)
{
  __m128 unpack_hi = _mm_unpackhi_ps(mm0, _MM_ZERO_S);
  __m128 unpack_lo = _mm_unpacklo_ps(mm0, _MM_ZERO_S);
  
  __m128 max = _mm_min_ps(unpack_hi, unpack_lo);
  
  unpack_hi = _mm_unpackhi_ps(max, _MM_ZERO_S);
  unpack_lo = _mm_unpacklo_ps(max, _MM_ZERO_S);
  
  max = _mm_min_ps(unpack_hi, unpack_lo);
  
  _mm_store_ss(&output, max);
}
#endif
