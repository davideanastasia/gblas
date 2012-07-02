/*
 *  msec_timer.h
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __TIMING_C_H__
#define __TIMING_C_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#define WIN_TIMER
#endif

#include <stdio.h>
#include <string>

// TIMER -----
#ifdef WIN_TIMER
#include <windows.h>
#elif __APPLE__
#include <stdint.h>
#include <mach/mach_time.h>
#else
//#include <ctime>
#include <sys/time.h> 
#endif

class msec_timer
{
private:
#ifdef WIN_TIMER
  LARGE_INTEGER start_t;
  LARGE_INTEGER stop_t;
  LARGE_INTEGER freq;
  double        wrk_time;
#elif __APPLE__
  uint64_t start_t;
  uint64_t stop_t;
  uint64_t wrk_time;
  double   conversion;
#else
  timeval start_t;
  timeval stop_t;
  double  wrk_time;
#endif
  
public:
  msec_timer();
  ~msec_timer();
  void start();
  void stop();
  void update();
  void stop_and_update();
  void reset();
  double get_time();
  
  void get_timer_type();
};

double convert_to_gigaflops(double, double);
std::string get_current_date();

#endif // __TIMING_C_H__