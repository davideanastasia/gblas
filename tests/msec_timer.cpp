/*
 *  msec_timer.cpp
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *  http://msdn.microsoft.com/en-us/library/ms644904%28v=VS.85%29.aspx
 *
 */

#include "msec_timer.h"
#include <time.h>

//costructor
msec_timer::msec_timer():
#ifdef WIN_TIMER
wrk_time(0.0)
#elif __APPLE__
start_t(0), stop_t(0), wrk_time(0), conversion(1.0)
#else
wrk_time(0.0)
#endif
{
#ifdef WIN_TIMER
  QueryPerformanceFrequency(&freq);
#elif __APPLE__
  mach_timebase_info_data_t info;
  kern_return_t err = mach_timebase_info( &info );
  
  //Convert the timebase into seconds
  if ( err == 0  ) conversion = (1e-9 * (double) info.numer / (double) info.denom);
#else

#endif
}

msec_timer::~msec_timer() // destructor
{
  // nothing to do
}

void msec_timer::start()
{
#ifdef WIN_TIMER
  QueryPerformanceCounter(&start_t);
#elif __APPLE__
  start_t = mach_absolute_time();
#else
  gettimeofday(&start_t, NULL);
#endif
}

void msec_timer::stop()
{
#ifdef WIN_TIMER
  QueryPerformanceCounter(&stop_t);
#elif __APPLE__
  stop_t = mach_absolute_time();
#else
  gettimeofday(&stop_t, NULL);
#endif
}

void msec_timer::update()
{
#ifdef WIN_TIMER
  wrk_time += ((double)(stop_t.QuadPart - start_t.QuadPart));
#elif __APPLE__
  wrk_time += stop_t - start_t;
#else
  wrk_time += (((stop_t.tv_sec - start_t.tv_sec)*1000.0) + (stop_t.tv_usec - start_t.tv_usec)/1000.0);
#endif
}

void msec_timer::stop_and_update()
{
  stop();
  update();
}

void msec_timer::reset()
{
  wrk_time = 0.0;
}

double msec_timer::get_time()
{
#ifdef WIN_TIMER
  return (wrk_time * 1000.0 / freq.QuadPart);
#elif __APPLE__
  return (conversion * (double) wrk_time * 1000.0);
#else
  return wrk_time;
#endif
}

void msec_timer::get_timer_type()
{
#ifdef WIN_TIMER
  printf("<windows.h> QueryPerformanceCounter()\n");
#elif __APPLE__
  printf("<mach/mach_time.h> mach_absolute_time()\n");
#else
  // clock_gettime(3)
  printf("<sys/time.h> gettimeofday()\n");
#endif
}

double convert_to_gigaflops(double msec_time, double scale_factor)
{
  return ( scale_factor / (msec_time*1000.0));
}

std::string get_current_date()
{
  char outstr[200];
  time_t t;
  struct tm *tmp;
  t = time(NULL);
  tmp = localtime(&t);
  strftime(outstr, sizeof(outstr), "%Y%m%d-%H%M", tmp);
  
  return std::string(outstr);
}