/*
 *  high_priority_process.h
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#ifndef __HIGH_PRIORITY_PROCESS_H__
#define __HIGH_PRIORITY_PROCESS_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#define WINDOWS_API (1)
#endif

#include <iostream>
#ifdef WINDOWS_API
#include <windows.h>
#else
//#include <sys/types.h>
//#include <sys/resource.h>
//#include <pthread.h>
//#include <sched.h>
#endif

using namespace std;

void start_high_priority();
void exit_high_priority();

#endif