/*
 *  high_priority_process.cpp
 *  gblas
 *
 *  Created by Davide Anastasia.
 *  <danastas@ee.ucl.ac.uk>
 *  Copyright 2010, 2011 University College London. All rights reserved.
 *
 */

#include "high_priority_process.h"

void start_high_priority()
{
#ifdef WINDOWS_API
  if(!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
  {
    cerr << "[!] Failed to enter HIGH_PRIORITY_CLASS (" << GetLastError() << " )" << endl;
  } 
  cout << " Current priority class is " << GetPriorityClass(GetCurrentProcess()) << endl;
#else
  
  
#endif
  

}

void exit_high_priority()
{
#ifdef WINDOWS_API
  if(!SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS))
  {
    cerr << "[!] Failed to exit HIGH_PRIORITY_CLASS (" << GetLastError() << ")" << endl;
  }

  cout << " Current priority class is " << GetPriorityClass(GetCurrentProcess()) << endl;
#else
  
#endif
}