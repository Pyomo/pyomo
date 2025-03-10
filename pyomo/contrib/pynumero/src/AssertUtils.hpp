/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/
#ifndef __ASSERTUTILS_HPP__
#define __ASSERTUTILS_HPP__

#include <cassert>
#include <iostream>
#include <stdlib.h>

#define _ASSERT_ assert
#define _ASSERT_MSG_ assert_msg
#define _ASSERT_EXIT_ assert_exit
#define _ASSERTION_FAILURE_ assertion_failure

inline void assert_msg(bool cond, const std::string &msg) {
  #if defined(_WIN32) || defined(_WIN64)
  #else
  if (!cond) {
      std::cout << "Assertion Failed: " << msg.c_str() << std::endl;
   }
   assert(msg.c_str() && cond);
  #endif
}

inline void assert_exit(bool cond, const std::string &msg, int exitcode = 1) {
  #if defined(_WIN32) || defined(_WIN64)
  #else
   if (!(cond)) {
      std::cout << msg << std::endl;
      exit(exitcode);
   }
  #endif
}

inline void assertion_failure(const std::string &msg) {
  #if defined(_WIN32) || defined(_WIN64)
  #else
   std::cout << "Assertion Failed: " << msg.c_str() << std::endl;
   assert(msg.c_str() && false);
  #endif
}

#endif
