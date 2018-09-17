#ifndef __ASSERTUTILS_HPP__
#define __ASSERTUTILS_HPP__

#include <cassert>
#include <iostream>

#define _ASSERT_ assert
#define _ASSERT_MSG_ assert_msg
#define _ASSERT_EXIT_ assert_exit
#define _ASSERTION_FAILURE_ assertion_failure

inline void assert_msg(bool cond, const std::string &msg) {
   if (!cond) {
      std::cout << "Assertion Failed: " << msg.c_str() << std::endl;
   }
   assert(msg.c_str() && cond);
}

inline void assert_exit(bool cond, const std::string &msg, int exitcode = 1) {
   if (!(cond)) {
      std::cout << msg << std::endl;
      exit(exitcode);
   }
}

inline void assertion_failure(const std::string &msg) {
   std::cout << "Assertion Failed: " << msg.c_str() << std::endl;
   assert(msg.c_str() && false);
}

#endif
