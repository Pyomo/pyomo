/**___________________________________________________________________________
 *
 * Pyomo: Python Optimization Modeling Objects
 * Copyright (c) 2008-2025
 * National Technology and Engineering Solutions of Sandia, LLC
 * Under the terms of Contract DE-NA0003525 with National Technology and
 * Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
 * rights in this software.
 * This software is distributed under the 3-clause BSD License.
 * ___________________________________________________________________________
**/

#include<iostream>
#include "AmplInterface.hpp"

int main()
{
  AmplInterface* ans = new AmplInterfaceFile();
  ans->initialize("simple_nlp.nl");
  delete ans;
  std::cout << "Done\n";
  return 0;
}
