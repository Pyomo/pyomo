#include<iostream>
#include "AmplInterface.hpp"

int main()
{
  AmplInterface* ans = new AmplInterfaceFile();
  ans->initialize("simple_nlp.nl");
  delete ans;
  return 0;
}
