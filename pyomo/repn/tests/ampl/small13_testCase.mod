#
# Author:  Gabe Hackebeil
# Purpose: For testing to ensure that the Pyomo NL writer properly
#          converts the nonlinear expression to the NL file format.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

var x := 0.5;

maximize obj: x;

s.t.
     c1: (x^3 - x) == 0;
     c2: 10*(x^3 - x) == 0;
     c3: (x^3 - x)/10.0 == 0;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gsmall13.ampl;
