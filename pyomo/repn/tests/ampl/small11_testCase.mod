#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly labels constraint ids in the "J"
#          section of the NL file when trivial constraints exist.
#          At the creation of this test, trivial constraints
#          (constraints with no variables) are being written to
#          the nl file as a feasibility check for the user.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

param n=3;
var x{k in 1..n, i in k..n};

minimize obj: x[n,n];

s.t.
     var_bnd{i in 1..n}: -1.0 <= x[1,i] <= 1.0;

fix x[1,1] := 1.0;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gsmall11.ampl;
