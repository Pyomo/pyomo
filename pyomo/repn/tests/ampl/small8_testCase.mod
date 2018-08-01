#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly modifies product expressions
#          with only constant terms in the denominator (that
#          are involved in linear expressions).
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

param a := 2.0;

var x >= 0;
var y >= 0;
var z >= 7;

minimize obj: z+x*x+y;

s.t.
        constr:
                y*y >= a;

        constr2:
                y <= x/a;

        constr3:
                z <= y + a;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gsmall8.ampl;
