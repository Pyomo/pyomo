#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly reclassifies nonlinear expressions
#          as linear or trivial when fixing variables or params
#          cause such a situation.
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

var x;
var y;
var z;
param q := 0.0;
param p := 0.0;

fix y := 0.0;

minimize obj: x;

s.t.
	con1:
		x*y*z + x == 1.0;
	con2:
		x*p*z + x == 1.0;
	con3:
		x*q*z + x == 1.0;
# AMPL differs from Pyomo in these cases that involve immutable params (q).
# These never actually become constraints in Pyomo, and for good reason.
	con4:
		x*y*z == 1.0;
	con5:
		x*p*z == 1.0;
#	con6:
#		x*q*z == 1.0;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gjunk;
