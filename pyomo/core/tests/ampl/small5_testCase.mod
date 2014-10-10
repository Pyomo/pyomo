#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly modifies product expressions 
#          with only constant terms in the denominator (that
#          are involved in nonlinear expressions).
#          The ASL differentiation routines seem to have a 
#          bug that causes the lagrangian hessian to become
#          dense unless this constant term in moved to the 
#          numerator. 
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

var x >= -1, <= 1;
var y >= -1, <= 1;
var v >= -1, <= 1;

param p := 2.0;
param q := 2.0;

minimize OBJ: (1.0/p)*x^2+(1.0/q)*x^2;
s.t.
        CON1:
                1.0/p*v*(x-y) = 2.0;
        CON2:
                v*1.0/p*(x-y) = 2.0;
        CON3:
                v*(x-y)/p = 2.0;
        CON4:
                v*(x/p-y/p) = 2.0;
        CON5:
                v*(x-y)*(1.0/p) = 2.0;
	CON6:
		v*(x-y) = 2.0*p;

	CON7:
		1.0/q*v*(x-y) = 2.0;
	CON8:
		v*1.0/q*(x-y) = 2.0;
	CON9:
		v*(x-y)/q = 2.0;
	CON10:
		v*(x/q-y/q) = 2.0;
	CON11:
		v*(x-y)*(1.0/q) = 2.0;
	CON12:
		v*(x-y) = 2.0*q;


data;
var x := 1.0;
var y := 2.0;
var v := 3.0;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gjunk;
