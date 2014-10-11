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

var p;
param q := 2.0;

minimize OBJ: x;
s.t.
        CON1a:
                1.0/p/q*v*(x-y) = 2.0;
        CON2a:
                v*(1.0/p/q)*(x-y) = 2.0;
        CON3a:
                v*(x-y)/p/q = 2.0;
        CON4a:
                v*(x/p/q-y/p/q) = 2.0;
        CON5a:
                v*(x-y)*(1.0/p/q) = 2.0;
	CON6a:
		v*(x-y) = 2.0*p*q;

        CON1b:
                1.0/(p*q)*v*(x-y) = 2.0;
        CON2b:  
		v*(1.0/(p*q))*(x-y) = 2.0;
        CON3b:
                v*(x-y)/(p*q) = 2.0;
        CON4b:
                v*(x/(p*q)-y/(p*q)) = 2.0;
        CON5b:
                v*(x-y)*(1.0/(p*q)) = 2.0;
	CON6b:
		v*(x-y) = 2.0*(p*q);

        CON1c:
                1.0/(p+q)*v*(x-y) = 2.0;
        CON2c:
                v*(1.0/(p+q))*(x-y) = 2.0;
        CON3c:
                v*(x-y)/(p+q) = 2.0;
        CON4c:
                v*(x/(p+q)-y/(p+q)) = 2.0;
        CON5c:
                v*(x-y)*(1.0/(p+q)) = 2.0;
	CON6c:
		v*(x-y) = 2.0*(p+q);


        CON1d:
                1.0/((p+q)^2)*v*(x-y) = 2.0;
        CON2d:
                v*(1.0/((p+q)^2))*(x-y) = 2.0;
        CON3d:
                v*(x-y)/((p+q)^2) = 2.0;
        CON4d:
                v*(x/((p+q)^2)-y/((p+q)^2)) = 2.0;
        CON5d:
                v*(x-y)*(1.0/((p+q)^2)) = 2.0;
	CON6d:
		v*(x-y) = 2.0*((p+q)^2);

data;
var x := 1.0;
var y := 2.0;
var v := 3.0;

fix p := 2.0;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gjunk;
