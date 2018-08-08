#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly reclassifies nonlinear expressions
#          as linear or trivial when fixing variables or params
#          cause such a situation.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

var x;
var y;
var z;
param q := 0.0;
param p := 0.0;

fix x := 1.0;
fix z := 0.0;

minimize obj: x*y +
              z*y +
              q*y +
              y*y*q +
              p*y +
              y*y*p +
              y*y*z +
              z*(y**2);

s.t.
        con1:
                x*y == 0;
        con2:
                z*y + y == 0;
        con3:
                q*(y**2) + y == 0;
        con4:
                q*y*x + y == 0;
        con5:
                p*(y**2) + y == 0;
        con6:
                p*y*x + y == 0;
        con7:
                z*(y**2) + y == 0;
        con8:
                z*y*x + y == 0;
# AMPL differs from Pyomo in these cases that involve immutable params (q).
# These never actually become constraints in Pyomo, and for good reason.
        con9:
                z*y == 0;
#        con10:
#                q*(y**2) == 0;
#        con11:
#                q*y*x == 0;
        con12:
                p*(y**2) == 0;
        con13:
                p*y*x == 0;
        con14:
                z*(y**2) == 0;
        con15:
                z*y*x == 0;
#        con16:
#                q*y == 0;
        con17:
                p*y == 0;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gsmall10.ampl;
