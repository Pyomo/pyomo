#
# Author:  Gabe Hackebeil
# Purpose: For testing to ensure that the Pyomo NL writer properly 
#          handles the Expr_if component.
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

var vTrue  := 1;
var vFalse := -1;

param pTrue  := 1;
param pFalse := -1;

var vN1 := -1;
var vP1 := 1;
var v0  := 0;
var vN2  := -2;
var vP2  := 2;

minimize obj: 10.0*if(v0)then(vTrue)else(vFalse);

s.t. 
     # True/False
     c1: if(0) then(vTrue) else(vFalse) = pFalse;
     c2: if(1) then(vTrue) else(vFalse) = pTrue;

     # x <= 0
     c3: if(vN1 <= 0) then(vTrue) else(vFalse) = pTrue;
     c4: if(v0  <= 0) then(vTrue) else(vFalse) = pTrue;
     c5: if(vP1 <= 0) then(vTrue) else(vFalse) = pFalse;

     # x < 0
     c6: if(vN1 < 0) then(vTrue) else(vFalse) = pTrue;
     c7: if(v0  < 0) then(vTrue) else(vFalse) = pFalse;
     c8: if(vP1 < 0) then(vTrue) else(vFalse) = pFalse;

     # x >= 0
     c9: if(vN1*10.0 >= 0) then(vTrue) else(vFalse)  = pFalse;
     c10: if(v0*10.0  >= 0) then(vTrue) else(vFalse) = pTrue;
     c11: if(vP1*10.0 >= 0) then(vTrue) else(vFalse) = pTrue;

     # x > 0
     c12: if(vN1*10.0 > 0) then(vTrue) else(vFalse) = pFalse;
     c13: if(v0*10.0  > 0) then(vTrue) else(vFalse) = pFalse;
     c14: if(vP1*10.0 > 0) then(vTrue) else(vFalse) = pTrue;
     
     # -1 <= x <= 1
     c15: if(-1        <= vN2 <= 1) then(vTrue) else(vFalse) = pFalse;
     c16: if(-1*vP1    <= vN1 <= 1) then(vTrue) else(vFalse) = pTrue;
     c17: if(-1*vP1**2 <= v0  <= 1) then(vTrue) else(vFalse) = pTrue;
     c18: if(vN1       <= vP1 <= 1) then(vTrue) else(vFalse) = pTrue;
     c19: if(-1        <= vP2 <= 1) then(vTrue) else(vFalse) = pFalse;

     # -1 < x < 1
     c20: if(-1 < vN2 < 1)        then(vTrue) else(vFalse) = pFalse;
     c21: if(-1 < vN1 < 1*vP1)    then(vTrue) else(vFalse) = pFalse;
     c22: if(-1 < v0  < 1*vP1**2) then(vTrue) else(vFalse) = pTrue;
     c23: if(-1 < vP1 < vP1)      then(vTrue) else(vFalse) = pFalse;
     c24: if(-1 < vP2 < 1)        then(vTrue) else(vFalse) = pFalse;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gjunk;
