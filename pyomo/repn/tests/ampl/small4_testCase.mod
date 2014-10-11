#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly reports the values corresponding
#          to the nl file header line with the label
#          '# nonlinear vars in constraints, objectives, both'
#

var x := 1.0;
var y := 1.0;

minimize OBJ: y^2;

s.t.
	CON1:
		y*x = 4;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gjunk;
