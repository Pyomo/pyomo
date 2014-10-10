
param N >= 1, integer;
set Locations := 1 .. N;

param P >= 1 <= N;

param M >= 1, integer;
set Customers := 1 .. M;

param d {Locations, Customers} := Uniform(1.0,2.0);

var x {Locations, Customers} >= 0 <= 1;
var y {Locations} binary;

minimize obj: sum {n in Locations, m in Customers} d[n,m] * x[n,m];

s.t. single_x {m in Customers}: sum {n in Locations} x[n,m] = 1;

s.t. bound_y {n in Locations, m in Customers}: x[n,m] <= y[n];

s.t. num_facilities: sum {n in Locations} y[n] = P;

# SUCASA SYMBOLS: *
