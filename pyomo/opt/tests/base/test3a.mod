# STIGLER'S NUTRITION MODEL
#
# This model determines a least cost diet which meets the daily
# allowances of nutrients for a moderately active man weighing 154 lbs.
#
#  References:
#              Dantzig G B, "Linear Programming and Extensions."
#              Princeton University Press, Princeton, New Jersey, 1963,
#              Chapter 27-1.

set N;
/* nutrients */

set F;
/* foods */

param b{N};
/* required daily allowances of nutrients */

param a{F,N};
/* nutritive value of foods (per dollar spent) */

var x{f in F} >= 0;
/* dollars of food f to be purchased daily */

s.t. nb{n in N}: sum{f in F} a[f,n] * x[f] = b[n];
/* nutrient balance (units) */

minimize cost: sum{f in F} x[f];
/* total food bill (dollars) */

