set MAT;             # materials
set ACT;             # activities
param io {MAT,ACT};  # input-output coefficients

param revenue {ACT};
param act_min {ACT} >= 0;
param act_max {j in ACT} >= act_min[j];

maximize Net_Profit;

subject to Balance {i in MAT}: to_come = 0;

var Run {j in ACT} >= act_min[j], <= act_max[j],
   obj Net_Profit revenue[j],
   coeff {i in MAT} Balance[i] io[i,j];

