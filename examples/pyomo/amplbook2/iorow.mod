set MAT;             # materials
set ACT;             # activities
param io {MAT,ACT};  # input-output coefficients

param revenue {ACT};
param act_min {ACT} >= 0;
param act_max {j in ACT} >= act_min[j];

var Run {j in ACT} >= act_min[j], <= act_max[j];

maximize Net_Profit:  sum {j in ACT} revenue[j] * Run[j];

subject to Balance {i in MAT}:
   sum {j in ACT} io[i,j] * Run[j] = 0;
