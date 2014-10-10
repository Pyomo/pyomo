set PROD;   # products
set ACT;    # activities

param cost {ACT} > 0;      # cost per unit of each activity
param io {PROD,ACT} >= 0;  # units of each product from
                           # 1 unit of each activity

param demzero {PROD} > 0;  # intercept and slope of the demand
param demrate {PROD} >= 0; # as a function of price

var Price {i in PROD};
var Level {j in ACT};

subject to Pri_Compl {i in PROD}:
   Price[i] >= 0 complements
      sum {j in ACT} io[i,j] * Level[j]
         >= demzero[i] - demrate[i] * Price[i];

subject to Lev_Compl {j in ACT}:
   Level[j] >= 0 complements
      sum {i in PROD} Price[i] * io[i,j] <= cost[j];
