set PROD;   # products
set ACT;    # activities

param cost {ACT} > 0;      # cost per unit of each activity
param demand {PROD} >= 0;  # units of demand for each product
param io {PROD,ACT} >= 0;  # units of each product from
                           # 1 unit of each activity

var Price {i in PROD};
var Level {j in ACT};

subject to Pri_Compl {i in PROD}:
   Price[i] >= 0 complements
      sum {j in ACT} io[i,j] * Level[j] >= demand[i];

subject to Lev_Compl {j in ACT}:
   Level[j] >= 0 complements
      sum {i in PROD} Price[i] * io[i,j] <= cost[j];
