set PROD;      # products
param T > 0;   # number of weeks

param rate {PROD} > 0;          # tons per hour produced
param inv0 {PROD} >= 0;         # initial inventory
param commit {PROD,1..T} >= 0;  # minimum tons sold in week
param market {PROD,1..T} >= 0;  # limit on tons sold in week

param avail_min {1..T} >= 0;                 # unpenalized hours available
param avail_max {t in 1..T} >= avail_min[t]; # total hours available
param time_penalty {1..T} > 0;

param prodcost {PROD} >= 0;     # cost per ton produced
param invcost {PROD} >= 0;      # carrying cost per ton of inventory
param revenue {PROD,1..T} >= 0; # revenue per ton sold

var Make {PROD,1..T} >= 0;                  # tons produced
var Inv {PROD,0..T} >= 0;                   # tons inventoried
var Sell1 {p in PROD, t in 1..T} 
   >= 0, <= market[p,t]-commit[p,t];        # tons sold above commitment
var Sell0 {p in PROD, t in 1..T}
   >= 0, <= commit[p,t];                    # tons sold below commitment

var Use1 {t in 1..T} >= 0, <= avail_min[t];
var Use2 {t in 1..T} >= 0, <= avail_max[t]-avail_min[t];

maximize Total_Profit: 
   sum {p in PROD, t in 1..T} 
     (revenue[p,t]*(commit[p,t]+Sell1[p,t]-Sell0[p,t]) -
      prodcost[p]*Make[p,t] - invcost[p]*Inv[p,t]) -
   sum {t in 1..T} time_penalty[t] * Use2[t] -
   sum {p in PROD, t in 1..T} 1000000*Sell0[p,t];

               # Objective: total revenue less costs in all weeks

subject to Time {t in 1..T}:  
   sum {p in PROD} (1/rate[p]) * Make[p,t] = Use1[t] + Use2[t];

               # Total of hours used by all products
               # may not exceed hours available, in each week

subject to Init_Inv {p in PROD}:  Inv[p,0] = inv0[p];

               # Initial inventory must equal given value

subject to Balance {p in PROD, t in 1..T}:
   Make[p,t] + Inv[p,t-1] = (commit[p,t]+Sell1[p,t]-Sell0[p,t]) + Inv[p,t];

               # Tons produced and taken from inventory
               # must equal tons sold and put into inventory
