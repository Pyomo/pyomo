set PROD;           # products
set WEEKS ordered;  # number of weeks

param rate {PROD} > 0;           # tons per hour produced
param inv0 {PROD} >= 0;          # initial inventory
param avail {WEEKS} >= 0;        # hours available in week
param market {PROD,WEEKS} >= 0;  # limit on tons sold in week

param prodcost {PROD} >= 0;      # cost per ton produced
param invcost {PROD} >= 0;       # carrying cost/ton of inventory
param revenue {PROD,WEEKS} >= 0; # revenue/ton sold

var Make {PROD,WEEKS} >= 0;      # tons produced
var Inv {PROD,WEEKS} >= 0;       # tons inventoried
var Sell {p in PROD, t in WEEKS} >= 0, <= market[p,t]; # tons sold

maximize Total_Profit: 
   sum {p in PROD, t in WEEKS} (revenue[p,t]*Sell[p,t] -
      prodcost[p]*Make[p,t] - invcost[p]*Inv[p,t]);

          # Objective: total revenue less costs in all weeks

subject to Time {t in WEEKS}:  
   sum {p in PROD} (1/rate[p]) * Make[p,t] <= avail[t];

          # Total of hours used by all products
          # may not exceed hours available, in each week

subject to Balance0 {p in PROD}:
   Make[p,first(WEEKS)] + inv0[p]
      = Sell[p,first(WEEKS)] + Inv[p,first(WEEKS)];

subject to Balance {p in PROD, t in WEEKS: ord(t) > 1}:
   Make[p,t] + Inv[p,prev(t)] = Sell[p,t] + Inv[p,t];

          # Tons produced and taken from inventory
          # must equal tons sold and put into inventory
