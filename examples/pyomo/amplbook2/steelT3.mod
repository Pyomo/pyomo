set PROD;         # products
set AREA {PROD};  # market areas for each product
param T > 0;      # number of weeks

param rate {PROD} > 0;          # tons per hour produced
param inv0 {PROD} >= 0;         # initial inventory
param avail {1..T} >= 0;        # hours available in week
param market {p in PROD, AREA[p], 1..T} >= 0;  
                                # limit on tons sold in week

param prodcost {PROD} >= 0;     # cost per ton produced
param invcost {PROD} >= 0;      # carrying cost/ton of inventory
param revenue {p in PROD, AREA[p], 1..T} >= 0; 
                                # revenue per ton sold

var Make {PROD,1..T} >= 0;      # tons produced
var Inv {PROD,0..T} >= 0;       # tons inventoried
var Sell {p in PROD, a in AREA[p], t in 1..T}   # tons sold
                    >= 0, <= market[p,a,t];

maximize Total_Profit:
   sum {p in PROD, t in 1..T} 
      (sum {a in AREA[p]} revenue[p,a,t]*Sell[p,a,t] -
         prodcost[p]*Make[p,t] - invcost[p]*Inv[p,t]);

           # Total revenue less costs for all products in all weeks

subject to Time {t in 1..T}:
   sum {p in PROD} (1/rate[p]) * Make[p,t] <= avail[t];

           # Total of hours used by all products
           # may not exceed hours available, in each week

subject to Init_Inv {p in PROD}:  Inv[p,0] = inv0[p];

           # Initial inventory must equal given value

subject to Balance {p in PROD, t in 1..T}:
   Make[p,t] + Inv[p,t-1]
      = sum {a in AREA[p]} Sell[p,a,t] + Inv[p,t];

           # Tons produced and taken from inventory
           # must equal tons sold and put into inventory
