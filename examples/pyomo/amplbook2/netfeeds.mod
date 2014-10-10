set CITIES;

set LINKS within (CITIES cross CITIES);

set PRODS;

param supply {PRODS,CITIES} >= 0;  # amounts available at cities

param demand {PRODS,CITIES} >= 0;  # amounts required at cities

   check {p in PRODS}: 
      sum {i in CITIES} supply[p,i] = sum {j in CITIES} demand[p,j];

param cost {PRODS,LINKS} >= 0;     # shipment costs/1000 packages
param capacity {PRODS,LINKS} >= 0; # max packages shipped of product

set FEEDS;

param yield {PRODS,FEEDS} >= 0;    # amounts derived from feedstocks
param limit {FEEDS,CITIES} >= 0;   # feedstocks available at cities

minimize Total_Cost;

var Feed {f in FEEDS, k in CITIES} >= 0, <= limit[f,k];

node Balance {p in PRODS, k in CITIES}: 
   net_out = supply[p,k] - demand[p,k]
      + sum {f in FEEDS} yield[p,f] * Feed[f,k];

arc Ship {p in PRODS, (i,j) in LINKS} >= 0, <= capacity[p,i,j],
   from Balance[p,i], to Balance[p,j],
   obj Total_Cost cost[p,i,j]; 

