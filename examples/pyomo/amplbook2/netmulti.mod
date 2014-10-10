set CITIES;
set LINKS within (CITIES cross CITIES);

set PRODS;

param supply {CITIES,PRODS} >= 0;  # amounts available at cities
param demand {CITIES,PRODS} >= 0;  # amounts required at cities

   check {p in PRODS}: 
      sum {i in CITIES} supply[i,p] = sum {j in CITIES} demand[j,p];

param cost {LINKS,PRODS} >= 0;     # shipment costs/1000 packages
param capacity {LINKS,PRODS} >= 0; # max packages shipped
param cap_joint {LINKS} >= 0;      # max total packages shipped/link

minimize Total_Cost;

node Balance {k in CITIES, p in PRODS}: 
   net_in = demand[k,p] - supply[k,p];

arc Ship {(i,j) in LINKS, p in PRODS} >= 0, <= capacity[i,j,p],
   from Balance[i,p], to Balance[j,p], obj Total_Cost cost[i,j,p]; 

subject to Multi {(i,j) in LINKS}:
   sum {p in PRODS} Ship[i,j,p] <= cap_joint[i,j];
