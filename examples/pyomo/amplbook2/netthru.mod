set CITIES;
set LINKS within (CITIES cross CITIES);

param supply {CITIES} >= 0;   # amounts available at cities
param demand {CITIES} >= 0;   # amounts required at cities

  check: sum {i in CITIES} supply[i] = sum {j in CITIES} demand[j];

param cost {LINKS} >= 0;      # shipment costs per ton

param city_cap {CITIES} >= 0; # max throughput at cities
param link_cap {LINKS} >= 0;  # max shipment over links

minimize Total_Cost;

node Supply {k in CITIES}: net_out = supply[k];
node Demand {k in CITIES}: net_in = demand[k];

arc Ship {(i,j) in LINKS} >= 0, <= link_cap[i,j],
   from Demand[i], to Supply[j], obj Total_Cost cost[i,j]; 

arc Through {k in CITIES} >= 0, <= city_cap[k],
   from Supply[k], to Demand[k];

