
set prd;
set raw;

param T > 0 integer;

param max_prd > 0;

param units {raw,prd} >= 0;

param init_stock {raw} >= 0;

param profit {prd,1..T};

param cost{raw} >= 0;

param values {raw};

var Make {prd,1..T} >= 0;

var Store {raw,1..T+1} >= 0;

maximize total_profit:
  sum {t in 1..T} ( sum {j in prd} profit[j,t] * Make[j,t] - sum {i in raw} cost[i] * Store[i,t] ) + sum {i in raw} values[i] * Store[i,T+1];

subject to limit {t in 1..T}:
  sum{j in prd} Make[j,t] <= max_prd;

subject to start {i in raw}:
  Store[i,1] <= init_stock[i];

subject to balance {i in raw, t in 1..T}:
  Store[i,t+1] = Store[i,t] - sum {j in prd} units[i,j] * Make[j,t];

