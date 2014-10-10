set ORIG;   # origins
set DEST;   # destinations

param supply {ORIG} >= 0;   # amounts available at origins
param demand {DEST} >= 0;   # amounts required at destinations

   check: sum {i in ORIG} supply[i] = sum {j in DEST} demand[j];

param npiece {ORIG,DEST} integer >= 1;

param rate {i in ORIG, j in DEST, p in 1..npiece[i,j]} 
  >= if p = 1 then 0 else rate[i,j,p-1];

param limit {i in ORIG, j in DEST, p in 1..npiece[i,j]-1} 
  > if p = 1 then 0 else limit[i,j,p-1];

var Trans {ORIG,DEST} >= 0;    # units to be shipped

minimize Total_Cost:
   sum {i in ORIG, j in DEST} 
      <<{p in 1..npiece[i,j]-1} limit[i,j,p]; 
        {p in 1..npiece[i,j]} rate[i,j,p]>> Trans[i,j];

subject to Supply {i in ORIG}:  
   sum {j in DEST} Trans[i,j] = supply[i];

subject to Demand {j in DEST}:  
   sum {i in ORIG} Trans[i,j] = demand[j];

