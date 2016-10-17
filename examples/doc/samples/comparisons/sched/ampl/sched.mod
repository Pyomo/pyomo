
set TASKS;
set PEOPLE;
set SLOTS;

param amt {TASKS,PEOPLE} >= 0;
param nrooms {SLOTS} >= 0;
param ntasks {TASKS} >= 0;
param minp {TASKS} >= 0;
param maxp {i in TASKS} >= minp[i];

var x {t in TASKS, p in PEOPLE, s in SLOTS} binary;
var xts {t in TASKS, s in SLOTS} binary;
var xtp {t in TASKS, p in PEOPLE} binary;

subject to rule1 {t in TASKS, s in SLOTS}:
   sum {p in PEOPLE} x[t,p,s] >= xts[t,s];
subject to rule2 {t in TASKS, p in PEOPLE, s in SLOTS}:
   x[t,p,s] <= xts[t,s];
subject to rule3 {t in TASKS, p in PEOPLE}:
   sum {s in SLOTS} x[t,p,s] == xtp[t,p];
subject to rule4 {t in TASKS}:
   sum {s in SLOTS} xts[t,s] == ntasks[t];
subject to rule5 {t in TASKS}:
   minp[t] <= sum {p in PEOPLE} xtp[t,p] <= maxp[t];
subject to rule6 {s in SLOTS}:
   sum {t in TASKS} xts[t,s] <= nrooms[s];
subject to rule7 {p in PEOPLE, s in SLOTS}:
   sum {t in TASKS} x[t,p,s] == 1;

maximize z:  sum {t in TASKS, p in PEOPLE} amt[t,p] * xtp[t,p];

