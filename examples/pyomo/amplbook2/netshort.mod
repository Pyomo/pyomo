set INTER;  # intersections

param entr symbolic in INTER;           # entrance to road network
param exit symbolic in INTER, <> entr;  # exit from road network

set ROADS within (INTER diff {exit}) cross (INTER diff {entr});

param time {ROADS} >= 0;         # times to travel roads
var Use {(i,j) in ROADS} >= 0;   # 1 iff (i,j) in shortest path

minimize Total_Time: sum {(i,j) in ROADS} time[i,j] * Use[i,j];

subject to Start:  sum {(entr,j) in ROADS} Use[entr,j] = 1;

subject to Balance {k in INTER diff {entr,exit}}:
   sum {(i,k) in ROADS} Use[i,k] = sum {(k,j) in ROADS} Use[k,j];

data;

set INTER := a b c d e f g ;

param entr := a ;
param exit := g ;

param:  ROADS:  time :=
         a b     50,	a c    100
         b d     40,	b e     20
         c d     60,	c f     20
         d e     50,	d f     60
         e g     70,	f g     70 ;

