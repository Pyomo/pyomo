set INTER;  # intersections

param entr symbolic in INTER;           # entrance to road network
param exit symbolic in INTER, <> entr;  # exit from road network

set ROADS within (INTER diff {exit}) cross (INTER diff {entr});

param cap {ROADS} >= 0;                        # capacities
var Traff {(i,j) in ROADS} >= 0, <= cap[i,j];  # traffic loads

maximize Entering_Traff: sum {(entr,j) in ROADS} Traff[entr,j];

subject to Balance {k in INTER diff {entr,exit}}:
  sum {(i,k) in ROADS} Traff[i,k] = sum {(k,j) in ROADS} Traff[k,j];

data;

set INTER := a b c d e f g ;

param entr := a ;
param exit := g ;

param:  ROADS:  cap :=
         a b     50,	a c    100
         b d     40,	b e     20
         c d     60,	c f     20
         d e     50,	d f     60
         e g     70,	f g     70 ;
