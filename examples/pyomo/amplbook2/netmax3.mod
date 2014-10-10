
set INTER;  # intersections

param entr symbolic in INTER;           # entrance to road network
param exit symbolic in INTER, <> entr;  # exit from road network

set ROADS within (INTER diff {exit}) cross (INTER diff {entr});

param cap {ROADS} >= 0;  # capacities of roads

node Intersection {k in INTER};

arc Traff_In >= 0, to Intersection[entr];
arc Traff_Out >= 0, from Intersection[exit];

arc Traff {(i,j) in ROADS} >= 0, <= cap[i,j],
   from Intersection[i], to Intersection[j];

maximize Entering_Traff: Traff_In;

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

