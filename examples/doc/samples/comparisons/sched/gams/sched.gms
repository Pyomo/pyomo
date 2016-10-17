set
 p 'participants' /p1*p4/
 t 'presenter'    /t1*t5/
 n 'talk number'  /n1*n1/
 s 'timeslot'     /slot1*slot2/

;

** Try reducing this to 1?
scalar rooms 'number of parallel sessions' /5/
scalar capacity 'capacity of people for each talk' /5/

set talk(t,n) 'each presenter can give several talks' /
(t1*t5).(n1)
/;

scalar numtalk 'total number of talks';
numtalk=card(talk);
display talk,numtalk;


** actually the count is 21
** abort$(numtalk<>22) "oops, check your sets";

parameter ntalk(t) 'numbers of times a talk is given';
ntalk(t) = sum(talk(t,n),1);
display ntalk;

parameter italk(t,n) 'number all talks (all 9 of them)';
scalar counter/0/;
loop(talk(t,n),
  counter=counter+1;
  italk(talk)=counter;
);
display italk;

table preferences(p,t) 'preference = 0..8, 8 is highest preference'

        t1    t2    t3    t4    t5

p1       1           2
p2             1     2
p3                   1     2
p4                         1     2
;

display "after",preferences

variable z 'objective variable';
binary variables
  x(t,p,s) 'assign talk/person/slot'
  xts(t,s) 'assign talk/slot'
  xtp(t,p) 'assign talk/person'
;

* we can relax xts and xtp
xts.prior(t,s) = INF;
xtp.prior(t,p) = INF;


equations
  defxts1(t,s)   'calculate xts: all x=0 => xts = 0'
  defxts2(t,p,s) 'calculate xts: any x=1 => xts = 1'

  defxtp(t,p)    'calculate xtp'

  talkcount(t)   'talk can be repeated ntalk times'
  slotcount(s)   'up to 5 talks each period'
  peoplecount(t) 'up to 5 people per talk'

  listen(p,s)    'p can visit only one talk in each time period'
  zdef           'objective: maximize preferences'
;

* x(t,p,s)=0 for all p => xts(t,s) = 0
defxts1(t,s).. xts(t,s) =L= sum(p, x(t,p,s));
* any x(t,p,s)=1 => xts(t,s) = 1
defxts2(t,p,s).. xts(t,s) =G= x(t,p,s);

* p visits talk t in any s
defxtp(t,p).. xtp(t,p) =E= sum(s, x(t,p,s));

* each talk happens exactly ntalk times
talkcount(t).. sum(s, xts(t,s)) =E= ntalk(t);

* talk can only hold up to 5 people
peoplecount(t).. sum(p, xtp(t,p)) =L= capacity;

* timeslot can only hold up to 5 talks
slotcount(s).. sum(t, xts(t,s)) =L= rooms;

* p can only visit one talk in each time period
listen(p,s)..  sum(t,x(t,p,s)) =E= 1;

* we don't allow here (t,p) combinations without preference
xtp.fx(t,p)$(preferences(p,t)=0) = 0;

display preferences;

* objective: maximize
zdef.. z =e= sum((t,p), preferences(p,t) * xtp(t,p));


model scheduling /all/;


option mip=cplex;
scheduling.optfile=1;
scheduling.iterlim=100000;
scheduling.optcr=0;
solve scheduling using mip maximizing z;


$onecho > cplex.opt
threads 4
$offecho
