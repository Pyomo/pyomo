#model morphEpanet3.mod;
#option solver ipopt;
#option ipopt_options "imaxiter=1000000 ISCALE=4 DMU0=1000 dtol=1e-6";
#solve;
#for {id in pipeID} {
#        display id, length[id], diam[id]*12;
#}

################################################################################
## morph.mod ###*****###########################################################
################################################################################
## Jon Berry
## 10/1/2004
################################################################################
set Pipes dimen 7;	# id,v1,v2,length, diameter,flow,headloss/1000ft
set Valves dimen 6;	# id,v1,v2,diameter,flow,headloss
set Pumps dimen 5;	# id,v1,v2,flow,hl, x1,y1
set DA dimen 3;		# directed connections (id,v1,v2)
set A dimen 3;
set V ordered;
set SourceID within V;
set BE ordered;
set ID;

param pipe := 1;
param pump := 2;
param valve := 3;
param typ {ID};
param source {ID};
param sink {ID};
param demand {V};

param a >= 0;                   
param absTolerance >= 0;                   
param hlTolerance >= 0;                   
param ratioTolerance >= 0;                   
param init_flow {ID};
param init_flow_mag {ID};
param init_flow_dir {ID};

param p symbolic;
param cur symbolic;
param nxt symbolic;
param pos;
param tmp  symbolic;
param v1  symbolic;
param v2  symbolic;
param v3  symbolic;
param v4  symbolic;
param v5  symbolic;
param id1 symbolic;
param id2 symbolic;
param id3 symbolic;
param ty  symbolic;
param done symbolic;
param noid symbolic := 0;

#
# In the EPANET input, there could be multiple pumps connecting any given 
# i and j in parallel.  We could have used a script to combine these into
# a single pump connection, but the pump curves might be quite different,
# so we deal with the pumps individually.

#data Arcs.dat;
let V := {
24,
25,
26,
27,
20,
21,
22,
23,
28,
29,
4,
8,
59,
58,
55,
54,
57,
56,
51,
50,
53,
52,
88,
89,
82,
83,
80,
81,
86,
87,
84,
85,
3,
7,
39,
38,
33,
32,
31,
30,
37,
36,
35,
34,
60,
61,
62,
63,
64,
65,
66,
67,
68,
69,
2,
6,
91,
90,
93,
92,
95,
94,
97,
96,
11,
10,
13,
12,
15,
14,
17,
16,
19,
18,
48,
49,
46,
47,
44,
45,
42,
43,
40,
41,
1,
5,
9,
77,
76,
75,
74,
73,
72,
71,
70,
79,
78};

let SourceID := {
93,
95,
94,
97,
96};

let demand[24] := 0.052722366666666673;
let demand[25] := 0.0;
let demand[26] := 0.12766121666666669;
let demand[27] := 0.017581550000000001;
let demand[20] := 0.0;
let demand[21] := 0.12429643333333334;
let demand[22] := 0.0;
let demand[23] := 0.13615116666666668;
let demand[28] := 0.029414000000000003;
let demand[29] := 0.018517450000000001;
let demand[4] := 3.6477816666666669;
let demand[8] := 0.0;
let demand[59] := 9.8915716666666675;
let demand[58] := 0.13320976666666667;
let demand[55] := 0.0;
let demand[54] := 0.21293953333333338;
let demand[57] := 0.35628821666666671;
let demand[56] := 0.050872849999999997;
let demand[51] := 0.0;
let demand[50] := 0.076587816666666669;
let demand[53] := 0.24455958333333336;
let demand[52] := 0.3222392833333334;
let demand[88] := 0.0;
let demand[89] := 0.0;
let demand[82] := 0.12059740000000001;
let demand[83] := 0.0;
let demand[80] := 0.072131149999999991;
let demand[81] := 0.16280203333333337;
let demand[86] := 0.0;
let demand[87] := 0.0;
let demand[84] := 0.0;
let demand[85] := 0.0;
let demand[3] := 0.0;
let demand[7] := 0.0;
let demand[39] := 0.0;
let demand[38] := 0.028121566666666667;
let demand[33] := 0.43140533333333336;
let demand[32] := 0.080821650000000023;
let demand[31] := 0.025536700000000006;
let demand[30] := 0.082492900000000022;
let demand[37] := 0.047173816666666674;
let demand[36] := 0.12338281666666667;
let demand[35] := 0.15464633333333336;
let demand[34] := 0.13189505000000001;
let demand[60] := 0.0;
let demand[61] := 0.19515743333333335;
let demand[62] := 0.0;
let demand[63] := 0.20719043333333337;
let demand[64] := 0.0;
let demand[65] := 0.0026071500000000004;
let demand[66] := 0.025893233333333335;
let demand[67] := 0.041625266666666667;
let demand[68] := 0.27526601666666667;
let demand[69] := 0.072309416666666682;
let demand[2] := 1.3815666666666668;
let demand[6] := 0.0;
let demand[91] := 0.0;
let demand[90] := 0.0;
let demand[93] := 1.0041538500000002;
let demand[92] := 0.0;
let demand[95] := 4.9965918333333343;
let demand[94] := -0.74268121666666675;
let demand[97] := -0.0;
let demand[96] := -29.280545116666669;
let demand[11] := 0.39773521666666672;
let demand[10] := 0.5671776833333334;
let demand[13] := 0.16315856666666667;
let demand[12] := 0.40421966666666675;
let demand[15] := 0.42382900000000007;
let demand[14] := 0.69096159999999995;
let demand[17] := 0.15555995000000003;
let demand[16] := 0.059741616666666671;
let demand[19] := 0.52590895000000004;
let demand[18] := 0.3514750166666667;
let demand[48] := 0.0;
let demand[49] := 0.0;
let demand[46] := 0.0;
let demand[47] := 0.0;
let demand[44] := 0.0;
let demand[45] := 0.17369858333333338;
let demand[42] := 0.0;
let demand[43] := 0.11747773333333335;
let demand[40] := 0.0077546000000000004;
let demand[41] := 0.043474783333333343;
let demand[1] := 0.0;
let demand[5] := 0.0;
let demand[9] := 0.0;
let demand[77] := 0.012968900000000002;
let demand[76] := 0.0;
let demand[75] := 0.13320976666666667;
let demand[74] := 0.046616733333333348;
let demand[73] := 0.049201599999999998;
let demand[72] := 0.19163666666666668;
let demand[71] := 0.068075583333333342;
let demand[70] := 0.12338281666666667;
let demand[79] := 0.0;
let demand[78] := 0.2101541166666667;

let Pipes := {
(24,23,24,3240,24/12,5.862009650000001,0.52090)
,(25,3,24,785,20/12,-4.9965918333333343,0.94168)
,(26,24,25,900,24/12,0.81269544999999999,0.01350)
,(27,25,26,6480,16/12,0.12766121666666669,0.00319)
,(20,21,20,2050,8/12,0.81971470000000013,2.87821)
,(21,21,19,2000,30/12,21.856919466666668,1.73201)
,(22,22,21,1500,30/12,29.280545116666669,2.97540)
,(23,21,23,930,24/12,6.4795922333333342,0.62696)
,(28,25,27,2750,8/12,0.68505651666666667,2.06543)
,(29,27,28,2050,8/12,0.66745268333333341,1.96853)
,(4,96,7,1231,24/12,29.280545116666669,8.93307)
,(8,12,13,1470,12/12,0.07152950000000001,0.00445)
,(59,60,50,1325.,12/12,0.75663058333333344,0.34485)
,(58,49,61,4530.,12/12,1.2068653333333335,0.81776)
,(55,5,46,1190,12/12,-1.0041538500000002,0.58099)
,(54,46,48,210,12/12,-0.13953823333333334,0.01519)
,(57,48,50,510,8/12,0.52684485000000003,1.27114)
,(56,50,49,99.9,8/12,1.2068653333333335,5.88708)
,(51,4,47,30,24/12,-3.6477816666666669,0.21681)
,(50,90,43,760,24/12,-4.3657729500000011,0.30190)
,(53,45,46,30,12/12,0.86463790000000007,0.44049)
,(52,47,45,30,12/12,1.0383364833333335,0.61841)
,(115,9,22,45500,30/12,29.280545116666669,3.01384)
,(114,89,90,1290,8/12,0.32032291666666673,0.50654)
,(88,74,72,790,8/12,0.15850134999999999,0.13814)
,(89,74,75,510,12/12,-0.030817850000000001,0.00095)
,(111,91,61,645,12/12,-0.0021837666666666669,0.00003)
,(110,91,92,2230,8/12,0.24725586666666668,0.31396)
,(113,58,92,300,12/12,-0.18377065000000001,0.02526)
,(112,38,87,1200,30/12,16.795928800000002,1.06389)
,(82,68,69,1660,16/12,0.26379010000000003,0.01215)
,(83,69,70,2050,14/12,0.12338281666666667,0.00573)
,(80,66,67,990,16/12,0.66301830000000017,0.06661)
,(81,67,68,4285,16/12,0.53905611666666664,0.04545)
,(86,72,73,1960,12/12,0.049201599999999998,0.00223)
,(87,66,74,2080,12/12,0.17430023333333333,0.02291)
,(84,69,71,1560,12/12,0.068075583333333342,0.00405)
,(85,67,72,2200,8/12,0.08233691666666669,0.04129)
,(3,94,6,99,99/12,0.74268121666666675,0.00001)
,(7,10,12,2540,12/12,-0.94093603333333342,0.51606)
,(108,89,52,646,12/12,1.0432611000000001,0.62459)
,(109,47,90,260,24/12,-4.6861181499999995,0.34414)
,(102,84,86,1400,8/12,0.52200936666666675,1.24935)
,(103,83,85,1400,8/12,0.56452596666666666,1.44414)
,(100,20,83,645,8/12,1.0865353333333334,4.84706)
,(101,83,84,350,8/12,0.52200936666666675,1.24933)
,(106,87,88,1580,8/12,0.46327050000000009,1.00200)
,(107,54,88,1170,12/12,-0.15974921666666667,0.01951)
,(104,18,85,645,12/12,0.33012758333333342,0.07448)
,(105,85,86,350,12/12,0.89465355000000002,0.47016)
,(39,19,35,2080,30/12,17.716898966666669,1.17434)
,(38,19,33,3460,12/12,1.0327879333333334,0.61312)
,(33,31,30,2200,12/12,0.8445383333333335,0.42260)
,(32,30,28,3510,12/12,0.7620231500000002,0.34946)
,(31,2,29,1650,8/12,-1.3815666666666668,7.56013)
,(30,29,28,1400,8/12,-1.4000841166666667,7.74856)
,(37,23,34,4560,8/12,0.48140913333333335,1.07577)
,(36,33,34,1170,12/12,-0.34953636666666671,0.08276)
,(35,32,33,1020,8/12,-0.95089668333333344,3.78789)
,(34,31,32,880,12/12,-0.87005275000000004,0.44655)
,(60,52,48,1350,12/12,0.66636080000000009,0.27260)
,(61,52,51,500,8/12,0.35818230000000006,0.62284)
,(62,42,89,646,12/12,1.3635840166666668,1.02484)
,(63,53,51,2560,12/12,0.39844828333333343,0.10542)
,(64,88,52,1230,12/12,0.30352128333333339,0.06375)
,(65,53,54,520,12/12,-0.54302254999999999,0.18681)
,(66,54,55,360,12/12,-0.55986875000000003,0.19767)
,(67,37,55,2300,8/12,0.55986875000000003,1.42208)
,(68,56,53,1150,12/12,0.099963033333333354,0.00822)
,(69,15,56,2790,12/12,0.15085816666666668,0.01755)
,(2,93,5,99,99/12,-1.0041538500000002,0.00002)
,(6,10,11,1350,16/12,0.3737360666666667,0.02312)
,(99,79,80,1450,12/12,-0.10158971666666668,0.00845)
,(98,82,80,1100,8/12,0.17372086666666667,0.16370)
,(91,76,77,2200,12/12,0.012968900000000002,0.00018)
,(90,75,76,35,12/12,-0.054683300000000004,0.00260)
,(93,75,79,430,12/12,-0.10934431666666669,0.00966)
,(92,76,78,445,10/12,-0.067652199999999996,0.00913)
,(95,78,82,1390,10/12,-0.28556091666666666,0.13836)
,(94,78,79,10,12/12,0.0077546000000000004,0.00015)
,(97,82,81,1100,10/12,0.16280203333333337,0.04902)
,(96,6,82,925,10/12,0.74268121666666675,0.81006)
,(11,17,15,1160,12/12,1.109420316666667,0.69984)
,(10,14,15,2000,12/12,-0.71493846666666661,0.31061)
,(13,17,16,2000,8/12,0.27629105000000004,0.38545)
,(12,15,16,1680,12/12,-0.18018303333333335,0.02435)
,(15,16,54,1660,12/12,0.036344116666666669,0.00133)
,(14,13,17,1950,8/12,-0.091629066666666661,0.05025)
,(17,17,18,2180,12/12,-1.6329226666666667,1.43042)
,(16,86,12,2725,12/12,1.4166629166666669,1.09988)
,(19,20,18,1870,12/12,2.3145029833333339,2.72733)
,(18,19,20,730,12/12,2.5813236166666669,3.33741)
,(117,8,9,1,30/12,-0.0,0.00000)
,(116,7,8,1,30/12,0.0,0.00000)
,(48,42,43,1270,30/12,14.925599500000002,0.85515)
,(49,43,44,50,30/12,10.442348816666669,0.44183)
,(46,41,42,60,8/12,-0.043474783333333343,0.01270)
,(47,51,60,99.9,8/12,0.75663058333333344,2.48180)
,(44,39,40,490,14/12,0.0077546000000000004,0.00006)
,(45,87,42,590,30/12,16.332658300000002,1.01022)
,(42,37,38,430,30/12,16.83182725,1.06811)
,(43,38,39,150,14/12,0.0077546000000000004,0.00006)
,(40,35,36,2910,30/12,17.562252633333337,1.15544)
,(41,36,37,2000,30/12,17.438869816666671,1.14047)
,(1,95,3,99,99/12,-4.9965918333333343,0.00032)
,(5,1,10,14200,18/12,0.0,0.00000)
,(9,11,14,3940,16/12,-0.023999150000000004,0.00011)
,(77,62,64,510,12/12,0.86579663333333345,0.44219)
,(76,92,63,1430,12/12,0.063462933333333346,0.00360)
,(75,63,62,450,12/12,0.86579663333333345,0.44219)
,(74,61,63,1200,12/12,1.0095241333333336,0.58774)
,(73,57,91,725,12/12,0.24507210000000004,0.04296)
,(72,58,59,120,24/12,9.8915716666666675,1.37174)
,(71,57,58,630,24/12,9.8409885000000017,1.17278)
,(70,44,57,4000,24/12,10.442348816666669,1.30878)
,(79,65,66,1210,16/12,0.86318948333333334,0.10842)
,(78,64,65,885,12/12,0.86579663333333345,0.44219)};
let Pumps := {
(119,7,9,29.280545116666669,-93.59961)
,(118,97,1,0.0,0.00000)};
let Valves := {};

set pipeID := setof {(id,n1,n2,l,d,f,h) in Pipes} id;
set pumpID := setof {(id,n1,n2,flw,hl) in Pumps} id;
set valveID := setof {(id,i,j,d,f,h) in Valves} id;

param init_length {pipeID};
param init_diam {pipeID};
param init_hl {ID};
param init_tau {pipeID} >= 0;
param proport_const {pipeID};
param pump_a {pumpID};		# pump curve   h_l = a - b * q^c
param pump_b {pumpID};
param pump_c;
param alpha;
param maxexp;

param expvalues {pipeID};

option randseed 1;	# make a seed param

let alpha := 0.1;

let absTolerance := 0.1;
let hlTolerance := 0.1;
let ratioTolerance := 0.1;

#for {id in pipeID} {
#	let expvalues[id] := Exponential();	
#}
#
#let maxexp := max {id in pipeID} expvalues[id];
#for {id in pipeID} {
#	let alpha[id] := (expvalues[id]/maxexp) - 0.5;
#	# Now alpha ranges between -0.5 and 0.5, with the smaller values favored
#	display id,alpha[id];
#	let beta[id] := -2 * alpha[id];
#}

let a := 0.78539816;          # pi/4


let DA := setof {(id,i,j,l,d,f,h) in Pipes} (id,i,j)
	  union setof {(id,i,j,d,f,h) in Valves} (id,i,j)
	  union setof {(id,i,j,f,h) in Pumps} (id,i,j);

let ID := setof {(id,i,j) in DA} id;

for {(idd,i,j,l,d,f,h) in Pipes} {
	let typ[idd] := pipe;
	let source[idd] := i;
	let sink[idd] := j;
}

for {(id,i,j,d,f,h) in Valves} {
	let typ[id] := valve;
	let source[id] := i;
	let sink[id] := j;
}

for {(id,i,j,f,h) in Pumps} {
	let typ[id] := pump;
	let source[id] := i;
	let sink[id] := j;
}

#
# We'll consider each direction of each connection in our spanning
# tree computation, but we will preserve the original id so that the
# original source and destination can be retrieved.
#
let A := DA union setof {(id,i,j) in DA} (id, j,i);

#set V ordered = union{(id,i,j) in A} {i,j};

#
# Closed pipes and inactive pumps during the instant of this
# morphing process aren't modeled.  This means that an otherwise
# connected network might be temporarily disconnected.  We'll 
# find a spanning forest in this network, remembering which spanning
# tree contains which vertices.  This is necessary since there
# will be a set of constraints guaranteeing that the algebraic
# head loss along a path between two sources (tanks, reservoirs)
# equals the difference in head between those sources.  They 
# might not be in the same spanning tree, in which case we do not
# generate the constraint.
#
param stnum;	
param stID {V};	

set FC {ID} ordered;
set S2SPath {SourceID, SourceID} ordered;
set uV;

set T ordered;          # set of vertices in current spanning forest so far
set ST;         	# spanning forest
set cross dimen 3;	# crossing edges from forest to frontier
set cross_j dimen 3;	# kludge
set cross_min dimen 3;  # kludge
set idset ordered;	# kludge
set induced dimen 3;	# edges of subgraph induced by current forest

let uV := V;		# find a ST of the subgraph induced by uV
let ST := {};
let stnum := 0;
let cross := {};
let induced := {};
let T := {};
set   PathFromRoot {V} ordered;
set   IDFromRoot {V} ordered;

repeat while card(T) < card(V)	# find spanning forest
{
  let stnum := stnum + 1;
  let uV := uV diff T;
  let p := min {v in uV} v;
  let T := if card(uV) > 0 then T union {p} else T;
  let done := if card(uV) == 0 then "true" else "false";
  let PathFromRoot[p] := {p};
  let IDFromRoot[p] := {noid};
  repeat while done != "true"	# find spanning tree of subgraph (whole graph
				# if connected)
  {
	# find edges crossing cut

	let p := last(T);
	let cross := cross union setof {(id,i,p) in A: i not in T} (id,p,i);
	let cross := cross union setof {(id,p,j) in A: j not in T} (id,p,j);
	
	# select one of them

	let tmp := min {(id,i,j) in cross} j;
	let cross_j := setof {(id,i,j) in cross: j <= tmp} (id,i,j);
	let tmp := min {(id,i,j) in cross_j} i;
	let cross_min := setof {(id,i,j) in cross_j: i <= tmp} (id,i,j);
	let cur := min {(id,i,j) in cross_min} i;  # we still care about
	let nxt := min {(id,i,j) in cross_min} j;  # i and j to construct
	if (card(cross_min)>0) 
		then { let idset := setof {(id,i,j) in cross_min} id;
		       let id1  := first(idset);
		     }
		else   let id1 := noid;

	# add it to the spanning tree (or quit)

	if (card(cross_min) <= 0)
	then {
		let done := "true";
	} else {
		let ST := ST union {id1};
		let T := T union {nxt};
		let PathFromRoot[nxt] := PathFromRoot[cur] union {nxt};
		let IDFromRoot[nxt] := IDFromRoot[cur] union {id1};
	}

	# update the set of edges crossing the new cut 

	let induced := setof {(id,i,j) in A: i in T && j in T} (id,i,j);
	let induced := induced union 
			setof {(id,i,j) in A:i in T && j in T} (id,j,i);
	let cross := cross diff induced;
  }
  for {v in uV intersect T} {
	let stID[v] := stnum;
  }
}
	
# Identify "back edges" (w.r.t. imaginary dfs)

let BE := ID diff ST;

# Find fundamental cycles by dynamic programming


for {id in BE} {
	let v1 := source[id];
	let v2 := sink[id];
	let FC[id] := {};
	let pos := 1;
	let v3 := first(PathFromRoot[v1]);
	let v4:= first(PathFromRoot[v2]);
	let id1 := first(IDFromRoot[v1]);
	let id2 := first(IDFromRoot[v2]);
	let v5 := v3;   # placeholder to let us trace one of the paths backwards
	repeat while ((v3 == v4) && 
		      pos < min(card(PathFromRoot[v1]),
				card(PathFromRoot[v2])))
	{
		let pos := pos + 1;
		let v5 := v3;   
		let v3 := next(v3, PathFromRoot[v1]);
		let id1 := next(id1, IDFromRoot[v1]);
		let v4 := next(v4, PathFromRoot[v2]);
		let id2 := next(id2, IDFromRoot[v2]);
	};
	if ((v3 == v4) && (v3 != last(PathFromRoot[v1])))
	   # reached the end of the line on the v2 branch   (v3 == v2)
           then {
		let v5 := v3;
		let v3 := next(v3, PathFromRoot[v1]);
		let id1 := next(id1, IDFromRoot[v1]);
	   } else if ((v3 == v4) && (v4 != last(PathFromRoot[v2])))
	   # reached the end of the line on the v1 branch   (v3 == v1)
           then {
		let v5 := v4;
		let v4 := next(v4, PathFromRoot[v2]);
		let id2 := next(id2, IDFromRoot[v2]);
	   };

	if v3 != v5
	    then { # trace this side, then add the back edge
		   let FC[id] := FC[id] union {id1};
		   repeat
	           {
			if v3 != last(PathFromRoot[v1])
			then {
				let id1:= next(id1, IDFromRoot[v1]);
				let v3 := next(v3, PathFromRoot[v1]);
		   		let FC[id] := FC[id] union {id1};
			}
		   } until v3 == last(PathFromRoot[v1]);
	    }

	let FC[id] := FC[id] union {id};

	# trace this side backward; we'll have a correct cycle order
	let v4 := last(PathFromRoot[v2]);
	let id2:= last(IDFromRoot[v2]);
	repeat while v4 != v5 
	{
		let FC[id] := FC[id] union {id2};
		let id2 := prev(id2, IDFromRoot[v2]);
		let v4 := prev(v4, PathFromRoot[v2]);
	}
}

#
# We have the cycle edges in an acceptable order, but we still need to
# establish which edges get traversed from source to sink and which get
# traversed from sink to source.  This will help us express the conservation
# of energy constraint (head loss around any fundamental cycle is 0).
#

param fcArcLeader {id in ID, eid in FC[id]};
param longestFundCycleID;
param longestFundCycleLength;

for {id in BE} {
   let id1 := first(FC[id]);
   if (ord(source[id1],T) < ord(sink[id1],T))
  	then { let nxt := sink[id1]; let fcArcLeader[id,id1] := source[id1]; }
  	else { let nxt := source[id1];let fcArcLeader[id,id1] := sink[id1]; }
   repeat while (id1 != last(FC[id])) {
       let id1 := next(id1,FC[id]);
       if (source[id1] == nxt)
	  then { let nxt := sink[id1]; let fcArcLeader[id,id1] := source[id1];}
	  else { let nxt := source[id1];let fcArcLeader[id,id1] := sink[id1];}
   }
}

# Find paths between pairs of source nodes (tanks and reservoirs)


for {v in SourceID, w in SourceID: v < w && stID[v] == stID[w]} {
	let S2SPath[v, w] := {};
	let pos := 1;
	let v3 := first(PathFromRoot[v]);
	let v4:= first(PathFromRoot[w]);
	let id1 := first(IDFromRoot[v]);
	let id2 := first(IDFromRoot[w]);
	let v5 := v3;   # placeholder to let us trace one of the paths backwards
	repeat while ((v3 == v4) && 
		      pos < min(card(PathFromRoot[v]),
				card(PathFromRoot[w])))
	{
		let pos := pos + 1;
		let v5 := v3;   
		let v3 := next(v3, PathFromRoot[v]);
		let id1 := next(id1, IDFromRoot[v]);
		let v4 := next(v4, PathFromRoot[w]);
		let id2 := next(id2, IDFromRoot[w]);
	};
	if ((v3 == v4) && (v3 != last(PathFromRoot[v])))
	   # reached the end of the line on the w branch   (v3 == w)
           then {
		let v5 := v3;
		let v3 := next(v3, PathFromRoot[v]);
		let id1 := next(id1, IDFromRoot[v]);
	   } else if ((v3 == v4) && (v4 != last(PathFromRoot[w])))
	   # reached the end of the line on the v branch   (v3 == v)
           then {
		let v5 := v4;
		let v4 := next(v4, PathFromRoot[w]);
		let id2 := next(id2, IDFromRoot[w]);
	   };

	# trace this side backward; we're moving from v up to v5, the
	# lowest common ancestor of v and w.
	let v3 := last(PathFromRoot[v]);
	let id1:= last(IDFromRoot[v]);
	repeat while v3 != v5 
	{
		let S2SPath[v,w] := S2SPath[v,w] union {id1};
		let id1 := prev(id1, IDFromRoot[v]);
		let v3 := prev(v3, PathFromRoot[v]);
	}

	# we've reached the lowest common ancestor of v and w; 
	# now proceed forward down to w.

	let v4 := v5;
	repeat while v4 != w 
	{
		let S2SPath[v,w] := S2SPath[v,w] union {id2};
		let v4 := next(v4, PathFromRoot[w]);
		if (v4 == w) then break;
		let id2 := next(id2, IDFromRoot[w]);
	}

}

#
# We have the path edges in an acceptable order, but we still need to
# establish which edges get traversed from source to sink and which get
# traversed from sink to source.  This will help us express the source to
# source head constraint (algebraic head loss between sources equals the
# difference in their heads).
#

param s2sPathLeader {v in SourceID, w in SourceID, 
		     eid in S2SPath[v, w]: v < w && stID[v] == stID[w] };

for {v in SourceID, w in SourceID: v < w && stID[v] == stID[w]} {
   let id3 := first(S2SPath[v, w]);
   if (ord(source[id3],T) < ord(sink[id3],T))
  	then {   let nxt := sink[id3]; 
	         let s2sPathLeader[v,w,id3] := source[id3]; 
	} else { let nxt := source[id3];
		 let s2sPathLeader[v,w,id3] := sink[id3]; }
   repeat while (id3 != last(S2SPath[v,w])) {
       let id3 := next(id3,S2SPath[v,w]);
       if (source[id3] == nxt)
	  then {   let nxt := sink[id3]; 
		   let s2sPathLeader[v,w,id3]:=source[id3];
	  } else { let nxt := source[id3];
		   let s2sPathLeader[v,w,id3] := sink[id3];}
   }
}

#
# End of the computation phase; traditional AMPL model below
#



var length {ID} >= 0;
var tau {pipeID} >= 0;
var diam {pipeID} >= 0;
#var flow {ID};
var flow_mag {ID} >= 0;
var flow_dir {ID};
var head_loss {ID} >= 0;
var head {SourceID} >= 0;
#var cycleLengthSQDiff {BE} >= 0;

for {(id,v,w,flw,hl) in Pumps} {
	let length[id] := 0;
	if (hl == 0.0) then {
		let init_hl[id] := 1e-04;
	} else {
		let init_hl[id] := hl;
	}
	if (flw == 0.0 || flw == -0.0) then {
		let init_flow[id] := 1e-04;
	} else {
		let init_flow[id] := flw;  # assume flw already in ft^3/s
	}
	if flw < 0 then {
		let init_flow_dir[id] := -1;
		let init_flow_mag[id] := -1*init_flow[id];
	} else {
		let init_flow_dir[id] := 1;
		let init_flow_mag[id] := init_flow[id];
	}
}

for {(id,i,j,d,flw,hl) in Valves} {
	let length[id] := 0;
	if (hl == 0.0) then {
		let init_hl[id] := 1e-04;
	} else {
		let init_hl[id] := hl;
	}
	if (flw == 0.0 || flw == -0.0) then {
		let init_flow[id] := 1e-04;
	} else {
		let init_flow[id] := flw;  # assume flw already in ft^3/s
	}
	if flw < 0 then {
		let init_flow_dir[id] := -1;
		let init_flow_mag[id] := -1*init_flow[id];
	} else {
		let init_flow_dir[id] := 1;
		let init_flow_mag[id] := init_flow[id];
	}
}

for {(id,v,w,len,di,flw,hl_per_1000) in Pipes} {
	let init_length[id] := len;
	let init_diam[id] := di;
	if (hl_per_1000 == 0.0) then {
		let init_hl[id] := 1e-04;
	} else {
		let init_hl[id] := hl_per_1000;
	}
	if (flw == 0.0 || flw == -0.0) then {
		let init_flow[id] := 1e-04;
		let proport_const[id]:= 1e-04;
	} else {
		let init_flow[id] := flw;  # assume flw already in ft^3/s
	}
	if flw < 0 then {
		let init_flow_dir[id] := -1;
		let init_flow_mag[id] := -1*init_flow[id];
	} else {
		let init_flow_dir[id] := 1;
		let init_flow_mag[id] := init_flow[id];
	}
	if (init_flow_mag[id] <= 0.0) then {
		let init_tau[id] := 100;
	} else {
		let init_tau[id] := a*len*(di^2)/(init_flow_mag[id]);
		let proport_const[id]:=len/1000 * init_hl[id] 
				        * (di^5)/(len*(init_flow_mag[id]^2));
	}
	##### initial values
	let head_loss[id] := init_hl[id];
	let length[id] := init_length[id];
	let diam[id] := init_diam[id];
	#let flow[id] := init_flow[id];
	let flow_mag[id] := init_flow_mag[id];
	let flow_dir[id] := init_flow_dir[id];
	let tau[id] := init_tau[id];
};

let longestFundCycleID := first(BE);
let longestFundCycleLength  := sum {idd in FC[longestFundCycleID]} length[idd];
for {id in BE} {
	let v1  := sum {idd in FC[id]} length[idd];
	if v1 > longestFundCycleLength 
	   then {  let longestFundCycleID := id; 
		   let longestFundCycleLength := v1; 
		}
}

let pump_c := 2;		# log_2 ( (4/3) / (1/3) )

for {(id,v,w,flw,hl) in Pumps} {
	let pump_a[id] := 4/3 * hl;
	if (flw > 0.0) then {
		let pump_b[id] := (1/3 * hl)/ (flw^pump_c);
	} else
		let pump_b[id] := 0.0;
	display id, pump_a[id], pump_b[id];
}

#minimize Objective: 
#	sum {id in BE} cycleLengthSQDiff[id];
minimize Objective: 
	sum {id in BE} (longestFundCycleLength -
	                       sum {idd in FC[id]} length[idd])^2;

#subject to flowDefinition {(id,i,j) in DA}:
#	flow[id] == flow_dir[id] * flow_mag[id];

subject to flowConservationU {i in V}:
	sum {(id,j,i) in DA} (flow_dir[id] *flow_mag[id])
        - ((sum {(id,i,j) in DA} (flow_dir[id]*flow_mag[id])) + demand[i])
				<= absTolerance;

subject to flowConservationL {i in V}:
	sum {(id,j,i) in DA} (flow_dir[id] *flow_mag[id])
        - ((sum {(id,i,j) in DA} (flow_dir[id]*flow_mag[id])) + demand[i]) 
				>= -absTolerance;

subject to pipeHeadLossDefinitionU {id in pipeID}:
	proport_const[id]*(length[id]/diam[id]^5) *flow_mag[id]^2
		<= head_loss[id] + head_loss[id] * ratioTolerance;

subject to pipeHeadLossDefinitionL {id in pipeID}:
	head_loss[id] <=
	(proport_const[id]*(length[id]/diam[id]^5) *flow_mag[id]^2) +  
	(proport_const[id]*(length[id]/diam[id]^5) *flow_mag[id]^2) 
			* ratioTolerance;

subject to pumpHeadLossDefinitionU {id in pumpID}:
	head_loss[id] <= (pump_a[id] - pump_b[id] * flow_mag[id]^pump_c)
				+ hlTolerance;

subject to pumpHeadLossDefinitionL {id in pumpID}:
	head_loss[id] >= (pump_a[id] - pump_b[id] * flow_mag[id]^pump_c)
				- hlTolerance;

#subject to pumpHeadLossDefinitionU {id in pumpID}:
#	head_loss[id] <= init_hl[id] + .001*hlTolerance;
#
#subject to pumpHeadLossDefinitionL {id in pumpID}:
#	head_loss[id] <= init_hl[id] - .001*hlTolerance;

# we won't modify valve flow and head loss; take from the input
subject to valveHeadLossDefinitionU {(id,v,w,di,flw,hl) in Valves}:
        head_loss[id] <= hl + absTolerance;
                                                                                
subject to valveHeadLossDefinition {(id,v,w,di,flw,hl) in Valves}:
        head_loss[id] >= hl - absTolerance;
                                                                                
subject to valveFlowDefinitionU {(id,v,w,di,flw,hl) in Valves}:
        (flow_dir[id]*flow_mag[id]) <= flw + absTolerance;
                                                                                
subject to valveFlowDefinition {(id,v,w,di,flw,hl) in Valves}:
        (flow_dir[id]*flow_mag[id]) >= flw - absTolerance;

subject to energyConservationU {id in BE}:
	sum {idd in FC[id]}
		(if fcArcLeader[id,idd] == source[idd]
		    then 
			if flow_dir[idd] == 1
				then head_loss[idd]
				else -head_loss[idd]
		    else 
			if flow_dir[idd] == -1
				then head_loss[idd]
				else -head_loss[idd])
		        <= absTolerance;

subject to energyConservationL {id in BE}:
	sum {idd in FC[id]}
		(if fcArcLeader[id,idd] == source[idd]
		    then 
			if flow_dir[idd] == 1
				then head_loss[idd]
				else -head_loss[idd]
		     else 
			if flow_dir[idd] == -1
				then head_loss[idd]
				else -head_loss[idd]
		    )    >= -absTolerance;

subject to source2sourceHeadLossU {v in SourceID, w in SourceID: 
					v < w && stID[v] == stID[w]}:
	sum {idd in S2SPath[v,w]}
		(if s2sPathLeader[v,w,idd] == source[idd]
		    then 
			if flow_dir[idd] == 1
				then head_loss[idd]
				else -head_loss[idd]
		    else 
			if flow_dir[idd] == -1
				then head_loss[idd]
				else -head_loss[idd])
		        <= (head[v] - head[w]) + absTolerance;

subject to source2sourceHeadLossL {v in SourceID, w in SourceID: 
					v < w && stID[v] == stID[w]}:
	sum {idd in S2SPath[v,w]}
		(if s2sPathLeader[v,w,idd] == source[idd]
		    then 
			if flow_dir[idd] == 1
				then head_loss[idd]
				else -head_loss[idd]
		    else 
			if flow_dir[idd] == -1
				then head_loss[idd]
				else -head_loss[idd])
		        >= (head[v] - head[w]) + absTolerance;

subject to flowRatioConservationU {(id,i,j,l,d,f,h) in Pipes}:
	flow_mag[id] * (sum {(idd,x,j) in DA} init_flow_mag[idd]) <= 
	(init_flow_mag[id] * (sum {(idd,x,j) in DA} flow_mag[idd]))
			   + ratioTolerance;

subject to flowRatioConservationL {(id,i,j,l,d,f,h) in Pipes}:
	flow_mag[id] * (sum {(idd,x,j) in DA} init_flow_mag[idd]) >= 
	(init_flow_mag[id] * (sum {(idd,x,j) in DA} flow_mag[idd]))
			   - ratioTolerance;

subject to transportTimeDefinitionU {(id,i,j,l,d,f,h) in Pipes}:
 	tau[id] <= (a*length[id]*(diam[id]^2)) /flow_mag[id] + absTolerance;

subject to transportTimeDefinitionL {(id,i,j,l,d,f,h) in Pipes}:
 	tau[id] >= (a*length[id]*(diam[id]^2)) /flow_mag[id] - absTolerance;

subject to transportTimeToleranceU {(id,i,j,l,d,f,h) in Pipes}:
	tau[id] <= init_tau[id]+alpha*init_tau[id];

subject to transportTimeToleranceL {(id,i,j,l,d,f,h) in Pipes}:
	init_tau[id] <= tau[id]+alpha*tau[id];

subject to flowDirSanity {id in pipeID}:
	flow_dir[id] = init_flow_dir[id];

subject to flowMagSanityU {id in ID}:
	flow_mag[id] <= init_flow_mag[id] +ratioTolerance*init_flow_mag[id];

subject to flowMagSanityL {id in ID}:
	flow_mag[id] >= init_flow_mag[id] -ratioTolerance*init_flow_mag[id];

subject to pumpLength {id in pumpID}:
	length[id] == 0;

subject to valveLength {id in valveID}:
	length[id] == 0;

subject to diamNonzero {id in pipeID}:
	diam[id] >= 1e-05;

#subject to headLossSanityU {id in pipeID}:
#	head_loss[id] <= 3*(init_hl[id]*(init_length[id]/1000));
#
#subject to headLossSanityL {id in pipeID}:
#	head_loss[id] >= 1/3*(init_hl[id]*(init_length[id]/1000));
