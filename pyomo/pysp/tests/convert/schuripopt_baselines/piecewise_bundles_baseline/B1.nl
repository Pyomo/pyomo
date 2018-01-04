g3 1 1 0	# problem B1
 39 40 1 1 11 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 12 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 93 6 	# nonzeros in Jacobian, obj. gradient
 46 40	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
S7 1 schurip_objective_weight
0 0.6666666666
S0 1 schurip_variable_id
0 368493142
C0	#MASTER_BLEND_CONSTRAINT_RootNode[1]
n0
C1	#MASTER_BLEND_CONSTRAINT_RootNode[2]
n0
C2	#Scenario1.c_first_stage
n0
C3	#Scenario1.c_second_stage
n0
C4	#Scenario1.r_second_stage
n0
C5	#Scenario1.p_first_stage.INC_constraint1
n0
C6	#Scenario1.p_first_stage.INC_constraint2
n0
C7	#Scenario1.p_first_stage.INC_constraint3[1]
n0
C8	#Scenario1.p_first_stage.INC_constraint3[2]
n0
C9	#Scenario1.p_first_stage.INC_constraint3[3]
n0
C10	#Scenario1.p_first_stage.INC_constraint4[1]
n0
C11	#Scenario1.p_first_stage.INC_constraint4[2]
n0
C12	#Scenario1.p_first_stage.INC_constraint4[3]
n0
C13	#Scenario1.p_second_stage[1].INC_constraint1
n0
C14	#Scenario1.p_second_stage[1].INC_constraint2
n0
C15	#Scenario1.p_second_stage[1].INC_constraint3[1]
n0
C16	#Scenario1.p_second_stage[1].INC_constraint3[2]
n0
C17	#Scenario1.p_second_stage[1].INC_constraint3[3]
n0
C18	#Scenario1.p_second_stage[1].INC_constraint4[1]
n0
C19	#Scenario1.p_second_stage[1].INC_constraint4[2]
n0
C20	#Scenario1.p_second_stage[1].INC_constraint4[3]
n0
C21	#Scenario2.c_first_stage
n0
C22	#Scenario2.c_second_stage
n0
C23	#Scenario2.r_second_stage
n0
C24	#Scenario2.p_first_stage.INC_constraint1
n0
C25	#Scenario2.p_first_stage.INC_constraint2
n0
C26	#Scenario2.p_first_stage.INC_constraint3[1]
n0
C27	#Scenario2.p_first_stage.INC_constraint3[2]
n0
C28	#Scenario2.p_first_stage.INC_constraint3[3]
n0
C29	#Scenario2.p_first_stage.INC_constraint4[1]
n0
C30	#Scenario2.p_first_stage.INC_constraint4[2]
n0
C31	#Scenario2.p_first_stage.INC_constraint4[3]
n0
C32	#Scenario2.p_second_stage[1].INC_constraint1
n0
C33	#Scenario2.p_second_stage[1].INC_constraint2
n0
C34	#Scenario2.p_second_stage[1].INC_constraint3[1]
n0
C35	#Scenario2.p_second_stage[1].INC_constraint3[2]
n0
C36	#Scenario2.p_second_stage[1].INC_constraint3[3]
n0
C37	#Scenario2.p_second_stage[1].INC_constraint4[1]
n0
C38	#Scenario2.p_second_stage[1].INC_constraint4[2]
n0
C39	#Scenario2.p_second_stage[1].INC_constraint4[3]
n0
O0 0	#MASTER
n-0.5
x0	# initial guess
r	#40 ranges (rhs's)
4 0.0
4 0.0
2 0.0
2 -100.0
4 0
4 0.0
4 10.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
4 -10.0
4 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
2 0.0
2 -100.0
0 -1.0 0.0
4 0.0
4 10.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
4 -10.0
4 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
b	#39 bounds (on variables)
3
0 0 10
3
0 -10 10
3
3
1 1
3
3
2 0
1 1
3
3
2 0
0 0 10
3
0 -10 10
3
3
1 1
3
3
2 0
1 1
3
3
2 0
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
k38	#intermediate Jacobian column lengths
2
6
7
8
9
10
12
16
20
22
24
28
32
35
39
40
41
42
44
46
50
54
56
58
62
66
69
71
73
75
77
79
81
83
85
87
89
91
J0 2
0 1
1 -1
J1 2
0 1
14 -1
J2 1
1 1
J3 1
1 1
J4 1
5 1
J5 5
1 1
6 -2.0
7 -3.0
8 -2.0
9 -3.0
J6 3
2 1
7 1.0
8 -1
J7 2
7 1
27 -1
J8 2
8 1
28 -1
J9 2
9 1
29 -1
J10 2
6 -1
27 1
J11 2
7 -1
28 1
J12 2
8 -1
29 1
J13 5
3 1
10 -5.0
11 -5.0
12 -5.0
13 -5.0
J14 4
4 1
11 1.0
12 -3.0
13 1.0
J15 2
11 1
30 -1
J16 2
12 1
31 -1
J17 2
13 1
32 -1
J18 2
10 -1
30 1
J19 2
11 -1
31 1
J20 2
12 -1
32 1
J21 1
14 1
J22 2
14 1
18 1
J23 1
18 1
J24 5
14 1
19 -2.0
20 -3.0
21 -2.0
22 -3.0
J25 3
15 1
20 1.0
21 -1
J26 2
20 1
33 -1
J27 2
21 1
34 -1
J28 2
22 1
35 -1
J29 2
19 -1
33 1
J30 2
20 -1
34 1
J31 2
21 -1
35 1
J32 5
16 1
23 -5.0
24 -5.0
25 -5.0
26 -5.0
J33 4
17 1
24 1.0
25 -4.0
26 2.0
J34 2
24 1
36 -1
J35 2
25 1
37 -1
J36 2
26 1
38 -1
J37 2
23 -1
36 1
J38 2
24 -1
37 1
J39 2
25 -1
38 1
G0 6
2 0.5
4 -0.5
5 0.5
15 0.5
17 -0.5
18 0.5
