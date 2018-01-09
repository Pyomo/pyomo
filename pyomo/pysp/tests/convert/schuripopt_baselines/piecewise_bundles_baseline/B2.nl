g3 1 1 0	# problem B2
 20 20 1 1 5 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 6 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 47 3 	# nonzeros in Jacobian, obj. gradient
 46 40	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
S7 1 schurip_objective_weight
0 0.3333333333
S0 1 schurip_variable_id
0 368493142
C0	#MASTER_BLEND_CONSTRAINT_RootNode[1]
n0
C1	#Scenario3.c_first_stage
n0
C2	#Scenario3.c_second_stage
n0
C3	#Scenario3.r_second_stage
n0
C4	#Scenario3.p_first_stage.INC_constraint1
n0
C5	#Scenario3.p_first_stage.INC_constraint2
n0
C6	#Scenario3.p_first_stage.INC_constraint3[1]
n0
C7	#Scenario3.p_first_stage.INC_constraint3[2]
n0
C8	#Scenario3.p_first_stage.INC_constraint3[3]
n0
C9	#Scenario3.p_first_stage.INC_constraint4[1]
n0
C10	#Scenario3.p_first_stage.INC_constraint4[2]
n0
C11	#Scenario3.p_first_stage.INC_constraint4[3]
n0
C12	#Scenario3.p_second_stage[1].INC_constraint1
n0
C13	#Scenario3.p_second_stage[1].INC_constraint2
n0
C14	#Scenario3.p_second_stage[1].INC_constraint3[1]
n0
C15	#Scenario3.p_second_stage[1].INC_constraint3[2]
n0
C16	#Scenario3.p_second_stage[1].INC_constraint3[3]
n0
C17	#Scenario3.p_second_stage[1].INC_constraint4[1]
n0
C18	#Scenario3.p_second_stage[1].INC_constraint4[2]
n0
C19	#Scenario3.p_second_stage[1].INC_constraint4[3]
n0
O0 0	#MASTER
n-2
x0	# initial guess
r	#20 ranges (rhs's)
4 0.0
2 0.0
2 -100.0
0 -2.0 0.0
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
b	#20 bounds (on variables)
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
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
0 0 1
k19	#intermediate Jacobian column lengths
1
5
6
7
8
10
12
16
20
22
24
28
32
35
37
39
41
43
45
J0 2
0 1
1 -1
J1 1
1 1
J2 2
1 1
5 2
J3 1
5 1
J4 5
1 1
6 -2.0
7 -3.0
8 -2.0
9 -3.0
J5 3
2 1
7 1.0
8 -1
J6 2
7 1
14 -1
J7 2
8 1
15 -1
J8 2
9 1
16 -1
J9 2
6 -1
14 1
J10 2
7 -1
15 1
J11 2
8 -1
16 1
J12 5
3 1
10 -5.0
11 -5.0
12 -5.0
13 -5.0
J13 4
4 1
11 1.0
12 -5.0
13 3.0
J14 2
11 1
17 -1
J15 2
12 1
18 -1
J16 2
13 1
19 -1
J17 2
10 -1
17 1
J18 2
11 -1
18 1
J19 2
12 -1
19 1
G0 3
2 1
4 -1
5 1
