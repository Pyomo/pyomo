g3 1 1 0	# problem Scenario2
 19 19 1 1 4 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 6 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 45 3 	# nonzeros in Jacobian, obj. gradient
 36 30	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
S7 1 schurip_objective_weight
0 0.3333333333
S0 1 schurip_variable_id
0 368493142
C0	#c_first_stage
n0
C1	#c_second_stage
n0
C2	#r_second_stage
n0
C3	#p_first_stage.INC_constraint1
n0
C4	#p_first_stage.INC_constraint2
n0
C5	#p_first_stage.INC_constraint3[1]
n0
C6	#p_first_stage.INC_constraint3[2]
n0
C7	#p_first_stage.INC_constraint3[3]
n0
C8	#p_first_stage.INC_constraint4[1]
n0
C9	#p_first_stage.INC_constraint4[2]
n0
C10	#p_first_stage.INC_constraint4[3]
n0
C11	#p_second_stage[1].INC_constraint1
n0
C12	#p_second_stage[1].INC_constraint2
n0
C13	#p_second_stage[1].INC_constraint3[1]
n0
C14	#p_second_stage[1].INC_constraint3[2]
n0
C15	#p_second_stage[1].INC_constraint3[3]
n0
C16	#p_second_stage[1].INC_constraint4[1]
n0
C17	#p_second_stage[1].INC_constraint4[2]
n0
C18	#p_second_stage[1].INC_constraint4[3]
n0
O0 0	#o
n-1
x0	# initial guess
r	#19 ranges (rhs's)
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
b	#19 bounds (on variables)
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
k18	#intermediate Jacobian column lengths
3
4
5
6
8
10
14
18
20
22
26
30
33
35
37
39
41
43
J0 1
0 1
J1 2
0 1
4 1
J2 1
4 1
J3 5
0 1
5 -2.0
6 -3.0
7 -2.0
8 -3.0
J4 3
1 1
6 1.0
7 -1
J5 2
6 1
13 -1
J6 2
7 1
14 -1
J7 2
8 1
15 -1
J8 2
5 -1
13 1
J9 2
6 -1
14 1
J10 2
7 -1
15 1
J11 5
2 1
9 -5.0
10 -5.0
11 -5.0
12 -5.0
J12 4
3 1
10 1.0
11 -4.0
12 2.0
J13 2
10 1
16 -1
J14 2
11 1
17 -1
J15 2
12 1
18 -1
J16 2
9 -1
16 1
J17 2
10 -1
17 1
J18 2
11 -1
18 1
G0 3
1 1
3 -1
4 1
