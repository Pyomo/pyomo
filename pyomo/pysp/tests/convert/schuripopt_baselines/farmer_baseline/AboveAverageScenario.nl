g3 1 1 0	# problem AboveAverageScenario
 12 10 1 3 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 27 10 	# nonzeros in Jacobian, obj. gradient
 41 35	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
S7 1 schurip_objective_weight
0 0.33333333
S0 3 schurip_variable_id
0 2021786751
1 1561833761
2 1002958202
C0	#ConstrainTotalAcreage
n0
C1	#EnforceCattleFeedRequirement[CORN]
n0
C2	#EnforceCattleFeedRequirement[SUGAR_BEETS]
n0
C3	#EnforceCattleFeedRequirement[WHEAT]
n0
C4	#LimitAmountSold[CORN]
n0
C5	#LimitAmountSold[SUGAR_BEETS]
n0
C6	#LimitAmountSold[WHEAT]
n0
C7	#EnforceQuotas[CORN]
n0
C8	#EnforceQuotas[SUGAR_BEETS]
n0
C9	#EnforceQuotas[WHEAT]
n0
O0 0	#Total_Cost_Objective
n0
x0	# initial guess
r	#10 ranges (rhs's)
1 500
2 240.0
2 0.0
2 200.0
1 0.0
1 0.0
1 0.0
0 0.0 100000.0
0 0.0 6000.0
0 0.0 100000.0
b	#12 bounds (on variables)
0 0.0 500
0 0.0 500
0 0.0 500
2 0.0
2 0.0
2 0.0
2 0.0
2 0.0
2 0.0
2 0.0
2 0.0
2 0.0
k11	#intermediate Jacobian column lengths
3
6
9
12
15
18
20
22
24
25
26
J0 3
0 1
1 1
2 1
J1 4
0 3.6
3 -1
6 -1
9 1
J2 4
1 24
4 -1
7 -1
10 1
J3 4
2 3.0
5 -1
8 -1
11 1
J4 3
0 -3.6
3 1
6 1
J5 3
1 -24
4 1
7 1
J6 3
2 -3.0
5 1
8 1
J7 1
3 1
J8 1
4 1
J9 1
5 1
G0 10
0 230
1 260
2 150
3 -150
4 -36
5 -170
7 -10
9 210
10 100000
11 238
