g3 1 1 0	# problem unknown
 13 7 1 0 7 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 3 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 27 2 	# nonzeros in Jacobian, obj. gradient
 22 19	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#pn_con
n0
C1	#con.DCC_constraint1
n0
C2	#con.DCC_constraint2
n0
C3	#con.DCC_constraint3[1]
n0
C4	#con.DCC_constraint3[2]
n0
C5	#con.DCC_constraint3[3]
n0
C6	#con.DCC_constraint4
n0
O0 0	#obj
n0
x0	# initial guess
r	#7 ranges (rhs's)
4 7.0
4 0.0
4 0.0
4 0.0
4 0.0
4 0.0
4 1.0
b	#13 bounds (on variables)
0 -1.0 10.0
3
2 0
2 0
2 0
2 0
2 0
2 0
2 0
2 0
0 0 1
0 0 1
0 0 1
k12	#intermediate Jacobian column lengths
1
3
4
5
8
10
12
15
18
21
23
25
J0 3
1 1
2 -1
3 1
J1 7
0 1
4 1.0
5 -2.0
6 -2.0
7 -6.0
8 -6.0
9 -10.0
J2 5
1 1
4 1.0
7 8.0
8 8.0
9 -12.0
J3 3
4 -1
5 -1
10 1
J4 3
6 -1
7 -1
11 1
J5 3
8 -1
9 -1
12 1
J6 3
10 1
11 1
12 1
G0 2
2 1
3 1
