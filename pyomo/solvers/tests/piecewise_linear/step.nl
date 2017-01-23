g3 1 1 0	# problem unknown
 11 10 1 0 2 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 4 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 23 2 	# nonzeros in Jacobian, obj. gradient
 22 16	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#con.INC_constraint1
n0
C1	#con.INC_constraint2
n0
C2	#con.INC_constraint3[1]
n0
C3	#con.INC_constraint3[2]
n0
C4	#con.INC_constraint3[3]
n0
C5	#con.INC_constraint3[4]
n0
C6	#con.INC_constraint4[1]
n0
C7	#con.INC_constraint4[2]
n0
C8	#con.INC_constraint4[3]
n0
C9	#con.INC_constraint4[4]
n0
O0 1	#obj
n0
x0	# initial guess
r	#10 ranges (rhs's)
4 0.0
4 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
1 0.0
b	#11 bounds (on variables)
0 0 3
3
1 1
3
3
3
2 0
0 0 1
0 0 1
0 0 1
0 0 1
k10	#intermediate Jacobian column lengths
1
2
4
7
10
13
15
17
19
21
J0 4
0 1
2 -1
4 -1
6 -1
J1 3
1 1
3 -2.0
5 1.5
J2 2
3 1
7 -1
J3 2
4 1
8 -1
J4 2
5 1
9 -1
J5 2
6 1
10 -1
J6 2
2 -1
7 1
J7 2
3 -1
8 1
J8 2
4 -1
9 1
J9 2
5 -1
10 1
G0 2
0 1
1 1
