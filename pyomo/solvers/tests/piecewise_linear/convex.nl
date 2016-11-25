g3 1 1 0	# problem unknown
 2 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 1 	# nonzeros in Jacobian, obj. gradient
 38 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#con.simplified_piecewise_constraint[1]
n0
C1	#con.simplified_piecewise_constraint[2]
n0
O0 0	#obj
n0
x0	# initial guess
r	#2 ranges (rhs's)
1 -2.0
1 0.0
b	#2 bounds (on variables)
0 -5 5
3
k1	#intermediate Jacobian column lengths
2
J0 2
0 -1.0
1 -1
J1 2
0 1
1 -1
G0 1
1 1
