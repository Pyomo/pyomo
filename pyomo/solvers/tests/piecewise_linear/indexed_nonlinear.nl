g3 1 1 0	# problem unknown
 4 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 2 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 25 6	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#nonlinear_constraint[0,1]
o46	#cos
o2	#*
n5.0
v0	#X[0,1]
C1	#nonlinear_constraint[8,3]
o46	#cos
o2	#*
n5.0
v1	#X[8,3]
O0 1	#obj
n0
x4	# initial guess
0 1.7
1 1.7
2 1.25
3 1.25
r	#2 ranges (rhs's)
4 0.0
4 0.0
b	#4 bounds (on variables)
0 -2 2
0 -2 2
3
3
k3	#intermediate Jacobian column lengths
1
2
3
J0 2
0 -0.1
2 1
J1 2
1 -0.1
3 1
G0 2
2 1
3 1
