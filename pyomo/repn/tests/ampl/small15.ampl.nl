g3 1 1 0	# problem unknown
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 1 1 	# nonzeros in Jacobian, obj. gradient
 4 3	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#CON1
o5	#pow
v0	#b.y
n2
O0 0	#OBJ
o5	#pow
v1	#x
n2
x2	# initial guess
0 1.0
1 1.0
r	#1 ranges (rhs's)
4 4.0
b	#2 bounds (on variables)
3
3
k1	#intermediate Jacobian column lengths
1
J0 1
0 0
G0 1
1 0
