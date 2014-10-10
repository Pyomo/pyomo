g3 1 1 0	# problem unknown
 2 0 1 0 0 	# vars, constraints, objectives, general inequalities, equalities  
 0 1 	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 2	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0	#f[None]
o54  #sum
3
o2  #*
n3
o5  #^
v0 #x[1]
n4 # numeric constant
o2  #*
n-2
o5  #^
o2  #*
v0 #x[1]
v1 #x[2]
n2 # numeric constant
o2  #*
n3
o5  #^
v1 #x[2]
n4 # numeric constant
x2
0 1.0 # x[1] initial
1 0.1 # x[2] initial
r
b
3  # v0  x[1]
3  # v1  x[2]
k1
0
G0 2
0   0
1   0
