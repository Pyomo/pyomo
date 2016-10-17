"""
Example:
(x0-5)^2 + (x2-5)^2 + ... +(x149-5)^2 -> min

subjected to

# lb<= x <= ub:
x4 <= 4
8 <= x5 <= 15

# Ax <= b
x9 + x19 <= 7
x10+x11 <= 9

# Aeq x = beq
x100+x101 = 11

# c(x) <= 0
2*x0^4-32 <= 0
x1^2+x2^2-8 <= 0

# h(x) = 0
(x[149]-1)**6 = 0
(x[148]-1.5)**6 = 0
"""

from openopt import NLP
from numpy import cos, arange, ones, asarray, zeros, mat, array
N = 150

# objective function:
f = lambda x: ((x-5)**2).sum()

# objective function gradient (optional):
df = lambda x: 2*(x-5)

# start point (initial estimation)
x0 = 8*cos(arange(N))

# c(x) <= 0 constraints
c = [lambda x: 2* x[0] **4-32, lambda x: x[1]**2+x[2]**2 - 8]

# dc(x)/dx: non-lin ineq constraints gradients (optional):
dc0 = lambda x: [8 * x[0]**3] + [0]*(N-1)
dc1 = lambda x: [0, 2 * x[1],  2 * x[2]] + [0]*(N-3)
dc = [dc0, dc1]

# h(x) = 0 constraints
def h(x):
    return (x[N-1]-1)**6, (x[N-2]-1.5)**6
    # other possible return types: numpy array, matrix, Python list, tuple
# or just h = lambda x: [(x[149]-1)**6, (x[148]-1.5)**6]


# dh(x)/dx: non-lin eq constraints gradients (optional):
def dh(x):
    r = zeros((2, N))
    r[0, -1] = 6*(x[N-1]-1)**5
    r[1, -2] = 6*(x[N-2]-1.5)**5
    return r
    
# lower and upper bounds on variables
lb = -6*ones(N)
ub = 6*ones(N)
ub[4] = 4
lb[5], ub[5] = 8, 15

# general linear inequality constraints
A = zeros((2, N))
A[0, 9] = 1
A[0, 19] = 1
A[1, 10:12] = 1
b = [7, 9]

# general linear equality constraints
Aeq = zeros(N)
Aeq[100:102] = 1
beq = 11

# Create a problem object
p = NLP(f, x0, df=df,  c=c,  dc=dc, h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq,  
        lb=lb, ub=ub, gtol=1e-7, contol=1e-7, iprint = 50, maxIter = 1e6, maxFunEvals = 1e7, name = 'NLP_1')

# solve the problem
r = p.solve('ralg')

print(r.xf)

# r.xf and r.ff are optim point and optim objFun value
# r.ff should be something like 132.05
