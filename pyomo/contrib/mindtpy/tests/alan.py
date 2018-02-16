#  MINLP written by GAMS Convert at 11/10/17 15:35:19
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#          8        3        0        5        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#          9        5        4        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         24       21        3        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,None),initialize=0.302884615384618)
m.x2 = Var(within=Reals,bounds=(0,None),initialize=0.0865384615384593)
m.x3 = Var(within=Reals,bounds=(0,None),initialize=0.504807692307693)
m.x4 = Var(within=Reals,bounds=(0,None),initialize=0.10576923076923)
m.b6 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b7 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b8 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b9 = Var(within=Binary,bounds=(0,1),initialize=0)

m.obj = Objective(expr=m.x1*(4*m.x1 + 3*m.x2 - m.x3) + m.x2*(3*m.x1 + 6*m.x2 + m.x3) + m.x3*(m.x2 - m.x1 + 10*m.x3)
                       , sense=minimize)

m.c1 = Constraint(expr=   m.x1 + m.x2 + m.x3 + m.x4 == 1)

m.c2 = Constraint(expr=   8*m.x1 + 9*m.x2 + 12*m.x3 + 7*m.x4 == 10)

m.c4 = Constraint(expr=   m.x1 - m.b6 <= 0)

m.c5 = Constraint(expr=   m.x2 - m.b7 <= 0)

m.c6 = Constraint(expr=   m.x3 - m.b8 <= 0)

m.c7 = Constraint(expr=   m.x4 - m.b9 <= 0)

m.c8 = Constraint(expr=   m.b6 + m.b7 + m.b8 + m.b9 <= 3)
