#  MINLP written by GAMS Convert at 11/10/17 15:35:21
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#         25        4        6       15        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#         27       15       12        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         87       84        3        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.x1 = Var(within=Reals,bounds=(0,29),initialize=0)
m.x2 = Var(within=Reals,bounds=(0,29),initialize=0)
m.x3 = Var(within=Reals,bounds=(0,29),initialize=0)
m.x4 = Var(within=Reals,bounds=(0,29),initialize=0)
m.x5 = Var(within=Reals,bounds=(0,29),initialize=0)
m.x6 = Var(within=Reals,bounds=(0,29),initialize=0)
m.x7 = Var(within=Reals,bounds=(1,40),initialize=1)
m.x8 = Var(within=Reals,bounds=(1,50),initialize=1)
m.x9 = Var(within=Reals,bounds=(1,60),initialize=1)
m.x10 = Var(within=Reals,bounds=(1,40),initialize=1)
m.x11 = Var(within=Reals,bounds=(1,50),initialize=1)
m.x12 = Var(within=Reals,bounds=(1,60),initialize=1)
m.x13 = Var(within=Reals,bounds=(0,30),initialize=0)
m.x14 = Var(within=Reals,bounds=(0,30),initialize=0)
m.b15 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b16 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b17 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b18 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b19 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b20 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b21 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b22 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b23 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b24 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b25 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b26 = Var(within=Binary,bounds=(0,1),initialize=0)

m.obj = Objective(expr=   2*m.x13 + 2*m.x14, sense=minimize)

m.c2 = Constraint(expr= - m.x1 - m.x7 + m.x13 >= 0)

m.c3 = Constraint(expr= - m.x2 - m.x8 + m.x13 >= 0)

m.c4 = Constraint(expr= - m.x3 - m.x9 + m.x13 >= 0)

m.c5 = Constraint(expr= - m.x4 - m.x10 + m.x14 >= 0)

m.c6 = Constraint(expr= - m.x5 - m.x11 + m.x14 >= 0)

m.c7 = Constraint(expr= - m.x6 - m.x12 + m.x14 >= 0)

m.c8 = Constraint(expr=40/m.x10 - m.x7 <= 0)

m.c9 = Constraint(expr=50/m.x11 - m.x8 <= 0)

m.c10 = Constraint(expr=60/m.x12 - m.x9 <= 0)

m.c11 = Constraint(expr=   m.x1 - m.x2 + m.x7 + 69*m.b15 <= 69)

m.c12 = Constraint(expr=   m.x1 - m.x3 + m.x7 + 69*m.b16 <= 69)

m.c13 = Constraint(expr=   m.x2 - m.x3 + m.x8 + 79*m.b17 <= 79)

m.c14 = Constraint(expr= - m.x1 + m.x2 + m.x8 + 79*m.b18 <= 79)

m.c15 = Constraint(expr= - m.x1 + m.x3 + m.x9 + 89*m.b19 <= 89)

m.c16 = Constraint(expr= - m.x2 + m.x3 + m.x9 + 89*m.b20 <= 89)

m.c17 = Constraint(expr=   m.x4 - m.x5 + m.x10 + 69*m.b21 <= 69)

m.c18 = Constraint(expr=   m.x4 - m.x6 + m.x10 + 69*m.b22 <= 69)

m.c19 = Constraint(expr=   m.x5 - m.x6 + m.x11 + 79*m.b23 <= 79)

m.c20 = Constraint(expr= - m.x4 + m.x5 + m.x11 + 79*m.b24 <= 79)

m.c21 = Constraint(expr= - m.x4 + m.x6 + m.x12 + 89*m.b25 <= 89)

m.c22 = Constraint(expr= - m.x5 + m.x6 + m.x12 + 89*m.b26 <= 89)

m.c23 = Constraint(expr=   m.b15 + m.b18 + m.b21 + m.b24 == 1)

m.c24 = Constraint(expr=   m.b16 + m.b19 + m.b22 + m.b25 == 1)

m.c25 = Constraint(expr=   m.b17 + m.b20 + m.b23 + m.b26 == 1)
