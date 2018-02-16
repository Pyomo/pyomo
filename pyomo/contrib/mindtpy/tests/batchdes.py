#  MINLP written by GAMS Convert at 11/10/17 15:35:19
#  
#  Equation counts
#      Total        E        G        L        N        X        C        B
#         20        7       12        1        0        0        0        0
#  
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#         20       11        9        0        0        0        0        0
#  FX      0        0        0        0        0        0        0        0
#  
#  Nonzero counts
#      Total    const       NL      DLL
#         53       43       10        0
# 
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *

model = m = ConcreteModel()


m.b1 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b2 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b3 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b4 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b5 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b6 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b7 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b8 = Var(within=Binary,bounds=(0,1),initialize=0)
m.b9 = Var(within=Binary,bounds=(0,1),initialize=0)
m.x10 = Var(within=Reals,bounds=(5.52146091786225,7.82404601085629),initialize=6.70502272492805)
m.x11 = Var(within=Reals,bounds=(5.52146091786225,7.82404601085629),initialize=7.11048783303622)
m.x12 = Var(within=Reals,bounds=(5.52146091786225,7.82404601085629),initialize=7.30700912709102)
m.x13 = Var(within=Reals,bounds=(5.40367788220586,6.4377516497364),initialize=5.92071476597113)
m.x14 = Var(within=Reals,bounds=(4.60517018598809,6.03228654162824),initialize=5.31872836380816)
m.x15 = Var(within=Reals,bounds=(1.89711998488588,2.99573227355399),initialize=1.89711998488588)
m.x16 = Var(within=Reals,bounds=(1.38629436111989,2.484906649788),initialize=1.38629436111989)
m.x17 = Var(within=Reals,bounds=(0,1.09861228866811),initialize=0)
m.x18 = Var(within=Reals,bounds=(0,1.09861228866811),initialize=0)
m.x19 = Var(within=Reals,bounds=(0,1.09861228866811),initialize=0)

m.obj = Objective(expr=250*exp(0.6*m.x10 + m.x17) + 500*exp(0.6*m.x11 + m.x18) + 340*exp(0.6*m.x12 + m.x19)
                       , sense=minimize)

m.c1 = Constraint(expr=   m.x10 - m.x13 >= 0.693147180559945)

m.c2 = Constraint(expr=   m.x11 - m.x13 >= 1.09861228866811)

m.c3 = Constraint(expr=   m.x12 - m.x13 >= 1.38629436111989)

m.c4 = Constraint(expr=   m.x10 - m.x14 >= 1.38629436111989)

m.c5 = Constraint(expr=   m.x11 - m.x14 >= 1.79175946922805)

m.c6 = Constraint(expr=   m.x12 - m.x14 >= 1.09861228866811)

m.c7 = Constraint(expr=   m.x15 + m.x17 >= 2.07944154167984)

m.c8 = Constraint(expr=   m.x15 + m.x18 >= 2.99573227355399)

m.c9 = Constraint(expr=   m.x15 + m.x19 >= 1.38629436111989)

m.c10 = Constraint(expr=   m.x16 + m.x17 >= 2.30258509299405)

m.c11 = Constraint(expr=   m.x16 + m.x18 >= 2.484906649788)

m.c12 = Constraint(expr=   m.x16 + m.x19 >= 1.09861228866811)

m.c13 = Constraint(expr=200000*exp(m.x15 - m.x13) + 150000*exp(m.x16 - m.x14) <= 6000)

m.c14 = Constraint(expr= - 0.693147180559945*m.b4 - 1.09861228866811*m.b7 + m.x17 == 0)

m.c15 = Constraint(expr= - 0.693147180559945*m.b5 - 1.09861228866811*m.b8 + m.x18 == 0)

m.c16 = Constraint(expr= - 0.693147180559945*m.b6 - 1.09861228866811*m.b9 + m.x19 == 0)

m.c17 = Constraint(expr=   m.b1 + m.b4 + m.b7 == 1)

m.c18 = Constraint(expr=   m.b2 + m.b5 + m.b8 == 1)

m.c19 = Constraint(expr=   m.b3 + m.b6 + m.b9 == 1)
