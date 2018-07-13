#  MINLP written by GAMS Convert at 11/10/17 15:35:21
#
#  Equation counts
#      Total        E        G        L        N        X        C        B
#        344        1        0      343        0        0        0        0
#
#  Variable counts
#                   x        b        i      s1s      s2s       sc       si
#      Total     cont   binary  integer     sos1     sos2    scont     sint
#        183      111       72        0        0        0        0        0
#  FX      2        2        0        0        0        0        0        0
#
#  Nonzero counts
#      Total    const       NL      DLL
#       1441     1423       18        0
#
#  Reformulation has removed 1 variable and 1 equation


from pyomo.environ import *


def build_model():
    model = m = ConcreteModel()

    m.b1 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b2 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b3 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b4 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b5 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b6 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b7 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b8 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b9 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b10 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b11 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b12 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b13 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b14 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b15 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b16 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b17 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b18 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b19 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b20 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b21 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b22 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b23 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b24 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b25 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b26 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b27 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b28 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b29 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b30 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b31 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b32 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b33 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b34 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b35 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b36 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b37 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b38 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b39 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b40 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b41 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b42 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b43 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b44 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b45 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b46 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b47 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b48 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b49 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b50 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b51 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b52 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b53 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b54 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b55 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b56 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b57 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b58 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b59 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b60 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b61 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b62 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b63 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b64 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b65 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b66 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b67 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b68 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b69 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b70 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b71 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.b72 = Var(within=Binary, bounds=(0, 1), initialize=0)
    m.x74 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x75 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x76 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x77 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x78 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x79 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x80 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x81 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x82 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x83 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x84 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x85 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x86 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x87 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x88 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x89 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x90 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x91 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x92 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x93 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x94 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x95 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x96 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x97 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x98 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x99 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x100 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x101 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x102 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x103 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x104 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x105 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x106 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x107 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x108 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x109 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x110 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x111 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x112 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x113 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x114 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x115 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x116 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x117 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x118 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x119 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x120 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x121 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x122 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x123 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x124 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x125 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x126 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x127 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x128 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x129 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x130 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x131 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x132 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x133 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x134 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x135 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x136 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x137 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x138 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x139 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x140 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x141 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x142 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x143 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x144 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x145 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x146 = Var(within=Reals, bounds=(2, 8), initialize=2)
    m.x147 = Var(within=Reals, bounds=(2, 8), initialize=2)
    m.x148 = Var(within=Reals, bounds=(2, 8), initialize=2)
    m.x149 = Var(within=Reals, bounds=(3, 12), initialize=3)
    m.x150 = Var(within=Reals, bounds=(3, 12), initialize=3)
    m.x151 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x152 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x153 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x154 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x155 = Var(within=Reals, bounds=(12, 12), initialize=12)
    m.x156 = Var(within=Reals, bounds=(2, 8), initialize=2)
    m.x157 = Var(within=Reals, bounds=(2, 8), initialize=2)
    m.x158 = Var(within=Reals, bounds=(2, 8), initialize=2)
    m.x159 = Var(within=Reals, bounds=(3, 12), initialize=3)
    m.x160 = Var(within=Reals, bounds=(3, 12), initialize=3)
    m.x161 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x162 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x163 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x164 = Var(within=Reals, bounds=(1.5, 6), initialize=1.5)
    m.x165 = Var(within=Reals, bounds=(13, 13), initialize=13)
    m.x166 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x167 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x168 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x169 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x170 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x171 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x172 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x173 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x174 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x175 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x176 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x177 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x178 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x179 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x180 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x181 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x182 = Var(within=Reals, bounds=(None, None), initialize=0)
    m.x183 = Var(within=Reals, bounds=(None, None), initialize=0)

    m.obj = Objective(expr=m.x74 + m.x75 + m.x90 + m.x91 + m.x104 + m.x105 + m.x116 + m.x117 + m.x126 + m.x127 + m.x134
                      + m.x135 + m.x140 + m.x141 + m.x144 + m.x145, sense=minimize)

    m.c2 = Constraint(expr=m.x166 - m.x167 <= 0)

    m.c3 = Constraint(expr=0.5 * m.x146 - m.x155 + m.x166 <= 0)

    m.c4 = Constraint(expr=0.5 * m.x146 - m.x166 <= 0)

    m.c5 = Constraint(expr=0.5 * m.x156 - m.x165 + m.x175 <= 0)

    m.c6 = Constraint(expr=0.5 * m.x156 - m.x175 <= 0)

    m.c7 = Constraint(expr=0.5 * m.x147 - m.x155 + m.x167 <= 0)

    m.c8 = Constraint(expr=0.5 * m.x147 - m.x167 <= 0)

    m.c9 = Constraint(expr=0.5 * m.x157 - m.x165 + m.x176 <= 0)

    m.c10 = Constraint(expr=0.5 * m.x157 - m.x176 <= 0)

    m.c11 = Constraint(expr=0.5 * m.x148 - m.x155 + m.x168 <= 0)

    m.c12 = Constraint(expr=0.5 * m.x148 - m.x168 <= 0)

    m.c13 = Constraint(expr=0.5 * m.x158 - m.x165 + m.x177 <= 0)

    m.c14 = Constraint(expr=0.5 * m.x158 - m.x177 <= 0)

    m.c15 = Constraint(expr=0.5 * m.x149 - m.x155 + m.x169 <= 0)

    m.c16 = Constraint(expr=0.5 * m.x149 - m.x169 <= 0)

    m.c17 = Constraint(expr=0.5 * m.x159 - m.x165 + m.x178 <= 0)

    m.c18 = Constraint(expr=0.5 * m.x159 - m.x178 <= 0)

    m.c19 = Constraint(expr=0.5 * m.x150 - m.x155 + m.x170 <= 0)

    m.c20 = Constraint(expr=0.5 * m.x150 - m.x170 <= 0)

    m.c21 = Constraint(expr=0.5 * m.x160 - m.x165 + m.x179 <= 0)

    m.c22 = Constraint(expr=0.5 * m.x160 - m.x179 <= 0)

    m.c23 = Constraint(expr=0.5 * m.x151 - m.x155 + m.x171 <= 0)

    m.c24 = Constraint(expr=0.5 * m.x151 - m.x171 <= 0)

    m.c25 = Constraint(expr=0.5 * m.x161 - m.x165 + m.x180 <= 0)

    m.c26 = Constraint(expr=0.5 * m.x161 - m.x180 <= 0)

    m.c27 = Constraint(expr=0.5 * m.x152 - m.x155 + m.x172 <= 0)

    m.c28 = Constraint(expr=0.5 * m.x152 - m.x172 <= 0)

    m.c29 = Constraint(expr=0.5 * m.x162 - m.x165 + m.x181 <= 0)

    m.c30 = Constraint(expr=0.5 * m.x162 - m.x181 <= 0)

    m.c31 = Constraint(expr=0.5 * m.x153 - m.x155 + m.x173 <= 0)

    m.c32 = Constraint(expr=0.5 * m.x153 - m.x173 <= 0)

    m.c33 = Constraint(expr=0.5 * m.x163 - m.x165 + m.x182 <= 0)

    m.c34 = Constraint(expr=0.5 * m.x163 - m.x182 <= 0)

    m.c35 = Constraint(expr=0.5 * m.x154 - m.x155 + m.x174 <= 0)

    m.c36 = Constraint(expr=0.5 * m.x154 - m.x174 <= 0)

    m.c37 = Constraint(expr=0.5 * m.x164 - m.x165 + m.x183 <= 0)

    m.c38 = Constraint(expr=0.5 * m.x164 - m.x183 <= 0)

    m.c39 = Constraint(expr=- m.x74 + m.x166 - m.x167 <= 0)

    m.c40 = Constraint(expr=- m.x74 - m.x166 + m.x167 <= 0)

    m.c41 = Constraint(expr=- m.x75 + m.x175 - m.x176 <= 0)

    m.c42 = Constraint(expr=- m.x75 - m.x175 + m.x176 <= 0)

    m.c43 = Constraint(expr=- 12 * m.b1 - 12 * m.b2 + 0.5 *
                       m.x146 + 0.5 * m.x147 - m.x166 + m.x167 <= 0)

    m.c44 = Constraint(expr=- 12 * m.b1 + 12 * m.b2 + 0.5 *
                       m.x146 + 0.5 * m.x147 + m.x166 - m.x167 <= 12)

    m.c45 = Constraint(expr=13 * m.b1 - 13 * m.b2 + 0.5 *
                       m.x156 + 0.5 * m.x157 - m.x175 + m.x176 <= 13)

    m.c46 = Constraint(expr=13 * m.b1 + 13 * m.b2 + 0.5 *
                       m.x156 + 0.5 * m.x157 + m.x175 - m.x176 <= 26)

    m.c47 = Constraint(expr=- m.x76 + m.x166 - m.x168 <= 0)

    m.c48 = Constraint(expr=- m.x76 - m.x166 + m.x168 <= 0)

    m.c49 = Constraint(expr=- m.x77 + m.x175 - m.x177 <= 0)

    m.c50 = Constraint(expr=- m.x77 - m.x175 + m.x177 <= 0)

    m.c51 = Constraint(expr=- 12 * m.b3 - 12 * m.b4 + 0.5 *
                       m.x146 + 0.5 * m.x148 - m.x166 + m.x168 <= 0)

    m.c52 = Constraint(expr=- 12 * m.b3 + 12 * m.b4 + 0.5 *
                       m.x146 + 0.5 * m.x148 + m.x166 - m.x168 <= 12)

    m.c53 = Constraint(expr=13 * m.b3 - 13 * m.b4 + 0.5 *
                       m.x156 + 0.5 * m.x158 - m.x175 + m.x177 <= 13)

    m.c54 = Constraint(expr=13 * m.b3 + 13 * m.b4 + 0.5 *
                       m.x156 + 0.5 * m.x158 + m.x175 - m.x177 <= 26)

    m.c55 = Constraint(expr=- m.x78 + m.x166 - m.x169 <= 0)

    m.c56 = Constraint(expr=- m.x78 - m.x166 + m.x169 <= 0)

    m.c57 = Constraint(expr=- m.x79 + m.x175 - m.x178 <= 0)

    m.c58 = Constraint(expr=- m.x79 - m.x175 + m.x178 <= 0)

    m.c59 = Constraint(expr=- 12 * m.b5 - 12 * m.b6 + 0.5 *
                       m.x146 + 0.5 * m.x149 - m.x166 + m.x169 <= 0)

    m.c60 = Constraint(expr=- 12 * m.b5 + 12 * m.b6 + 0.5 *
                       m.x146 + 0.5 * m.x149 + m.x166 - m.x169 <= 12)

    m.c61 = Constraint(expr=13 * m.b5 - 13 * m.b6 + 0.5 *
                       m.x156 + 0.5 * m.x159 - m.x175 + m.x178 <= 13)

    m.c62 = Constraint(expr=13 * m.b5 + 13 * m.b6 + 0.5 *
                       m.x156 + 0.5 * m.x159 + m.x175 - m.x178 <= 26)

    m.c63 = Constraint(expr=- m.x80 + m.x166 - m.x170 <= 0)

    m.c64 = Constraint(expr=- m.x80 - m.x166 + m.x170 <= 0)

    m.c65 = Constraint(expr=- m.x81 + m.x175 - m.x179 <= 0)

    m.c66 = Constraint(expr=- m.x81 - m.x175 + m.x179 <= 0)

    m.c67 = Constraint(expr=- 12 * m.b7 - 12 * m.b8 + 0.5 *
                       m.x146 + 0.5 * m.x150 - m.x166 + m.x170 <= 0)

    m.c68 = Constraint(expr=- 12 * m.b7 + 12 * m.b8 + 0.5 *
                       m.x146 + 0.5 * m.x150 + m.x166 - m.x170 <= 12)

    m.c69 = Constraint(expr=13 * m.b7 - 13 * m.b8 + 0.5 *
                       m.x156 + 0.5 * m.x160 - m.x175 + m.x179 <= 13)

    m.c70 = Constraint(expr=13 * m.b7 + 13 * m.b8 + 0.5 *
                       m.x156 + 0.5 * m.x160 + m.x175 - m.x179 <= 26)

    m.c71 = Constraint(expr=- m.x82 + m.x166 - m.x171 <= 0)

    m.c72 = Constraint(expr=- m.x82 - m.x166 + m.x171 <= 0)

    m.c73 = Constraint(expr=- m.x83 + m.x175 - m.x180 <= 0)

    m.c74 = Constraint(expr=- m.x83 - m.x175 + m.x180 <= 0)

    m.c75 = Constraint(expr=- 12 * m.b9 - 12 * m.b10 + 0.5 *
                       m.x146 + 0.5 * m.x151 - m.x166 + m.x171 <= 0)

    m.c76 = Constraint(expr=- 12 * m.b9 + 12 * m.b10 + 0.5 *
                       m.x146 + 0.5 * m.x151 + m.x166 - m.x171 <= 12)

    m.c77 = Constraint(expr=13 * m.b9 - 13 * m.b10 + 0.5 *
                       m.x156 + 0.5 * m.x161 - m.x175 + m.x180 <= 13)

    m.c78 = Constraint(expr=13 * m.b9 + 13 * m.b10 + 0.5 *
                       m.x156 + 0.5 * m.x161 + m.x175 - m.x180 <= 26)

    m.c79 = Constraint(expr=- m.x84 + m.x166 - m.x172 <= 0)

    m.c80 = Constraint(expr=- m.x84 - m.x166 + m.x172 <= 0)

    m.c81 = Constraint(expr=- m.x85 + m.x175 - m.x181 <= 0)

    m.c82 = Constraint(expr=- m.x85 - m.x175 + m.x181 <= 0)

    m.c83 = Constraint(expr=- 12 * m.b11 - 12 * m.b12 + 0.5 *
                       m.x146 + 0.5 * m.x152 - m.x166 + m.x172 <= 0)

    m.c84 = Constraint(expr=- 12 * m.b11 + 12 * m.b12 + 0.5 *
                       m.x146 + 0.5 * m.x152 + m.x166 - m.x172 <= 12)

    m.c85 = Constraint(expr=13 * m.b11 - 13 * m.b12 + 0.5 *
                       m.x156 + 0.5 * m.x162 - m.x175 + m.x181 <= 13)

    m.c86 = Constraint(expr=13 * m.b11 + 13 * m.b12 + 0.5 *
                       m.x156 + 0.5 * m.x162 + m.x175 - m.x181 <= 26)

    m.c87 = Constraint(expr=- m.x86 + m.x166 - m.x173 <= 0)

    m.c88 = Constraint(expr=- m.x86 - m.x166 + m.x173 <= 0)

    m.c89 = Constraint(expr=- m.x87 + m.x175 - m.x182 <= 0)

    m.c90 = Constraint(expr=- m.x87 - m.x175 + m.x182 <= 0)

    m.c91 = Constraint(expr=- 12 * m.b13 - 12 * m.b14 + 0.5 *
                       m.x146 + 0.5 * m.x153 - m.x166 + m.x173 <= 0)

    m.c92 = Constraint(expr=- 12 * m.b13 + 12 * m.b14 + 0.5 *
                       m.x146 + 0.5 * m.x153 + m.x166 - m.x173 <= 12)

    m.c93 = Constraint(expr=13 * m.b13 - 13 * m.b14 + 0.5 *
                       m.x156 + 0.5 * m.x163 - m.x175 + m.x182 <= 13)

    m.c94 = Constraint(expr=13 * m.b13 + 13 * m.b14 + 0.5 *
                       m.x156 + 0.5 * m.x163 + m.x175 - m.x182 <= 26)

    m.c95 = Constraint(expr=- m.x88 + m.x166 - m.x174 <= 0)

    m.c96 = Constraint(expr=- m.x88 - m.x166 + m.x174 <= 0)

    m.c97 = Constraint(expr=- m.x89 + m.x175 - m.x183 <= 0)

    m.c98 = Constraint(expr=- m.x89 - m.x175 + m.x183 <= 0)

    m.c99 = Constraint(expr=- 12 * m.b15 - 12 * m.b16 + 0.5 *
                       m.x146 + 0.5 * m.x154 - m.x166 + m.x174 <= 0)

    m.c100 = Constraint(expr=- 12 * m.b15 + 12 * m.b16 +
                        0.5 * m.x146 + 0.5 * m.x154 + m.x166 - m.x174 <= 12)

    m.c101 = Constraint(expr=13 * m.b15 - 13 * m.b16 + 0.5 *
                        m.x156 + 0.5 * m.x164 - m.x175 + m.x183 <= 13)

    m.c102 = Constraint(expr=13 * m.b15 + 13 * m.b16 + 0.5 *
                        m.x156 + 0.5 * m.x164 + m.x175 - m.x183 <= 26)

    m.c103 = Constraint(expr=- m.x90 + m.x167 - m.x168 <= 0)

    m.c104 = Constraint(expr=- m.x90 - m.x167 + m.x168 <= 0)

    m.c105 = Constraint(expr=- m.x91 + m.x176 - m.x177 <= 0)

    m.c106 = Constraint(expr=- m.x91 - m.x176 + m.x177 <= 0)

    m.c107 = Constraint(expr=- 12 * m.b17 - 12 * m.b18 +
                        0.5 * m.x147 + 0.5 * m.x148 - m.x167 + m.x168 <= 0)

    m.c108 = Constraint(expr=- 12 * m.b17 + 12 * m.b18 +
                        0.5 * m.x147 + 0.5 * m.x148 + m.x167 - m.x168 <= 12)

    m.c109 = Constraint(expr=13 * m.b17 - 13 * m.b18 + 0.5 *
                        m.x157 + 0.5 * m.x158 - m.x176 + m.x177 <= 13)

    m.c110 = Constraint(expr=13 * m.b17 + 13 * m.b18 + 0.5 *
                        m.x157 + 0.5 * m.x158 + m.x176 - m.x177 <= 26)

    m.c111 = Constraint(expr=- m.x92 + m.x167 - m.x169 <= 0)

    m.c112 = Constraint(expr=- m.x92 - m.x167 + m.x169 <= 0)

    m.c113 = Constraint(expr=- m.x93 + m.x176 - m.x178 <= 0)

    m.c114 = Constraint(expr=- m.x93 - m.x176 + m.x178 <= 0)

    m.c115 = Constraint(expr=- 12 * m.b19 - 12 * m.b20 +
                        0.5 * m.x147 + 0.5 * m.x149 - m.x167 + m.x169 <= 0)

    m.c116 = Constraint(expr=- 12 * m.b19 + 12 * m.b20 +
                        0.5 * m.x147 + 0.5 * m.x149 + m.x167 - m.x169 <= 12)

    m.c117 = Constraint(expr=13 * m.b19 - 13 * m.b20 + 0.5 *
                        m.x157 + 0.5 * m.x159 - m.x176 + m.x178 <= 13)

    m.c118 = Constraint(expr=13 * m.b19 + 13 * m.b20 + 0.5 *
                        m.x157 + 0.5 * m.x159 + m.x176 - m.x178 <= 26)

    m.c119 = Constraint(expr=- m.x94 + m.x167 - m.x170 <= 0)

    m.c120 = Constraint(expr=- m.x94 - m.x167 + m.x170 <= 0)

    m.c121 = Constraint(expr=- m.x95 + m.x176 - m.x179 <= 0)

    m.c122 = Constraint(expr=- m.x95 - m.x176 + m.x179 <= 0)

    m.c123 = Constraint(expr=- 12 * m.b21 - 12 * m.b22 +
                        0.5 * m.x147 + 0.5 * m.x150 - m.x167 + m.x170 <= 0)

    m.c124 = Constraint(expr=- 12 * m.b21 + 12 * m.b22 +
                        0.5 * m.x147 + 0.5 * m.x150 + m.x167 - m.x170 <= 12)

    m.c125 = Constraint(expr=13 * m.b21 - 13 * m.b22 + 0.5 *
                        m.x157 + 0.5 * m.x160 - m.x176 + m.x179 <= 13)

    m.c126 = Constraint(expr=13 * m.b21 + 13 * m.b22 + 0.5 *
                        m.x157 + 0.5 * m.x160 + m.x176 - m.x179 <= 26)

    m.c127 = Constraint(expr=- m.x96 + m.x167 - m.x171 <= 0)

    m.c128 = Constraint(expr=- m.x96 - m.x167 + m.x171 <= 0)

    m.c129 = Constraint(expr=- m.x97 + m.x176 - m.x180 <= 0)

    m.c130 = Constraint(expr=- m.x97 - m.x176 + m.x180 <= 0)

    m.c131 = Constraint(expr=- 12 * m.b23 - 12 * m.b24 +
                        0.5 * m.x147 + 0.5 * m.x151 - m.x167 + m.x171 <= 0)

    m.c132 = Constraint(expr=- 12 * m.b23 + 12 * m.b24 +
                        0.5 * m.x147 + 0.5 * m.x151 + m.x167 - m.x171 <= 12)

    m.c133 = Constraint(expr=13 * m.b23 - 13 * m.b24 + 0.5 *
                        m.x157 + 0.5 * m.x161 - m.x176 + m.x180 <= 13)

    m.c134 = Constraint(expr=13 * m.b23 + 13 * m.b24 + 0.5 *
                        m.x157 + 0.5 * m.x161 + m.x176 - m.x180 <= 26)

    m.c135 = Constraint(expr=- m.x98 + m.x167 - m.x172 <= 0)

    m.c136 = Constraint(expr=- m.x98 - m.x167 + m.x172 <= 0)

    m.c137 = Constraint(expr=- m.x99 + m.x176 - m.x181 <= 0)

    m.c138 = Constraint(expr=- m.x99 - m.x176 + m.x181 <= 0)

    m.c139 = Constraint(expr=- 12 * m.b25 - 12 * m.b26 +
                        0.5 * m.x147 + 0.5 * m.x152 - m.x167 + m.x172 <= 0)

    m.c140 = Constraint(expr=- 12 * m.b25 + 12 * m.b26 +
                        0.5 * m.x147 + 0.5 * m.x152 + m.x167 - m.x172 <= 12)

    m.c141 = Constraint(expr=13 * m.b25 - 13 * m.b26 + 0.5 *
                        m.x157 + 0.5 * m.x162 - m.x176 + m.x181 <= 13)

    m.c142 = Constraint(expr=13 * m.b25 + 13 * m.b26 + 0.5 *
                        m.x157 + 0.5 * m.x162 + m.x176 - m.x181 <= 26)

    m.c143 = Constraint(expr=- m.x100 + m.x167 - m.x173 <= 0)

    m.c144 = Constraint(expr=- m.x100 - m.x167 + m.x173 <= 0)

    m.c145 = Constraint(expr=- m.x101 + m.x176 - m.x182 <= 0)

    m.c146 = Constraint(expr=- m.x101 - m.x176 + m.x182 <= 0)

    m.c147 = Constraint(expr=- 12 * m.b27 - 12 * m.b28 +
                        0.5 * m.x147 + 0.5 * m.x153 - m.x167 + m.x173 <= 0)

    m.c148 = Constraint(expr=- 12 * m.b27 + 12 * m.b28 +
                        0.5 * m.x147 + 0.5 * m.x153 + m.x167 - m.x173 <= 12)

    m.c149 = Constraint(expr=13 * m.b27 - 13 * m.b28 + 0.5 *
                        m.x157 + 0.5 * m.x163 - m.x176 + m.x182 <= 13)

    m.c150 = Constraint(expr=13 * m.b27 + 13 * m.b28 + 0.5 *
                        m.x157 + 0.5 * m.x163 + m.x176 - m.x182 <= 26)

    m.c151 = Constraint(expr=- m.x102 + m.x167 - m.x174 <= 0)

    m.c152 = Constraint(expr=- m.x102 - m.x167 + m.x174 <= 0)

    m.c153 = Constraint(expr=- m.x103 + m.x176 - m.x183 <= 0)

    m.c154 = Constraint(expr=- m.x103 - m.x176 + m.x183 <= 0)

    m.c155 = Constraint(expr=- 12 * m.b29 - 12 * m.b30 +
                        0.5 * m.x147 + 0.5 * m.x154 - m.x167 + m.x174 <= 0)

    m.c156 = Constraint(expr=- 12 * m.b29 + 12 * m.b30 +
                        0.5 * m.x147 + 0.5 * m.x154 + m.x167 - m.x174 <= 12)

    m.c157 = Constraint(expr=13 * m.b29 - 13 * m.b30 + 0.5 *
                        m.x157 + 0.5 * m.x164 - m.x176 + m.x183 <= 13)

    m.c158 = Constraint(expr=13 * m.b29 + 13 * m.b30 + 0.5 *
                        m.x157 + 0.5 * m.x164 + m.x176 - m.x183 <= 26)

    m.c159 = Constraint(expr=- m.x104 + m.x168 - m.x169 <= 0)

    m.c160 = Constraint(expr=- m.x104 - m.x168 + m.x169 <= 0)

    m.c161 = Constraint(expr=- m.x105 + m.x177 - m.x178 <= 0)

    m.c162 = Constraint(expr=- m.x105 - m.x177 + m.x178 <= 0)

    m.c163 = Constraint(expr=- 12 * m.b31 - 12 * m.b32 +
                        0.5 * m.x148 + 0.5 * m.x149 - m.x168 + m.x169 <= 0)

    m.c164 = Constraint(expr=- 12 * m.b31 + 12 * m.b32 +
                        0.5 * m.x148 + 0.5 * m.x149 + m.x168 - m.x169 <= 12)

    m.c165 = Constraint(expr=13 * m.b31 - 13 * m.b32 + 0.5 *
                        m.x158 + 0.5 * m.x159 - m.x177 + m.x178 <= 13)

    m.c166 = Constraint(expr=13 * m.b31 + 13 * m.b32 + 0.5 *
                        m.x158 + 0.5 * m.x159 + m.x177 - m.x178 <= 26)

    m.c167 = Constraint(expr=- m.x106 + m.x168 - m.x170 <= 0)

    m.c168 = Constraint(expr=- m.x106 - m.x168 + m.x170 <= 0)

    m.c169 = Constraint(expr=- m.x107 + m.x177 - m.x179 <= 0)

    m.c170 = Constraint(expr=- m.x107 - m.x177 + m.x179 <= 0)

    m.c171 = Constraint(expr=- 12 * m.b33 - 12 * m.b34 +
                        0.5 * m.x148 + 0.5 * m.x150 - m.x168 + m.x170 <= 0)

    m.c172 = Constraint(expr=- 12 * m.b33 + 12 * m.b34 +
                        0.5 * m.x148 + 0.5 * m.x150 + m.x168 - m.x170 <= 12)

    m.c173 = Constraint(expr=13 * m.b33 - 13 * m.b34 + 0.5 *
                        m.x158 + 0.5 * m.x160 - m.x177 + m.x179 <= 13)

    m.c174 = Constraint(expr=13 * m.b33 + 13 * m.b34 + 0.5 *
                        m.x158 + 0.5 * m.x160 + m.x177 - m.x179 <= 26)

    m.c175 = Constraint(expr=- m.x108 + m.x168 - m.x171 <= 0)

    m.c176 = Constraint(expr=- m.x108 - m.x168 + m.x171 <= 0)

    m.c177 = Constraint(expr=- m.x109 + m.x177 - m.x180 <= 0)

    m.c178 = Constraint(expr=- m.x109 - m.x177 + m.x180 <= 0)

    m.c179 = Constraint(expr=- 12 * m.b35 - 12 * m.b36 +
                        0.5 * m.x148 + 0.5 * m.x151 - m.x168 + m.x171 <= 0)

    m.c180 = Constraint(expr=- 12 * m.b35 + 12 * m.b36 +
                        0.5 * m.x148 + 0.5 * m.x151 + m.x168 - m.x171 <= 12)

    m.c181 = Constraint(expr=13 * m.b35 - 13 * m.b36 + 0.5 *
                        m.x158 + 0.5 * m.x161 - m.x177 + m.x180 <= 13)

    m.c182 = Constraint(expr=13 * m.b35 + 13 * m.b36 + 0.5 *
                        m.x158 + 0.5 * m.x161 + m.x177 - m.x180 <= 26)

    m.c183 = Constraint(expr=- m.x110 + m.x168 - m.x172 <= 0)

    m.c184 = Constraint(expr=- m.x110 - m.x168 + m.x172 <= 0)

    m.c185 = Constraint(expr=- m.x111 + m.x177 - m.x181 <= 0)

    m.c186 = Constraint(expr=- m.x111 - m.x177 + m.x181 <= 0)

    m.c187 = Constraint(expr=- 12 * m.b37 - 12 * m.b38 +
                        0.5 * m.x148 + 0.5 * m.x152 - m.x168 + m.x172 <= 0)

    m.c188 = Constraint(expr=- 12 * m.b37 + 12 * m.b38 +
                        0.5 * m.x148 + 0.5 * m.x152 + m.x168 - m.x172 <= 12)

    m.c189 = Constraint(expr=13 * m.b37 - 13 * m.b38 + 0.5 *
                        m.x158 + 0.5 * m.x162 - m.x177 + m.x181 <= 13)

    m.c190 = Constraint(expr=13 * m.b37 + 13 * m.b38 + 0.5 *
                        m.x158 + 0.5 * m.x162 + m.x177 - m.x181 <= 26)

    m.c191 = Constraint(expr=- m.x112 + m.x168 - m.x173 <= 0)

    m.c192 = Constraint(expr=- m.x112 - m.x168 + m.x173 <= 0)

    m.c193 = Constraint(expr=- m.x113 + m.x177 - m.x182 <= 0)

    m.c194 = Constraint(expr=- m.x113 - m.x177 + m.x182 <= 0)

    m.c195 = Constraint(expr=- 12 * m.b39 - 12 * m.b40 +
                        0.5 * m.x148 + 0.5 * m.x153 - m.x168 + m.x173 <= 0)

    m.c196 = Constraint(expr=- 12 * m.b39 + 12 * m.b40 +
                        0.5 * m.x148 + 0.5 * m.x153 + m.x168 - m.x173 <= 12)

    m.c197 = Constraint(expr=13 * m.b39 - 13 * m.b40 + 0.5 *
                        m.x158 + 0.5 * m.x163 - m.x177 + m.x182 <= 13)

    m.c198 = Constraint(expr=13 * m.b39 + 13 * m.b40 + 0.5 *
                        m.x158 + 0.5 * m.x163 + m.x177 - m.x182 <= 26)

    m.c199 = Constraint(expr=- m.x114 + m.x168 - m.x174 <= 0)

    m.c200 = Constraint(expr=- m.x114 - m.x168 + m.x174 <= 0)

    m.c201 = Constraint(expr=- m.x115 + m.x177 - m.x183 <= 0)

    m.c202 = Constraint(expr=- m.x115 - m.x177 + m.x183 <= 0)

    m.c203 = Constraint(expr=- 12 * m.b41 - 12 * m.b42 +
                        0.5 * m.x148 + 0.5 * m.x154 - m.x168 + m.x174 <= 0)

    m.c204 = Constraint(expr=- 12 * m.b41 + 12 * m.b42 +
                        0.5 * m.x148 + 0.5 * m.x154 + m.x168 - m.x174 <= 12)

    m.c205 = Constraint(expr=13 * m.b41 - 13 * m.b42 + 0.5 *
                        m.x158 + 0.5 * m.x164 - m.x177 + m.x183 <= 13)

    m.c206 = Constraint(expr=13 * m.b41 + 13 * m.b42 + 0.5 *
                        m.x158 + 0.5 * m.x164 + m.x177 - m.x183 <= 26)

    m.c207 = Constraint(expr=- m.x116 + m.x169 - m.x170 <= 0)

    m.c208 = Constraint(expr=- m.x116 - m.x169 + m.x170 <= 0)

    m.c209 = Constraint(expr=- m.x117 + m.x178 - m.x179 <= 0)

    m.c210 = Constraint(expr=- m.x117 - m.x178 + m.x179 <= 0)

    m.c211 = Constraint(expr=- 12 * m.b43 - 12 * m.b44 +
                        0.5 * m.x149 + 0.5 * m.x150 - m.x169 + m.x170 <= 0)

    m.c212 = Constraint(expr=- 12 * m.b43 + 12 * m.b44 +
                        0.5 * m.x149 + 0.5 * m.x150 + m.x169 - m.x170 <= 12)

    m.c213 = Constraint(expr=13 * m.b43 - 13 * m.b44 + 0.5 *
                        m.x159 + 0.5 * m.x160 - m.x178 + m.x179 <= 13)

    m.c214 = Constraint(expr=13 * m.b43 + 13 * m.b44 + 0.5 *
                        m.x159 + 0.5 * m.x160 + m.x178 - m.x179 <= 26)

    m.c215 = Constraint(expr=- m.x118 + m.x169 - m.x171 <= 0)

    m.c216 = Constraint(expr=- m.x118 - m.x169 + m.x171 <= 0)

    m.c217 = Constraint(expr=- m.x119 + m.x178 - m.x180 <= 0)

    m.c218 = Constraint(expr=- m.x119 - m.x178 + m.x180 <= 0)

    m.c219 = Constraint(expr=- 12 * m.b45 - 12 * m.b46 +
                        0.5 * m.x149 + 0.5 * m.x151 - m.x169 + m.x171 <= 0)

    m.c220 = Constraint(expr=- 12 * m.b45 + 12 * m.b46 +
                        0.5 * m.x149 + 0.5 * m.x151 + m.x169 - m.x171 <= 12)

    m.c221 = Constraint(expr=13 * m.b45 - 13 * m.b46 + 0.5 *
                        m.x159 + 0.5 * m.x161 - m.x178 + m.x180 <= 13)

    m.c222 = Constraint(expr=13 * m.b45 + 13 * m.b46 + 0.5 *
                        m.x159 + 0.5 * m.x161 + m.x178 - m.x180 <= 26)

    m.c223 = Constraint(expr=- m.x120 + m.x169 - m.x172 <= 0)

    m.c224 = Constraint(expr=- m.x120 - m.x169 + m.x172 <= 0)

    m.c225 = Constraint(expr=- m.x121 + m.x178 - m.x181 <= 0)

    m.c226 = Constraint(expr=- m.x121 - m.x178 + m.x181 <= 0)

    m.c227 = Constraint(expr=- 12 * m.b47 - 12 * m.b48 +
                        0.5 * m.x149 + 0.5 * m.x152 - m.x169 + m.x172 <= 0)

    m.c228 = Constraint(expr=- 12 * m.b47 + 12 * m.b48 +
                        0.5 * m.x149 + 0.5 * m.x152 + m.x169 - m.x172 <= 12)

    m.c229 = Constraint(expr=13 * m.b47 - 13 * m.b48 + 0.5 *
                        m.x159 + 0.5 * m.x162 - m.x178 + m.x181 <= 13)

    m.c230 = Constraint(expr=13 * m.b47 + 13 * m.b48 + 0.5 *
                        m.x159 + 0.5 * m.x162 + m.x178 - m.x181 <= 26)

    m.c231 = Constraint(expr=- m.x122 + m.x169 - m.x173 <= 0)

    m.c232 = Constraint(expr=- m.x122 - m.x169 + m.x173 <= 0)

    m.c233 = Constraint(expr=- m.x123 + m.x178 - m.x182 <= 0)

    m.c234 = Constraint(expr=- m.x123 - m.x178 + m.x182 <= 0)

    m.c235 = Constraint(expr=- 12 * m.b49 - 12 * m.b50 +
                        0.5 * m.x149 + 0.5 * m.x153 - m.x169 + m.x173 <= 0)

    m.c236 = Constraint(expr=- 12 * m.b49 + 12 * m.b50 +
                        0.5 * m.x149 + 0.5 * m.x153 + m.x169 - m.x173 <= 12)

    m.c237 = Constraint(expr=13 * m.b49 - 13 * m.b50 + 0.5 *
                        m.x159 + 0.5 * m.x163 - m.x178 + m.x182 <= 13)

    m.c238 = Constraint(expr=13 * m.b49 + 13 * m.b50 + 0.5 *
                        m.x159 + 0.5 * m.x163 + m.x178 - m.x182 <= 26)

    m.c239 = Constraint(expr=- m.x124 + m.x169 - m.x174 <= 0)

    m.c240 = Constraint(expr=- m.x124 - m.x169 + m.x174 <= 0)

    m.c241 = Constraint(expr=- m.x125 + m.x178 - m.x183 <= 0)

    m.c242 = Constraint(expr=- m.x125 - m.x178 + m.x183 <= 0)

    m.c243 = Constraint(expr=- 12 * m.b51 - 12 * m.b52 +
                        0.5 * m.x149 + 0.5 * m.x154 - m.x169 + m.x174 <= 0)

    m.c244 = Constraint(expr=- 12 * m.b51 + 12 * m.b52 +
                        0.5 * m.x149 + 0.5 * m.x154 + m.x169 - m.x174 <= 12)

    m.c245 = Constraint(expr=13 * m.b51 - 13 * m.b52 + 0.5 *
                        m.x159 + 0.5 * m.x164 - m.x178 + m.x183 <= 13)

    m.c246 = Constraint(expr=13 * m.b51 + 13 * m.b52 + 0.5 *
                        m.x159 + 0.5 * m.x164 + m.x178 - m.x183 <= 26)

    m.c247 = Constraint(expr=- m.x126 + m.x170 - m.x171 <= 0)

    m.c248 = Constraint(expr=- m.x126 - m.x170 + m.x171 <= 0)

    m.c249 = Constraint(expr=- m.x127 + m.x179 - m.x180 <= 0)

    m.c250 = Constraint(expr=- m.x127 - m.x179 + m.x180 <= 0)

    m.c251 = Constraint(expr=- 12 * m.b53 - 12 * m.b54 +
                        0.5 * m.x150 + 0.5 * m.x151 - m.x170 + m.x171 <= 0)

    m.c252 = Constraint(expr=- 12 * m.b53 + 12 * m.b54 +
                        0.5 * m.x150 + 0.5 * m.x151 + m.x170 - m.x171 <= 12)

    m.c253 = Constraint(expr=13 * m.b53 - 13 * m.b54 + 0.5 *
                        m.x160 + 0.5 * m.x161 - m.x179 + m.x180 <= 13)

    m.c254 = Constraint(expr=13 * m.b53 + 13 * m.b54 + 0.5 *
                        m.x160 + 0.5 * m.x161 + m.x179 - m.x180 <= 26)

    m.c255 = Constraint(expr=- m.x128 + m.x170 - m.x172 <= 0)

    m.c256 = Constraint(expr=- m.x128 - m.x170 + m.x172 <= 0)

    m.c257 = Constraint(expr=- m.x129 + m.x179 - m.x181 <= 0)

    m.c258 = Constraint(expr=- m.x129 - m.x179 + m.x181 <= 0)

    m.c259 = Constraint(expr=- 12 * m.b55 - 12 * m.b56 +
                        0.5 * m.x150 + 0.5 * m.x152 - m.x170 + m.x172 <= 0)

    m.c260 = Constraint(expr=- 12 * m.b55 + 12 * m.b56 +
                        0.5 * m.x150 + 0.5 * m.x152 + m.x170 - m.x172 <= 12)

    m.c261 = Constraint(expr=13 * m.b55 - 13 * m.b56 + 0.5 *
                        m.x160 + 0.5 * m.x162 - m.x179 + m.x181 <= 13)

    m.c262 = Constraint(expr=13 * m.b55 + 13 * m.b56 + 0.5 *
                        m.x160 + 0.5 * m.x162 + m.x179 - m.x181 <= 26)

    m.c263 = Constraint(expr=- m.x130 + m.x170 - m.x173 <= 0)

    m.c264 = Constraint(expr=- m.x130 - m.x170 + m.x173 <= 0)

    m.c265 = Constraint(expr=- m.x131 + m.x179 - m.x182 <= 0)

    m.c266 = Constraint(expr=- m.x131 - m.x179 + m.x182 <= 0)

    m.c267 = Constraint(expr=- 12 * m.b57 - 12 * m.b58 +
                        0.5 * m.x150 + 0.5 * m.x153 - m.x170 + m.x173 <= 0)

    m.c268 = Constraint(expr=- 12 * m.b57 + 12 * m.b58 +
                        0.5 * m.x150 + 0.5 * m.x153 + m.x170 - m.x173 <= 12)

    m.c269 = Constraint(expr=13 * m.b57 - 13 * m.b58 + 0.5 *
                        m.x160 + 0.5 * m.x163 - m.x179 + m.x182 <= 13)

    m.c270 = Constraint(expr=13 * m.b57 + 13 * m.b58 + 0.5 *
                        m.x160 + 0.5 * m.x163 + m.x179 - m.x182 <= 26)

    m.c271 = Constraint(expr=- m.x132 + m.x170 - m.x174 <= 0)

    m.c272 = Constraint(expr=- m.x132 - m.x170 + m.x174 <= 0)

    m.c273 = Constraint(expr=- m.x133 + m.x179 - m.x183 <= 0)

    m.c274 = Constraint(expr=- m.x133 - m.x179 + m.x183 <= 0)

    m.c275 = Constraint(expr=- 12 * m.b59 - 12 * m.b60 +
                        0.5 * m.x150 + 0.5 * m.x154 - m.x170 + m.x174 <= 0)

    m.c276 = Constraint(expr=- 12 * m.b59 + 12 * m.b60 +
                        0.5 * m.x150 + 0.5 * m.x154 + m.x170 - m.x174 <= 12)

    m.c277 = Constraint(expr=13 * m.b59 - 13 * m.b60 + 0.5 *
                        m.x160 + 0.5 * m.x164 - m.x179 + m.x183 <= 13)

    m.c278 = Constraint(expr=13 * m.b59 + 13 * m.b60 + 0.5 *
                        m.x160 + 0.5 * m.x164 + m.x179 - m.x183 <= 26)

    m.c279 = Constraint(expr=- m.x134 + m.x171 - m.x172 <= 0)

    m.c280 = Constraint(expr=- m.x134 - m.x171 + m.x172 <= 0)

    m.c281 = Constraint(expr=- m.x135 + m.x180 - m.x181 <= 0)

    m.c282 = Constraint(expr=- m.x135 - m.x180 + m.x181 <= 0)

    m.c283 = Constraint(expr=- 12 * m.b61 - 12 * m.b62 +
                        0.5 * m.x151 + 0.5 * m.x152 - m.x171 + m.x172 <= 0)

    m.c284 = Constraint(expr=- 12 * m.b61 + 12 * m.b62 +
                        0.5 * m.x151 + 0.5 * m.x152 + m.x171 - m.x172 <= 12)

    m.c285 = Constraint(expr=13 * m.b61 - 13 * m.b62 + 0.5 *
                        m.x161 + 0.5 * m.x162 - m.x180 + m.x181 <= 13)

    m.c286 = Constraint(expr=13 * m.b61 + 13 * m.b62 + 0.5 *
                        m.x161 + 0.5 * m.x162 + m.x180 - m.x181 <= 26)

    m.c287 = Constraint(expr=- m.x136 + m.x171 - m.x173 <= 0)

    m.c288 = Constraint(expr=- m.x136 - m.x171 + m.x173 <= 0)

    m.c289 = Constraint(expr=- m.x137 + m.x180 - m.x182 <= 0)

    m.c290 = Constraint(expr=- m.x137 - m.x180 + m.x182 <= 0)

    m.c291 = Constraint(expr=- 12 * m.b63 - 12 * m.b64 +
                        0.5 * m.x151 + 0.5 * m.x153 - m.x171 + m.x173 <= 0)

    m.c292 = Constraint(expr=- 12 * m.b63 + 12 * m.b64 +
                        0.5 * m.x151 + 0.5 * m.x153 + m.x171 - m.x173 <= 12)

    m.c293 = Constraint(expr=13 * m.b63 - 13 * m.b64 + 0.5 *
                        m.x161 + 0.5 * m.x163 - m.x180 + m.x182 <= 13)

    m.c294 = Constraint(expr=13 * m.b63 + 13 * m.b64 + 0.5 *
                        m.x161 + 0.5 * m.x163 + m.x180 - m.x182 <= 26)

    m.c295 = Constraint(expr=- m.x138 + m.x171 - m.x174 <= 0)

    m.c296 = Constraint(expr=- m.x138 - m.x171 + m.x174 <= 0)

    m.c297 = Constraint(expr=- m.x139 + m.x180 - m.x183 <= 0)

    m.c298 = Constraint(expr=- m.x139 - m.x180 + m.x183 <= 0)

    m.c299 = Constraint(expr=- 12 * m.b65 - 12 * m.b66 +
                        0.5 * m.x151 + 0.5 * m.x154 - m.x171 + m.x174 <= 0)

    m.c300 = Constraint(expr=- 12 * m.b65 + 12 * m.b66 +
                        0.5 * m.x151 + 0.5 * m.x154 + m.x171 - m.x174 <= 12)

    m.c301 = Constraint(expr=13 * m.b65 - 13 * m.b66 + 0.5 *
                        m.x161 + 0.5 * m.x164 - m.x180 + m.x183 <= 13)

    m.c302 = Constraint(expr=13 * m.b65 + 13 * m.b66 + 0.5 *
                        m.x161 + 0.5 * m.x164 + m.x180 - m.x183 <= 26)

    m.c303 = Constraint(expr=- m.x140 + m.x172 - m.x173 <= 0)

    m.c304 = Constraint(expr=- m.x140 - m.x172 + m.x173 <= 0)

    m.c305 = Constraint(expr=- m.x141 + m.x181 - m.x182 <= 0)

    m.c306 = Constraint(expr=- m.x141 - m.x181 + m.x182 <= 0)

    m.c307 = Constraint(expr=- 12 * m.b67 - 12 * m.b68 +
                        0.5 * m.x152 + 0.5 * m.x153 - m.x172 + m.x173 <= 0)

    m.c308 = Constraint(expr=- 12 * m.b67 + 12 * m.b68 +
                        0.5 * m.x152 + 0.5 * m.x153 + m.x172 - m.x173 <= 12)

    m.c309 = Constraint(expr=13 * m.b67 - 13 * m.b68 + 0.5 *
                        m.x162 + 0.5 * m.x163 - m.x181 + m.x182 <= 13)

    m.c310 = Constraint(expr=13 * m.b67 + 13 * m.b68 + 0.5 *
                        m.x162 + 0.5 * m.x163 + m.x181 - m.x182 <= 26)

    m.c311 = Constraint(expr=- m.x142 + m.x172 - m.x174 <= 0)

    m.c312 = Constraint(expr=- m.x142 - m.x172 + m.x174 <= 0)

    m.c313 = Constraint(expr=- m.x143 + m.x181 - m.x183 <= 0)

    m.c314 = Constraint(expr=- m.x143 - m.x181 + m.x183 <= 0)

    m.c315 = Constraint(expr=- 12 * m.b69 - 12 * m.b70 +
                        0.5 * m.x152 + 0.5 * m.x154 - m.x172 + m.x174 <= 0)

    m.c316 = Constraint(expr=- 12 * m.b69 + 12 * m.b70 +
                        0.5 * m.x152 + 0.5 * m.x154 + m.x172 - m.x174 <= 12)

    m.c317 = Constraint(expr=13 * m.b69 - 13 * m.b70 + 0.5 *
                        m.x162 + 0.5 * m.x164 - m.x181 + m.x183 <= 13)

    m.c318 = Constraint(expr=13 * m.b69 + 13 * m.b70 + 0.5 *
                        m.x162 + 0.5 * m.x164 + m.x181 - m.x183 <= 26)

    m.c319 = Constraint(expr=- m.x144 + m.x173 - m.x174 <= 0)

    m.c320 = Constraint(expr=- m.x144 - m.x173 + m.x174 <= 0)

    m.c321 = Constraint(expr=- m.x145 + m.x182 - m.x183 <= 0)

    m.c322 = Constraint(expr=- m.x145 - m.x182 + m.x183 <= 0)

    m.c323 = Constraint(expr=- 12 * m.b71 - 12 * m.b72 +
                        0.5 * m.x153 + 0.5 * m.x154 - m.x173 + m.x174 <= 0)

    m.c324 = Constraint(expr=- 12 * m.b71 + 12 * m.b72 +
                        0.5 * m.x153 + 0.5 * m.x154 + m.x173 - m.x174 <= 12)

    m.c325 = Constraint(expr=13 * m.b71 - 13 * m.b72 + 0.5 *
                        m.x163 + 0.5 * m.x164 - m.x182 + m.x183 <= 13)

    m.c326 = Constraint(expr=13 * m.b71 + 13 * m.b72 + 0.5 *
                        m.x163 + 0.5 * m.x164 + m.x182 - m.x183 <= 26)

    m.c327 = Constraint(expr=16 / m.x146 - m.x156 <= 0)

    m.c328 = Constraint(expr=16 / m.x156 - m.x146 <= 0)

    m.c329 = Constraint(expr=16 / m.x147 - m.x157 <= 0)

    m.c330 = Constraint(expr=16 / m.x157 - m.x147 <= 0)

    m.c331 = Constraint(expr=16 / m.x148 - m.x158 <= 0)

    m.c332 = Constraint(expr=16 / m.x158 - m.x148 <= 0)

    m.c333 = Constraint(expr=36 / m.x149 - m.x159 <= 0)

    m.c334 = Constraint(expr=36 / m.x159 - m.x149 <= 0)

    m.c335 = Constraint(expr=36 / m.x150 - m.x160 <= 0)

    m.c336 = Constraint(expr=36 / m.x160 - m.x150 <= 0)

    m.c337 = Constraint(expr=9 / m.x151 - m.x161 <= 0)

    m.c338 = Constraint(expr=9 / m.x161 - m.x151 <= 0)

    m.c339 = Constraint(expr=9 / m.x152 - m.x162 <= 0)

    m.c340 = Constraint(expr=9 / m.x162 - m.x152 <= 0)

    m.c341 = Constraint(expr=9 / m.x153 - m.x163 <= 0)

    m.c342 = Constraint(expr=9 / m.x163 - m.x153 <= 0)

    m.c343 = Constraint(expr=9 / m.x154 - m.x164 <= 0)

    m.c344 = Constraint(expr=9 / m.x164 - m.x154 <= 0)
