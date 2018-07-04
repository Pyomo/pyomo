\* Source Pyomo model name=unknown *\

min 
x4:
+ [
+2 x1 ^ 2
] / 2

s.t.

c_e_x5_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x6_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x7_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x8_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x9_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x10_:
+ [
+1 x1 * x3
-1 x2 * x3
]
= 4

c_e_x11_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x12_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x13_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x14_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x15_:
+ [
+0.5 x1 * x3
-0.5 x2 * x3
]
= 2

c_e_x16_:
+ [
+1 x1 * x3
-1 x2 * x3
]
= 4

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   -1 <= x1 <= 1
   -1 <= x2 <= 1
   -1 <= x3 <= 1
end
