\* Source Pyomo model name=unknown *\

min 
x4:
+1 x2
+1 x3
+ [
+2 x1 ^ 2
] / 2

s.t.

c_l_x5_:
+ [
+1 x2 ^ 2
]
>= 2

c_u_x6_:
-0.5 x1
+1 x2
<= 0

c_u_x7_:
-1 x2
+1 x3
<= 2

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= +inf
   0 <= x2 <= +inf
   7 <= x3 <= +inf
end
