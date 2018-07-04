\* Source Pyomo model name=unknown *\

min 
x3:
+ [
+2 x1 * x2
] / 2

s.t.

c_e_x4_:
+ [
+1 x2 ^ 2
]
= 4

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x1 <= +inf
    -inf <= x2 <= +inf
end
