\* Source Pyomo model name=unknown *\

min 
x4:
+1 x1

s.t.

c_e_x5_:
+1 x1
= 1

c_e_x6_:
+1 x1
= 1

c_e_x7_:
+1 x1
= 1

c_e_x8_:
+0 ONE_VAR_CONSTANT
= 1

c_e_x9_:
+0 ONE_VAR_CONSTANT
= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= x1 <= +inf
end
