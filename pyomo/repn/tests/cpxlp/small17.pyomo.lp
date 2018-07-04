\* Source Pyomo model name=unknown *\

min 
x4:
+1 x1
+1 x2

s.t.

c_e_x5_:
+1 x1
+1 x3
= 1

c_e_x6_:
+1 x2
+1 x3
= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   2 <= x1 <= +inf
   1 <= x2 <= +inf
   0 <= x3 <= +inf
end
