\* Source Pyomo model name=unknown *\

min 
x3:
+1 x1
+1 x2

s.t.

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= +inf
   0 <= x2 <= +inf
end
