\* Source Pyomo model name=unknown *\

min 
x3:
+1 x1
+2 x2

s.t.

c_l_x4_:
+3 x1
+4 x2
>= 1

c_l_x5_:
+2 x1
+5 x2
>= 2

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x1 <= +inf
   0 <= x2 <= +inf
general
  x1
  x2
end
