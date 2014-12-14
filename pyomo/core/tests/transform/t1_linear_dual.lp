\* Source Pyomo model name=unknown *\

max 
o:
+6 x1
+4 x2
+2 x3

s.t.

c_l_c1_:
+4 x1
+2 x2
+1 x3
>= 5

c_l_c2_:
+1 x1
+1 x2
>= 3

c_l_c3_:
+1 x2
+1 x3
>= 4

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
   0 <= x1 <= +inf
   0 <= x2 <= +inf
   0 <= x3 <= +inf
end 
