\* Source Pyomo model name=unknown *\

max 
o:
+5 c1
+3 c2
+4 c3

s.t.

c_u_x1_:
+4 c1
+1 c2
<= 6

c_u_x2_:
+2 c1
+1 c2
+1 c3
<= 4

c_u_x3_:
+1 c1
+1 c3
<= 2

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= c1 <= +inf
   0 <= c2 <= +inf
   0 <= c3 <= +inf
end
