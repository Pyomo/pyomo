\* Source Pyomo model name=unknown *\

min 
o:
+3 x1
+2.5 x2

s.t.

c_u_c1_:
+4.4400000000000004 x1
<= 100

c_u_c2_:
+6.6699999999999999 x2
<= 100

c_u_c3_:
+4 x1
+2.8599999999999999 x2
<= 100

c_u_c4_:
+3 x1
+6 x2
<= 100

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
   0 <= x1 <= +inf
   0 <= x2 <= +inf
end 
