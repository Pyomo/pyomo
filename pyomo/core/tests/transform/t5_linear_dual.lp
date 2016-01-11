\* Source Pyomo model name=unknown *\

min 
o:
-100 c1
-100 c2
-100 c3
-100 c4

s.t.

c_u_x1_:
+4.4400000000000004 c1
+4 c3
+3 c4
<= -3

c_u_x2_:
+6.6699999999999999 c2
+2.8599999999999999 c3
+6 c4
<= -2.5

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
    -inf <= c1 <= 0
    -inf <= c2 <= 0
    -inf <= c3 <= 0
    -inf <= c4 <= 0
end
