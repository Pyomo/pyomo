\* Source Pyomo model name=unknown *\

min 
o:
-100.0 c1
-100.0 c2
-100.0 c3
-100.0 c4

s.t.

c_u_x1_:
+4.44 c1
+4 c3
+3 c4
<= -3

c_u_x2_:
+6.67 c2
+2.86 c3
+6 c4
<= -2.5

bounds
   -inf <= c1 <= 0
   -inf <= c2 <= 0
   -inf <= c3 <= 0
   -inf <= c4 <= 0
end
