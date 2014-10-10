\* Source Pyomo model name=unknown *\

max 
obj:
+1 X
+1 Z

s.t.

c_e_con_INC_constraint1_:
+1 X
-1 con_INC_delta(1)
-1 con_INC_delta(3)
-1 con_INC_delta(5)
= 0

c_e_con_INC_constraint2_:
+1 Z
-2 con_INC_delta(2)
+1.5 con_INC_delta(4)
= 0

c_u_con_INC_constraint3(1)_:
-1 con_INC_bin_y(1)
+1 con_INC_delta(2)
<= 0

c_u_con_INC_constraint3(2)_:
-1 con_INC_bin_y(2)
+1 con_INC_delta(3)
<= 0

c_u_con_INC_constraint3(3)_:
-1 con_INC_bin_y(3)
+1 con_INC_delta(4)
<= 0

c_u_con_INC_constraint3(4)_:
-1 con_INC_bin_y(4)
+1 con_INC_delta(5)
<= 0

c_u_con_INC_constraint4(1)_:
+1 con_INC_bin_y(1)
-1 con_INC_delta(1)
<= 0

c_u_con_INC_constraint4(2)_:
+1 con_INC_bin_y(2)
-1 con_INC_delta(2)
<= 0

c_u_con_INC_constraint4(3)_:
+1 con_INC_bin_y(3)
-1 con_INC_delta(3)
<= 0

c_u_con_INC_constraint4(4)_:
+1 con_INC_bin_y(4)
-1 con_INC_delta(4)
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
   0 <= X <= 3
    -inf <= Z <= +inf
    -inf <= con_INC_delta(1) <= 1
    -inf <= con_INC_delta(2) <= +inf
    -inf <= con_INC_delta(3) <= +inf
    -inf <= con_INC_delta(4) <= +inf
   0 <= con_INC_delta(5) <= +inf
   0 <= con_INC_bin_y(1) <= 1
   0 <= con_INC_bin_y(2) <= 1
   0 <= con_INC_bin_y(3) <= 1
   0 <= con_INC_bin_y(4) <= 1
binary
  con_INC_bin_y(1)
  con_INC_bin_y(2)
  con_INC_bin_y(3)
  con_INC_bin_y(4)
end 
