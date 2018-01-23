\* Source Pyomo model name=Scenario1 *\

min 
o:
+1 fx
-1 fz
+1 r
+0 ONE_VAR_CONSTANT

s.t.

c_l_c_first_stage_:
+1 x__DDSIP_FIRSTSTAGE
>= 0

c_e_p_first_stage_INC_constraint1_:
+1 x__DDSIP_FIRSTSTAGE
-2 p_first_stage_INC_delta(1)
-3 p_first_stage_INC_delta(2)
-2 p_first_stage_INC_delta(3)
-3 p_first_stage_INC_delta(4)
= 0

c_e_p_first_stage_INC_constraint2_:
+1 fx
+1 p_first_stage_INC_delta(2)
-1 p_first_stage_INC_delta(3)
= 10

c_e_p_second_stage(1)_INC_constraint1_:
-5 p_second_stage(1)_INC_delta(1)
-5 p_second_stage(1)_INC_delta(2)
-5 p_second_stage(1)_INC_delta(3)
-5 p_second_stage(1)_INC_delta(4)
+1 z
= -10

c_e_p_second_stage(1)_INC_constraint2_:
+1 fz
+1 p_second_stage(1)_INC_delta(2)
-3 p_second_stage(1)_INC_delta(3)
+1 p_second_stage(1)_INC_delta(4)
= 0

c_l_c_second_stage_:
+1 x__DDSIP_FIRSTSTAGE
+0 r
>= -100

c_u_p_first_stage_INC_constraint3(1)_:
-1 p_first_stage_INC_bin_y(1)
+1 p_first_stage_INC_delta(2)
<= 0

c_u_p_first_stage_INC_constraint3(2)_:
-1 p_first_stage_INC_bin_y(2)
+1 p_first_stage_INC_delta(3)
<= 0

c_u_p_first_stage_INC_constraint3(3)_:
-1 p_first_stage_INC_bin_y(3)
+1 p_first_stage_INC_delta(4)
<= 0

c_u_p_first_stage_INC_constraint4(1)_:
+1 p_first_stage_INC_bin_y(1)
-1 p_first_stage_INC_delta(1)
<= 0

c_u_p_first_stage_INC_constraint4(2)_:
+1 p_first_stage_INC_bin_y(2)
-1 p_first_stage_INC_delta(2)
<= 0

c_u_p_first_stage_INC_constraint4(3)_:
+1 p_first_stage_INC_bin_y(3)
-1 p_first_stage_INC_delta(3)
<= 0

c_u_p_second_stage(1)_INC_constraint3(1)_:
-1 p_second_stage(1)_INC_bin_y(1)
+1 p_second_stage(1)_INC_delta(2)
<= 0

c_u_p_second_stage(1)_INC_constraint3(2)_:
-1 p_second_stage(1)_INC_bin_y(2)
+1 p_second_stage(1)_INC_delta(3)
<= 0

c_u_p_second_stage(1)_INC_constraint3(3)_:
-1 p_second_stage(1)_INC_bin_y(3)
+1 p_second_stage(1)_INC_delta(4)
<= 0

c_u_p_second_stage(1)_INC_constraint4(1)_:
+1 p_second_stage(1)_INC_bin_y(1)
-1 p_second_stage(1)_INC_delta(1)
<= 0

c_u_p_second_stage(1)_INC_constraint4(2)_:
+1 p_second_stage(1)_INC_bin_y(2)
-1 p_second_stage(1)_INC_delta(2)
<= 0

c_u_p_second_stage(1)_INC_constraint4(3)_:
+1 p_second_stage(1)_INC_bin_y(3)
-1 p_second_stage(1)_INC_delta(3)
<= 0

r_l_r_second_stage_:
+1 r
>= 0

r_u_r_second_stage_:
+1 r
<= 0

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x__DDSIP_FIRSTSTAGE <= 10
    -inf <= fx <= +inf
   -10 <= z <= 10
    -inf <= fz <= +inf
    -inf <= r <= +inf
    -inf <= p_first_stage_INC_delta(1) <= 1
    -inf <= p_first_stage_INC_delta(2) <= +inf
    -inf <= p_first_stage_INC_delta(3) <= +inf
   0 <= p_first_stage_INC_delta(4) <= +inf
   0 <= p_first_stage_INC_bin_y(1) <= 1
   0 <= p_first_stage_INC_bin_y(2) <= 1
   0 <= p_first_stage_INC_bin_y(3) <= 1
    -inf <= p_second_stage(1)_INC_delta(1) <= 1
    -inf <= p_second_stage(1)_INC_delta(2) <= +inf
    -inf <= p_second_stage(1)_INC_delta(3) <= +inf
   0 <= p_second_stage(1)_INC_delta(4) <= +inf
   0 <= p_second_stage(1)_INC_bin_y(1) <= 1
   0 <= p_second_stage(1)_INC_bin_y(2) <= 1
   0 <= p_second_stage(1)_INC_bin_y(3) <= 1
binary
  p_first_stage_INC_bin_y(1)
  p_first_stage_INC_bin_y(2)
  p_first_stage_INC_bin_y(3)
  p_second_stage(1)_INC_bin_y(1)
  p_second_stage(1)_INC_bin_y(2)
  p_second_stage(1)_INC_bin_y(3)
end
