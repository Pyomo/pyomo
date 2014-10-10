\* Source Pyomo model name=unknown *\

min 
makespan:
+1 ms

s.t.

c_u_Feas(A)_:
-1 ms
+1 t(A)
<= -8

c_u_Feas(B)_:
-1 ms
+1 t(B)
<= -5

c_u_Feas(C)_:
-1 ms
+1 t(C)
<= -6

c_e__gdp_relax_bigm_disj(A_B_3)_:
+1 _gdp_relax_bigm_NoClash(A_B_3_0)_indicator_var
+1 _gdp_relax_bigm_NoClash(A_B_3_1)_indicator_var
= 1

c_e__gdp_relax_bigm_disj(A_C_1)_:
+1 _gdp_relax_bigm_NoClash(A_C_1_0)_indicator_var
+1 _gdp_relax_bigm_NoClash(A_C_1_1)_indicator_var
= 1

c_e__gdp_relax_bigm_disj(B_C_2)_:
+1 _gdp_relax_bigm_NoClash(B_C_2_0)_indicator_var
+1 _gdp_relax_bigm_NoClash(B_C_2_1)_indicator_var
= 1

c_u__gdp_relax_bigm_NoClash(A_B_3_0)_c_eq_:
+19 _gdp_relax_bigm_NoClash(A_B_3_0)_indicator_var
-1 t(A)
+1 t(B)
<= 19

c_u__gdp_relax_bigm_NoClash(A_B_3_1)_c_eq_:
+24 _gdp_relax_bigm_NoClash(A_B_3_1)_indicator_var
+1 t(A)
-1 t(B)
<= 19

c_u__gdp_relax_bigm_NoClash(A_C_1_0)_c_eq_:
+21 _gdp_relax_bigm_NoClash(A_C_1_0)_indicator_var
-1 t(A)
+1 t(C)
<= 19

c_u__gdp_relax_bigm_NoClash(A_C_1_1)_c_eq_:
+24 _gdp_relax_bigm_NoClash(A_C_1_1)_indicator_var
+1 t(A)
-1 t(C)
<= 19

c_u__gdp_relax_bigm_NoClash(B_C_2_0)_c_eq_:
+25 _gdp_relax_bigm_NoClash(B_C_2_0)_indicator_var
-1 t(B)
+1 t(C)
<= 19

c_u__gdp_relax_bigm_NoClash(B_C_2_1)_c_eq_:
+20 _gdp_relax_bigm_NoClash(B_C_2_1)_indicator_var
+1 t(B)
-1 t(C)
<= 19

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
    -inf <= ms <= +inf
   0 <= t(A) <= 19
   0 <= t(B) <= 19
   0 <= t(C) <= 19
   0 <= _gdp_relax_bigm_NoClash(A_B_3_0)_indicator_var <= 1
   0 <= _gdp_relax_bigm_NoClash(A_B_3_1)_indicator_var <= 1
   0 <= _gdp_relax_bigm_NoClash(A_C_1_0)_indicator_var <= 1
   0 <= _gdp_relax_bigm_NoClash(A_C_1_1)_indicator_var <= 1
   0 <= _gdp_relax_bigm_NoClash(B_C_2_0)_indicator_var <= 1
   0 <= _gdp_relax_bigm_NoClash(B_C_2_1)_indicator_var <= 1
binary
  _gdp_relax_bigm_NoClash(A_B_3_0)_indicator_var
  _gdp_relax_bigm_NoClash(A_B_3_1)_indicator_var
  _gdp_relax_bigm_NoClash(A_C_1_0)_indicator_var
  _gdp_relax_bigm_NoClash(A_C_1_1)_indicator_var
  _gdp_relax_bigm_NoClash(B_C_2_0)_indicator_var
  _gdp_relax_bigm_NoClash(B_C_2_1)_indicator_var
end 
