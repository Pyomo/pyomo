\* Source Pyomo model name=unknown *\

min 
makespan:
+1 ms

s.t.

c_u_Feas(A)_:
-1 ms
+1 t(A)
<= -10

c_u_Feas(B)_:
-1 ms
+1 t(B)
<= -10

c_u_Feas(C)_:
-1 ms
+1 t(C)
<= -15

c_u_Feas(D)_:
-1 ms
+1 t(D)
<= -14

c_u_Feas(E)_:
-1 ms
+1 t(E)
<= -12

c_u_Feas(F)_:
-1 ms
+1 t(F)
<= -14

c_u_Feas(G)_:
-1 ms
+1 t(G)
<= -17

c_u_NoClash(A_B_3_0)c_hi_:
+4 NoClash(A_B_3_0)indicator_var
-1 NoClash(A_B_3_0)t(A)
+1 NoClash(A_B_3_0)t(B)
<= 0

c_u_NoClash(A_B_3_1)c_hi_:
+5 NoClash(A_B_3_1)indicator_var
+1 NoClash(A_B_3_1)t(A)
-1 NoClash(A_B_3_1)t(B)
<= 0

c_e_disj_(A_B_3)_t(A)_:
-1 NoClash(A_B_3_0)t(A)
-1 NoClash(A_B_3_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_B_3)_t(B)_:
-1 NoClash(A_B_3_0)t(B)
-1 NoClash(A_B_3_1)t(B)
+1 t(B)
= 0

c_u_NoClash(A_B_3_0)t(A)_hi_:
-92 NoClash(A_B_3_0)indicator_var
+1 NoClash(A_B_3_0)t(A)
<= 0

c_u_NoClash(A_B_3_0)t(B)_hi_:
-92 NoClash(A_B_3_0)indicator_var
+1 NoClash(A_B_3_0)t(B)
<= 0

c_u_NoClash(A_B_3_1)t(A)_hi_:
-92 NoClash(A_B_3_1)indicator_var
+1 NoClash(A_B_3_1)t(A)
<= 0

c_u_NoClash(A_B_3_1)t(B)_hi_:
-92 NoClash(A_B_3_1)indicator_var
+1 NoClash(A_B_3_1)t(B)
<= 0

c_u_NoClash(A_B_5_0)c_hi_:
+2 NoClash(A_B_5_0)indicator_var
-1 NoClash(A_B_5_0)t(A)
+1 NoClash(A_B_5_0)t(B)
<= 0

c_u_NoClash(A_B_5_1)c_hi_:
+3 NoClash(A_B_5_1)indicator_var
+1 NoClash(A_B_5_1)t(A)
-1 NoClash(A_B_5_1)t(B)
<= 0

c_e_disj_(A_B_5)_t(A)_:
-1 NoClash(A_B_5_0)t(A)
-1 NoClash(A_B_5_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_B_5)_t(B)_:
-1 NoClash(A_B_5_0)t(B)
-1 NoClash(A_B_5_1)t(B)
+1 t(B)
= 0

c_u_NoClash(A_B_5_0)t(A)_hi_:
-92 NoClash(A_B_5_0)indicator_var
+1 NoClash(A_B_5_0)t(A)
<= 0

c_u_NoClash(A_B_5_0)t(B)_hi_:
-92 NoClash(A_B_5_0)indicator_var
+1 NoClash(A_B_5_0)t(B)
<= 0

c_u_NoClash(A_B_5_1)t(A)_hi_:
-92 NoClash(A_B_5_1)indicator_var
+1 NoClash(A_B_5_1)t(A)
<= 0

c_u_NoClash(A_B_5_1)t(B)_hi_:
-92 NoClash(A_B_5_1)indicator_var
+1 NoClash(A_B_5_1)t(B)
<= 0

c_u_NoClash(A_C_1_0)c_hi_:
+6 NoClash(A_C_1_0)indicator_var
-1 NoClash(A_C_1_0)t(A)
+1 NoClash(A_C_1_0)t(C)
<= 0

c_u_NoClash(A_C_1_1)c_hi_:
+3 NoClash(A_C_1_1)indicator_var
+1 NoClash(A_C_1_1)t(A)
-1 NoClash(A_C_1_1)t(C)
<= 0

c_e_disj_(A_C_1)_t(A)_:
-1 NoClash(A_C_1_0)t(A)
-1 NoClash(A_C_1_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_C_1)_t(C)_:
-1 NoClash(A_C_1_0)t(C)
-1 NoClash(A_C_1_1)t(C)
+1 t(C)
= 0

c_u_NoClash(A_C_1_0)t(A)_hi_:
-92 NoClash(A_C_1_0)indicator_var
+1 NoClash(A_C_1_0)t(A)
<= 0

c_u_NoClash(A_C_1_0)t(C)_hi_:
-92 NoClash(A_C_1_0)indicator_var
+1 NoClash(A_C_1_0)t(C)
<= 0

c_u_NoClash(A_C_1_1)t(A)_hi_:
-92 NoClash(A_C_1_1)indicator_var
+1 NoClash(A_C_1_1)t(A)
<= 0

c_u_NoClash(A_C_1_1)t(C)_hi_:
-92 NoClash(A_C_1_1)indicator_var
+1 NoClash(A_C_1_1)t(C)
<= 0

c_u_NoClash(A_D_3_0)c_hi_:
+10 NoClash(A_D_3_0)indicator_var
-1 NoClash(A_D_3_0)t(A)
+1 NoClash(A_D_3_0)t(D)
<= 0

c_u_NoClash(A_D_3_1)c_hi_:
+1 NoClash(A_D_3_1)t(A)
-1 NoClash(A_D_3_1)t(D)
<= 0

c_e_disj_(A_D_3)_t(A)_:
-1 NoClash(A_D_3_0)t(A)
-1 NoClash(A_D_3_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_D_3)_t(D)_:
-1 NoClash(A_D_3_0)t(D)
-1 NoClash(A_D_3_1)t(D)
+1 t(D)
= 0

c_u_NoClash(A_D_3_0)t(A)_hi_:
-92 NoClash(A_D_3_0)indicator_var
+1 NoClash(A_D_3_0)t(A)
<= 0

c_u_NoClash(A_D_3_0)t(D)_hi_:
-92 NoClash(A_D_3_0)indicator_var
+1 NoClash(A_D_3_0)t(D)
<= 0

c_u_NoClash(A_D_3_1)t(A)_hi_:
-92 NoClash(A_D_3_1)indicator_var
+1 NoClash(A_D_3_1)t(A)
<= 0

c_u_NoClash(A_D_3_1)t(D)_hi_:
-92 NoClash(A_D_3_1)indicator_var
+1 NoClash(A_D_3_1)t(D)
<= 0

c_u_NoClash(A_E_3_0)c_hi_:
+7 NoClash(A_E_3_0)indicator_var
-1 NoClash(A_E_3_0)t(A)
+1 NoClash(A_E_3_0)t(E)
<= 0

c_u_NoClash(A_E_3_1)c_hi_:
+4 NoClash(A_E_3_1)indicator_var
+1 NoClash(A_E_3_1)t(A)
-1 NoClash(A_E_3_1)t(E)
<= 0

c_e_disj_(A_E_3)_t(A)_:
-1 NoClash(A_E_3_0)t(A)
-1 NoClash(A_E_3_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_E_3)_t(E)_:
-1 NoClash(A_E_3_0)t(E)
-1 NoClash(A_E_3_1)t(E)
+1 t(E)
= 0

c_u_NoClash(A_E_3_0)t(A)_hi_:
-92 NoClash(A_E_3_0)indicator_var
+1 NoClash(A_E_3_0)t(A)
<= 0

c_u_NoClash(A_E_3_0)t(E)_hi_:
-92 NoClash(A_E_3_0)indicator_var
+1 NoClash(A_E_3_0)t(E)
<= 0

c_u_NoClash(A_E_3_1)t(A)_hi_:
-92 NoClash(A_E_3_1)indicator_var
+1 NoClash(A_E_3_1)t(A)
<= 0

c_u_NoClash(A_E_3_1)t(E)_hi_:
-92 NoClash(A_E_3_1)indicator_var
+1 NoClash(A_E_3_1)t(E)
<= 0

c_u_NoClash(A_E_5_0)c_hi_:
+4 NoClash(A_E_5_0)indicator_var
-1 NoClash(A_E_5_0)t(A)
+1 NoClash(A_E_5_0)t(E)
<= 0

c_u_NoClash(A_E_5_1)c_hi_:
+1 NoClash(A_E_5_1)t(A)
-1 NoClash(A_E_5_1)t(E)
<= 0

c_e_disj_(A_E_5)_t(A)_:
-1 NoClash(A_E_5_0)t(A)
-1 NoClash(A_E_5_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_E_5)_t(E)_:
-1 NoClash(A_E_5_0)t(E)
-1 NoClash(A_E_5_1)t(E)
+1 t(E)
= 0

c_u_NoClash(A_E_5_0)t(A)_hi_:
-92 NoClash(A_E_5_0)indicator_var
+1 NoClash(A_E_5_0)t(A)
<= 0

c_u_NoClash(A_E_5_0)t(E)_hi_:
-92 NoClash(A_E_5_0)indicator_var
+1 NoClash(A_E_5_0)t(E)
<= 0

c_u_NoClash(A_E_5_1)t(A)_hi_:
-92 NoClash(A_E_5_1)indicator_var
+1 NoClash(A_E_5_1)t(A)
<= 0

c_u_NoClash(A_E_5_1)t(E)_hi_:
-92 NoClash(A_E_5_1)indicator_var
+1 NoClash(A_E_5_1)t(E)
<= 0

c_u_NoClash(A_F_1_0)c_hi_:
+2 NoClash(A_F_1_0)indicator_var
-1 NoClash(A_F_1_0)t(A)
+1 NoClash(A_F_1_0)t(F)
<= 0

c_u_NoClash(A_F_1_1)c_hi_:
+3 NoClash(A_F_1_1)indicator_var
+1 NoClash(A_F_1_1)t(A)
-1 NoClash(A_F_1_1)t(F)
<= 0

c_e_disj_(A_F_1)_t(A)_:
-1 NoClash(A_F_1_0)t(A)
-1 NoClash(A_F_1_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_F_1)_t(F)_:
-1 NoClash(A_F_1_0)t(F)
-1 NoClash(A_F_1_1)t(F)
+1 t(F)
= 0

c_u_NoClash(A_F_1_0)t(A)_hi_:
-92 NoClash(A_F_1_0)indicator_var
+1 NoClash(A_F_1_0)t(A)
<= 0

c_u_NoClash(A_F_1_0)t(F)_hi_:
-92 NoClash(A_F_1_0)indicator_var
+1 NoClash(A_F_1_0)t(F)
<= 0

c_u_NoClash(A_F_1_1)t(A)_hi_:
-92 NoClash(A_F_1_1)indicator_var
+1 NoClash(A_F_1_1)t(A)
<= 0

c_u_NoClash(A_F_1_1)t(F)_hi_:
-92 NoClash(A_F_1_1)indicator_var
+1 NoClash(A_F_1_1)t(F)
<= 0

c_u_NoClash(A_F_3_0)c_hi_:
+4 NoClash(A_F_3_0)indicator_var
-1 NoClash(A_F_3_0)t(A)
+1 NoClash(A_F_3_0)t(F)
<= 0

c_u_NoClash(A_F_3_1)c_hi_:
+6 NoClash(A_F_3_1)indicator_var
+1 NoClash(A_F_3_1)t(A)
-1 NoClash(A_F_3_1)t(F)
<= 0

c_e_disj_(A_F_3)_t(A)_:
-1 NoClash(A_F_3_0)t(A)
-1 NoClash(A_F_3_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_F_3)_t(F)_:
-1 NoClash(A_F_3_0)t(F)
-1 NoClash(A_F_3_1)t(F)
+1 t(F)
= 0

c_u_NoClash(A_F_3_0)t(A)_hi_:
-92 NoClash(A_F_3_0)indicator_var
+1 NoClash(A_F_3_0)t(A)
<= 0

c_u_NoClash(A_F_3_0)t(F)_hi_:
-92 NoClash(A_F_3_0)indicator_var
+1 NoClash(A_F_3_0)t(F)
<= 0

c_u_NoClash(A_F_3_1)t(A)_hi_:
-92 NoClash(A_F_3_1)indicator_var
+1 NoClash(A_F_3_1)t(A)
<= 0

c_u_NoClash(A_F_3_1)t(F)_hi_:
-92 NoClash(A_F_3_1)indicator_var
+1 NoClash(A_F_3_1)t(F)
<= 0

c_u_NoClash(A_G_5_0)c_hi_:
+9 NoClash(A_G_5_0)indicator_var
-1 NoClash(A_G_5_0)t(A)
+1 NoClash(A_G_5_0)t(G)
<= 0

c_u_NoClash(A_G_5_1)c_hi_:
-3 NoClash(A_G_5_1)indicator_var
+1 NoClash(A_G_5_1)t(A)
-1 NoClash(A_G_5_1)t(G)
<= 0

c_e_disj_(A_G_5)_t(A)_:
-1 NoClash(A_G_5_0)t(A)
-1 NoClash(A_G_5_1)t(A)
+1 t(A)
= 0

c_e_disj_(A_G_5)_t(G)_:
-1 NoClash(A_G_5_0)t(G)
-1 NoClash(A_G_5_1)t(G)
+1 t(G)
= 0

c_u_NoClash(A_G_5_0)t(A)_hi_:
-92 NoClash(A_G_5_0)indicator_var
+1 NoClash(A_G_5_0)t(A)
<= 0

c_u_NoClash(A_G_5_0)t(G)_hi_:
-92 NoClash(A_G_5_0)indicator_var
+1 NoClash(A_G_5_0)t(G)
<= 0

c_u_NoClash(A_G_5_1)t(A)_hi_:
-92 NoClash(A_G_5_1)indicator_var
+1 NoClash(A_G_5_1)t(A)
<= 0

c_u_NoClash(A_G_5_1)t(G)_hi_:
-92 NoClash(A_G_5_1)indicator_var
+1 NoClash(A_G_5_1)t(G)
<= 0

c_u_NoClash(B_C_2_0)c_hi_:
+9 NoClash(B_C_2_0)indicator_var
-1 NoClash(B_C_2_0)t(B)
+1 NoClash(B_C_2_0)t(C)
<= 0

c_u_NoClash(B_C_2_1)c_hi_:
-3 NoClash(B_C_2_1)indicator_var
+1 NoClash(B_C_2_1)t(B)
-1 NoClash(B_C_2_1)t(C)
<= 0

c_e_disj_(B_C_2)_t(B)_:
-1 NoClash(B_C_2_0)t(B)
-1 NoClash(B_C_2_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_C_2)_t(C)_:
-1 NoClash(B_C_2_0)t(C)
-1 NoClash(B_C_2_1)t(C)
+1 t(C)
= 0

c_u_NoClash(B_C_2_0)t(B)_hi_:
-92 NoClash(B_C_2_0)indicator_var
+1 NoClash(B_C_2_0)t(B)
<= 0

c_u_NoClash(B_C_2_0)t(C)_hi_:
-92 NoClash(B_C_2_0)indicator_var
+1 NoClash(B_C_2_0)t(C)
<= 0

c_u_NoClash(B_C_2_1)t(B)_hi_:
-92 NoClash(B_C_2_1)indicator_var
+1 NoClash(B_C_2_1)t(B)
<= 0

c_u_NoClash(B_C_2_1)t(C)_hi_:
-92 NoClash(B_C_2_1)indicator_var
+1 NoClash(B_C_2_1)t(C)
<= 0

c_u_NoClash(B_D_2_0)c_hi_:
+8 NoClash(B_D_2_0)indicator_var
-1 NoClash(B_D_2_0)t(B)
+1 NoClash(B_D_2_0)t(D)
<= 0

c_u_NoClash(B_D_2_1)c_hi_:
+3 NoClash(B_D_2_1)indicator_var
+1 NoClash(B_D_2_1)t(B)
-1 NoClash(B_D_2_1)t(D)
<= 0

c_e_disj_(B_D_2)_t(B)_:
-1 NoClash(B_D_2_0)t(B)
-1 NoClash(B_D_2_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_D_2)_t(D)_:
-1 NoClash(B_D_2_0)t(D)
-1 NoClash(B_D_2_1)t(D)
+1 t(D)
= 0

c_u_NoClash(B_D_2_0)t(B)_hi_:
-92 NoClash(B_D_2_0)indicator_var
+1 NoClash(B_D_2_0)t(B)
<= 0

c_u_NoClash(B_D_2_0)t(D)_hi_:
-92 NoClash(B_D_2_0)indicator_var
+1 NoClash(B_D_2_0)t(D)
<= 0

c_u_NoClash(B_D_2_1)t(B)_hi_:
-92 NoClash(B_D_2_1)indicator_var
+1 NoClash(B_D_2_1)t(B)
<= 0

c_u_NoClash(B_D_2_1)t(D)_hi_:
-92 NoClash(B_D_2_1)indicator_var
+1 NoClash(B_D_2_1)t(D)
<= 0

c_u_NoClash(B_D_3_0)c_hi_:
+10 NoClash(B_D_3_0)indicator_var
-1 NoClash(B_D_3_0)t(B)
+1 NoClash(B_D_3_0)t(D)
<= 0

c_u_NoClash(B_D_3_1)c_hi_:
-1 NoClash(B_D_3_1)indicator_var
+1 NoClash(B_D_3_1)t(B)
-1 NoClash(B_D_3_1)t(D)
<= 0

c_e_disj_(B_D_3)_t(B)_:
-1 NoClash(B_D_3_0)t(B)
-1 NoClash(B_D_3_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_D_3)_t(D)_:
-1 NoClash(B_D_3_0)t(D)
-1 NoClash(B_D_3_1)t(D)
+1 t(D)
= 0

c_u_NoClash(B_D_3_0)t(B)_hi_:
-92 NoClash(B_D_3_0)indicator_var
+1 NoClash(B_D_3_0)t(B)
<= 0

c_u_NoClash(B_D_3_0)t(D)_hi_:
-92 NoClash(B_D_3_0)indicator_var
+1 NoClash(B_D_3_0)t(D)
<= 0

c_u_NoClash(B_D_3_1)t(B)_hi_:
-92 NoClash(B_D_3_1)indicator_var
+1 NoClash(B_D_3_1)t(B)
<= 0

c_u_NoClash(B_D_3_1)t(D)_hi_:
-92 NoClash(B_D_3_1)indicator_var
+1 NoClash(B_D_3_1)t(D)
<= 0

c_u_NoClash(B_E_2_0)c_hi_:
+4 NoClash(B_E_2_0)indicator_var
-1 NoClash(B_E_2_0)t(B)
+1 NoClash(B_E_2_0)t(E)
<= 0

c_u_NoClash(B_E_2_1)c_hi_:
+3 NoClash(B_E_2_1)indicator_var
+1 NoClash(B_E_2_1)t(B)
-1 NoClash(B_E_2_1)t(E)
<= 0

c_e_disj_(B_E_2)_t(B)_:
-1 NoClash(B_E_2_0)t(B)
-1 NoClash(B_E_2_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_E_2)_t(E)_:
-1 NoClash(B_E_2_0)t(E)
-1 NoClash(B_E_2_1)t(E)
+1 t(E)
= 0

c_u_NoClash(B_E_2_0)t(B)_hi_:
-92 NoClash(B_E_2_0)indicator_var
+1 NoClash(B_E_2_0)t(B)
<= 0

c_u_NoClash(B_E_2_0)t(E)_hi_:
-92 NoClash(B_E_2_0)indicator_var
+1 NoClash(B_E_2_0)t(E)
<= 0

c_u_NoClash(B_E_2_1)t(B)_hi_:
-92 NoClash(B_E_2_1)indicator_var
+1 NoClash(B_E_2_1)t(B)
<= 0

c_u_NoClash(B_E_2_1)t(E)_hi_:
-92 NoClash(B_E_2_1)indicator_var
+1 NoClash(B_E_2_1)t(E)
<= 0

c_u_NoClash(B_E_3_0)c_hi_:
+7 NoClash(B_E_3_0)indicator_var
-1 NoClash(B_E_3_0)t(B)
+1 NoClash(B_E_3_0)t(E)
<= 0

c_u_NoClash(B_E_3_1)c_hi_:
+3 NoClash(B_E_3_1)indicator_var
+1 NoClash(B_E_3_1)t(B)
-1 NoClash(B_E_3_1)t(E)
<= 0

c_e_disj_(B_E_3)_t(B)_:
-1 NoClash(B_E_3_0)t(B)
-1 NoClash(B_E_3_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_E_3)_t(E)_:
-1 NoClash(B_E_3_0)t(E)
-1 NoClash(B_E_3_1)t(E)
+1 t(E)
= 0

c_u_NoClash(B_E_3_0)t(B)_hi_:
-92 NoClash(B_E_3_0)indicator_var
+1 NoClash(B_E_3_0)t(B)
<= 0

c_u_NoClash(B_E_3_0)t(E)_hi_:
-92 NoClash(B_E_3_0)indicator_var
+1 NoClash(B_E_3_0)t(E)
<= 0

c_u_NoClash(B_E_3_1)t(B)_hi_:
-92 NoClash(B_E_3_1)indicator_var
+1 NoClash(B_E_3_1)t(B)
<= 0

c_u_NoClash(B_E_3_1)t(E)_hi_:
-92 NoClash(B_E_3_1)indicator_var
+1 NoClash(B_E_3_1)t(E)
<= 0

c_u_NoClash(B_E_5_0)c_hi_:
+5 NoClash(B_E_5_0)indicator_var
-1 NoClash(B_E_5_0)t(B)
+1 NoClash(B_E_5_0)t(E)
<= 0

c_u_NoClash(B_E_5_1)c_hi_:
+1 NoClash(B_E_5_1)t(B)
-1 NoClash(B_E_5_1)t(E)
<= 0

c_e_disj_(B_E_5)_t(B)_:
-1 NoClash(B_E_5_0)t(B)
-1 NoClash(B_E_5_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_E_5)_t(E)_:
-1 NoClash(B_E_5_0)t(E)
-1 NoClash(B_E_5_1)t(E)
+1 t(E)
= 0

c_u_NoClash(B_E_5_0)t(B)_hi_:
-92 NoClash(B_E_5_0)indicator_var
+1 NoClash(B_E_5_0)t(B)
<= 0

c_u_NoClash(B_E_5_0)t(E)_hi_:
-92 NoClash(B_E_5_0)indicator_var
+1 NoClash(B_E_5_0)t(E)
<= 0

c_u_NoClash(B_E_5_1)t(B)_hi_:
-92 NoClash(B_E_5_1)indicator_var
+1 NoClash(B_E_5_1)t(B)
<= 0

c_u_NoClash(B_E_5_1)t(E)_hi_:
-92 NoClash(B_E_5_1)indicator_var
+1 NoClash(B_E_5_1)t(E)
<= 0

c_u_NoClash(B_F_3_0)c_hi_:
+4 NoClash(B_F_3_0)indicator_var
-1 NoClash(B_F_3_0)t(B)
+1 NoClash(B_F_3_0)t(F)
<= 0

c_u_NoClash(B_F_3_1)c_hi_:
+5 NoClash(B_F_3_1)indicator_var
+1 NoClash(B_F_3_1)t(B)
-1 NoClash(B_F_3_1)t(F)
<= 0

c_e_disj_(B_F_3)_t(B)_:
-1 NoClash(B_F_3_0)t(B)
-1 NoClash(B_F_3_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_F_3)_t(F)_:
-1 NoClash(B_F_3_0)t(F)
-1 NoClash(B_F_3_1)t(F)
+1 t(F)
= 0

c_u_NoClash(B_F_3_0)t(B)_hi_:
-92 NoClash(B_F_3_0)indicator_var
+1 NoClash(B_F_3_0)t(B)
<= 0

c_u_NoClash(B_F_3_0)t(F)_hi_:
-92 NoClash(B_F_3_0)indicator_var
+1 NoClash(B_F_3_0)t(F)
<= 0

c_u_NoClash(B_F_3_1)t(B)_hi_:
-92 NoClash(B_F_3_1)indicator_var
+1 NoClash(B_F_3_1)t(B)
<= 0

c_u_NoClash(B_F_3_1)t(F)_hi_:
-92 NoClash(B_F_3_1)indicator_var
+1 NoClash(B_F_3_1)t(F)
<= 0

c_u_NoClash(B_G_2_0)c_hi_:
+8 NoClash(B_G_2_0)indicator_var
-1 NoClash(B_G_2_0)t(B)
+1 NoClash(B_G_2_0)t(G)
<= 0

c_u_NoClash(B_G_2_1)c_hi_:
+3 NoClash(B_G_2_1)indicator_var
+1 NoClash(B_G_2_1)t(B)
-1 NoClash(B_G_2_1)t(G)
<= 0

c_e_disj_(B_G_2)_t(B)_:
-1 NoClash(B_G_2_0)t(B)
-1 NoClash(B_G_2_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_G_2)_t(G)_:
-1 NoClash(B_G_2_0)t(G)
-1 NoClash(B_G_2_1)t(G)
+1 t(G)
= 0

c_u_NoClash(B_G_2_0)t(B)_hi_:
-92 NoClash(B_G_2_0)indicator_var
+1 NoClash(B_G_2_0)t(B)
<= 0

c_u_NoClash(B_G_2_0)t(G)_hi_:
-92 NoClash(B_G_2_0)indicator_var
+1 NoClash(B_G_2_0)t(G)
<= 0

c_u_NoClash(B_G_2_1)t(B)_hi_:
-92 NoClash(B_G_2_1)indicator_var
+1 NoClash(B_G_2_1)t(B)
<= 0

c_u_NoClash(B_G_2_1)t(G)_hi_:
-92 NoClash(B_G_2_1)indicator_var
+1 NoClash(B_G_2_1)t(G)
<= 0

c_u_NoClash(B_G_5_0)c_hi_:
+10 NoClash(B_G_5_0)indicator_var
-1 NoClash(B_G_5_0)t(B)
+1 NoClash(B_G_5_0)t(G)
<= 0

c_u_NoClash(B_G_5_1)c_hi_:
-3 NoClash(B_G_5_1)indicator_var
+1 NoClash(B_G_5_1)t(B)
-1 NoClash(B_G_5_1)t(G)
<= 0

c_e_disj_(B_G_5)_t(B)_:
-1 NoClash(B_G_5_0)t(B)
-1 NoClash(B_G_5_1)t(B)
+1 t(B)
= 0

c_e_disj_(B_G_5)_t(G)_:
-1 NoClash(B_G_5_0)t(G)
-1 NoClash(B_G_5_1)t(G)
+1 t(G)
= 0

c_u_NoClash(B_G_5_0)t(B)_hi_:
-92 NoClash(B_G_5_0)indicator_var
+1 NoClash(B_G_5_0)t(B)
<= 0

c_u_NoClash(B_G_5_0)t(G)_hi_:
-92 NoClash(B_G_5_0)indicator_var
+1 NoClash(B_G_5_0)t(G)
<= 0

c_u_NoClash(B_G_5_1)t(B)_hi_:
-92 NoClash(B_G_5_1)indicator_var
+1 NoClash(B_G_5_1)t(B)
<= 0

c_u_NoClash(B_G_5_1)t(G)_hi_:
-92 NoClash(B_G_5_1)indicator_var
+1 NoClash(B_G_5_1)t(G)
<= 0

c_u_NoClash(C_D_2_0)c_hi_:
+2 NoClash(C_D_2_0)indicator_var
-1 NoClash(C_D_2_0)t(C)
+1 NoClash(C_D_2_0)t(D)
<= 0

c_u_NoClash(C_D_2_1)c_hi_:
+9 NoClash(C_D_2_1)indicator_var
+1 NoClash(C_D_2_1)t(C)
-1 NoClash(C_D_2_1)t(D)
<= 0

c_e_disj_(C_D_2)_t(C)_:
-1 NoClash(C_D_2_0)t(C)
-1 NoClash(C_D_2_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_D_2)_t(D)_:
-1 NoClash(C_D_2_0)t(D)
-1 NoClash(C_D_2_1)t(D)
+1 t(D)
= 0

c_u_NoClash(C_D_2_0)t(C)_hi_:
-92 NoClash(C_D_2_0)indicator_var
+1 NoClash(C_D_2_0)t(C)
<= 0

c_u_NoClash(C_D_2_0)t(D)_hi_:
-92 NoClash(C_D_2_0)indicator_var
+1 NoClash(C_D_2_0)t(D)
<= 0

c_u_NoClash(C_D_2_1)t(C)_hi_:
-92 NoClash(C_D_2_1)indicator_var
+1 NoClash(C_D_2_1)t(C)
<= 0

c_u_NoClash(C_D_2_1)t(D)_hi_:
-92 NoClash(C_D_2_1)indicator_var
+1 NoClash(C_D_2_1)t(D)
<= 0

c_u_NoClash(C_D_4_0)c_hi_:
+5 NoClash(C_D_4_0)indicator_var
-1 NoClash(C_D_4_0)t(C)
+1 NoClash(C_D_4_0)t(D)
<= 0

c_u_NoClash(C_D_4_1)c_hi_:
+2 NoClash(C_D_4_1)indicator_var
+1 NoClash(C_D_4_1)t(C)
-1 NoClash(C_D_4_1)t(D)
<= 0

c_e_disj_(C_D_4)_t(C)_:
-1 NoClash(C_D_4_0)t(C)
-1 NoClash(C_D_4_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_D_4)_t(D)_:
-1 NoClash(C_D_4_0)t(D)
-1 NoClash(C_D_4_1)t(D)
+1 t(D)
= 0

c_u_NoClash(C_D_4_0)t(C)_hi_:
-92 NoClash(C_D_4_0)indicator_var
+1 NoClash(C_D_4_0)t(C)
<= 0

c_u_NoClash(C_D_4_0)t(D)_hi_:
-92 NoClash(C_D_4_0)indicator_var
+1 NoClash(C_D_4_0)t(D)
<= 0

c_u_NoClash(C_D_4_1)t(C)_hi_:
-92 NoClash(C_D_4_1)indicator_var
+1 NoClash(C_D_4_1)t(C)
<= 0

c_u_NoClash(C_D_4_1)t(D)_hi_:
-92 NoClash(C_D_4_1)indicator_var
+1 NoClash(C_D_4_1)t(D)
<= 0

c_u_NoClash(C_E_2_0)c_hi_:
-2 NoClash(C_E_2_0)indicator_var
-1 NoClash(C_E_2_0)t(C)
+1 NoClash(C_E_2_0)t(E)
<= 0

c_u_NoClash(C_E_2_1)c_hi_:
+9 NoClash(C_E_2_1)indicator_var
+1 NoClash(C_E_2_1)t(C)
-1 NoClash(C_E_2_1)t(E)
<= 0

c_e_disj_(C_E_2)_t(C)_:
-1 NoClash(C_E_2_0)t(C)
-1 NoClash(C_E_2_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_E_2)_t(E)_:
-1 NoClash(C_E_2_0)t(E)
-1 NoClash(C_E_2_1)t(E)
+1 t(E)
= 0

c_u_NoClash(C_E_2_0)t(C)_hi_:
-92 NoClash(C_E_2_0)indicator_var
+1 NoClash(C_E_2_0)t(C)
<= 0

c_u_NoClash(C_E_2_0)t(E)_hi_:
-92 NoClash(C_E_2_0)indicator_var
+1 NoClash(C_E_2_0)t(E)
<= 0

c_u_NoClash(C_E_2_1)t(C)_hi_:
-92 NoClash(C_E_2_1)indicator_var
+1 NoClash(C_E_2_1)t(C)
<= 0

c_u_NoClash(C_E_2_1)t(E)_hi_:
-92 NoClash(C_E_2_1)indicator_var
+1 NoClash(C_E_2_1)t(E)
<= 0

c_u_NoClash(C_F_1_0)c_hi_:
+2 NoClash(C_F_1_0)indicator_var
-1 NoClash(C_F_1_0)t(C)
+1 NoClash(C_F_1_0)t(F)
<= 0

c_u_NoClash(C_F_1_1)c_hi_:
+6 NoClash(C_F_1_1)indicator_var
+1 NoClash(C_F_1_1)t(C)
-1 NoClash(C_F_1_1)t(F)
<= 0

c_e_disj_(C_F_1)_t(C)_:
-1 NoClash(C_F_1_0)t(C)
-1 NoClash(C_F_1_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_F_1)_t(F)_:
-1 NoClash(C_F_1_0)t(F)
-1 NoClash(C_F_1_1)t(F)
+1 t(F)
= 0

c_u_NoClash(C_F_1_0)t(C)_hi_:
-92 NoClash(C_F_1_0)indicator_var
+1 NoClash(C_F_1_0)t(C)
<= 0

c_u_NoClash(C_F_1_0)t(F)_hi_:
-92 NoClash(C_F_1_0)indicator_var
+1 NoClash(C_F_1_0)t(F)
<= 0

c_u_NoClash(C_F_1_1)t(C)_hi_:
-92 NoClash(C_F_1_1)indicator_var
+1 NoClash(C_F_1_1)t(C)
<= 0

c_u_NoClash(C_F_1_1)t(F)_hi_:
-92 NoClash(C_F_1_1)indicator_var
+1 NoClash(C_F_1_1)t(F)
<= 0

c_u_NoClash(C_F_4_0)c_hi_:
+5 NoClash(C_F_4_0)indicator_var
-1 NoClash(C_F_4_0)t(C)
+1 NoClash(C_F_4_0)t(F)
<= 0

c_u_NoClash(C_F_4_1)c_hi_:
+8 NoClash(C_F_4_1)indicator_var
+1 NoClash(C_F_4_1)t(C)
-1 NoClash(C_F_4_1)t(F)
<= 0

c_e_disj_(C_F_4)_t(C)_:
-1 NoClash(C_F_4_0)t(C)
-1 NoClash(C_F_4_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_F_4)_t(F)_:
-1 NoClash(C_F_4_0)t(F)
-1 NoClash(C_F_4_1)t(F)
+1 t(F)
= 0

c_u_NoClash(C_F_4_0)t(C)_hi_:
-92 NoClash(C_F_4_0)indicator_var
+1 NoClash(C_F_4_0)t(C)
<= 0

c_u_NoClash(C_F_4_0)t(F)_hi_:
-92 NoClash(C_F_4_0)indicator_var
+1 NoClash(C_F_4_0)t(F)
<= 0

c_u_NoClash(C_F_4_1)t(C)_hi_:
-92 NoClash(C_F_4_1)indicator_var
+1 NoClash(C_F_4_1)t(C)
<= 0

c_u_NoClash(C_F_4_1)t(F)_hi_:
-92 NoClash(C_F_4_1)indicator_var
+1 NoClash(C_F_4_1)t(F)
<= 0

c_u_NoClash(C_G_2_0)c_hi_:
+2 NoClash(C_G_2_0)indicator_var
-1 NoClash(C_G_2_0)t(C)
+1 NoClash(C_G_2_0)t(G)
<= 0

c_u_NoClash(C_G_2_1)c_hi_:
+9 NoClash(C_G_2_1)indicator_var
+1 NoClash(C_G_2_1)t(C)
-1 NoClash(C_G_2_1)t(G)
<= 0

c_e_disj_(C_G_2)_t(C)_:
-1 NoClash(C_G_2_0)t(C)
-1 NoClash(C_G_2_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_G_2)_t(G)_:
-1 NoClash(C_G_2_0)t(G)
-1 NoClash(C_G_2_1)t(G)
+1 t(G)
= 0

c_u_NoClash(C_G_2_0)t(C)_hi_:
-92 NoClash(C_G_2_0)indicator_var
+1 NoClash(C_G_2_0)t(C)
<= 0

c_u_NoClash(C_G_2_0)t(G)_hi_:
-92 NoClash(C_G_2_0)indicator_var
+1 NoClash(C_G_2_0)t(G)
<= 0

c_u_NoClash(C_G_2_1)t(C)_hi_:
-92 NoClash(C_G_2_1)indicator_var
+1 NoClash(C_G_2_1)t(C)
<= 0

c_u_NoClash(C_G_2_1)t(G)_hi_:
-92 NoClash(C_G_2_1)indicator_var
+1 NoClash(C_G_2_1)t(G)
<= 0

c_u_NoClash(C_G_4_0)c_hi_:
+4 NoClash(C_G_4_0)indicator_var
-1 NoClash(C_G_4_0)t(C)
+1 NoClash(C_G_4_0)t(G)
<= 0

c_u_NoClash(C_G_4_1)c_hi_:
+7 NoClash(C_G_4_1)indicator_var
+1 NoClash(C_G_4_1)t(C)
-1 NoClash(C_G_4_1)t(G)
<= 0

c_e_disj_(C_G_4)_t(C)_:
-1 NoClash(C_G_4_0)t(C)
-1 NoClash(C_G_4_1)t(C)
+1 t(C)
= 0

c_e_disj_(C_G_4)_t(G)_:
-1 NoClash(C_G_4_0)t(G)
-1 NoClash(C_G_4_1)t(G)
+1 t(G)
= 0

c_u_NoClash(C_G_4_0)t(C)_hi_:
-92 NoClash(C_G_4_0)indicator_var
+1 NoClash(C_G_4_0)t(C)
<= 0

c_u_NoClash(C_G_4_0)t(G)_hi_:
-92 NoClash(C_G_4_0)indicator_var
+1 NoClash(C_G_4_0)t(G)
<= 0

c_u_NoClash(C_G_4_1)t(C)_hi_:
-92 NoClash(C_G_4_1)indicator_var
+1 NoClash(C_G_4_1)t(C)
<= 0

c_u_NoClash(C_G_4_1)t(G)_hi_:
-92 NoClash(C_G_4_1)indicator_var
+1 NoClash(C_G_4_1)t(G)
<= 0

c_u_NoClash(D_E_2_0)c_hi_:
+4 NoClash(D_E_2_0)indicator_var
-1 NoClash(D_E_2_0)t(D)
+1 NoClash(D_E_2_0)t(E)
<= 0

c_u_NoClash(D_E_2_1)c_hi_:
+8 NoClash(D_E_2_1)indicator_var
+1 NoClash(D_E_2_1)t(D)
-1 NoClash(D_E_2_1)t(E)
<= 0

c_e_disj_(D_E_2)_t(D)_:
-1 NoClash(D_E_2_0)t(D)
-1 NoClash(D_E_2_1)t(D)
+1 t(D)
= 0

c_e_disj_(D_E_2)_t(E)_:
-1 NoClash(D_E_2_0)t(E)
-1 NoClash(D_E_2_1)t(E)
+1 t(E)
= 0

c_u_NoClash(D_E_2_0)t(D)_hi_:
-92 NoClash(D_E_2_0)indicator_var
+1 NoClash(D_E_2_0)t(D)
<= 0

c_u_NoClash(D_E_2_0)t(E)_hi_:
-92 NoClash(D_E_2_0)indicator_var
+1 NoClash(D_E_2_0)t(E)
<= 0

c_u_NoClash(D_E_2_1)t(D)_hi_:
-92 NoClash(D_E_2_1)indicator_var
+1 NoClash(D_E_2_1)t(D)
<= 0

c_u_NoClash(D_E_2_1)t(E)_hi_:
-92 NoClash(D_E_2_1)indicator_var
+1 NoClash(D_E_2_1)t(E)
<= 0

c_u_NoClash(D_E_3_0)c_hi_:
+2 NoClash(D_E_3_0)indicator_var
-1 NoClash(D_E_3_0)t(D)
+1 NoClash(D_E_3_0)t(E)
<= 0

c_u_NoClash(D_E_3_1)c_hi_:
+9 NoClash(D_E_3_1)indicator_var
+1 NoClash(D_E_3_1)t(D)
-1 NoClash(D_E_3_1)t(E)
<= 0

c_e_disj_(D_E_3)_t(D)_:
-1 NoClash(D_E_3_0)t(D)
-1 NoClash(D_E_3_1)t(D)
+1 t(D)
= 0

c_e_disj_(D_E_3)_t(E)_:
-1 NoClash(D_E_3_0)t(E)
-1 NoClash(D_E_3_1)t(E)
+1 t(E)
= 0

c_u_NoClash(D_E_3_0)t(D)_hi_:
-92 NoClash(D_E_3_0)indicator_var
+1 NoClash(D_E_3_0)t(D)
<= 0

c_u_NoClash(D_E_3_0)t(E)_hi_:
-92 NoClash(D_E_3_0)indicator_var
+1 NoClash(D_E_3_0)t(E)
<= 0

c_u_NoClash(D_E_3_1)t(D)_hi_:
-92 NoClash(D_E_3_1)indicator_var
+1 NoClash(D_E_3_1)t(D)
<= 0

c_u_NoClash(D_E_3_1)t(E)_hi_:
-92 NoClash(D_E_3_1)indicator_var
+1 NoClash(D_E_3_1)t(E)
<= 0

c_u_NoClash(D_F_3_0)c_hi_:
-1 NoClash(D_F_3_0)indicator_var
-1 NoClash(D_F_3_0)t(D)
+1 NoClash(D_F_3_0)t(F)
<= 0

c_u_NoClash(D_F_3_1)c_hi_:
+11 NoClash(D_F_3_1)indicator_var
+1 NoClash(D_F_3_1)t(D)
-1 NoClash(D_F_3_1)t(F)
<= 0

c_e_disj_(D_F_3)_t(D)_:
-1 NoClash(D_F_3_0)t(D)
-1 NoClash(D_F_3_1)t(D)
+1 t(D)
= 0

c_e_disj_(D_F_3)_t(F)_:
-1 NoClash(D_F_3_0)t(F)
-1 NoClash(D_F_3_1)t(F)
+1 t(F)
= 0

c_u_NoClash(D_F_3_0)t(D)_hi_:
-92 NoClash(D_F_3_0)indicator_var
+1 NoClash(D_F_3_0)t(D)
<= 0

c_u_NoClash(D_F_3_0)t(F)_hi_:
-92 NoClash(D_F_3_0)indicator_var
+1 NoClash(D_F_3_0)t(F)
<= 0

c_u_NoClash(D_F_3_1)t(D)_hi_:
-92 NoClash(D_F_3_1)indicator_var
+1 NoClash(D_F_3_1)t(D)
<= 0

c_u_NoClash(D_F_3_1)t(F)_hi_:
-92 NoClash(D_F_3_1)indicator_var
+1 NoClash(D_F_3_1)t(F)
<= 0

c_u_NoClash(D_F_4_0)c_hi_:
+1 NoClash(D_F_4_0)indicator_var
-1 NoClash(D_F_4_0)t(D)
+1 NoClash(D_F_4_0)t(F)
<= 0

c_u_NoClash(D_F_4_1)c_hi_:
+7 NoClash(D_F_4_1)indicator_var
+1 NoClash(D_F_4_1)t(D)
-1 NoClash(D_F_4_1)t(F)
<= 0

c_e_disj_(D_F_4)_t(D)_:
-1 NoClash(D_F_4_0)t(D)
-1 NoClash(D_F_4_1)t(D)
+1 t(D)
= 0

c_e_disj_(D_F_4)_t(F)_:
-1 NoClash(D_F_4_0)t(F)
-1 NoClash(D_F_4_1)t(F)
+1 t(F)
= 0

c_u_NoClash(D_F_4_0)t(D)_hi_:
-92 NoClash(D_F_4_0)indicator_var
+1 NoClash(D_F_4_0)t(D)
<= 0

c_u_NoClash(D_F_4_0)t(F)_hi_:
-92 NoClash(D_F_4_0)indicator_var
+1 NoClash(D_F_4_0)t(F)
<= 0

c_u_NoClash(D_F_4_1)t(D)_hi_:
-92 NoClash(D_F_4_1)indicator_var
+1 NoClash(D_F_4_1)t(D)
<= 0

c_u_NoClash(D_F_4_1)t(F)_hi_:
-92 NoClash(D_F_4_1)indicator_var
+1 NoClash(D_F_4_1)t(F)
<= 0

c_u_NoClash(D_G_2_0)c_hi_:
+8 NoClash(D_G_2_0)indicator_var
-1 NoClash(D_G_2_0)t(D)
+1 NoClash(D_G_2_0)t(G)
<= 0

c_u_NoClash(D_G_2_1)c_hi_:
+8 NoClash(D_G_2_1)indicator_var
+1 NoClash(D_G_2_1)t(D)
-1 NoClash(D_G_2_1)t(G)
<= 0

c_e_disj_(D_G_2)_t(D)_:
-1 NoClash(D_G_2_0)t(D)
-1 NoClash(D_G_2_1)t(D)
+1 t(D)
= 0

c_e_disj_(D_G_2)_t(G)_:
-1 NoClash(D_G_2_0)t(G)
-1 NoClash(D_G_2_1)t(G)
+1 t(G)
= 0

c_u_NoClash(D_G_2_0)t(D)_hi_:
-92 NoClash(D_G_2_0)indicator_var
+1 NoClash(D_G_2_0)t(D)
<= 0

c_u_NoClash(D_G_2_0)t(G)_hi_:
-92 NoClash(D_G_2_0)indicator_var
+1 NoClash(D_G_2_0)t(G)
<= 0

c_u_NoClash(D_G_2_1)t(D)_hi_:
-92 NoClash(D_G_2_1)indicator_var
+1 NoClash(D_G_2_1)t(D)
<= 0

c_u_NoClash(D_G_2_1)t(G)_hi_:
-92 NoClash(D_G_2_1)indicator_var
+1 NoClash(D_G_2_1)t(G)
<= 0

c_u_NoClash(D_G_4_0)c_hi_:
-1 NoClash(D_G_4_0)t(D)
+1 NoClash(D_G_4_0)t(G)
<= 0

c_u_NoClash(D_G_4_1)c_hi_:
+6 NoClash(D_G_4_1)indicator_var
+1 NoClash(D_G_4_1)t(D)
-1 NoClash(D_G_4_1)t(G)
<= 0

c_e_disj_(D_G_4)_t(D)_:
-1 NoClash(D_G_4_0)t(D)
-1 NoClash(D_G_4_1)t(D)
+1 t(D)
= 0

c_e_disj_(D_G_4)_t(G)_:
-1 NoClash(D_G_4_0)t(G)
-1 NoClash(D_G_4_1)t(G)
+1 t(G)
= 0

c_u_NoClash(D_G_4_0)t(D)_hi_:
-92 NoClash(D_G_4_0)indicator_var
+1 NoClash(D_G_4_0)t(D)
<= 0

c_u_NoClash(D_G_4_0)t(G)_hi_:
-92 NoClash(D_G_4_0)indicator_var
+1 NoClash(D_G_4_0)t(G)
<= 0

c_u_NoClash(D_G_4_1)t(D)_hi_:
-92 NoClash(D_G_4_1)indicator_var
+1 NoClash(D_G_4_1)t(D)
<= 0

c_u_NoClash(D_G_4_1)t(G)_hi_:
-92 NoClash(D_G_4_1)indicator_var
+1 NoClash(D_G_4_1)t(G)
<= 0

c_u_NoClash(E_F_3_0)c_hi_:
+3 NoClash(E_F_3_0)indicator_var
-1 NoClash(E_F_3_0)t(E)
+1 NoClash(E_F_3_0)t(F)
<= 0

c_u_NoClash(E_F_3_1)c_hi_:
+8 NoClash(E_F_3_1)indicator_var
+1 NoClash(E_F_3_1)t(E)
-1 NoClash(E_F_3_1)t(F)
<= 0

c_e_disj_(E_F_3)_t(E)_:
-1 NoClash(E_F_3_0)t(E)
-1 NoClash(E_F_3_1)t(E)
+1 t(E)
= 0

c_e_disj_(E_F_3)_t(F)_:
-1 NoClash(E_F_3_0)t(F)
-1 NoClash(E_F_3_1)t(F)
+1 t(F)
= 0

c_u_NoClash(E_F_3_0)t(E)_hi_:
-92 NoClash(E_F_3_0)indicator_var
+1 NoClash(E_F_3_0)t(E)
<= 0

c_u_NoClash(E_F_3_0)t(F)_hi_:
-92 NoClash(E_F_3_0)indicator_var
+1 NoClash(E_F_3_0)t(F)
<= 0

c_u_NoClash(E_F_3_1)t(E)_hi_:
-92 NoClash(E_F_3_1)indicator_var
+1 NoClash(E_F_3_1)t(E)
<= 0

c_u_NoClash(E_F_3_1)t(F)_hi_:
-92 NoClash(E_F_3_1)indicator_var
+1 NoClash(E_F_3_1)t(F)
<= 0

c_u_NoClash(E_G_2_0)c_hi_:
+8 NoClash(E_G_2_0)indicator_var
-1 NoClash(E_G_2_0)t(E)
+1 NoClash(E_G_2_0)t(G)
<= 0

c_u_NoClash(E_G_2_1)c_hi_:
+4 NoClash(E_G_2_1)indicator_var
+1 NoClash(E_G_2_1)t(E)
-1 NoClash(E_G_2_1)t(G)
<= 0

c_e_disj_(E_G_2)_t(E)_:
-1 NoClash(E_G_2_0)t(E)
-1 NoClash(E_G_2_1)t(E)
+1 t(E)
= 0

c_e_disj_(E_G_2)_t(G)_:
-1 NoClash(E_G_2_0)t(G)
-1 NoClash(E_G_2_1)t(G)
+1 t(G)
= 0

c_u_NoClash(E_G_2_0)t(E)_hi_:
-92 NoClash(E_G_2_0)indicator_var
+1 NoClash(E_G_2_0)t(E)
<= 0

c_u_NoClash(E_G_2_0)t(G)_hi_:
-92 NoClash(E_G_2_0)indicator_var
+1 NoClash(E_G_2_0)t(G)
<= 0

c_u_NoClash(E_G_2_1)t(E)_hi_:
-92 NoClash(E_G_2_1)indicator_var
+1 NoClash(E_G_2_1)t(E)
<= 0

c_u_NoClash(E_G_2_1)t(G)_hi_:
-92 NoClash(E_G_2_1)indicator_var
+1 NoClash(E_G_2_1)t(G)
<= 0

c_u_NoClash(E_G_5_0)c_hi_:
+7 NoClash(E_G_5_0)indicator_var
-1 NoClash(E_G_5_0)t(E)
+1 NoClash(E_G_5_0)t(G)
<= 0

c_u_NoClash(E_G_5_1)c_hi_:
-1 NoClash(E_G_5_1)indicator_var
+1 NoClash(E_G_5_1)t(E)
-1 NoClash(E_G_5_1)t(G)
<= 0

c_e_disj_(E_G_5)_t(E)_:
-1 NoClash(E_G_5_0)t(E)
-1 NoClash(E_G_5_1)t(E)
+1 t(E)
= 0

c_e_disj_(E_G_5)_t(G)_:
-1 NoClash(E_G_5_0)t(G)
-1 NoClash(E_G_5_1)t(G)
+1 t(G)
= 0

c_u_NoClash(E_G_5_0)t(E)_hi_:
-92 NoClash(E_G_5_0)indicator_var
+1 NoClash(E_G_5_0)t(E)
<= 0

c_u_NoClash(E_G_5_0)t(G)_hi_:
-92 NoClash(E_G_5_0)indicator_var
+1 NoClash(E_G_5_0)t(G)
<= 0

c_u_NoClash(E_G_5_1)t(E)_hi_:
-92 NoClash(E_G_5_1)indicator_var
+1 NoClash(E_G_5_1)t(E)
<= 0

c_u_NoClash(E_G_5_1)t(G)_hi_:
-92 NoClash(E_G_5_1)indicator_var
+1 NoClash(E_G_5_1)t(G)
<= 0

c_u_NoClash(F_G_4_0)c_hi_:
+6 NoClash(F_G_4_0)indicator_var
-1 NoClash(F_G_4_0)t(F)
+1 NoClash(F_G_4_0)t(G)
<= 0

c_u_NoClash(F_G_4_1)c_hi_:
+6 NoClash(F_G_4_1)indicator_var
+1 NoClash(F_G_4_1)t(F)
-1 NoClash(F_G_4_1)t(G)
<= 0

c_e_disj_(F_G_4)_t(F)_:
-1 NoClash(F_G_4_0)t(F)
-1 NoClash(F_G_4_1)t(F)
+1 t(F)
= 0

c_e_disj_(F_G_4)_t(G)_:
-1 NoClash(F_G_4_0)t(G)
-1 NoClash(F_G_4_1)t(G)
+1 t(G)
= 0

c_u_NoClash(F_G_4_0)t(F)_hi_:
-92 NoClash(F_G_4_0)indicator_var
+1 NoClash(F_G_4_0)t(F)
<= 0

c_u_NoClash(F_G_4_0)t(G)_hi_:
-92 NoClash(F_G_4_0)indicator_var
+1 NoClash(F_G_4_0)t(G)
<= 0

c_u_NoClash(F_G_4_1)t(F)_hi_:
-92 NoClash(F_G_4_1)indicator_var
+1 NoClash(F_G_4_1)t(F)
<= 0

c_u_NoClash(F_G_4_1)t(G)_hi_:
-92 NoClash(F_G_4_1)indicator_var
+1 NoClash(F_G_4_1)t(G)
<= 0

c_e__gdp_relax_chull_disj(A_B_3)_:
+1 NoClash(A_B_3_0)indicator_var
+1 NoClash(A_B_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_B_5)_:
+1 NoClash(A_B_5_0)indicator_var
+1 NoClash(A_B_5_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_C_1)_:
+1 NoClash(A_C_1_0)indicator_var
+1 NoClash(A_C_1_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_D_3)_:
+1 NoClash(A_D_3_0)indicator_var
+1 NoClash(A_D_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_E_3)_:
+1 NoClash(A_E_3_0)indicator_var
+1 NoClash(A_E_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_E_5)_:
+1 NoClash(A_E_5_0)indicator_var
+1 NoClash(A_E_5_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_F_1)_:
+1 NoClash(A_F_1_0)indicator_var
+1 NoClash(A_F_1_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_F_3)_:
+1 NoClash(A_F_3_0)indicator_var
+1 NoClash(A_F_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(A_G_5)_:
+1 NoClash(A_G_5_0)indicator_var
+1 NoClash(A_G_5_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_C_2)_:
+1 NoClash(B_C_2_0)indicator_var
+1 NoClash(B_C_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_D_2)_:
+1 NoClash(B_D_2_0)indicator_var
+1 NoClash(B_D_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_D_3)_:
+1 NoClash(B_D_3_0)indicator_var
+1 NoClash(B_D_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_E_2)_:
+1 NoClash(B_E_2_0)indicator_var
+1 NoClash(B_E_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_E_3)_:
+1 NoClash(B_E_3_0)indicator_var
+1 NoClash(B_E_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_E_5)_:
+1 NoClash(B_E_5_0)indicator_var
+1 NoClash(B_E_5_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_F_3)_:
+1 NoClash(B_F_3_0)indicator_var
+1 NoClash(B_F_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_G_2)_:
+1 NoClash(B_G_2_0)indicator_var
+1 NoClash(B_G_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(B_G_5)_:
+1 NoClash(B_G_5_0)indicator_var
+1 NoClash(B_G_5_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_D_2)_:
+1 NoClash(C_D_2_0)indicator_var
+1 NoClash(C_D_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_D_4)_:
+1 NoClash(C_D_4_0)indicator_var
+1 NoClash(C_D_4_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_E_2)_:
+1 NoClash(C_E_2_0)indicator_var
+1 NoClash(C_E_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_F_1)_:
+1 NoClash(C_F_1_0)indicator_var
+1 NoClash(C_F_1_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_F_4)_:
+1 NoClash(C_F_4_0)indicator_var
+1 NoClash(C_F_4_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_G_2)_:
+1 NoClash(C_G_2_0)indicator_var
+1 NoClash(C_G_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(C_G_4)_:
+1 NoClash(C_G_4_0)indicator_var
+1 NoClash(C_G_4_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(D_E_2)_:
+1 NoClash(D_E_2_0)indicator_var
+1 NoClash(D_E_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(D_E_3)_:
+1 NoClash(D_E_3_0)indicator_var
+1 NoClash(D_E_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(D_F_3)_:
+1 NoClash(D_F_3_0)indicator_var
+1 NoClash(D_F_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(D_F_4)_:
+1 NoClash(D_F_4_0)indicator_var
+1 NoClash(D_F_4_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(D_G_2)_:
+1 NoClash(D_G_2_0)indicator_var
+1 NoClash(D_G_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(D_G_4)_:
+1 NoClash(D_G_4_0)indicator_var
+1 NoClash(D_G_4_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(E_F_3)_:
+1 NoClash(E_F_3_0)indicator_var
+1 NoClash(E_F_3_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(E_G_2)_:
+1 NoClash(E_G_2_0)indicator_var
+1 NoClash(E_G_2_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(E_G_5)_:
+1 NoClash(E_G_5_0)indicator_var
+1 NoClash(E_G_5_1)indicator_var
= 1

c_e__gdp_relax_chull_disj(F_G_4)_:
+1 NoClash(F_G_4_0)indicator_var
+1 NoClash(F_G_4_1)indicator_var
= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
    -inf <= ms <= +inf
   0 <= t(A) <= 92
   0 <= t(B) <= 92
   0 <= t(C) <= 92
   0 <= t(D) <= 92
   0 <= t(E) <= 92
   0 <= t(F) <= 92
   0 <= t(G) <= 92
   0 <= NoClash(A_B_3_0)t(A) <= 92
   0 <= NoClash(A_B_3_0)t(B) <= 92
   0 <= NoClash(A_B_3_1)t(A) <= 92
   0 <= NoClash(A_B_3_1)t(B) <= 92
   0 <= NoClash(A_B_5_0)t(A) <= 92
   0 <= NoClash(A_B_5_0)t(B) <= 92
   0 <= NoClash(A_B_5_1)t(A) <= 92
   0 <= NoClash(A_B_5_1)t(B) <= 92
   0 <= NoClash(A_C_1_0)t(A) <= 92
   0 <= NoClash(A_C_1_0)t(C) <= 92
   0 <= NoClash(A_C_1_1)t(A) <= 92
   0 <= NoClash(A_C_1_1)t(C) <= 92
   0 <= NoClash(A_D_3_0)t(A) <= 92
   0 <= NoClash(A_D_3_0)t(D) <= 92
   0 <= NoClash(A_D_3_1)t(A) <= 92
   0 <= NoClash(A_D_3_1)t(D) <= 92
   0 <= NoClash(A_E_3_0)t(A) <= 92
   0 <= NoClash(A_E_3_0)t(E) <= 92
   0 <= NoClash(A_E_3_1)t(A) <= 92
   0 <= NoClash(A_E_3_1)t(E) <= 92
   0 <= NoClash(A_E_5_0)t(A) <= 92
   0 <= NoClash(A_E_5_0)t(E) <= 92
   0 <= NoClash(A_E_5_1)t(A) <= 92
   0 <= NoClash(A_E_5_1)t(E) <= 92
   0 <= NoClash(A_F_1_0)t(A) <= 92
   0 <= NoClash(A_F_1_0)t(F) <= 92
   0 <= NoClash(A_F_1_1)t(A) <= 92
   0 <= NoClash(A_F_1_1)t(F) <= 92
   0 <= NoClash(A_F_3_0)t(A) <= 92
   0 <= NoClash(A_F_3_0)t(F) <= 92
   0 <= NoClash(A_F_3_1)t(A) <= 92
   0 <= NoClash(A_F_3_1)t(F) <= 92
   0 <= NoClash(A_G_5_0)t(A) <= 92
   0 <= NoClash(A_G_5_0)t(G) <= 92
   0 <= NoClash(A_G_5_1)t(A) <= 92
   0 <= NoClash(A_G_5_1)t(G) <= 92
   0 <= NoClash(B_C_2_0)t(B) <= 92
   0 <= NoClash(B_C_2_0)t(C) <= 92
   0 <= NoClash(B_C_2_1)t(B) <= 92
   0 <= NoClash(B_C_2_1)t(C) <= 92
   0 <= NoClash(B_D_2_0)t(B) <= 92
   0 <= NoClash(B_D_2_0)t(D) <= 92
   0 <= NoClash(B_D_2_1)t(B) <= 92
   0 <= NoClash(B_D_2_1)t(D) <= 92
   0 <= NoClash(B_D_3_0)t(B) <= 92
   0 <= NoClash(B_D_3_0)t(D) <= 92
   0 <= NoClash(B_D_3_1)t(B) <= 92
   0 <= NoClash(B_D_3_1)t(D) <= 92
   0 <= NoClash(B_E_2_0)t(B) <= 92
   0 <= NoClash(B_E_2_0)t(E) <= 92
   0 <= NoClash(B_E_2_1)t(B) <= 92
   0 <= NoClash(B_E_2_1)t(E) <= 92
   0 <= NoClash(B_E_3_0)t(B) <= 92
   0 <= NoClash(B_E_3_0)t(E) <= 92
   0 <= NoClash(B_E_3_1)t(B) <= 92
   0 <= NoClash(B_E_3_1)t(E) <= 92
   0 <= NoClash(B_E_5_0)t(B) <= 92
   0 <= NoClash(B_E_5_0)t(E) <= 92
   0 <= NoClash(B_E_5_1)t(B) <= 92
   0 <= NoClash(B_E_5_1)t(E) <= 92
   0 <= NoClash(B_F_3_0)t(B) <= 92
   0 <= NoClash(B_F_3_0)t(F) <= 92
   0 <= NoClash(B_F_3_1)t(B) <= 92
   0 <= NoClash(B_F_3_1)t(F) <= 92
   0 <= NoClash(B_G_2_0)t(B) <= 92
   0 <= NoClash(B_G_2_0)t(G) <= 92
   0 <= NoClash(B_G_2_1)t(B) <= 92
   0 <= NoClash(B_G_2_1)t(G) <= 92
   0 <= NoClash(B_G_5_0)t(B) <= 92
   0 <= NoClash(B_G_5_0)t(G) <= 92
   0 <= NoClash(B_G_5_1)t(B) <= 92
   0 <= NoClash(B_G_5_1)t(G) <= 92
   0 <= NoClash(C_D_2_0)t(C) <= 92
   0 <= NoClash(C_D_2_0)t(D) <= 92
   0 <= NoClash(C_D_2_1)t(C) <= 92
   0 <= NoClash(C_D_2_1)t(D) <= 92
   0 <= NoClash(C_D_4_0)t(C) <= 92
   0 <= NoClash(C_D_4_0)t(D) <= 92
   0 <= NoClash(C_D_4_1)t(C) <= 92
   0 <= NoClash(C_D_4_1)t(D) <= 92
   0 <= NoClash(C_E_2_0)t(C) <= 92
   0 <= NoClash(C_E_2_0)t(E) <= 92
   0 <= NoClash(C_E_2_1)t(C) <= 92
   0 <= NoClash(C_E_2_1)t(E) <= 92
   0 <= NoClash(C_F_1_0)t(C) <= 92
   0 <= NoClash(C_F_1_0)t(F) <= 92
   0 <= NoClash(C_F_1_1)t(C) <= 92
   0 <= NoClash(C_F_1_1)t(F) <= 92
   0 <= NoClash(C_F_4_0)t(C) <= 92
   0 <= NoClash(C_F_4_0)t(F) <= 92
   0 <= NoClash(C_F_4_1)t(C) <= 92
   0 <= NoClash(C_F_4_1)t(F) <= 92
   0 <= NoClash(C_G_2_0)t(C) <= 92
   0 <= NoClash(C_G_2_0)t(G) <= 92
   0 <= NoClash(C_G_2_1)t(C) <= 92
   0 <= NoClash(C_G_2_1)t(G) <= 92
   0 <= NoClash(C_G_4_0)t(C) <= 92
   0 <= NoClash(C_G_4_0)t(G) <= 92
   0 <= NoClash(C_G_4_1)t(C) <= 92
   0 <= NoClash(C_G_4_1)t(G) <= 92
   0 <= NoClash(D_E_2_0)t(D) <= 92
   0 <= NoClash(D_E_2_0)t(E) <= 92
   0 <= NoClash(D_E_2_1)t(D) <= 92
   0 <= NoClash(D_E_2_1)t(E) <= 92
   0 <= NoClash(D_E_3_0)t(D) <= 92
   0 <= NoClash(D_E_3_0)t(E) <= 92
   0 <= NoClash(D_E_3_1)t(D) <= 92
   0 <= NoClash(D_E_3_1)t(E) <= 92
   0 <= NoClash(D_F_3_0)t(D) <= 92
   0 <= NoClash(D_F_3_0)t(F) <= 92
   0 <= NoClash(D_F_3_1)t(D) <= 92
   0 <= NoClash(D_F_3_1)t(F) <= 92
   0 <= NoClash(D_F_4_0)t(D) <= 92
   0 <= NoClash(D_F_4_0)t(F) <= 92
   0 <= NoClash(D_F_4_1)t(D) <= 92
   0 <= NoClash(D_F_4_1)t(F) <= 92
   0 <= NoClash(D_G_2_0)t(D) <= 92
   0 <= NoClash(D_G_2_0)t(G) <= 92
   0 <= NoClash(D_G_2_1)t(D) <= 92
   0 <= NoClash(D_G_2_1)t(G) <= 92
   0 <= NoClash(D_G_4_0)t(D) <= 92
   0 <= NoClash(D_G_4_0)t(G) <= 92
   0 <= NoClash(D_G_4_1)t(D) <= 92
   0 <= NoClash(D_G_4_1)t(G) <= 92
   0 <= NoClash(E_F_3_0)t(E) <= 92
   0 <= NoClash(E_F_3_0)t(F) <= 92
   0 <= NoClash(E_F_3_1)t(E) <= 92
   0 <= NoClash(E_F_3_1)t(F) <= 92
   0 <= NoClash(E_G_2_0)t(E) <= 92
   0 <= NoClash(E_G_2_0)t(G) <= 92
   0 <= NoClash(E_G_2_1)t(E) <= 92
   0 <= NoClash(E_G_2_1)t(G) <= 92
   0 <= NoClash(E_G_5_0)t(E) <= 92
   0 <= NoClash(E_G_5_0)t(G) <= 92
   0 <= NoClash(E_G_5_1)t(E) <= 92
   0 <= NoClash(E_G_5_1)t(G) <= 92
   0 <= NoClash(F_G_4_0)t(F) <= 92
   0 <= NoClash(F_G_4_0)t(G) <= 92
   0 <= NoClash(F_G_4_1)t(F) <= 92
   0 <= NoClash(F_G_4_1)t(G) <= 92
   0 <= NoClash(A_B_3_0)indicator_var <= 1
   0 <= NoClash(A_B_3_1)indicator_var <= 1
   0 <= NoClash(A_B_5_0)indicator_var <= 1
   0 <= NoClash(A_B_5_1)indicator_var <= 1
   0 <= NoClash(A_C_1_0)indicator_var <= 1
   0 <= NoClash(A_C_1_1)indicator_var <= 1
   0 <= NoClash(A_D_3_0)indicator_var <= 1
   0 <= NoClash(A_D_3_1)indicator_var <= 1
   0 <= NoClash(A_E_3_0)indicator_var <= 1
   0 <= NoClash(A_E_3_1)indicator_var <= 1
   0 <= NoClash(A_E_5_0)indicator_var <= 1
   0 <= NoClash(A_E_5_1)indicator_var <= 1
   0 <= NoClash(A_F_1_0)indicator_var <= 1
   0 <= NoClash(A_F_1_1)indicator_var <= 1
   0 <= NoClash(A_F_3_0)indicator_var <= 1
   0 <= NoClash(A_F_3_1)indicator_var <= 1
   0 <= NoClash(A_G_5_0)indicator_var <= 1
   0 <= NoClash(A_G_5_1)indicator_var <= 1
   0 <= NoClash(B_C_2_0)indicator_var <= 1
   0 <= NoClash(B_C_2_1)indicator_var <= 1
   0 <= NoClash(B_D_2_0)indicator_var <= 1
   0 <= NoClash(B_D_2_1)indicator_var <= 1
   0 <= NoClash(B_D_3_0)indicator_var <= 1
   0 <= NoClash(B_D_3_1)indicator_var <= 1
   0 <= NoClash(B_E_2_0)indicator_var <= 1
   0 <= NoClash(B_E_2_1)indicator_var <= 1
   0 <= NoClash(B_E_3_0)indicator_var <= 1
   0 <= NoClash(B_E_3_1)indicator_var <= 1
   0 <= NoClash(B_E_5_0)indicator_var <= 1
   0 <= NoClash(B_E_5_1)indicator_var <= 1
   0 <= NoClash(B_F_3_0)indicator_var <= 1
   0 <= NoClash(B_F_3_1)indicator_var <= 1
   0 <= NoClash(B_G_2_0)indicator_var <= 1
   0 <= NoClash(B_G_2_1)indicator_var <= 1
   0 <= NoClash(B_G_5_0)indicator_var <= 1
   0 <= NoClash(B_G_5_1)indicator_var <= 1
   0 <= NoClash(C_D_2_0)indicator_var <= 1
   0 <= NoClash(C_D_2_1)indicator_var <= 1
   0 <= NoClash(C_D_4_0)indicator_var <= 1
   0 <= NoClash(C_D_4_1)indicator_var <= 1
   0 <= NoClash(C_E_2_0)indicator_var <= 1
   0 <= NoClash(C_E_2_1)indicator_var <= 1
   0 <= NoClash(C_F_1_0)indicator_var <= 1
   0 <= NoClash(C_F_1_1)indicator_var <= 1
   0 <= NoClash(C_F_4_0)indicator_var <= 1
   0 <= NoClash(C_F_4_1)indicator_var <= 1
   0 <= NoClash(C_G_2_0)indicator_var <= 1
   0 <= NoClash(C_G_2_1)indicator_var <= 1
   0 <= NoClash(C_G_4_0)indicator_var <= 1
   0 <= NoClash(C_G_4_1)indicator_var <= 1
   0 <= NoClash(D_E_2_0)indicator_var <= 1
   0 <= NoClash(D_E_2_1)indicator_var <= 1
   0 <= NoClash(D_E_3_0)indicator_var <= 1
   0 <= NoClash(D_E_3_1)indicator_var <= 1
   0 <= NoClash(D_F_3_0)indicator_var <= 1
   0 <= NoClash(D_F_3_1)indicator_var <= 1
   0 <= NoClash(D_F_4_0)indicator_var <= 1
   0 <= NoClash(D_F_4_1)indicator_var <= 1
   0 <= NoClash(D_G_2_0)indicator_var <= 1
   0 <= NoClash(D_G_2_1)indicator_var <= 1
   0 <= NoClash(D_G_4_0)indicator_var <= 1
   0 <= NoClash(D_G_4_1)indicator_var <= 1
   0 <= NoClash(E_F_3_0)indicator_var <= 1
   0 <= NoClash(E_F_3_1)indicator_var <= 1
   0 <= NoClash(E_G_2_0)indicator_var <= 1
   0 <= NoClash(E_G_2_1)indicator_var <= 1
   0 <= NoClash(E_G_5_0)indicator_var <= 1
   0 <= NoClash(E_G_5_1)indicator_var <= 1
   0 <= NoClash(F_G_4_0)indicator_var <= 1
   0 <= NoClash(F_G_4_1)indicator_var <= 1
binary
  NoClash(A_B_3_0)indicator_var
  NoClash(A_B_3_1)indicator_var
  NoClash(A_B_5_0)indicator_var
  NoClash(A_B_5_1)indicator_var
  NoClash(A_C_1_0)indicator_var
  NoClash(A_C_1_1)indicator_var
  NoClash(A_D_3_0)indicator_var
  NoClash(A_D_3_1)indicator_var
  NoClash(A_E_3_0)indicator_var
  NoClash(A_E_3_1)indicator_var
  NoClash(A_E_5_0)indicator_var
  NoClash(A_E_5_1)indicator_var
  NoClash(A_F_1_0)indicator_var
  NoClash(A_F_1_1)indicator_var
  NoClash(A_F_3_0)indicator_var
  NoClash(A_F_3_1)indicator_var
  NoClash(A_G_5_0)indicator_var
  NoClash(A_G_5_1)indicator_var
  NoClash(B_C_2_0)indicator_var
  NoClash(B_C_2_1)indicator_var
  NoClash(B_D_2_0)indicator_var
  NoClash(B_D_2_1)indicator_var
  NoClash(B_D_3_0)indicator_var
  NoClash(B_D_3_1)indicator_var
  NoClash(B_E_2_0)indicator_var
  NoClash(B_E_2_1)indicator_var
  NoClash(B_E_3_0)indicator_var
  NoClash(B_E_3_1)indicator_var
  NoClash(B_E_5_0)indicator_var
  NoClash(B_E_5_1)indicator_var
  NoClash(B_F_3_0)indicator_var
  NoClash(B_F_3_1)indicator_var
  NoClash(B_G_2_0)indicator_var
  NoClash(B_G_2_1)indicator_var
  NoClash(B_G_5_0)indicator_var
  NoClash(B_G_5_1)indicator_var
  NoClash(C_D_2_0)indicator_var
  NoClash(C_D_2_1)indicator_var
  NoClash(C_D_4_0)indicator_var
  NoClash(C_D_4_1)indicator_var
  NoClash(C_E_2_0)indicator_var
  NoClash(C_E_2_1)indicator_var
  NoClash(C_F_1_0)indicator_var
  NoClash(C_F_1_1)indicator_var
  NoClash(C_F_4_0)indicator_var
  NoClash(C_F_4_1)indicator_var
  NoClash(C_G_2_0)indicator_var
  NoClash(C_G_2_1)indicator_var
  NoClash(C_G_4_0)indicator_var
  NoClash(C_G_4_1)indicator_var
  NoClash(D_E_2_0)indicator_var
  NoClash(D_E_2_1)indicator_var
  NoClash(D_E_3_0)indicator_var
  NoClash(D_E_3_1)indicator_var
  NoClash(D_F_3_0)indicator_var
  NoClash(D_F_3_1)indicator_var
  NoClash(D_F_4_0)indicator_var
  NoClash(D_F_4_1)indicator_var
  NoClash(D_G_2_0)indicator_var
  NoClash(D_G_2_1)indicator_var
  NoClash(D_G_4_0)indicator_var
  NoClash(D_G_4_1)indicator_var
  NoClash(E_F_3_0)indicator_var
  NoClash(E_F_3_1)indicator_var
  NoClash(E_G_2_0)indicator_var
  NoClash(E_G_2_1)indicator_var
  NoClash(E_G_5_0)indicator_var
  NoClash(E_G_5_1)indicator_var
  NoClash(F_G_4_0)indicator_var
  NoClash(F_G_4_1)indicator_var
end 
