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

c_e__gdp_chull_relaxation_disj_disaggregation(A_B_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_B_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_B_5_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_B_5_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_C_1_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_C_1_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_D_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_D_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_E_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_E_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_E_5_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_E_5_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_F_1_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_F_1_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_F_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_F_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_G_5_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(A_G_5_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(A)
+1 t(A)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_C_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_C_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_D_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_D_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_D_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_D_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_E_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_E_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_E_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_E_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_E_5_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_E_5_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_F_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_F_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_G_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_G_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_G_5_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(B_G_5_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(B)
+1 t(B)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_D_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_D_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_D_4_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_D_4_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_E_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_E_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_F_1_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_F_1_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_F_4_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_F_4_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_G_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_G_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_G_4_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(C_G_4_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(C)
+1 t(C)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_E_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_E_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_E_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_E_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_F_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_F_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_F_4_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_F_4_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_G_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_G_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_G_4_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(D_G_4_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(D)
+1 t(D)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(E_F_3_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(E_F_3_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(E_G_2_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(E_G_2_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(E_G_5_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(E_G_5_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(E)
+1 t(E)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(F_G_4_0)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(G)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(G)
+1 t(G)
= 0

c_e__gdp_chull_relaxation_disj_disaggregation(F_G_4_1)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(F)
+1 t(F)
= 0

c_e__gdp_chull_relaxation_disj_xor(A_B_3)_:
+1 NoClash(A_B_3_0)_indicator_var
+1 NoClash(A_B_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_B_5)_:
+1 NoClash(A_B_5_0)_indicator_var
+1 NoClash(A_B_5_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_C_1)_:
+1 NoClash(A_C_1_0)_indicator_var
+1 NoClash(A_C_1_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_D_3)_:
+1 NoClash(A_D_3_0)_indicator_var
+1 NoClash(A_D_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_E_3)_:
+1 NoClash(A_E_3_0)_indicator_var
+1 NoClash(A_E_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_E_5)_:
+1 NoClash(A_E_5_0)_indicator_var
+1 NoClash(A_E_5_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_F_1)_:
+1 NoClash(A_F_1_0)_indicator_var
+1 NoClash(A_F_1_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_F_3)_:
+1 NoClash(A_F_3_0)_indicator_var
+1 NoClash(A_F_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(A_G_5)_:
+1 NoClash(A_G_5_0)_indicator_var
+1 NoClash(A_G_5_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_C_2)_:
+1 NoClash(B_C_2_0)_indicator_var
+1 NoClash(B_C_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_D_2)_:
+1 NoClash(B_D_2_0)_indicator_var
+1 NoClash(B_D_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_D_3)_:
+1 NoClash(B_D_3_0)_indicator_var
+1 NoClash(B_D_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_E_2)_:
+1 NoClash(B_E_2_0)_indicator_var
+1 NoClash(B_E_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_E_3)_:
+1 NoClash(B_E_3_0)_indicator_var
+1 NoClash(B_E_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_E_5)_:
+1 NoClash(B_E_5_0)_indicator_var
+1 NoClash(B_E_5_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_F_3)_:
+1 NoClash(B_F_3_0)_indicator_var
+1 NoClash(B_F_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_G_2)_:
+1 NoClash(B_G_2_0)_indicator_var
+1 NoClash(B_G_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(B_G_5)_:
+1 NoClash(B_G_5_0)_indicator_var
+1 NoClash(B_G_5_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_D_2)_:
+1 NoClash(C_D_2_0)_indicator_var
+1 NoClash(C_D_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_D_4)_:
+1 NoClash(C_D_4_0)_indicator_var
+1 NoClash(C_D_4_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_E_2)_:
+1 NoClash(C_E_2_0)_indicator_var
+1 NoClash(C_E_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_F_1)_:
+1 NoClash(C_F_1_0)_indicator_var
+1 NoClash(C_F_1_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_F_4)_:
+1 NoClash(C_F_4_0)_indicator_var
+1 NoClash(C_F_4_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_G_2)_:
+1 NoClash(C_G_2_0)_indicator_var
+1 NoClash(C_G_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(C_G_4)_:
+1 NoClash(C_G_4_0)_indicator_var
+1 NoClash(C_G_4_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(D_E_2)_:
+1 NoClash(D_E_2_0)_indicator_var
+1 NoClash(D_E_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(D_E_3)_:
+1 NoClash(D_E_3_0)_indicator_var
+1 NoClash(D_E_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(D_F_3)_:
+1 NoClash(D_F_3_0)_indicator_var
+1 NoClash(D_F_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(D_F_4)_:
+1 NoClash(D_F_4_0)_indicator_var
+1 NoClash(D_F_4_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(D_G_2)_:
+1 NoClash(D_G_2_0)_indicator_var
+1 NoClash(D_G_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(D_G_4)_:
+1 NoClash(D_G_4_0)_indicator_var
+1 NoClash(D_G_4_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(E_F_3)_:
+1 NoClash(E_F_3_0)_indicator_var
+1 NoClash(E_F_3_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(E_G_2)_:
+1 NoClash(E_G_2_0)_indicator_var
+1 NoClash(E_G_2_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(E_G_5)_:
+1 NoClash(E_G_5_0)_indicator_var
+1 NoClash(E_G_5_1)_indicator_var
= 1

c_e__gdp_chull_relaxation_disj_xor(F_G_4)_:
+1 NoClash(F_G_4_0)_indicator_var
+1 NoClash(F_G_4_1)_indicator_var
= 1

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(B)_bounds(ub)_:
-92 NoClash(A_B_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(A)_bounds(ub)_:
-92 NoClash(A_B_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_NoClash(A_B_3_0)_c(ub)_:
+4 NoClash(A_B_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(B)_bounds(ub)_:
-92 NoClash(A_B_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(A)_bounds(ub)_:
-92 NoClash(A_B_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_NoClash(A_B_3_1)_c(ub)_:
+5 NoClash(A_B_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(B)_bounds(ub)_:
-92 NoClash(A_B_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(A)_bounds(ub)_:
-92 NoClash(A_B_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_NoClash(A_B_5_0)_c(ub)_:
+2 NoClash(A_B_5_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(B)_bounds(ub)_:
-92 NoClash(A_B_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(A)_bounds(ub)_:
-92 NoClash(A_B_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_NoClash(A_B_5_1)_c(ub)_:
+3 NoClash(A_B_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(C)_bounds(ub)_:
-92 NoClash(A_C_1_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(A)_bounds(ub)_:
-92 NoClash(A_C_1_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_NoClash(A_C_1_0)_c(ub)_:
+6 NoClash(A_C_1_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(C)_bounds(ub)_:
-92 NoClash(A_C_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(A)_bounds(ub)_:
-92 NoClash(A_C_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_NoClash(A_C_1_1)_c(ub)_:
+3 NoClash(A_C_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(D)_bounds(ub)_:
-92 NoClash(A_D_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(A)_bounds(ub)_:
-92 NoClash(A_D_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_NoClash(A_D_3_0)_c(ub)_:
+10 NoClash(A_D_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(D)_bounds(ub)_:
-92 NoClash(A_D_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(A)_bounds(ub)_:
-92 NoClash(A_D_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_NoClash(A_D_3_1)_c(ub)_:
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(E)_bounds(ub)_:
-92 NoClash(A_E_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(A)_bounds(ub)_:
-92 NoClash(A_E_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_NoClash(A_E_3_0)_c(ub)_:
+7 NoClash(A_E_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(E)_bounds(ub)_:
-92 NoClash(A_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(A)_bounds(ub)_:
-92 NoClash(A_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_NoClash(A_E_3_1)_c(ub)_:
+4 NoClash(A_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(E)_bounds(ub)_:
-92 NoClash(A_E_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(A)_bounds(ub)_:
-92 NoClash(A_E_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_NoClash(A_E_5_0)_c(ub)_:
+4 NoClash(A_E_5_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(E)_bounds(ub)_:
-92 NoClash(A_E_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(A)_bounds(ub)_:
-92 NoClash(A_E_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_NoClash(A_E_5_1)_c(ub)_:
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(F)_bounds(ub)_:
-92 NoClash(A_F_1_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(A)_bounds(ub)_:
-92 NoClash(A_F_1_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_NoClash(A_F_1_0)_c(ub)_:
+2 NoClash(A_F_1_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(F)_bounds(ub)_:
-92 NoClash(A_F_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(A)_bounds(ub)_:
-92 NoClash(A_F_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_NoClash(A_F_1_1)_c(ub)_:
+3 NoClash(A_F_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(F)_bounds(ub)_:
-92 NoClash(A_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(A)_bounds(ub)_:
-92 NoClash(A_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_NoClash(A_F_3_0)_c(ub)_:
+4 NoClash(A_F_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(F)_bounds(ub)_:
-92 NoClash(A_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(A)_bounds(ub)_:
-92 NoClash(A_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_NoClash(A_F_3_1)_c(ub)_:
+6 NoClash(A_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(G)_bounds(ub)_:
-92 NoClash(A_G_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(A)_bounds(ub)_:
-92 NoClash(A_G_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_NoClash(A_G_5_0)_c(ub)_:
+9 NoClash(A_G_5_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(A)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(G)_bounds(ub)_:
-92 NoClash(A_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(A)_bounds(ub)_:
-92 NoClash(A_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(A)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_NoClash(A_G_5_1)_c(ub)_:
-3 NoClash(A_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(A)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(C)_bounds(ub)_:
-92 NoClash(B_C_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(B)_bounds(ub)_:
-92 NoClash(B_C_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_NoClash(B_C_2_0)_c(ub)_:
+9 NoClash(B_C_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(C)_bounds(ub)_:
-92 NoClash(B_C_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(B)_bounds(ub)_:
-92 NoClash(B_C_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_NoClash(B_C_2_1)_c(ub)_:
-3 NoClash(B_C_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(D)_bounds(ub)_:
-92 NoClash(B_D_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(B)_bounds(ub)_:
-92 NoClash(B_D_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_NoClash(B_D_2_0)_c(ub)_:
+8 NoClash(B_D_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(D)_bounds(ub)_:
-92 NoClash(B_D_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(B)_bounds(ub)_:
-92 NoClash(B_D_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_NoClash(B_D_2_1)_c(ub)_:
+3 NoClash(B_D_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(D)_bounds(ub)_:
-92 NoClash(B_D_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(B)_bounds(ub)_:
-92 NoClash(B_D_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_NoClash(B_D_3_0)_c(ub)_:
+10 NoClash(B_D_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(D)_bounds(ub)_:
-92 NoClash(B_D_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(B)_bounds(ub)_:
-92 NoClash(B_D_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_NoClash(B_D_3_1)_c(ub)_:
-1 NoClash(B_D_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(E)_bounds(ub)_:
-92 NoClash(B_E_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(B)_bounds(ub)_:
-92 NoClash(B_E_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_NoClash(B_E_2_0)_c(ub)_:
+4 NoClash(B_E_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(E)_bounds(ub)_:
-92 NoClash(B_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(B)_bounds(ub)_:
-92 NoClash(B_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_NoClash(B_E_2_1)_c(ub)_:
+3 NoClash(B_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(E)_bounds(ub)_:
-92 NoClash(B_E_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(B)_bounds(ub)_:
-92 NoClash(B_E_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_NoClash(B_E_3_0)_c(ub)_:
+7 NoClash(B_E_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(E)_bounds(ub)_:
-92 NoClash(B_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(B)_bounds(ub)_:
-92 NoClash(B_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_NoClash(B_E_3_1)_c(ub)_:
+3 NoClash(B_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(E)_bounds(ub)_:
-92 NoClash(B_E_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(B)_bounds(ub)_:
-92 NoClash(B_E_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_NoClash(B_E_5_0)_c(ub)_:
+5 NoClash(B_E_5_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(E)_bounds(ub)_:
-92 NoClash(B_E_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(B)_bounds(ub)_:
-92 NoClash(B_E_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_NoClash(B_E_5_1)_c(ub)_:
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(F)_bounds(ub)_:
-92 NoClash(B_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(B)_bounds(ub)_:
-92 NoClash(B_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_NoClash(B_F_3_0)_c(ub)_:
+4 NoClash(B_F_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(F)_bounds(ub)_:
-92 NoClash(B_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(B)_bounds(ub)_:
-92 NoClash(B_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_NoClash(B_F_3_1)_c(ub)_:
+5 NoClash(B_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(G)_bounds(ub)_:
-92 NoClash(B_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(B)_bounds(ub)_:
-92 NoClash(B_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_NoClash(B_G_2_0)_c(ub)_:
+8 NoClash(B_G_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(G)_bounds(ub)_:
-92 NoClash(B_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(B)_bounds(ub)_:
-92 NoClash(B_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_NoClash(B_G_2_1)_c(ub)_:
+3 NoClash(B_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(G)_bounds(ub)_:
-92 NoClash(B_G_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(B)_bounds(ub)_:
-92 NoClash(B_G_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_NoClash(B_G_5_0)_c(ub)_:
+10 NoClash(B_G_5_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(B)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(G)_bounds(ub)_:
-92 NoClash(B_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(B)_bounds(ub)_:
-92 NoClash(B_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(B)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_NoClash(B_G_5_1)_c(ub)_:
-3 NoClash(B_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(B)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(D)_bounds(ub)_:
-92 NoClash(C_D_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(C)_bounds(ub)_:
-92 NoClash(C_D_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_NoClash(C_D_2_0)_c(ub)_:
+2 NoClash(C_D_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(D)_bounds(ub)_:
-92 NoClash(C_D_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(C)_bounds(ub)_:
-92 NoClash(C_D_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_NoClash(C_D_2_1)_c(ub)_:
+9 NoClash(C_D_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(D)_bounds(ub)_:
-92 NoClash(C_D_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(C)_bounds(ub)_:
-92 NoClash(C_D_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_NoClash(C_D_4_0)_c(ub)_:
+5 NoClash(C_D_4_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(D)_bounds(ub)_:
-92 NoClash(C_D_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(C)_bounds(ub)_:
-92 NoClash(C_D_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_NoClash(C_D_4_1)_c(ub)_:
+2 NoClash(C_D_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(E)_bounds(ub)_:
-92 NoClash(C_E_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(C)_bounds(ub)_:
-92 NoClash(C_E_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_NoClash(C_E_2_0)_c(ub)_:
-2 NoClash(C_E_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(E)_bounds(ub)_:
-92 NoClash(C_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(C)_bounds(ub)_:
-92 NoClash(C_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_NoClash(C_E_2_1)_c(ub)_:
+9 NoClash(C_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(F)_bounds(ub)_:
-92 NoClash(C_F_1_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(C)_bounds(ub)_:
-92 NoClash(C_F_1_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_NoClash(C_F_1_0)_c(ub)_:
+2 NoClash(C_F_1_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(F)_bounds(ub)_:
-92 NoClash(C_F_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(C)_bounds(ub)_:
-92 NoClash(C_F_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_NoClash(C_F_1_1)_c(ub)_:
+6 NoClash(C_F_1_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(F)_bounds(ub)_:
-92 NoClash(C_F_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(C)_bounds(ub)_:
-92 NoClash(C_F_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_NoClash(C_F_4_0)_c(ub)_:
+5 NoClash(C_F_4_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(F)_bounds(ub)_:
-92 NoClash(C_F_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(C)_bounds(ub)_:
-92 NoClash(C_F_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_NoClash(C_F_4_1)_c(ub)_:
+8 NoClash(C_F_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(G)_bounds(ub)_:
-92 NoClash(C_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(C)_bounds(ub)_:
-92 NoClash(C_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_NoClash(C_G_2_0)_c(ub)_:
+2 NoClash(C_G_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(G)_bounds(ub)_:
-92 NoClash(C_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(C)_bounds(ub)_:
-92 NoClash(C_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_NoClash(C_G_2_1)_c(ub)_:
+9 NoClash(C_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(G)_bounds(ub)_:
-92 NoClash(C_G_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(C)_bounds(ub)_:
-92 NoClash(C_G_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_NoClash(C_G_4_0)_c(ub)_:
+4 NoClash(C_G_4_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(C)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(G)_bounds(ub)_:
-92 NoClash(C_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(C)_bounds(ub)_:
-92 NoClash(C_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(C)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_NoClash(C_G_4_1)_c(ub)_:
+7 NoClash(C_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(C)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(E)_bounds(ub)_:
-92 NoClash(D_E_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(D)_bounds(ub)_:
-92 NoClash(D_E_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_NoClash(D_E_2_0)_c(ub)_:
+4 NoClash(D_E_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(D)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(E)_bounds(ub)_:
-92 NoClash(D_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(D)_bounds(ub)_:
-92 NoClash(D_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_NoClash(D_E_2_1)_c(ub)_:
+8 NoClash(D_E_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(E)_bounds(ub)_:
-92 NoClash(D_E_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(D)_bounds(ub)_:
-92 NoClash(D_E_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_NoClash(D_E_3_0)_c(ub)_:
+2 NoClash(D_E_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(D)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(E)_bounds(ub)_:
-92 NoClash(D_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(D)_bounds(ub)_:
-92 NoClash(D_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_NoClash(D_E_3_1)_c(ub)_:
+9 NoClash(D_E_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(F)_bounds(ub)_:
-92 NoClash(D_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(D)_bounds(ub)_:
-92 NoClash(D_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_NoClash(D_F_3_0)_c(ub)_:
-1 NoClash(D_F_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(D)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(F)_bounds(ub)_:
-92 NoClash(D_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(D)_bounds(ub)_:
-92 NoClash(D_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_NoClash(D_F_3_1)_c(ub)_:
+11 NoClash(D_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(F)_bounds(ub)_:
-92 NoClash(D_F_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(D)_bounds(ub)_:
-92 NoClash(D_F_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_NoClash(D_F_4_0)_c(ub)_:
+1 NoClash(D_F_4_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(D)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(F)_bounds(ub)_:
-92 NoClash(D_F_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(D)_bounds(ub)_:
-92 NoClash(D_F_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_NoClash(D_F_4_1)_c(ub)_:
+7 NoClash(D_F_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(G)_bounds(ub)_:
-92 NoClash(D_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(D)_bounds(ub)_:
-92 NoClash(D_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_NoClash(D_G_2_0)_c(ub)_:
+8 NoClash(D_G_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(D)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(G)_bounds(ub)_:
-92 NoClash(D_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(D)_bounds(ub)_:
-92 NoClash(D_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_NoClash(D_G_2_1)_c(ub)_:
+8 NoClash(D_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(G)_bounds(ub)_:
-92 NoClash(D_G_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(D)_bounds(ub)_:
-92 NoClash(D_G_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_NoClash(D_G_4_0)_c(ub)_:
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(D)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(G)_bounds(ub)_:
-92 NoClash(D_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(D)_bounds(ub)_:
-92 NoClash(D_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(D)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_NoClash(D_G_4_1)_c(ub)_:
+6 NoClash(D_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(D)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(F)_bounds(ub)_:
-92 NoClash(E_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(E)_bounds(ub)_:
-92 NoClash(E_F_3_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_NoClash(E_F_3_0)_c(ub)_:
+3 NoClash(E_F_3_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(E)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(F)_bounds(ub)_:
-92 NoClash(E_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(E)_bounds(ub)_:
-92 NoClash(E_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_NoClash(E_F_3_1)_c(ub)_:
+8 NoClash(E_F_3_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(G)_bounds(ub)_:
-92 NoClash(E_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(E)_bounds(ub)_:
-92 NoClash(E_G_2_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_NoClash(E_G_2_0)_c(ub)_:
+8 NoClash(E_G_2_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(E)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(G)_bounds(ub)_:
-92 NoClash(E_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(E)_bounds(ub)_:
-92 NoClash(E_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_NoClash(E_G_2_1)_c(ub)_:
+4 NoClash(E_G_2_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(G)_bounds(ub)_:
-92 NoClash(E_G_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(E)_bounds(ub)_:
-92 NoClash(E_G_5_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_NoClash(E_G_5_0)_c(ub)_:
+7 NoClash(E_G_5_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(E)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(G)_bounds(ub)_:
-92 NoClash(E_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(E)_bounds(ub)_:
-92 NoClash(E_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(E)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_NoClash(E_G_5_1)_c(ub)_:
-1 NoClash(E_G_5_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(E)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(G)_bounds(ub)_:
-92 NoClash(F_G_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(F)_bounds(ub)_:
-92 NoClash(F_G_4_0)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_NoClash(F_G_4_0)_c(ub)_:
+6 NoClash(F_G_4_0)_indicator_var
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(F)
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(G)_bounds(ub)_:
-92 NoClash(F_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(G)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(F)_bounds(ub)_:
-92 NoClash(F_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(F)
<= 0

c_u__pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_NoClash(F_G_4_1)_c(ub)_:
+6 NoClash(F_G_4_1)_indicator_var
+1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(F)
-1 _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(G)
<= 0

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
   0 <= NoClash(A_B_3_0)_indicator_var <= 1
   0 <= NoClash(A_B_3_1)_indicator_var <= 1
   0 <= NoClash(A_B_5_0)_indicator_var <= 1
   0 <= NoClash(A_B_5_1)_indicator_var <= 1
   0 <= NoClash(A_C_1_0)_indicator_var <= 1
   0 <= NoClash(A_C_1_1)_indicator_var <= 1
   0 <= NoClash(A_D_3_0)_indicator_var <= 1
   0 <= NoClash(A_D_3_1)_indicator_var <= 1
   0 <= NoClash(A_E_3_0)_indicator_var <= 1
   0 <= NoClash(A_E_3_1)_indicator_var <= 1
   0 <= NoClash(A_E_5_0)_indicator_var <= 1
   0 <= NoClash(A_E_5_1)_indicator_var <= 1
   0 <= NoClash(A_F_1_0)_indicator_var <= 1
   0 <= NoClash(A_F_1_1)_indicator_var <= 1
   0 <= NoClash(A_F_3_0)_indicator_var <= 1
   0 <= NoClash(A_F_3_1)_indicator_var <= 1
   0 <= NoClash(A_G_5_0)_indicator_var <= 1
   0 <= NoClash(A_G_5_1)_indicator_var <= 1
   0 <= NoClash(B_C_2_0)_indicator_var <= 1
   0 <= NoClash(B_C_2_1)_indicator_var <= 1
   0 <= NoClash(B_D_2_0)_indicator_var <= 1
   0 <= NoClash(B_D_2_1)_indicator_var <= 1
   0 <= NoClash(B_D_3_0)_indicator_var <= 1
   0 <= NoClash(B_D_3_1)_indicator_var <= 1
   0 <= NoClash(B_E_2_0)_indicator_var <= 1
   0 <= NoClash(B_E_2_1)_indicator_var <= 1
   0 <= NoClash(B_E_3_0)_indicator_var <= 1
   0 <= NoClash(B_E_3_1)_indicator_var <= 1
   0 <= NoClash(B_E_5_0)_indicator_var <= 1
   0 <= NoClash(B_E_5_1)_indicator_var <= 1
   0 <= NoClash(B_F_3_0)_indicator_var <= 1
   0 <= NoClash(B_F_3_1)_indicator_var <= 1
   0 <= NoClash(B_G_2_0)_indicator_var <= 1
   0 <= NoClash(B_G_2_1)_indicator_var <= 1
   0 <= NoClash(B_G_5_0)_indicator_var <= 1
   0 <= NoClash(B_G_5_1)_indicator_var <= 1
   0 <= NoClash(C_D_2_0)_indicator_var <= 1
   0 <= NoClash(C_D_2_1)_indicator_var <= 1
   0 <= NoClash(C_D_4_0)_indicator_var <= 1
   0 <= NoClash(C_D_4_1)_indicator_var <= 1
   0 <= NoClash(C_E_2_0)_indicator_var <= 1
   0 <= NoClash(C_E_2_1)_indicator_var <= 1
   0 <= NoClash(C_F_1_0)_indicator_var <= 1
   0 <= NoClash(C_F_1_1)_indicator_var <= 1
   0 <= NoClash(C_F_4_0)_indicator_var <= 1
   0 <= NoClash(C_F_4_1)_indicator_var <= 1
   0 <= NoClash(C_G_2_0)_indicator_var <= 1
   0 <= NoClash(C_G_2_1)_indicator_var <= 1
   0 <= NoClash(C_G_4_0)_indicator_var <= 1
   0 <= NoClash(C_G_4_1)_indicator_var <= 1
   0 <= NoClash(D_E_2_0)_indicator_var <= 1
   0 <= NoClash(D_E_2_1)_indicator_var <= 1
   0 <= NoClash(D_E_3_0)_indicator_var <= 1
   0 <= NoClash(D_E_3_1)_indicator_var <= 1
   0 <= NoClash(D_F_3_0)_indicator_var <= 1
   0 <= NoClash(D_F_3_1)_indicator_var <= 1
   0 <= NoClash(D_F_4_0)_indicator_var <= 1
   0 <= NoClash(D_F_4_1)_indicator_var <= 1
   0 <= NoClash(D_G_2_0)_indicator_var <= 1
   0 <= NoClash(D_G_2_1)_indicator_var <= 1
   0 <= NoClash(D_G_4_0)_indicator_var <= 1
   0 <= NoClash(D_G_4_1)_indicator_var <= 1
   0 <= NoClash(E_F_3_0)_indicator_var <= 1
   0 <= NoClash(E_F_3_1)_indicator_var <= 1
   0 <= NoClash(E_G_2_0)_indicator_var <= 1
   0 <= NoClash(E_G_2_1)_indicator_var <= 1
   0 <= NoClash(E_G_5_0)_indicator_var <= 1
   0 <= NoClash(E_G_5_1)_indicator_var <= 1
   0 <= NoClash(F_G_4_0)_indicator_var <= 1
   0 <= NoClash(F_G_4_1)_indicator_var <= 1
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(0)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(1)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(2)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(3)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(4)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(5)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(6)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(7)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(8)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(9)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(10)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(11)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(12)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(13)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(14)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(15)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(16)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(17)_t(A) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(18)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(19)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(20)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(21)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(22)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(23)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(24)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(25)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(26)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(27)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(28)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(29)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(30)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(31)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(32)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(33)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(34)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(35)_t(B) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(36)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(37)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(38)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(39)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(40)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(41)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(42)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(43)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(44)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(45)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(46)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(47)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(48)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(49)_t(C) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(50)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(51)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(52)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(53)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(54)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(55)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(56)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(57)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(58)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(59)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(60)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(61)_t(D) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(62)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(63)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(64)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(65)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(66)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(67)_t(E) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(68)_t(F) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(G) <= 92
   0 <= _pyomo_gdp_chull_relaxation_relaxedDisjuncts(69)_t(F) <= 92
binary
  NoClash(A_B_3_0)_indicator_var
  NoClash(A_B_3_1)_indicator_var
  NoClash(A_B_5_0)_indicator_var
  NoClash(A_B_5_1)_indicator_var
  NoClash(A_C_1_0)_indicator_var
  NoClash(A_C_1_1)_indicator_var
  NoClash(A_D_3_0)_indicator_var
  NoClash(A_D_3_1)_indicator_var
  NoClash(A_E_3_0)_indicator_var
  NoClash(A_E_3_1)_indicator_var
  NoClash(A_E_5_0)_indicator_var
  NoClash(A_E_5_1)_indicator_var
  NoClash(A_F_1_0)_indicator_var
  NoClash(A_F_1_1)_indicator_var
  NoClash(A_F_3_0)_indicator_var
  NoClash(A_F_3_1)_indicator_var
  NoClash(A_G_5_0)_indicator_var
  NoClash(A_G_5_1)_indicator_var
  NoClash(B_C_2_0)_indicator_var
  NoClash(B_C_2_1)_indicator_var
  NoClash(B_D_2_0)_indicator_var
  NoClash(B_D_2_1)_indicator_var
  NoClash(B_D_3_0)_indicator_var
  NoClash(B_D_3_1)_indicator_var
  NoClash(B_E_2_0)_indicator_var
  NoClash(B_E_2_1)_indicator_var
  NoClash(B_E_3_0)_indicator_var
  NoClash(B_E_3_1)_indicator_var
  NoClash(B_E_5_0)_indicator_var
  NoClash(B_E_5_1)_indicator_var
  NoClash(B_F_3_0)_indicator_var
  NoClash(B_F_3_1)_indicator_var
  NoClash(B_G_2_0)_indicator_var
  NoClash(B_G_2_1)_indicator_var
  NoClash(B_G_5_0)_indicator_var
  NoClash(B_G_5_1)_indicator_var
  NoClash(C_D_2_0)_indicator_var
  NoClash(C_D_2_1)_indicator_var
  NoClash(C_D_4_0)_indicator_var
  NoClash(C_D_4_1)_indicator_var
  NoClash(C_E_2_0)_indicator_var
  NoClash(C_E_2_1)_indicator_var
  NoClash(C_F_1_0)_indicator_var
  NoClash(C_F_1_1)_indicator_var
  NoClash(C_F_4_0)_indicator_var
  NoClash(C_F_4_1)_indicator_var
  NoClash(C_G_2_0)_indicator_var
  NoClash(C_G_2_1)_indicator_var
  NoClash(C_G_4_0)_indicator_var
  NoClash(C_G_4_1)_indicator_var
  NoClash(D_E_2_0)_indicator_var
  NoClash(D_E_2_1)_indicator_var
  NoClash(D_E_3_0)_indicator_var
  NoClash(D_E_3_1)_indicator_var
  NoClash(D_F_3_0)_indicator_var
  NoClash(D_F_3_1)_indicator_var
  NoClash(D_F_4_0)_indicator_var
  NoClash(D_F_4_1)_indicator_var
  NoClash(D_G_2_0)_indicator_var
  NoClash(D_G_2_1)_indicator_var
  NoClash(D_G_4_0)_indicator_var
  NoClash(D_G_4_1)_indicator_var
  NoClash(E_F_3_0)_indicator_var
  NoClash(E_F_3_1)_indicator_var
  NoClash(E_G_2_0)_indicator_var
  NoClash(E_G_2_1)_indicator_var
  NoClash(E_G_5_0)_indicator_var
  NoClash(E_G_5_1)_indicator_var
  NoClash(F_G_4_0)_indicator_var
  NoClash(F_G_4_1)_indicator_var
end
