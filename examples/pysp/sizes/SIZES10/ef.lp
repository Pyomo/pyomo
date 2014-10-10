\ Begin objective block for master
min 
obj:  1.0 MASTER_EXPECTED_COST_RootNode 
\ End objective block for master

s.t.

\ Begin constraint block for master
MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario1_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario1_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario8_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario8_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario7_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario7_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario8_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario8_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario4_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario4_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario6_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario6_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_COST_Scenario8Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario8Node - 1.0 MASTER_Scenario8Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario8Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario8Node - 1.0 MASTER_Scenario8Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario6_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario6_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario4_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario4_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario4_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario4_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario8_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario8_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario1_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario1_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario9_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario9_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario10_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario10_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario9_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario9_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario9_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario9_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario2_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario2_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario10_Scenario10Node_SecondStageCost_None_:  1.0 MASTER_Scenario10Node_SecondStageCost - 1.0 Scenario10_SecondStageCost >= 0.0

MASTER_c_u_Scenario10_Scenario10Node_SecondStageCost_None_:  1.0 MASTER_Scenario10Node_SecondStageCost - 1.0 Scenario10_SecondStageCost <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario5_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario5_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario10_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario10_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario7_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario7_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario4_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario4_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario4_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario4_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario6_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario6_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario4_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario4_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario10_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario10_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario5_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario5_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario9_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario9_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario7_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario7_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario8_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario8_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario9_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario9_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario3_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario3_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario5_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario5_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario8_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario8_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario1_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario1_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario4_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario4_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario7_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario7_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario4_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario4_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario1_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario1_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario3_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario3_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario8_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario8_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario5_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario5_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario3_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario3_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario6_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario6_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario3_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario3_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario3_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario3_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario3_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario3_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario3_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario3_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario5_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario5_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario3_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario3_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario10_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario10_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario7_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario7_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario6_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario6_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario5_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario5_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario3_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario3_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario3_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario3_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario2_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario2_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario3_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario3_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario2_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario2_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario1_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario1_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario4_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario4_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_COST_Scenario10Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario10Node - 1.0 MASTER_Scenario10Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario10Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario10Node - 1.0 MASTER_Scenario10Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario1_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario1_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario10_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario10_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario1_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario1_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario2_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario2_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario5_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario5_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario1_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario1_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario5_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario5_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario6_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario6_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario1_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario1_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario8_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario8_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario3_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario3_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario6_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario6_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario9_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario9_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario2_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario2_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario5_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario5_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario1_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario1_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario3_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario3_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario10_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario10_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario7_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario7_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario10_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario10_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario7_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario7_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario6_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario6_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario5_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario5_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario10_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario10_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario9_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario9_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario4_FirstStageCost >= 0.0

MASTER_c_u_Scenario4_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario4_FirstStageCost <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario5_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario5_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario5_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario5_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario5_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario5_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario5_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario5_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario5_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario5_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario1_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario1_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario5_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario5_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario5_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario5_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario5_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario5_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario5_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario5_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario5_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario5_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario3_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario3_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario4_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario4_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario6_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario6_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario1_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario1_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario2_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario2_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario10_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario10_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario10_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario10_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario1_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario1_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario10_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario10_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario2_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario2_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario7_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario7_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario9_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario9_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario10_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario10_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario6_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario6_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario6_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario6_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario6_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario6_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario2_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario2_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario5_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario5_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario5_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario5_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario6_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario6_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario3_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario3_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario5_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario5_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario5_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario5_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario5_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario5_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario5_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario5_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario4_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario4_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario5_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario5_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario5_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario5_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario10_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario10_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario10_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario10_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario10_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario10_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario10_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario10_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario10_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario10_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario10_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario10_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario10_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario10_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario2_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario2_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario10_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario10_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario10_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario10_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario7_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario7_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario8_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario8_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario2_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario2_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario7_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario7_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario10_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario10_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario9_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario9_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario7_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario7_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario6_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario6_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario6_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario6_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario1_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario1_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario9_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario9_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario3_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario3_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario2_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario2_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario5_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario5_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario4_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario4_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario3_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario3_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario4_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario4_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario8_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario8_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario8_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario8_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario10_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario10_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario7_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario7_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario7_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario7_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario9_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario9_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario5_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario5_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario6_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario6_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario5_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario5_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario2_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario2_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario5_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario5_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario4_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario4_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario1_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario1_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario6_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario6_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario6_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario6_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario6_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario6_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario5_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario5_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario4_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario4_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario6_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario6_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario6_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario6_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario6_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario6_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario7_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario7_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario6_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario6_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario6_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario6_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario6_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario6_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario6_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario6_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario6_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario6_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario5_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario5_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario2_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario2_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario8_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario8_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario4_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario4_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario5_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario5_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario3_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario3_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario7_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario7_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario10_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario10_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario3_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario3_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario5_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario5_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario8_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario8_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario7_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario7_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario8_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario8_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario1_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario1_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario10_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario10_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario2_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_Scenario2Node_SecondStageCost - 1.0 Scenario2_SecondStageCost >= 0.0

MASTER_c_u_Scenario2_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_Scenario2Node_SecondStageCost - 1.0 Scenario2_SecondStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario3_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario3_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario1_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario1_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario8_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario8_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario2_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario2_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario3_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_Scenario3Node_SecondStageCost - 1.0 Scenario3_SecondStageCost >= 0.0

MASTER_c_u_Scenario3_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_Scenario3Node_SecondStageCost - 1.0 Scenario3_SecondStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario3_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario3_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario9_Scenario9Node_SecondStageCost_None_:  1.0 MASTER_Scenario9Node_SecondStageCost - 1.0 Scenario9_SecondStageCost >= 0.0

MASTER_c_u_Scenario9_Scenario9Node_SecondStageCost_None_:  1.0 MASTER_Scenario9Node_SecondStageCost - 1.0 Scenario9_SecondStageCost <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario9_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario9_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario7_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario7_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario2_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario2_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario9_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario9_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario9_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario9_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario6_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario6_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario5_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario5_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario8_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario8_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario10_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario10_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario9_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario9_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario9_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario9_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario3_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario3_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario3_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario3_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario1_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario1_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario2_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario2_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario5_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario5_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario2_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario2_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario2_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario2_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario4_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario4_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario9_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario9_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario1_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario1_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario1_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario1_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario10_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario10_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario8_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario8_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario6_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario6_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario2_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario2_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario1_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario1_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario9_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario9_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario4_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario4_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_COST_Scenario4Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario4Node - 1.0 MASTER_Scenario4Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario4Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario4Node - 1.0 MASTER_Scenario4Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario1_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario1_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario4_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario4_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario7_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario7_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario10_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario10_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario5_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario5_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_COST_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario3Node - 1.0 MASTER_Scenario3Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario3Node - 1.0 MASTER_Scenario3Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario5_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario5_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario6_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario6_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario9_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario9_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario3_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario3_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario9_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario9_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario9_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario9_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario9_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario9_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario9_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario9_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario9_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario9_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario9_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario9_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario9_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario9_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario9_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario9_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario9_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario9_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario8_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario8_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario2_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario2_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario2_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario2_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario2_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario2_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario2_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario2_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario2_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario2_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario2_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario2_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario2_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario2_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario2_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario2_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario2_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario2_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario2_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario2_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario4_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario4_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario6_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario6_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario3_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario3_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario8_Scenario8Node_SecondStageCost_None_:  1.0 MASTER_Scenario8Node_SecondStageCost - 1.0 Scenario8_SecondStageCost >= 0.0

MASTER_c_u_Scenario8_Scenario8Node_SecondStageCost_None_:  1.0 MASTER_Scenario8Node_SecondStageCost - 1.0 Scenario8_SecondStageCost <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario5_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario5_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario7_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario7_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario4_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario4_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario8_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario8_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario7_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario7_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario4_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario4_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario2_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario2_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario8_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario8_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario2_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario2_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario4_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario4_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario10_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario10_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario9_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario9_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario6_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario6_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario2_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario2_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario6_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario6_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario1_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario1_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario9_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario9_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario6_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario6_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario8_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario8_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario10_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario10_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario7_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario7_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario7_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario7_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario7_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario7_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario1_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario1_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario10_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario10_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_COST_Scenario7Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario7Node - 1.0 MASTER_Scenario7Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario7Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario7Node - 1.0 MASTER_Scenario7Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario6_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario6_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario1_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario1_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario1_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario1_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario10_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario10_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario1_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario1_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario2_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario2_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario1_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario1_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario1_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario1_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario1_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario1_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario1_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario1_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario6_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario6_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario8_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario8_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario8_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario8_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario1_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario1_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario4_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario4_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario10_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario10_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario4_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario4_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario8_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario8_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario7_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario7_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario9_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario9_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario3_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario3_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario5_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario5_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario1_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario1_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario7_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario7_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario4_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario4_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario8_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario8_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario2_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario2_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario10_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario10_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario3_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario3_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario8_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario8_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario4_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario4_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario1_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario1_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario9_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario9_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario1_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario1_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario7_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario7_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario1_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario1_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario9_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario9_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario4_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario4_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario4_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario4_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario5_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario5_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario8_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario8_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario7_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario7_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario7_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario7_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario1_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario1_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario7_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario7_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario9_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario9_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario7_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario7_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario7_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario7_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario7_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario7_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario7_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario7_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario7_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario7_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario7_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario7_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario7_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario7_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario1_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario1_FirstStageCost >= 0.0

MASTER_c_u_Scenario1_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario1_FirstStageCost <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario8_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario8_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario10_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario10_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario6_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario6_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario2_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario2_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario1_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario1_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario10_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario10_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario6_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario6_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario8_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario8_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario9_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario9_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario3_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario3_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario8_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario8_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario8_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario8_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario3_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario3_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario2_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario2_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario9_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario9_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario3_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario3_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario3_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario3_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario5_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario5_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario10_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario10_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario9_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario9_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario8_FirstStageCost >= 0.0

MASTER_c_u_Scenario8_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario8_FirstStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario3_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario3_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_COST_Scenario5Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario5Node - 1.0 MASTER_Scenario5Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario5Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario5Node - 1.0 MASTER_Scenario5Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario9_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario9_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario8_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario8_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario6_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario6_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario6_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario6_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario1_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario1_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario6_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario6_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario3_FirstStageCost >= 0.0

MASTER_c_u_Scenario3_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario3_FirstStageCost <= 0.0

MASTER_c_l_Scenario2_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario2_FirstStageCost >= 0.0

MASTER_c_u_Scenario2_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario2_FirstStageCost <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario4_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario4_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario7_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario7_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario1_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario1_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario7_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario7_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario6_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario6_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario2_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario2_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario1_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario1_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario7_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario7_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario3_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario3_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario9_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario9_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario5_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario5_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario5_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario5_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario4_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario4_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario6_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario6_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario4_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario4_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario9_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario9_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario8_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario8_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario10_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario10_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario6_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario6_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario7_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario7_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario6_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario6_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario1_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario1_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario1_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario1_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario1_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario1_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario1_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario1_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario2_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario2_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario8_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario8_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario2_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario2_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario6_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario6_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario2_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario2_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario5_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario5_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario2_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario2_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario3_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario3_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario2_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario2_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario2_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario2_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario2_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario2_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario2_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario2_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario1_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario1_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario8_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario8_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario1_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario1_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario1_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario1_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario1_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario1_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario1_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario1_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario5_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario5_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario5_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario5_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario2_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario2_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario7_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario7_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario4_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario4_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario10_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario10_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario1_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario1_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario10_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario10_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario6_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario6_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario10_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario10_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario10_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario10_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario10_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario10_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario10_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario10_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario10_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario10_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario10_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario10_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario10_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario10_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario7_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario7_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario4_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario4_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario2_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario2_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario8_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario8_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario5_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario5_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario3_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario3_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario10_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario10_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_COST_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario1Node - 1.0 MASTER_Scenario1Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario1Node - 1.0 MASTER_Scenario1Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario9_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario9_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario6_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario6_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario7_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario7_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario4_Scenario4Node_SecondStageCost_None_:  1.0 MASTER_Scenario4Node_SecondStageCost - 1.0 Scenario4_SecondStageCost >= 0.0

MASTER_c_u_Scenario4_Scenario4Node_SecondStageCost_None_:  1.0 MASTER_Scenario4Node_SecondStageCost - 1.0 Scenario4_SecondStageCost <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario10_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario10_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario4_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario4_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario2_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario2_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario4_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario4_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario1_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario1_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario4_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario4_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario4_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario4_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario7_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario7_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario9_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario9_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario4_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario4_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario4_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario4_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario4_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario4_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario4_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario4_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario4_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario4_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario4_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario4_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario4_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario4_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario7_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario7_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario10_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario10_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario6_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario6_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario3_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario3_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_COST_Scenario6Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario6Node - 1.0 MASTER_Scenario6Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario6Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario6Node - 1.0 MASTER_Scenario6Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario8_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario8_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario5_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario5_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario9_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario9_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario4_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario4_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario2_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario2_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario3_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario3_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario3_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario3_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario8_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario8_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario7_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario7_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario9_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario9_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario10_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario10_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario6_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario6_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario1_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario1_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario3_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario3_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario3_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario3_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario7_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario7_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario8_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario8_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario5_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario5_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario4_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario4_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario6_Scenario6Node_SecondStageCost_None_:  1.0 MASTER_Scenario6Node_SecondStageCost - 1.0 Scenario6_SecondStageCost >= 0.0

MASTER_c_u_Scenario6_Scenario6Node_SecondStageCost_None_:  1.0 MASTER_Scenario6Node_SecondStageCost - 1.0 Scenario6_SecondStageCost <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario2_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario2_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario5_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario5_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario2_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario2_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario10_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario10_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario2_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario2_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario7_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario7_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario3_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario3_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario3_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario3_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario9_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario9_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario2_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario2_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario5_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario5_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario10_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario10_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario5_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario5_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario1_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario1_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario3_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario3_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario10_FirstStageCost >= 0.0

MASTER_c_u_Scenario10_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario10_FirstStageCost <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario6_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario6_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario2_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario2_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario1_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario1_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario7_Scenario7Node_SecondStageCost_None_:  1.0 MASTER_Scenario7Node_SecondStageCost - 1.0 Scenario7_SecondStageCost >= 0.0

MASTER_c_u_Scenario7_Scenario7Node_SecondStageCost_None_:  1.0 MASTER_Scenario7Node_SecondStageCost - 1.0 Scenario7_SecondStageCost <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario6_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario6_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario8_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario8_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario3_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario3_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario4_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario4_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario6_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario6_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario5_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario5_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario6_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario6_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario10_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario10_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario7_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario7_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario7_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario7_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario4_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario4_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario7_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario7_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario6_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario6_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario9_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario9_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario6_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario6_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario6_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario6_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario6_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario6_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario5_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario5_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario4_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario4_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario6_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario6_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario6_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario6_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario2_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario2_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario6_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario6_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario1_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario1_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario2_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario2_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario10_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario10_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario1_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario1_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario1_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario1_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario2_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario2_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario8_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario8_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario5_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario5_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario10_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario10_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario3_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario3_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario7_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario7_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario8_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario8_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario8_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario8_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario2_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario2_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario10_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario10_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario1_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario1_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario8_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario8_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario7_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario7_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario9_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario9_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario2_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario2_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario3_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario3_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario1_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario1_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario6_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario6_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario10_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario10_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario7_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario7_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario9_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario9_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario3_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario3_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario7_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario7_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario6_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario6_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario4_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario4_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario7_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario7_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario8_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario8_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario10_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario10_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario2_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario2_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario9_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario9_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario8_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario8_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario7_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario7_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario10_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario10_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario6_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario6_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario10_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario10_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario7_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario7_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario3_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario3_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_COST_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario2Node - 1.0 MASTER_Scenario2Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario2Node - 1.0 MASTER_Scenario2Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario3_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario3_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario8_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario8_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario6_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario6_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario8_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario8_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario8_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario8_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario5_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario5_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario4_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario4_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario2_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario2_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario7_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario7_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario7_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario7_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario3_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario3_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario3_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario3_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario9_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario9_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario5_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario5_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario1_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario1_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario3_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario3_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario4_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario4_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario9_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario9_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario7_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario7_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario8_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario8_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario6_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario6_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario1_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario1_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario5_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario5_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario4_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario4_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario4_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario4_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario9_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario9_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario4_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario4_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario4_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario4_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario4_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario4_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario3_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario3_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario1_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario1_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario4_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario4_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario4_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario4_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario4_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario4_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario4_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario4_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario4_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario4_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario2_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario2_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario5_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario5_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario4_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario4_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario5_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario5_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario2_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario2_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario5_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario5_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario7_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario7_FirstStageCost >= 0.0

MASTER_c_u_Scenario7_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario7_FirstStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario1_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario1_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario1_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario1_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario1_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario1_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario9_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario9_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario9_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario9_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario8_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario8_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario5_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario5_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario9_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario9_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario4_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario4_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario3_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario3_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario3_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario3_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario5_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario5_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario4_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario4_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario1_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario1_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario7_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario7_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario10_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario10_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario7_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario7_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario9_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario9_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario4_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario4_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario8_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario8_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario6_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario6_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario1_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario1_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario10_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario10_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario10_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario10_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario9_FirstStageCost >= 0.0

MASTER_c_u_Scenario9_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario9_FirstStageCost <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario10_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario10_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario7_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario7_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario2_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario2_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario8_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario8_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario3_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario3_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario2_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario2_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario10_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario10_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario3_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario3_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario2_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario2_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario7_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario7_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario3_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario3_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario3_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario3_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario1_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario1_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario8_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario8_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario4_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario4_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario3_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario3_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario3_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario3_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario4_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario4_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario3_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario3_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario3_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario3_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario6_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario6_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario9_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario9_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario10_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario10_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario7_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario7_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario2_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario2_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario9_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario9_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario9_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario9_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario9_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario9_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario5_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario5_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario9_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario9_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario9_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario9_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario9_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario9_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario9_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario9_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario9_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario9_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario6_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario6_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario1_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario1_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario4_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario4_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario9_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario9_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario9_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario9_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario9_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario9_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario5_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario5_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario6_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario6_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario8_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario8_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario3_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario3_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario4_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario4_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario9_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario9_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario8_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario8_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario7_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario7_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario8_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario8_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario8_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario8_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario8_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario8_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario8_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario8_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario8_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario8_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario7_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario7_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario7_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario7_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario7_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario7_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario7_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario7_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario8_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario8_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario9_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario9_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario7_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario7_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario1_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario1_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario3_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario3_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario9_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario9_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario6_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario6_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario2_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario2_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario9_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario9_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario3_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario3_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_COST_RootNode_FirstStageCost_None_:  1.0 MASTER_EXPECTED_COST_RootNode - 0.1 MASTER_EXPECTED_COST_Scenario10Node - 0.1 MASTER_EXPECTED_COST_Scenario1Node - 0.1 MASTER_EXPECTED_COST_Scenario2Node - 0.1 MASTER_EXPECTED_COST_Scenario3Node 
- 0.1 MASTER_EXPECTED_COST_Scenario4Node - 0.1 MASTER_EXPECTED_COST_Scenario5Node - 0.1 MASTER_EXPECTED_COST_Scenario6Node - 0.1 MASTER_EXPECTED_COST_Scenario7Node - 0.1 MASTER_EXPECTED_COST_Scenario8Node 
- 0.1 MASTER_EXPECTED_COST_Scenario9Node - 1.0 MASTER_RootNode_FirstStageCost >= 0.0

MASTER_c_u_COST_RootNode_FirstStageCost_None_:  1.0 MASTER_EXPECTED_COST_RootNode - 0.1 MASTER_EXPECTED_COST_Scenario10Node - 0.1 MASTER_EXPECTED_COST_Scenario1Node - 0.1 MASTER_EXPECTED_COST_Scenario2Node - 0.1 MASTER_EXPECTED_COST_Scenario3Node 
- 0.1 MASTER_EXPECTED_COST_Scenario4Node - 0.1 MASTER_EXPECTED_COST_Scenario5Node - 0.1 MASTER_EXPECTED_COST_Scenario6Node - 0.1 MASTER_EXPECTED_COST_Scenario7Node - 0.1 MASTER_EXPECTED_COST_Scenario8Node 
- 0.1 MASTER_EXPECTED_COST_Scenario9Node - 1.0 MASTER_RootNode_FirstStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario3_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario3_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario9_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario9_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario5_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario5_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario8_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario8_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario1_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario1_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario10_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario10_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario6_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario6_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario8_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario8_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario5_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario5_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario4_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario4_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario2_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario2_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario6_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario6_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario7_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario7_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario8_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario8_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario2_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario2_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario10_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario10_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario3_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario3_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario9_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario9_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario9_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario9_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario2_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario2_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario1_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_Scenario1Node_SecondStageCost - 1.0 Scenario1_SecondStageCost >= 0.0

MASTER_c_u_Scenario1_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_Scenario1Node_SecondStageCost - 1.0 Scenario1_SecondStageCost <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario4_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario4_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario5_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario5_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_COST_Scenario9Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario9Node - 1.0 MASTER_Scenario9Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario9Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario9Node - 1.0 MASTER_Scenario9Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario10_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario10_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario10_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario10_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario6_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario6_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario4_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario4_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario5_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario5_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario5_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario5_Scenario5Node_SecondStageCost_None_:  1.0 MASTER_Scenario5Node_SecondStageCost - 1.0 Scenario5_SecondStageCost >= 0.0

MASTER_c_u_Scenario5_Scenario5Node_SecondStageCost_None_:  1.0 MASTER_Scenario5Node_SecondStageCost - 1.0 Scenario5_SecondStageCost <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario8_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario8_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario8_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario8_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario8_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario8_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario8_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario8_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario8_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario8_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario8_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario8_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario8_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario8_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario10_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario10_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario2_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario2_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario6_FirstStageCost >= 0.0

MASTER_c_u_Scenario6_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario6_FirstStageCost <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario8_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario8_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario3_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario3_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario2_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario2_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario10_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario10_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario5_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario5_FirstStageCost >= 0.0

MASTER_c_u_Scenario5_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario5_FirstStageCost <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario8_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario8_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario4_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario4_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario6_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario6_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario3_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario3_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario10_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario10_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario7_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario7_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario3_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario3_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario7_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario7_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario5_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario5_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario6_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario6_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario6_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario8_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario8_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario8_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario8_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario3_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario3_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario1_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario1_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario3_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario3_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario1_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario1_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario9_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario9_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario9_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario9_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario8_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario8_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario8_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario5_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario5_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario3_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario3_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario5_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario5_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario5_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario2_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario2_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario2_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario2_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario3_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario3_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario10_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario10_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario7_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario7_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario7_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario7_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario10_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario10_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario10_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario1_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario1_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario4_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario4_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario4_NumUnitsCutFirstStage(9,8) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for master 
\ Begin constraint block for scenario Scenario10
Scenario10_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario10_NumUnitsCutSecondStage(1,1) + 1.0 Scenario10_NumUnitsCutSecondStage(10,1) + 1.0 Scenario10_NumUnitsCutSecondStage(2,1) + 1.0 Scenario10_NumUnitsCutSecondStage(3,1) + 1.0 Scenario10_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(5,1) + 1.0 Scenario10_NumUnitsCutSecondStage(6,1) + 1.0 Scenario10_NumUnitsCutSecondStage(7,1) + 1.0 Scenario10_NumUnitsCutSecondStage(8,1) + 1.0 Scenario10_NumUnitsCutSecondStage(9,1) 
>= 3750.0

Scenario10_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,2) + 1.0 Scenario10_NumUnitsCutSecondStage(2,2) + 1.0 Scenario10_NumUnitsCutSecondStage(3,2) + 1.0 Scenario10_NumUnitsCutSecondStage(4,2) + 1.0 Scenario10_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(6,2) + 1.0 Scenario10_NumUnitsCutSecondStage(7,2) + 1.0 Scenario10_NumUnitsCutSecondStage(8,2) + 1.0 Scenario10_NumUnitsCutSecondStage(9,2) >= 11250.0

Scenario10_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,3) + 1.0 Scenario10_NumUnitsCutSecondStage(3,3) + 1.0 Scenario10_NumUnitsCutSecondStage(4,3) + 1.0 Scenario10_NumUnitsCutSecondStage(5,3) + 1.0 Scenario10_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(7,3) + 1.0 Scenario10_NumUnitsCutSecondStage(8,3) + 1.0 Scenario10_NumUnitsCutSecondStage(9,3) >= 18750.0

Scenario10_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,4) + 1.0 Scenario10_NumUnitsCutSecondStage(4,4) + 1.0 Scenario10_NumUnitsCutSecondStage(5,4) + 1.0 Scenario10_NumUnitsCutSecondStage(6,4) + 1.0 Scenario10_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(8,4) + 1.0 Scenario10_NumUnitsCutSecondStage(9,4) >= 15000.0

Scenario10_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,5) + 1.0 Scenario10_NumUnitsCutSecondStage(5,5) + 1.0 Scenario10_NumUnitsCutSecondStage(6,5) + 1.0 Scenario10_NumUnitsCutSecondStage(7,5) + 1.0 Scenario10_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(9,5) >= 52500.0

Scenario10_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,6) + 1.0 Scenario10_NumUnitsCutSecondStage(6,6) + 1.0 Scenario10_NumUnitsCutSecondStage(7,6) + 1.0 Scenario10_NumUnitsCutSecondStage(8,6) + 1.0 Scenario10_NumUnitsCutSecondStage(9,6) 
>= 37500.0

Scenario10_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,7) + 1.0 Scenario10_NumUnitsCutSecondStage(7,7) + 1.0 Scenario10_NumUnitsCutSecondStage(8,7) + 1.0 Scenario10_NumUnitsCutSecondStage(9,7) >= 22500.0

Scenario10_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,8) + 1.0 Scenario10_NumUnitsCutSecondStage(8,8) + 1.0 Scenario10_NumUnitsCutSecondStage(9,8) >= 18750.0

Scenario10_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,9) + 1.0 Scenario10_NumUnitsCutSecondStage(9,9) >= 18750.0

Scenario10_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario10_NumUnitsCutSecondStage(10,10) >= 7500.0

Scenario10_c_e_ComputeSecondStageCost_:  -0.8416 Scenario10_NumProducedSecondStage(10) - 0.748 Scenario10_NumProducedSecondStage(1) - 0.7584 Scenario10_NumProducedSecondStage(2) - 0.7688 Scenario10_NumProducedSecondStage(3) - 0.7792 Scenario10_NumProducedSecondStage(4) 
- 0.7896 Scenario10_NumProducedSecondStage(5) - 0.8 Scenario10_NumProducedSecondStage(6) - 0.8104 Scenario10_NumProducedSecondStage(7) - 0.8208 Scenario10_NumProducedSecondStage(8) - 0.8312 Scenario10_NumProducedSecondStage(9) 
- 0.008 Scenario10_NumUnitsCutSecondStage(10,1) - 0.008 Scenario10_NumUnitsCutSecondStage(10,2) - 0.008 Scenario10_NumUnitsCutSecondStage(10,3) - 0.008 Scenario10_NumUnitsCutSecondStage(10,4) - 0.008 Scenario10_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario10_NumUnitsCutSecondStage(10,6) - 0.008 Scenario10_NumUnitsCutSecondStage(10,7) - 0.008 Scenario10_NumUnitsCutSecondStage(10,8) - 0.008 Scenario10_NumUnitsCutSecondStage(10,9) - 0.008 Scenario10_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario10_NumUnitsCutSecondStage(3,1) - 0.008 Scenario10_NumUnitsCutSecondStage(3,2) - 0.008 Scenario10_NumUnitsCutSecondStage(4,1) - 0.008 Scenario10_NumUnitsCutSecondStage(4,2) - 0.008 Scenario10_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario10_NumUnitsCutSecondStage(5,1) - 0.008 Scenario10_NumUnitsCutSecondStage(5,2) - 0.008 Scenario10_NumUnitsCutSecondStage(5,3) - 0.008 Scenario10_NumUnitsCutSecondStage(5,4) - 0.008 Scenario10_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario10_NumUnitsCutSecondStage(6,2) - 0.008 Scenario10_NumUnitsCutSecondStage(6,3) - 0.008 Scenario10_NumUnitsCutSecondStage(6,4) - 0.008 Scenario10_NumUnitsCutSecondStage(6,5) - 0.008 Scenario10_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario10_NumUnitsCutSecondStage(7,2) - 0.008 Scenario10_NumUnitsCutSecondStage(7,3) - 0.008 Scenario10_NumUnitsCutSecondStage(7,4) - 0.008 Scenario10_NumUnitsCutSecondStage(7,5) - 0.008 Scenario10_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario10_NumUnitsCutSecondStage(8,1) - 0.008 Scenario10_NumUnitsCutSecondStage(8,2) - 0.008 Scenario10_NumUnitsCutSecondStage(8,3) - 0.008 Scenario10_NumUnitsCutSecondStage(8,4) - 0.008 Scenario10_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario10_NumUnitsCutSecondStage(8,6) - 0.008 Scenario10_NumUnitsCutSecondStage(8,7) - 0.008 Scenario10_NumUnitsCutSecondStage(9,1) - 0.008 Scenario10_NumUnitsCutSecondStage(9,2) - 0.008 Scenario10_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario10_NumUnitsCutSecondStage(9,4) - 0.008 Scenario10_NumUnitsCutSecondStage(9,5) - 0.008 Scenario10_NumUnitsCutSecondStage(9,6) - 0.008 Scenario10_NumUnitsCutSecondStage(9,7) - 0.008 Scenario10_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario10_ProduceSizeSecondStage(10) - 453.0 Scenario10_ProduceSizeSecondStage(1) - 453.0 Scenario10_ProduceSizeSecondStage(2) - 453.0 Scenario10_ProduceSizeSecondStage(3) - 453.0 Scenario10_ProduceSizeSecondStage(4) 
- 453.0 Scenario10_ProduceSizeSecondStage(5) - 453.0 Scenario10_ProduceSizeSecondStage(6) - 453.0 Scenario10_ProduceSizeSecondStage(7) - 453.0 Scenario10_ProduceSizeSecondStage(8) - 453.0 Scenario10_ProduceSizeSecondStage(9) 
+ 1.0 Scenario10_SecondStageCost = 0.0

Scenario10_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario10_NumProducedSecondStage(10) + 1.0 Scenario10_NumProducedSecondStage(1) + 1.0 Scenario10_NumProducedSecondStage(2) + 1.0 Scenario10_NumProducedSecondStage(3) + 1.0 Scenario10_NumProducedSecondStage(4) 
+ 1.0 Scenario10_NumProducedSecondStage(5) + 1.0 Scenario10_NumProducedSecondStage(6) + 1.0 Scenario10_NumProducedSecondStage(7) + 1.0 Scenario10_NumProducedSecondStage(8) + 1.0 Scenario10_NumProducedSecondStage(9) 
<= 200000.0

Scenario10_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario10_NumProducedSecondStage(1) - 200000.0 Scenario10_ProduceSizeSecondStage(1) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario10_NumProducedSecondStage(2) - 200000.0 Scenario10_ProduceSizeSecondStage(2) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario10_NumProducedSecondStage(3) - 200000.0 Scenario10_ProduceSizeSecondStage(3) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario10_NumProducedSecondStage(4) - 200000.0 Scenario10_ProduceSizeSecondStage(4) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario10_NumProducedSecondStage(5) - 200000.0 Scenario10_ProduceSizeSecondStage(5) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario10_NumProducedSecondStage(6) - 200000.0 Scenario10_ProduceSizeSecondStage(6) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario10_NumProducedSecondStage(7) - 200000.0 Scenario10_ProduceSizeSecondStage(7) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario10_NumProducedSecondStage(8) - 200000.0 Scenario10_ProduceSizeSecondStage(8) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario10_NumProducedSecondStage(9) - 200000.0 Scenario10_ProduceSizeSecondStage(9) <= 0.0

Scenario10_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario10_NumProducedSecondStage(10) - 200000.0 Scenario10_ProduceSizeSecondStage(10) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario10_NumProducedFirstStage(1) - 200000.0 Scenario10_ProduceSizeFirstStage(1) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario10_NumProducedFirstStage(2) - 200000.0 Scenario10_ProduceSizeFirstStage(2) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario10_NumProducedFirstStage(3) - 200000.0 Scenario10_ProduceSizeFirstStage(3) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario10_NumProducedFirstStage(4) - 200000.0 Scenario10_ProduceSizeFirstStage(4) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario10_NumProducedFirstStage(5) - 200000.0 Scenario10_ProduceSizeFirstStage(5) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario10_NumProducedFirstStage(6) - 200000.0 Scenario10_ProduceSizeFirstStage(6) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario10_NumProducedFirstStage(7) - 200000.0 Scenario10_ProduceSizeFirstStage(7) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario10_NumProducedFirstStage(8) - 200000.0 Scenario10_ProduceSizeFirstStage(8) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario10_NumProducedFirstStage(9) - 200000.0 Scenario10_ProduceSizeFirstStage(9) <= 0.0

Scenario10_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario10_NumProducedFirstStage(10) - 200000.0 Scenario10_ProduceSizeFirstStage(10) <= 0.0

Scenario10_c_e_ComputeFirstStageCost_:  1.0 Scenario10_FirstStageCost - 0.8416 Scenario10_NumProducedFirstStage(10) - 0.748 Scenario10_NumProducedFirstStage(1) - 0.7584 Scenario10_NumProducedFirstStage(2) - 0.7688 Scenario10_NumProducedFirstStage(3) 
- 0.7792 Scenario10_NumProducedFirstStage(4) - 0.7896 Scenario10_NumProducedFirstStage(5) - 0.8 Scenario10_NumProducedFirstStage(6) - 0.8104 Scenario10_NumProducedFirstStage(7) - 0.8208 Scenario10_NumProducedFirstStage(8) 
- 0.8312 Scenario10_NumProducedFirstStage(9) - 0.008 Scenario10_NumUnitsCutFirstStage(10,1) - 0.008 Scenario10_NumUnitsCutFirstStage(10,2) - 0.008 Scenario10_NumUnitsCutFirstStage(10,3) - 0.008 Scenario10_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario10_NumUnitsCutFirstStage(10,5) - 0.008 Scenario10_NumUnitsCutFirstStage(10,6) - 0.008 Scenario10_NumUnitsCutFirstStage(10,7) - 0.008 Scenario10_NumUnitsCutFirstStage(10,8) - 0.008 Scenario10_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario10_NumUnitsCutFirstStage(2,1) - 0.008 Scenario10_NumUnitsCutFirstStage(3,1) - 0.008 Scenario10_NumUnitsCutFirstStage(3,2) - 0.008 Scenario10_NumUnitsCutFirstStage(4,1) - 0.008 Scenario10_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario10_NumUnitsCutFirstStage(4,3) - 0.008 Scenario10_NumUnitsCutFirstStage(5,1) - 0.008 Scenario10_NumUnitsCutFirstStage(5,2) - 0.008 Scenario10_NumUnitsCutFirstStage(5,3) - 0.008 Scenario10_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario10_NumUnitsCutFirstStage(6,1) - 0.008 Scenario10_NumUnitsCutFirstStage(6,2) - 0.008 Scenario10_NumUnitsCutFirstStage(6,3) - 0.008 Scenario10_NumUnitsCutFirstStage(6,4) - 0.008 Scenario10_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario10_NumUnitsCutFirstStage(7,1) - 0.008 Scenario10_NumUnitsCutFirstStage(7,2) - 0.008 Scenario10_NumUnitsCutFirstStage(7,3) - 0.008 Scenario10_NumUnitsCutFirstStage(7,4) - 0.008 Scenario10_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario10_NumUnitsCutFirstStage(7,6) - 0.008 Scenario10_NumUnitsCutFirstStage(8,1) - 0.008 Scenario10_NumUnitsCutFirstStage(8,2) - 0.008 Scenario10_NumUnitsCutFirstStage(8,3) - 0.008 Scenario10_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario10_NumUnitsCutFirstStage(8,5) - 0.008 Scenario10_NumUnitsCutFirstStage(8,6) - 0.008 Scenario10_NumUnitsCutFirstStage(8,7) - 0.008 Scenario10_NumUnitsCutFirstStage(9,1) - 0.008 Scenario10_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario10_NumUnitsCutFirstStage(9,3) - 0.008 Scenario10_NumUnitsCutFirstStage(9,4) - 0.008 Scenario10_NumUnitsCutFirstStage(9,5) - 0.008 Scenario10_NumUnitsCutFirstStage(9,6) - 0.008 Scenario10_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario10_NumUnitsCutFirstStage(9,8) - 453.0 Scenario10_ProduceSizeFirstStage(10) - 453.0 Scenario10_ProduceSizeFirstStage(1) - 453.0 Scenario10_ProduceSizeFirstStage(2) - 453.0 Scenario10_ProduceSizeFirstStage(3) 
- 453.0 Scenario10_ProduceSizeFirstStage(4) - 453.0 Scenario10_ProduceSizeFirstStage(5) - 453.0 Scenario10_ProduceSizeFirstStage(6) - 453.0 Scenario10_ProduceSizeFirstStage(7) - 453.0 Scenario10_ProduceSizeFirstStage(8) 
- 453.0 Scenario10_ProduceSizeFirstStage(9) = 0.0

Scenario10_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario10_NumProducedFirstStage(1) + 1.0 Scenario10_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario10_NumProducedFirstStage(2) + 1.0 Scenario10_NumUnitsCutFirstStage(2,1) + 1.0 Scenario10_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario10_NumProducedFirstStage(3) + 1.0 Scenario10_NumUnitsCutFirstStage(3,1) + 1.0 Scenario10_NumUnitsCutFirstStage(3,2) + 1.0 Scenario10_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario10_NumProducedFirstStage(4) + 1.0 Scenario10_NumUnitsCutFirstStage(4,1) + 1.0 Scenario10_NumUnitsCutFirstStage(4,2) + 1.0 Scenario10_NumUnitsCutFirstStage(4,3) + 1.0 Scenario10_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario10_NumProducedFirstStage(5) + 1.0 Scenario10_NumUnitsCutFirstStage(5,1) + 1.0 Scenario10_NumUnitsCutFirstStage(5,2) + 1.0 Scenario10_NumUnitsCutFirstStage(5,3) + 1.0 Scenario10_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario10_NumProducedFirstStage(6) + 1.0 Scenario10_NumUnitsCutFirstStage(6,1) + 1.0 Scenario10_NumUnitsCutFirstStage(6,2) + 1.0 Scenario10_NumUnitsCutFirstStage(6,3) + 1.0 Scenario10_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(6,5) + 1.0 Scenario10_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario10_NumProducedFirstStage(7) + 1.0 Scenario10_NumUnitsCutFirstStage(7,1) + 1.0 Scenario10_NumUnitsCutFirstStage(7,2) + 1.0 Scenario10_NumUnitsCutFirstStage(7,3) + 1.0 Scenario10_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(7,5) + 1.0 Scenario10_NumUnitsCutFirstStage(7,6) + 1.0 Scenario10_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario10_NumProducedFirstStage(8) + 1.0 Scenario10_NumUnitsCutFirstStage(8,1) + 1.0 Scenario10_NumUnitsCutFirstStage(8,2) + 1.0 Scenario10_NumUnitsCutFirstStage(8,3) + 1.0 Scenario10_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(8,5) + 1.0 Scenario10_NumUnitsCutFirstStage(8,6) + 1.0 Scenario10_NumUnitsCutFirstStage(8,7) + 1.0 Scenario10_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario10_NumProducedFirstStage(9) + 1.0 Scenario10_NumUnitsCutFirstStage(9,1) + 1.0 Scenario10_NumUnitsCutFirstStage(9,2) + 1.0 Scenario10_NumUnitsCutFirstStage(9,3) + 1.0 Scenario10_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(9,5) + 1.0 Scenario10_NumUnitsCutFirstStage(9,6) + 1.0 Scenario10_NumUnitsCutFirstStage(9,7) + 1.0 Scenario10_NumUnitsCutFirstStage(9,8) + 1.0 Scenario10_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario10_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario10_NumProducedFirstStage(10) + 1.0 Scenario10_NumUnitsCutFirstStage(10,10) + 1.0 Scenario10_NumUnitsCutFirstStage(10,1) + 1.0 Scenario10_NumUnitsCutFirstStage(10,2) + 1.0 Scenario10_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(10,4) + 1.0 Scenario10_NumUnitsCutFirstStage(10,5) + 1.0 Scenario10_NumUnitsCutFirstStage(10,6) + 1.0 Scenario10_NumUnitsCutFirstStage(10,7) + 1.0 Scenario10_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario10_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario10_NumProducedFirstStage(10) + 1.0 Scenario10_NumProducedFirstStage(1) + 1.0 Scenario10_NumProducedFirstStage(2) + 1.0 Scenario10_NumProducedFirstStage(3) + 1.0 Scenario10_NumProducedFirstStage(4) 
+ 1.0 Scenario10_NumProducedFirstStage(5) + 1.0 Scenario10_NumProducedFirstStage(6) + 1.0 Scenario10_NumProducedFirstStage(7) + 1.0 Scenario10_NumProducedFirstStage(8) + 1.0 Scenario10_NumProducedFirstStage(9) 
<= 200000.0

Scenario10_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario10_NumUnitsCutFirstStage(1,1) + 1.0 Scenario10_NumUnitsCutFirstStage(10,1) + 1.0 Scenario10_NumUnitsCutFirstStage(2,1) + 1.0 Scenario10_NumUnitsCutFirstStage(3,1) + 1.0 Scenario10_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(5,1) + 1.0 Scenario10_NumUnitsCutFirstStage(6,1) + 1.0 Scenario10_NumUnitsCutFirstStage(7,1) + 1.0 Scenario10_NumUnitsCutFirstStage(8,1) + 1.0 Scenario10_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario10_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,2) + 1.0 Scenario10_NumUnitsCutFirstStage(2,2) + 1.0 Scenario10_NumUnitsCutFirstStage(3,2) + 1.0 Scenario10_NumUnitsCutFirstStage(4,2) + 1.0 Scenario10_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(6,2) + 1.0 Scenario10_NumUnitsCutFirstStage(7,2) + 1.0 Scenario10_NumUnitsCutFirstStage(8,2) + 1.0 Scenario10_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario10_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,3) + 1.0 Scenario10_NumUnitsCutFirstStage(3,3) + 1.0 Scenario10_NumUnitsCutFirstStage(4,3) + 1.0 Scenario10_NumUnitsCutFirstStage(5,3) + 1.0 Scenario10_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(7,3) + 1.0 Scenario10_NumUnitsCutFirstStage(8,3) + 1.0 Scenario10_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario10_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,4) + 1.0 Scenario10_NumUnitsCutFirstStage(4,4) + 1.0 Scenario10_NumUnitsCutFirstStage(5,4) + 1.0 Scenario10_NumUnitsCutFirstStage(6,4) + 1.0 Scenario10_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(8,4) + 1.0 Scenario10_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario10_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,5) + 1.0 Scenario10_NumUnitsCutFirstStage(5,5) + 1.0 Scenario10_NumUnitsCutFirstStage(6,5) + 1.0 Scenario10_NumUnitsCutFirstStage(7,5) + 1.0 Scenario10_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario10_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,6) + 1.0 Scenario10_NumUnitsCutFirstStage(6,6) + 1.0 Scenario10_NumUnitsCutFirstStage(7,6) + 1.0 Scenario10_NumUnitsCutFirstStage(8,6) + 1.0 Scenario10_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario10_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,7) + 1.0 Scenario10_NumUnitsCutFirstStage(7,7) + 1.0 Scenario10_NumUnitsCutFirstStage(8,7) + 1.0 Scenario10_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario10_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,8) + 1.0 Scenario10_NumUnitsCutFirstStage(8,8) + 1.0 Scenario10_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario10_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,9) + 1.0 Scenario10_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario10_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario10_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario10_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario10_NumProducedFirstStage(1) - 1.0 Scenario10_NumProducedSecondStage(1) + 1.0 Scenario10_NumUnitsCutFirstStage(1,1) + 1.0 Scenario10_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario10_NumProducedFirstStage(2) - 1.0 Scenario10_NumProducedSecondStage(2) + 1.0 Scenario10_NumUnitsCutFirstStage(2,1) + 1.0 Scenario10_NumUnitsCutFirstStage(2,2) + 1.0 Scenario10_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario10_NumProducedFirstStage(3) - 1.0 Scenario10_NumProducedSecondStage(3) + 1.0 Scenario10_NumUnitsCutFirstStage(3,1) + 1.0 Scenario10_NumUnitsCutFirstStage(3,2) + 1.0 Scenario10_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(3,1) + 1.0 Scenario10_NumUnitsCutSecondStage(3,2) + 1.0 Scenario10_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario10_NumProducedFirstStage(4) - 1.0 Scenario10_NumProducedSecondStage(4) + 1.0 Scenario10_NumUnitsCutFirstStage(4,1) + 1.0 Scenario10_NumUnitsCutFirstStage(4,2) + 1.0 Scenario10_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(4,4) + 1.0 Scenario10_NumUnitsCutSecondStage(4,1) + 1.0 Scenario10_NumUnitsCutSecondStage(4,2) + 1.0 Scenario10_NumUnitsCutSecondStage(4,3) + 1.0 Scenario10_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario10_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario10_NumProducedFirstStage(5) - 1.0 Scenario10_NumProducedSecondStage(5) + 1.0 Scenario10_NumUnitsCutFirstStage(5,1) + 1.0 Scenario10_NumUnitsCutFirstStage(5,2) + 1.0 Scenario10_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(5,4) + 1.0 Scenario10_NumUnitsCutFirstStage(5,5) + 1.0 Scenario10_NumUnitsCutSecondStage(5,1) + 1.0 Scenario10_NumUnitsCutSecondStage(5,2) + 1.0 Scenario10_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(5,4) + 1.0 Scenario10_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario10_NumProducedFirstStage(6) - 1.0 Scenario10_NumProducedSecondStage(6) + 1.0 Scenario10_NumUnitsCutFirstStage(6,1) + 1.0 Scenario10_NumUnitsCutFirstStage(6,2) + 1.0 Scenario10_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(6,4) + 1.0 Scenario10_NumUnitsCutFirstStage(6,5) + 1.0 Scenario10_NumUnitsCutFirstStage(6,6) + 1.0 Scenario10_NumUnitsCutSecondStage(6,1) + 1.0 Scenario10_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(6,3) + 1.0 Scenario10_NumUnitsCutSecondStage(6,4) + 1.0 Scenario10_NumUnitsCutSecondStage(6,5) + 1.0 Scenario10_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario10_NumProducedFirstStage(7) - 1.0 Scenario10_NumProducedSecondStage(7) + 1.0 Scenario10_NumUnitsCutFirstStage(7,1) + 1.0 Scenario10_NumUnitsCutFirstStage(7,2) + 1.0 Scenario10_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(7,4) + 1.0 Scenario10_NumUnitsCutFirstStage(7,5) + 1.0 Scenario10_NumUnitsCutFirstStage(7,6) + 1.0 Scenario10_NumUnitsCutFirstStage(7,7) + 1.0 Scenario10_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(7,2) + 1.0 Scenario10_NumUnitsCutSecondStage(7,3) + 1.0 Scenario10_NumUnitsCutSecondStage(7,4) + 1.0 Scenario10_NumUnitsCutSecondStage(7,5) + 1.0 Scenario10_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario10_NumProducedFirstStage(8) - 1.0 Scenario10_NumProducedSecondStage(8) + 1.0 Scenario10_NumUnitsCutFirstStage(8,1) + 1.0 Scenario10_NumUnitsCutFirstStage(8,2) + 1.0 Scenario10_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(8,4) + 1.0 Scenario10_NumUnitsCutFirstStage(8,5) + 1.0 Scenario10_NumUnitsCutFirstStage(8,6) + 1.0 Scenario10_NumUnitsCutFirstStage(8,7) + 1.0 Scenario10_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(8,1) + 1.0 Scenario10_NumUnitsCutSecondStage(8,2) + 1.0 Scenario10_NumUnitsCutSecondStage(8,3) + 1.0 Scenario10_NumUnitsCutSecondStage(8,4) + 1.0 Scenario10_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(8,6) + 1.0 Scenario10_NumUnitsCutSecondStage(8,7) + 1.0 Scenario10_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario10_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario10_NumProducedFirstStage(9) - 1.0 Scenario10_NumProducedSecondStage(9) + 1.0 Scenario10_NumUnitsCutFirstStage(9,1) + 1.0 Scenario10_NumUnitsCutFirstStage(9,2) + 1.0 Scenario10_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(9,4) + 1.0 Scenario10_NumUnitsCutFirstStage(9,5) + 1.0 Scenario10_NumUnitsCutFirstStage(9,6) + 1.0 Scenario10_NumUnitsCutFirstStage(9,7) + 1.0 Scenario10_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(9,9) + 1.0 Scenario10_NumUnitsCutSecondStage(9,1) + 1.0 Scenario10_NumUnitsCutSecondStage(9,2) + 1.0 Scenario10_NumUnitsCutSecondStage(9,3) + 1.0 Scenario10_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(9,5) + 1.0 Scenario10_NumUnitsCutSecondStage(9,6) + 1.0 Scenario10_NumUnitsCutSecondStage(9,7) + 1.0 Scenario10_NumUnitsCutSecondStage(9,8) + 1.0 Scenario10_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario10_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario10_NumProducedFirstStage(10) - 1.0 Scenario10_NumProducedSecondStage(10) + 1.0 Scenario10_NumUnitsCutFirstStage(10,10) + 1.0 Scenario10_NumUnitsCutFirstStage(10,1) + 1.0 Scenario10_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(10,3) + 1.0 Scenario10_NumUnitsCutFirstStage(10,4) + 1.0 Scenario10_NumUnitsCutFirstStage(10,5) + 1.0 Scenario10_NumUnitsCutFirstStage(10,6) + 1.0 Scenario10_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario10_NumUnitsCutFirstStage(10,8) + 1.0 Scenario10_NumUnitsCutFirstStage(10,9) + 1.0 Scenario10_NumUnitsCutSecondStage(10,10) + 1.0 Scenario10_NumUnitsCutSecondStage(10,1) + 1.0 Scenario10_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(10,3) + 1.0 Scenario10_NumUnitsCutSecondStage(10,4) + 1.0 Scenario10_NumUnitsCutSecondStage(10,5) + 1.0 Scenario10_NumUnitsCutSecondStage(10,6) + 1.0 Scenario10_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario10_NumUnitsCutSecondStage(10,8) + 1.0 Scenario10_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario10

\ Begin constraint block for scenario Scenario3
Scenario3_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario3_NumUnitsCutSecondStage(1,1) + 1.0 Scenario3_NumUnitsCutSecondStage(10,1) + 1.0 Scenario3_NumUnitsCutSecondStage(2,1) + 1.0 Scenario3_NumUnitsCutSecondStage(3,1) + 1.0 Scenario3_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(5,1) + 1.0 Scenario3_NumUnitsCutSecondStage(6,1) + 1.0 Scenario3_NumUnitsCutSecondStage(7,1) + 1.0 Scenario3_NumUnitsCutSecondStage(8,1) + 1.0 Scenario3_NumUnitsCutSecondStage(9,1) 
>= 1750.0

Scenario3_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,2) + 1.0 Scenario3_NumUnitsCutSecondStage(2,2) + 1.0 Scenario3_NumUnitsCutSecondStage(3,2) + 1.0 Scenario3_NumUnitsCutSecondStage(4,2) + 1.0 Scenario3_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(6,2) + 1.0 Scenario3_NumUnitsCutSecondStage(7,2) + 1.0 Scenario3_NumUnitsCutSecondStage(8,2) + 1.0 Scenario3_NumUnitsCutSecondStage(9,2) >= 5250.0

Scenario3_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,3) + 1.0 Scenario3_NumUnitsCutSecondStage(3,3) + 1.0 Scenario3_NumUnitsCutSecondStage(4,3) + 1.0 Scenario3_NumUnitsCutSecondStage(5,3) + 1.0 Scenario3_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(7,3) + 1.0 Scenario3_NumUnitsCutSecondStage(8,3) + 1.0 Scenario3_NumUnitsCutSecondStage(9,3) >= 8750.0

Scenario3_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,4) + 1.0 Scenario3_NumUnitsCutSecondStage(4,4) + 1.0 Scenario3_NumUnitsCutSecondStage(5,4) + 1.0 Scenario3_NumUnitsCutSecondStage(6,4) + 1.0 Scenario3_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(8,4) + 1.0 Scenario3_NumUnitsCutSecondStage(9,4) >= 7000.0

Scenario3_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,5) + 1.0 Scenario3_NumUnitsCutSecondStage(5,5) + 1.0 Scenario3_NumUnitsCutSecondStage(6,5) + 1.0 Scenario3_NumUnitsCutSecondStage(7,5) + 1.0 Scenario3_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(9,5) >= 24500.0

Scenario3_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,6) + 1.0 Scenario3_NumUnitsCutSecondStage(6,6) + 1.0 Scenario3_NumUnitsCutSecondStage(7,6) + 1.0 Scenario3_NumUnitsCutSecondStage(8,6) + 1.0 Scenario3_NumUnitsCutSecondStage(9,6) 
>= 17500.0

Scenario3_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,7) + 1.0 Scenario3_NumUnitsCutSecondStage(7,7) + 1.0 Scenario3_NumUnitsCutSecondStage(8,7) + 1.0 Scenario3_NumUnitsCutSecondStage(9,7) >= 10500.0

Scenario3_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,8) + 1.0 Scenario3_NumUnitsCutSecondStage(8,8) + 1.0 Scenario3_NumUnitsCutSecondStage(9,8) >= 8750.0

Scenario3_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,9) + 1.0 Scenario3_NumUnitsCutSecondStage(9,9) >= 8750.0

Scenario3_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,10) >= 3500.0

Scenario3_c_e_ComputeSecondStageCost_:  -0.8416 Scenario3_NumProducedSecondStage(10) - 0.748 Scenario3_NumProducedSecondStage(1) - 0.7584 Scenario3_NumProducedSecondStage(2) - 0.7688 Scenario3_NumProducedSecondStage(3) - 0.7792 Scenario3_NumProducedSecondStage(4) 
- 0.7896 Scenario3_NumProducedSecondStage(5) - 0.8 Scenario3_NumProducedSecondStage(6) - 0.8104 Scenario3_NumProducedSecondStage(7) - 0.8208 Scenario3_NumProducedSecondStage(8) - 0.8312 Scenario3_NumProducedSecondStage(9) 
- 0.008 Scenario3_NumUnitsCutSecondStage(10,1) - 0.008 Scenario3_NumUnitsCutSecondStage(10,2) - 0.008 Scenario3_NumUnitsCutSecondStage(10,3) - 0.008 Scenario3_NumUnitsCutSecondStage(10,4) - 0.008 Scenario3_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario3_NumUnitsCutSecondStage(10,6) - 0.008 Scenario3_NumUnitsCutSecondStage(10,7) - 0.008 Scenario3_NumUnitsCutSecondStage(10,8) - 0.008 Scenario3_NumUnitsCutSecondStage(10,9) - 0.008 Scenario3_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario3_NumUnitsCutSecondStage(3,1) - 0.008 Scenario3_NumUnitsCutSecondStage(3,2) - 0.008 Scenario3_NumUnitsCutSecondStage(4,1) - 0.008 Scenario3_NumUnitsCutSecondStage(4,2) - 0.008 Scenario3_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario3_NumUnitsCutSecondStage(5,1) - 0.008 Scenario3_NumUnitsCutSecondStage(5,2) - 0.008 Scenario3_NumUnitsCutSecondStage(5,3) - 0.008 Scenario3_NumUnitsCutSecondStage(5,4) - 0.008 Scenario3_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario3_NumUnitsCutSecondStage(6,2) - 0.008 Scenario3_NumUnitsCutSecondStage(6,3) - 0.008 Scenario3_NumUnitsCutSecondStage(6,4) - 0.008 Scenario3_NumUnitsCutSecondStage(6,5) - 0.008 Scenario3_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario3_NumUnitsCutSecondStage(7,2) - 0.008 Scenario3_NumUnitsCutSecondStage(7,3) - 0.008 Scenario3_NumUnitsCutSecondStage(7,4) - 0.008 Scenario3_NumUnitsCutSecondStage(7,5) - 0.008 Scenario3_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario3_NumUnitsCutSecondStage(8,1) - 0.008 Scenario3_NumUnitsCutSecondStage(8,2) - 0.008 Scenario3_NumUnitsCutSecondStage(8,3) - 0.008 Scenario3_NumUnitsCutSecondStage(8,4) - 0.008 Scenario3_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario3_NumUnitsCutSecondStage(8,6) - 0.008 Scenario3_NumUnitsCutSecondStage(8,7) - 0.008 Scenario3_NumUnitsCutSecondStage(9,1) - 0.008 Scenario3_NumUnitsCutSecondStage(9,2) - 0.008 Scenario3_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario3_NumUnitsCutSecondStage(9,4) - 0.008 Scenario3_NumUnitsCutSecondStage(9,5) - 0.008 Scenario3_NumUnitsCutSecondStage(9,6) - 0.008 Scenario3_NumUnitsCutSecondStage(9,7) - 0.008 Scenario3_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario3_ProduceSizeSecondStage(10) - 453.0 Scenario3_ProduceSizeSecondStage(1) - 453.0 Scenario3_ProduceSizeSecondStage(2) - 453.0 Scenario3_ProduceSizeSecondStage(3) - 453.0 Scenario3_ProduceSizeSecondStage(4) 
- 453.0 Scenario3_ProduceSizeSecondStage(5) - 453.0 Scenario3_ProduceSizeSecondStage(6) - 453.0 Scenario3_ProduceSizeSecondStage(7) - 453.0 Scenario3_ProduceSizeSecondStage(8) - 453.0 Scenario3_ProduceSizeSecondStage(9) 
+ 1.0 Scenario3_SecondStageCost = 0.0

Scenario3_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario3_NumProducedSecondStage(10) + 1.0 Scenario3_NumProducedSecondStage(1) + 1.0 Scenario3_NumProducedSecondStage(2) + 1.0 Scenario3_NumProducedSecondStage(3) + 1.0 Scenario3_NumProducedSecondStage(4) 
+ 1.0 Scenario3_NumProducedSecondStage(5) + 1.0 Scenario3_NumProducedSecondStage(6) + 1.0 Scenario3_NumProducedSecondStage(7) + 1.0 Scenario3_NumProducedSecondStage(8) + 1.0 Scenario3_NumProducedSecondStage(9) 
<= 200000.0

Scenario3_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario3_NumProducedSecondStage(1) - 200000.0 Scenario3_ProduceSizeSecondStage(1) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario3_NumProducedSecondStage(2) - 200000.0 Scenario3_ProduceSizeSecondStage(2) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario3_NumProducedSecondStage(3) - 200000.0 Scenario3_ProduceSizeSecondStage(3) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario3_NumProducedSecondStage(4) - 200000.0 Scenario3_ProduceSizeSecondStage(4) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario3_NumProducedSecondStage(5) - 200000.0 Scenario3_ProduceSizeSecondStage(5) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario3_NumProducedSecondStage(6) - 200000.0 Scenario3_ProduceSizeSecondStage(6) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario3_NumProducedSecondStage(7) - 200000.0 Scenario3_ProduceSizeSecondStage(7) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario3_NumProducedSecondStage(8) - 200000.0 Scenario3_ProduceSizeSecondStage(8) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario3_NumProducedSecondStage(9) - 200000.0 Scenario3_ProduceSizeSecondStage(9) <= 0.0

Scenario3_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario3_NumProducedSecondStage(10) - 200000.0 Scenario3_ProduceSizeSecondStage(10) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario3_NumProducedFirstStage(1) - 200000.0 Scenario3_ProduceSizeFirstStage(1) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario3_NumProducedFirstStage(2) - 200000.0 Scenario3_ProduceSizeFirstStage(2) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario3_NumProducedFirstStage(3) - 200000.0 Scenario3_ProduceSizeFirstStage(3) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario3_NumProducedFirstStage(4) - 200000.0 Scenario3_ProduceSizeFirstStage(4) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario3_NumProducedFirstStage(5) - 200000.0 Scenario3_ProduceSizeFirstStage(5) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario3_NumProducedFirstStage(6) - 200000.0 Scenario3_ProduceSizeFirstStage(6) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario3_NumProducedFirstStage(7) - 200000.0 Scenario3_ProduceSizeFirstStage(7) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario3_NumProducedFirstStage(8) - 200000.0 Scenario3_ProduceSizeFirstStage(8) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario3_NumProducedFirstStage(9) - 200000.0 Scenario3_ProduceSizeFirstStage(9) <= 0.0

Scenario3_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario3_NumProducedFirstStage(10) - 200000.0 Scenario3_ProduceSizeFirstStage(10) <= 0.0

Scenario3_c_e_ComputeFirstStageCost_:  1.0 Scenario3_FirstStageCost - 0.8416 Scenario3_NumProducedFirstStage(10) - 0.748 Scenario3_NumProducedFirstStage(1) - 0.7584 Scenario3_NumProducedFirstStage(2) - 0.7688 Scenario3_NumProducedFirstStage(3) 
- 0.7792 Scenario3_NumProducedFirstStage(4) - 0.7896 Scenario3_NumProducedFirstStage(5) - 0.8 Scenario3_NumProducedFirstStage(6) - 0.8104 Scenario3_NumProducedFirstStage(7) - 0.8208 Scenario3_NumProducedFirstStage(8) 
- 0.8312 Scenario3_NumProducedFirstStage(9) - 0.008 Scenario3_NumUnitsCutFirstStage(10,1) - 0.008 Scenario3_NumUnitsCutFirstStage(10,2) - 0.008 Scenario3_NumUnitsCutFirstStage(10,3) - 0.008 Scenario3_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario3_NumUnitsCutFirstStage(10,5) - 0.008 Scenario3_NumUnitsCutFirstStage(10,6) - 0.008 Scenario3_NumUnitsCutFirstStage(10,7) - 0.008 Scenario3_NumUnitsCutFirstStage(10,8) - 0.008 Scenario3_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario3_NumUnitsCutFirstStage(2,1) - 0.008 Scenario3_NumUnitsCutFirstStage(3,1) - 0.008 Scenario3_NumUnitsCutFirstStage(3,2) - 0.008 Scenario3_NumUnitsCutFirstStage(4,1) - 0.008 Scenario3_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario3_NumUnitsCutFirstStage(4,3) - 0.008 Scenario3_NumUnitsCutFirstStage(5,1) - 0.008 Scenario3_NumUnitsCutFirstStage(5,2) - 0.008 Scenario3_NumUnitsCutFirstStage(5,3) - 0.008 Scenario3_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario3_NumUnitsCutFirstStage(6,1) - 0.008 Scenario3_NumUnitsCutFirstStage(6,2) - 0.008 Scenario3_NumUnitsCutFirstStage(6,3) - 0.008 Scenario3_NumUnitsCutFirstStage(6,4) - 0.008 Scenario3_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario3_NumUnitsCutFirstStage(7,1) - 0.008 Scenario3_NumUnitsCutFirstStage(7,2) - 0.008 Scenario3_NumUnitsCutFirstStage(7,3) - 0.008 Scenario3_NumUnitsCutFirstStage(7,4) - 0.008 Scenario3_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario3_NumUnitsCutFirstStage(7,6) - 0.008 Scenario3_NumUnitsCutFirstStage(8,1) - 0.008 Scenario3_NumUnitsCutFirstStage(8,2) - 0.008 Scenario3_NumUnitsCutFirstStage(8,3) - 0.008 Scenario3_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario3_NumUnitsCutFirstStage(8,5) - 0.008 Scenario3_NumUnitsCutFirstStage(8,6) - 0.008 Scenario3_NumUnitsCutFirstStage(8,7) - 0.008 Scenario3_NumUnitsCutFirstStage(9,1) - 0.008 Scenario3_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario3_NumUnitsCutFirstStage(9,3) - 0.008 Scenario3_NumUnitsCutFirstStage(9,4) - 0.008 Scenario3_NumUnitsCutFirstStage(9,5) - 0.008 Scenario3_NumUnitsCutFirstStage(9,6) - 0.008 Scenario3_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario3_NumUnitsCutFirstStage(9,8) - 453.0 Scenario3_ProduceSizeFirstStage(10) - 453.0 Scenario3_ProduceSizeFirstStage(1) - 453.0 Scenario3_ProduceSizeFirstStage(2) - 453.0 Scenario3_ProduceSizeFirstStage(3) 
- 453.0 Scenario3_ProduceSizeFirstStage(4) - 453.0 Scenario3_ProduceSizeFirstStage(5) - 453.0 Scenario3_ProduceSizeFirstStage(6) - 453.0 Scenario3_ProduceSizeFirstStage(7) - 453.0 Scenario3_ProduceSizeFirstStage(8) 
- 453.0 Scenario3_ProduceSizeFirstStage(9) = 0.0

Scenario3_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario3_NumProducedFirstStage(1) + 1.0 Scenario3_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario3_NumProducedFirstStage(2) + 1.0 Scenario3_NumUnitsCutFirstStage(2,1) + 1.0 Scenario3_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario3_NumProducedFirstStage(3) + 1.0 Scenario3_NumUnitsCutFirstStage(3,1) + 1.0 Scenario3_NumUnitsCutFirstStage(3,2) + 1.0 Scenario3_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario3_NumProducedFirstStage(4) + 1.0 Scenario3_NumUnitsCutFirstStage(4,1) + 1.0 Scenario3_NumUnitsCutFirstStage(4,2) + 1.0 Scenario3_NumUnitsCutFirstStage(4,3) + 1.0 Scenario3_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario3_NumProducedFirstStage(5) + 1.0 Scenario3_NumUnitsCutFirstStage(5,1) + 1.0 Scenario3_NumUnitsCutFirstStage(5,2) + 1.0 Scenario3_NumUnitsCutFirstStage(5,3) + 1.0 Scenario3_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario3_NumProducedFirstStage(6) + 1.0 Scenario3_NumUnitsCutFirstStage(6,1) + 1.0 Scenario3_NumUnitsCutFirstStage(6,2) + 1.0 Scenario3_NumUnitsCutFirstStage(6,3) + 1.0 Scenario3_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(6,5) + 1.0 Scenario3_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario3_NumProducedFirstStage(7) + 1.0 Scenario3_NumUnitsCutFirstStage(7,1) + 1.0 Scenario3_NumUnitsCutFirstStage(7,2) + 1.0 Scenario3_NumUnitsCutFirstStage(7,3) + 1.0 Scenario3_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(7,5) + 1.0 Scenario3_NumUnitsCutFirstStage(7,6) + 1.0 Scenario3_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario3_NumProducedFirstStage(8) + 1.0 Scenario3_NumUnitsCutFirstStage(8,1) + 1.0 Scenario3_NumUnitsCutFirstStage(8,2) + 1.0 Scenario3_NumUnitsCutFirstStage(8,3) + 1.0 Scenario3_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(8,5) + 1.0 Scenario3_NumUnitsCutFirstStage(8,6) + 1.0 Scenario3_NumUnitsCutFirstStage(8,7) + 1.0 Scenario3_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario3_NumProducedFirstStage(9) + 1.0 Scenario3_NumUnitsCutFirstStage(9,1) + 1.0 Scenario3_NumUnitsCutFirstStage(9,2) + 1.0 Scenario3_NumUnitsCutFirstStage(9,3) + 1.0 Scenario3_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(9,5) + 1.0 Scenario3_NumUnitsCutFirstStage(9,6) + 1.0 Scenario3_NumUnitsCutFirstStage(9,7) + 1.0 Scenario3_NumUnitsCutFirstStage(9,8) + 1.0 Scenario3_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario3_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario3_NumProducedFirstStage(10) + 1.0 Scenario3_NumUnitsCutFirstStage(10,10) + 1.0 Scenario3_NumUnitsCutFirstStage(10,1) + 1.0 Scenario3_NumUnitsCutFirstStage(10,2) + 1.0 Scenario3_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(10,4) + 1.0 Scenario3_NumUnitsCutFirstStage(10,5) + 1.0 Scenario3_NumUnitsCutFirstStage(10,6) + 1.0 Scenario3_NumUnitsCutFirstStage(10,7) + 1.0 Scenario3_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario3_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario3_NumProducedFirstStage(10) + 1.0 Scenario3_NumProducedFirstStage(1) + 1.0 Scenario3_NumProducedFirstStage(2) + 1.0 Scenario3_NumProducedFirstStage(3) + 1.0 Scenario3_NumProducedFirstStage(4) 
+ 1.0 Scenario3_NumProducedFirstStage(5) + 1.0 Scenario3_NumProducedFirstStage(6) + 1.0 Scenario3_NumProducedFirstStage(7) + 1.0 Scenario3_NumProducedFirstStage(8) + 1.0 Scenario3_NumProducedFirstStage(9) 
<= 200000.0

Scenario3_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario3_NumUnitsCutFirstStage(1,1) + 1.0 Scenario3_NumUnitsCutFirstStage(10,1) + 1.0 Scenario3_NumUnitsCutFirstStage(2,1) + 1.0 Scenario3_NumUnitsCutFirstStage(3,1) + 1.0 Scenario3_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(5,1) + 1.0 Scenario3_NumUnitsCutFirstStage(6,1) + 1.0 Scenario3_NumUnitsCutFirstStage(7,1) + 1.0 Scenario3_NumUnitsCutFirstStage(8,1) + 1.0 Scenario3_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario3_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,2) + 1.0 Scenario3_NumUnitsCutFirstStage(2,2) + 1.0 Scenario3_NumUnitsCutFirstStage(3,2) + 1.0 Scenario3_NumUnitsCutFirstStage(4,2) + 1.0 Scenario3_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(6,2) + 1.0 Scenario3_NumUnitsCutFirstStage(7,2) + 1.0 Scenario3_NumUnitsCutFirstStage(8,2) + 1.0 Scenario3_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario3_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,3) + 1.0 Scenario3_NumUnitsCutFirstStage(3,3) + 1.0 Scenario3_NumUnitsCutFirstStage(4,3) + 1.0 Scenario3_NumUnitsCutFirstStage(5,3) + 1.0 Scenario3_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(7,3) + 1.0 Scenario3_NumUnitsCutFirstStage(8,3) + 1.0 Scenario3_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario3_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,4) + 1.0 Scenario3_NumUnitsCutFirstStage(4,4) + 1.0 Scenario3_NumUnitsCutFirstStage(5,4) + 1.0 Scenario3_NumUnitsCutFirstStage(6,4) + 1.0 Scenario3_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(8,4) + 1.0 Scenario3_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario3_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,5) + 1.0 Scenario3_NumUnitsCutFirstStage(5,5) + 1.0 Scenario3_NumUnitsCutFirstStage(6,5) + 1.0 Scenario3_NumUnitsCutFirstStage(7,5) + 1.0 Scenario3_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario3_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,6) + 1.0 Scenario3_NumUnitsCutFirstStage(6,6) + 1.0 Scenario3_NumUnitsCutFirstStage(7,6) + 1.0 Scenario3_NumUnitsCutFirstStage(8,6) + 1.0 Scenario3_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario3_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,7) + 1.0 Scenario3_NumUnitsCutFirstStage(7,7) + 1.0 Scenario3_NumUnitsCutFirstStage(8,7) + 1.0 Scenario3_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario3_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,8) + 1.0 Scenario3_NumUnitsCutFirstStage(8,8) + 1.0 Scenario3_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario3_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,9) + 1.0 Scenario3_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario3_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario3_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario3_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario3_NumProducedFirstStage(1) - 1.0 Scenario3_NumProducedSecondStage(1) + 1.0 Scenario3_NumUnitsCutFirstStage(1,1) + 1.0 Scenario3_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario3_NumProducedFirstStage(2) - 1.0 Scenario3_NumProducedSecondStage(2) + 1.0 Scenario3_NumUnitsCutFirstStage(2,1) + 1.0 Scenario3_NumUnitsCutFirstStage(2,2) + 1.0 Scenario3_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario3_NumProducedFirstStage(3) - 1.0 Scenario3_NumProducedSecondStage(3) + 1.0 Scenario3_NumUnitsCutFirstStage(3,1) + 1.0 Scenario3_NumUnitsCutFirstStage(3,2) + 1.0 Scenario3_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(3,1) + 1.0 Scenario3_NumUnitsCutSecondStage(3,2) + 1.0 Scenario3_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario3_NumProducedFirstStage(4) - 1.0 Scenario3_NumProducedSecondStage(4) + 1.0 Scenario3_NumUnitsCutFirstStage(4,1) + 1.0 Scenario3_NumUnitsCutFirstStage(4,2) + 1.0 Scenario3_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(4,4) + 1.0 Scenario3_NumUnitsCutSecondStage(4,1) + 1.0 Scenario3_NumUnitsCutSecondStage(4,2) + 1.0 Scenario3_NumUnitsCutSecondStage(4,3) + 1.0 Scenario3_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario3_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario3_NumProducedFirstStage(5) - 1.0 Scenario3_NumProducedSecondStage(5) + 1.0 Scenario3_NumUnitsCutFirstStage(5,1) + 1.0 Scenario3_NumUnitsCutFirstStage(5,2) + 1.0 Scenario3_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(5,4) + 1.0 Scenario3_NumUnitsCutFirstStage(5,5) + 1.0 Scenario3_NumUnitsCutSecondStage(5,1) + 1.0 Scenario3_NumUnitsCutSecondStage(5,2) + 1.0 Scenario3_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(5,4) + 1.0 Scenario3_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario3_NumProducedFirstStage(6) - 1.0 Scenario3_NumProducedSecondStage(6) + 1.0 Scenario3_NumUnitsCutFirstStage(6,1) + 1.0 Scenario3_NumUnitsCutFirstStage(6,2) + 1.0 Scenario3_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(6,4) + 1.0 Scenario3_NumUnitsCutFirstStage(6,5) + 1.0 Scenario3_NumUnitsCutFirstStage(6,6) + 1.0 Scenario3_NumUnitsCutSecondStage(6,1) + 1.0 Scenario3_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(6,3) + 1.0 Scenario3_NumUnitsCutSecondStage(6,4) + 1.0 Scenario3_NumUnitsCutSecondStage(6,5) + 1.0 Scenario3_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario3_NumProducedFirstStage(7) - 1.0 Scenario3_NumProducedSecondStage(7) + 1.0 Scenario3_NumUnitsCutFirstStage(7,1) + 1.0 Scenario3_NumUnitsCutFirstStage(7,2) + 1.0 Scenario3_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(7,4) + 1.0 Scenario3_NumUnitsCutFirstStage(7,5) + 1.0 Scenario3_NumUnitsCutFirstStage(7,6) + 1.0 Scenario3_NumUnitsCutFirstStage(7,7) + 1.0 Scenario3_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(7,2) + 1.0 Scenario3_NumUnitsCutSecondStage(7,3) + 1.0 Scenario3_NumUnitsCutSecondStage(7,4) + 1.0 Scenario3_NumUnitsCutSecondStage(7,5) + 1.0 Scenario3_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario3_NumProducedFirstStage(8) - 1.0 Scenario3_NumProducedSecondStage(8) + 1.0 Scenario3_NumUnitsCutFirstStage(8,1) + 1.0 Scenario3_NumUnitsCutFirstStage(8,2) + 1.0 Scenario3_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(8,4) + 1.0 Scenario3_NumUnitsCutFirstStage(8,5) + 1.0 Scenario3_NumUnitsCutFirstStage(8,6) + 1.0 Scenario3_NumUnitsCutFirstStage(8,7) + 1.0 Scenario3_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(8,1) + 1.0 Scenario3_NumUnitsCutSecondStage(8,2) + 1.0 Scenario3_NumUnitsCutSecondStage(8,3) + 1.0 Scenario3_NumUnitsCutSecondStage(8,4) + 1.0 Scenario3_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(8,6) + 1.0 Scenario3_NumUnitsCutSecondStage(8,7) + 1.0 Scenario3_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario3_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario3_NumProducedFirstStage(9) - 1.0 Scenario3_NumProducedSecondStage(9) + 1.0 Scenario3_NumUnitsCutFirstStage(9,1) + 1.0 Scenario3_NumUnitsCutFirstStage(9,2) + 1.0 Scenario3_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(9,4) + 1.0 Scenario3_NumUnitsCutFirstStage(9,5) + 1.0 Scenario3_NumUnitsCutFirstStage(9,6) + 1.0 Scenario3_NumUnitsCutFirstStage(9,7) + 1.0 Scenario3_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(9,9) + 1.0 Scenario3_NumUnitsCutSecondStage(9,1) + 1.0 Scenario3_NumUnitsCutSecondStage(9,2) + 1.0 Scenario3_NumUnitsCutSecondStage(9,3) + 1.0 Scenario3_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(9,5) + 1.0 Scenario3_NumUnitsCutSecondStage(9,6) + 1.0 Scenario3_NumUnitsCutSecondStage(9,7) + 1.0 Scenario3_NumUnitsCutSecondStage(9,8) + 1.0 Scenario3_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario3_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario3_NumProducedFirstStage(10) - 1.0 Scenario3_NumProducedSecondStage(10) + 1.0 Scenario3_NumUnitsCutFirstStage(10,10) + 1.0 Scenario3_NumUnitsCutFirstStage(10,1) + 1.0 Scenario3_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(10,3) + 1.0 Scenario3_NumUnitsCutFirstStage(10,4) + 1.0 Scenario3_NumUnitsCutFirstStage(10,5) + 1.0 Scenario3_NumUnitsCutFirstStage(10,6) + 1.0 Scenario3_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario3_NumUnitsCutFirstStage(10,8) + 1.0 Scenario3_NumUnitsCutFirstStage(10,9) + 1.0 Scenario3_NumUnitsCutSecondStage(10,10) + 1.0 Scenario3_NumUnitsCutSecondStage(10,1) + 1.0 Scenario3_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(10,3) + 1.0 Scenario3_NumUnitsCutSecondStage(10,4) + 1.0 Scenario3_NumUnitsCutSecondStage(10,5) + 1.0 Scenario3_NumUnitsCutSecondStage(10,6) + 1.0 Scenario3_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(10,8) + 1.0 Scenario3_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario3

\ Begin constraint block for scenario Scenario2
Scenario2_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario2_NumUnitsCutSecondStage(1,1) + 1.0 Scenario2_NumUnitsCutSecondStage(10,1) + 1.0 Scenario2_NumUnitsCutSecondStage(2,1) + 1.0 Scenario2_NumUnitsCutSecondStage(3,1) + 1.0 Scenario2_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(5,1) + 1.0 Scenario2_NumUnitsCutSecondStage(6,1) + 1.0 Scenario2_NumUnitsCutSecondStage(7,1) + 1.0 Scenario2_NumUnitsCutSecondStage(8,1) + 1.0 Scenario2_NumUnitsCutSecondStage(9,1) 
>= 1500.0

Scenario2_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,2) + 1.0 Scenario2_NumUnitsCutSecondStage(2,2) + 1.0 Scenario2_NumUnitsCutSecondStage(3,2) + 1.0 Scenario2_NumUnitsCutSecondStage(4,2) + 1.0 Scenario2_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(6,2) + 1.0 Scenario2_NumUnitsCutSecondStage(7,2) + 1.0 Scenario2_NumUnitsCutSecondStage(8,2) + 1.0 Scenario2_NumUnitsCutSecondStage(9,2) >= 4500.0

Scenario2_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,3) + 1.0 Scenario2_NumUnitsCutSecondStage(3,3) + 1.0 Scenario2_NumUnitsCutSecondStage(4,3) + 1.0 Scenario2_NumUnitsCutSecondStage(5,3) + 1.0 Scenario2_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(7,3) + 1.0 Scenario2_NumUnitsCutSecondStage(8,3) + 1.0 Scenario2_NumUnitsCutSecondStage(9,3) >= 7500.0

Scenario2_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,4) + 1.0 Scenario2_NumUnitsCutSecondStage(4,4) + 1.0 Scenario2_NumUnitsCutSecondStage(5,4) + 1.0 Scenario2_NumUnitsCutSecondStage(6,4) + 1.0 Scenario2_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(8,4) + 1.0 Scenario2_NumUnitsCutSecondStage(9,4) >= 6000.0

Scenario2_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,5) + 1.0 Scenario2_NumUnitsCutSecondStage(5,5) + 1.0 Scenario2_NumUnitsCutSecondStage(6,5) + 1.0 Scenario2_NumUnitsCutSecondStage(7,5) + 1.0 Scenario2_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(9,5) >= 21000.0

Scenario2_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,6) + 1.0 Scenario2_NumUnitsCutSecondStage(6,6) + 1.0 Scenario2_NumUnitsCutSecondStage(7,6) + 1.0 Scenario2_NumUnitsCutSecondStage(8,6) + 1.0 Scenario2_NumUnitsCutSecondStage(9,6) 
>= 15000.0

Scenario2_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,7) + 1.0 Scenario2_NumUnitsCutSecondStage(7,7) + 1.0 Scenario2_NumUnitsCutSecondStage(8,7) + 1.0 Scenario2_NumUnitsCutSecondStage(9,7) >= 9000.0

Scenario2_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,8) + 1.0 Scenario2_NumUnitsCutSecondStage(8,8) + 1.0 Scenario2_NumUnitsCutSecondStage(9,8) >= 7500.0

Scenario2_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,9) + 1.0 Scenario2_NumUnitsCutSecondStage(9,9) >= 7500.0

Scenario2_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,10) >= 3000.0

Scenario2_c_e_ComputeSecondStageCost_:  -0.8416 Scenario2_NumProducedSecondStage(10) - 0.748 Scenario2_NumProducedSecondStage(1) - 0.7584 Scenario2_NumProducedSecondStage(2) - 0.7688 Scenario2_NumProducedSecondStage(3) - 0.7792 Scenario2_NumProducedSecondStage(4) 
- 0.7896 Scenario2_NumProducedSecondStage(5) - 0.8 Scenario2_NumProducedSecondStage(6) - 0.8104 Scenario2_NumProducedSecondStage(7) - 0.8208 Scenario2_NumProducedSecondStage(8) - 0.8312 Scenario2_NumProducedSecondStage(9) 
- 0.008 Scenario2_NumUnitsCutSecondStage(10,1) - 0.008 Scenario2_NumUnitsCutSecondStage(10,2) - 0.008 Scenario2_NumUnitsCutSecondStage(10,3) - 0.008 Scenario2_NumUnitsCutSecondStage(10,4) - 0.008 Scenario2_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario2_NumUnitsCutSecondStage(10,6) - 0.008 Scenario2_NumUnitsCutSecondStage(10,7) - 0.008 Scenario2_NumUnitsCutSecondStage(10,8) - 0.008 Scenario2_NumUnitsCutSecondStage(10,9) - 0.008 Scenario2_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario2_NumUnitsCutSecondStage(3,1) - 0.008 Scenario2_NumUnitsCutSecondStage(3,2) - 0.008 Scenario2_NumUnitsCutSecondStage(4,1) - 0.008 Scenario2_NumUnitsCutSecondStage(4,2) - 0.008 Scenario2_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario2_NumUnitsCutSecondStage(5,1) - 0.008 Scenario2_NumUnitsCutSecondStage(5,2) - 0.008 Scenario2_NumUnitsCutSecondStage(5,3) - 0.008 Scenario2_NumUnitsCutSecondStage(5,4) - 0.008 Scenario2_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario2_NumUnitsCutSecondStage(6,2) - 0.008 Scenario2_NumUnitsCutSecondStage(6,3) - 0.008 Scenario2_NumUnitsCutSecondStage(6,4) - 0.008 Scenario2_NumUnitsCutSecondStage(6,5) - 0.008 Scenario2_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario2_NumUnitsCutSecondStage(7,2) - 0.008 Scenario2_NumUnitsCutSecondStage(7,3) - 0.008 Scenario2_NumUnitsCutSecondStage(7,4) - 0.008 Scenario2_NumUnitsCutSecondStage(7,5) - 0.008 Scenario2_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario2_NumUnitsCutSecondStage(8,1) - 0.008 Scenario2_NumUnitsCutSecondStage(8,2) - 0.008 Scenario2_NumUnitsCutSecondStage(8,3) - 0.008 Scenario2_NumUnitsCutSecondStage(8,4) - 0.008 Scenario2_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario2_NumUnitsCutSecondStage(8,6) - 0.008 Scenario2_NumUnitsCutSecondStage(8,7) - 0.008 Scenario2_NumUnitsCutSecondStage(9,1) - 0.008 Scenario2_NumUnitsCutSecondStage(9,2) - 0.008 Scenario2_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario2_NumUnitsCutSecondStage(9,4) - 0.008 Scenario2_NumUnitsCutSecondStage(9,5) - 0.008 Scenario2_NumUnitsCutSecondStage(9,6) - 0.008 Scenario2_NumUnitsCutSecondStage(9,7) - 0.008 Scenario2_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario2_ProduceSizeSecondStage(10) - 453.0 Scenario2_ProduceSizeSecondStage(1) - 453.0 Scenario2_ProduceSizeSecondStage(2) - 453.0 Scenario2_ProduceSizeSecondStage(3) - 453.0 Scenario2_ProduceSizeSecondStage(4) 
- 453.0 Scenario2_ProduceSizeSecondStage(5) - 453.0 Scenario2_ProduceSizeSecondStage(6) - 453.0 Scenario2_ProduceSizeSecondStage(7) - 453.0 Scenario2_ProduceSizeSecondStage(8) - 453.0 Scenario2_ProduceSizeSecondStage(9) 
+ 1.0 Scenario2_SecondStageCost = 0.0

Scenario2_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario2_NumProducedSecondStage(10) + 1.0 Scenario2_NumProducedSecondStage(1) + 1.0 Scenario2_NumProducedSecondStage(2) + 1.0 Scenario2_NumProducedSecondStage(3) + 1.0 Scenario2_NumProducedSecondStage(4) 
+ 1.0 Scenario2_NumProducedSecondStage(5) + 1.0 Scenario2_NumProducedSecondStage(6) + 1.0 Scenario2_NumProducedSecondStage(7) + 1.0 Scenario2_NumProducedSecondStage(8) + 1.0 Scenario2_NumProducedSecondStage(9) 
<= 200000.0

Scenario2_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario2_NumProducedSecondStage(1) - 200000.0 Scenario2_ProduceSizeSecondStage(1) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario2_NumProducedSecondStage(2) - 200000.0 Scenario2_ProduceSizeSecondStage(2) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario2_NumProducedSecondStage(3) - 200000.0 Scenario2_ProduceSizeSecondStage(3) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario2_NumProducedSecondStage(4) - 200000.0 Scenario2_ProduceSizeSecondStage(4) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario2_NumProducedSecondStage(5) - 200000.0 Scenario2_ProduceSizeSecondStage(5) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario2_NumProducedSecondStage(6) - 200000.0 Scenario2_ProduceSizeSecondStage(6) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario2_NumProducedSecondStage(7) - 200000.0 Scenario2_ProduceSizeSecondStage(7) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario2_NumProducedSecondStage(8) - 200000.0 Scenario2_ProduceSizeSecondStage(8) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario2_NumProducedSecondStage(9) - 200000.0 Scenario2_ProduceSizeSecondStage(9) <= 0.0

Scenario2_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario2_NumProducedSecondStage(10) - 200000.0 Scenario2_ProduceSizeSecondStage(10) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario2_NumProducedFirstStage(1) - 200000.0 Scenario2_ProduceSizeFirstStage(1) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario2_NumProducedFirstStage(2) - 200000.0 Scenario2_ProduceSizeFirstStage(2) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario2_NumProducedFirstStage(3) - 200000.0 Scenario2_ProduceSizeFirstStage(3) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario2_NumProducedFirstStage(4) - 200000.0 Scenario2_ProduceSizeFirstStage(4) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario2_NumProducedFirstStage(5) - 200000.0 Scenario2_ProduceSizeFirstStage(5) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario2_NumProducedFirstStage(6) - 200000.0 Scenario2_ProduceSizeFirstStage(6) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario2_NumProducedFirstStage(7) - 200000.0 Scenario2_ProduceSizeFirstStage(7) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario2_NumProducedFirstStage(8) - 200000.0 Scenario2_ProduceSizeFirstStage(8) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario2_NumProducedFirstStage(9) - 200000.0 Scenario2_ProduceSizeFirstStage(9) <= 0.0

Scenario2_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario2_NumProducedFirstStage(10) - 200000.0 Scenario2_ProduceSizeFirstStage(10) <= 0.0

Scenario2_c_e_ComputeFirstStageCost_:  1.0 Scenario2_FirstStageCost - 0.8416 Scenario2_NumProducedFirstStage(10) - 0.748 Scenario2_NumProducedFirstStage(1) - 0.7584 Scenario2_NumProducedFirstStage(2) - 0.7688 Scenario2_NumProducedFirstStage(3) 
- 0.7792 Scenario2_NumProducedFirstStage(4) - 0.7896 Scenario2_NumProducedFirstStage(5) - 0.8 Scenario2_NumProducedFirstStage(6) - 0.8104 Scenario2_NumProducedFirstStage(7) - 0.8208 Scenario2_NumProducedFirstStage(8) 
- 0.8312 Scenario2_NumProducedFirstStage(9) - 0.008 Scenario2_NumUnitsCutFirstStage(10,1) - 0.008 Scenario2_NumUnitsCutFirstStage(10,2) - 0.008 Scenario2_NumUnitsCutFirstStage(10,3) - 0.008 Scenario2_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario2_NumUnitsCutFirstStage(10,5) - 0.008 Scenario2_NumUnitsCutFirstStage(10,6) - 0.008 Scenario2_NumUnitsCutFirstStage(10,7) - 0.008 Scenario2_NumUnitsCutFirstStage(10,8) - 0.008 Scenario2_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario2_NumUnitsCutFirstStage(2,1) - 0.008 Scenario2_NumUnitsCutFirstStage(3,1) - 0.008 Scenario2_NumUnitsCutFirstStage(3,2) - 0.008 Scenario2_NumUnitsCutFirstStage(4,1) - 0.008 Scenario2_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario2_NumUnitsCutFirstStage(4,3) - 0.008 Scenario2_NumUnitsCutFirstStage(5,1) - 0.008 Scenario2_NumUnitsCutFirstStage(5,2) - 0.008 Scenario2_NumUnitsCutFirstStage(5,3) - 0.008 Scenario2_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario2_NumUnitsCutFirstStage(6,1) - 0.008 Scenario2_NumUnitsCutFirstStage(6,2) - 0.008 Scenario2_NumUnitsCutFirstStage(6,3) - 0.008 Scenario2_NumUnitsCutFirstStage(6,4) - 0.008 Scenario2_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario2_NumUnitsCutFirstStage(7,1) - 0.008 Scenario2_NumUnitsCutFirstStage(7,2) - 0.008 Scenario2_NumUnitsCutFirstStage(7,3) - 0.008 Scenario2_NumUnitsCutFirstStage(7,4) - 0.008 Scenario2_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario2_NumUnitsCutFirstStage(7,6) - 0.008 Scenario2_NumUnitsCutFirstStage(8,1) - 0.008 Scenario2_NumUnitsCutFirstStage(8,2) - 0.008 Scenario2_NumUnitsCutFirstStage(8,3) - 0.008 Scenario2_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario2_NumUnitsCutFirstStage(8,5) - 0.008 Scenario2_NumUnitsCutFirstStage(8,6) - 0.008 Scenario2_NumUnitsCutFirstStage(8,7) - 0.008 Scenario2_NumUnitsCutFirstStage(9,1) - 0.008 Scenario2_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario2_NumUnitsCutFirstStage(9,3) - 0.008 Scenario2_NumUnitsCutFirstStage(9,4) - 0.008 Scenario2_NumUnitsCutFirstStage(9,5) - 0.008 Scenario2_NumUnitsCutFirstStage(9,6) - 0.008 Scenario2_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario2_NumUnitsCutFirstStage(9,8) - 453.0 Scenario2_ProduceSizeFirstStage(10) - 453.0 Scenario2_ProduceSizeFirstStage(1) - 453.0 Scenario2_ProduceSizeFirstStage(2) - 453.0 Scenario2_ProduceSizeFirstStage(3) 
- 453.0 Scenario2_ProduceSizeFirstStage(4) - 453.0 Scenario2_ProduceSizeFirstStage(5) - 453.0 Scenario2_ProduceSizeFirstStage(6) - 453.0 Scenario2_ProduceSizeFirstStage(7) - 453.0 Scenario2_ProduceSizeFirstStage(8) 
- 453.0 Scenario2_ProduceSizeFirstStage(9) = 0.0

Scenario2_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario2_NumProducedFirstStage(1) + 1.0 Scenario2_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario2_NumProducedFirstStage(2) + 1.0 Scenario2_NumUnitsCutFirstStage(2,1) + 1.0 Scenario2_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario2_NumProducedFirstStage(3) + 1.0 Scenario2_NumUnitsCutFirstStage(3,1) + 1.0 Scenario2_NumUnitsCutFirstStage(3,2) + 1.0 Scenario2_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario2_NumProducedFirstStage(4) + 1.0 Scenario2_NumUnitsCutFirstStage(4,1) + 1.0 Scenario2_NumUnitsCutFirstStage(4,2) + 1.0 Scenario2_NumUnitsCutFirstStage(4,3) + 1.0 Scenario2_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario2_NumProducedFirstStage(5) + 1.0 Scenario2_NumUnitsCutFirstStage(5,1) + 1.0 Scenario2_NumUnitsCutFirstStage(5,2) + 1.0 Scenario2_NumUnitsCutFirstStage(5,3) + 1.0 Scenario2_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario2_NumProducedFirstStage(6) + 1.0 Scenario2_NumUnitsCutFirstStage(6,1) + 1.0 Scenario2_NumUnitsCutFirstStage(6,2) + 1.0 Scenario2_NumUnitsCutFirstStage(6,3) + 1.0 Scenario2_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(6,5) + 1.0 Scenario2_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario2_NumProducedFirstStage(7) + 1.0 Scenario2_NumUnitsCutFirstStage(7,1) + 1.0 Scenario2_NumUnitsCutFirstStage(7,2) + 1.0 Scenario2_NumUnitsCutFirstStage(7,3) + 1.0 Scenario2_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(7,5) + 1.0 Scenario2_NumUnitsCutFirstStage(7,6) + 1.0 Scenario2_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario2_NumProducedFirstStage(8) + 1.0 Scenario2_NumUnitsCutFirstStage(8,1) + 1.0 Scenario2_NumUnitsCutFirstStage(8,2) + 1.0 Scenario2_NumUnitsCutFirstStage(8,3) + 1.0 Scenario2_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(8,5) + 1.0 Scenario2_NumUnitsCutFirstStage(8,6) + 1.0 Scenario2_NumUnitsCutFirstStage(8,7) + 1.0 Scenario2_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario2_NumProducedFirstStage(9) + 1.0 Scenario2_NumUnitsCutFirstStage(9,1) + 1.0 Scenario2_NumUnitsCutFirstStage(9,2) + 1.0 Scenario2_NumUnitsCutFirstStage(9,3) + 1.0 Scenario2_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(9,5) + 1.0 Scenario2_NumUnitsCutFirstStage(9,6) + 1.0 Scenario2_NumUnitsCutFirstStage(9,7) + 1.0 Scenario2_NumUnitsCutFirstStage(9,8) + 1.0 Scenario2_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario2_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario2_NumProducedFirstStage(10) + 1.0 Scenario2_NumUnitsCutFirstStage(10,10) + 1.0 Scenario2_NumUnitsCutFirstStage(10,1) + 1.0 Scenario2_NumUnitsCutFirstStage(10,2) + 1.0 Scenario2_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(10,4) + 1.0 Scenario2_NumUnitsCutFirstStage(10,5) + 1.0 Scenario2_NumUnitsCutFirstStage(10,6) + 1.0 Scenario2_NumUnitsCutFirstStage(10,7) + 1.0 Scenario2_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario2_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario2_NumProducedFirstStage(10) + 1.0 Scenario2_NumProducedFirstStage(1) + 1.0 Scenario2_NumProducedFirstStage(2) + 1.0 Scenario2_NumProducedFirstStage(3) + 1.0 Scenario2_NumProducedFirstStage(4) 
+ 1.0 Scenario2_NumProducedFirstStage(5) + 1.0 Scenario2_NumProducedFirstStage(6) + 1.0 Scenario2_NumProducedFirstStage(7) + 1.0 Scenario2_NumProducedFirstStage(8) + 1.0 Scenario2_NumProducedFirstStage(9) 
<= 200000.0

Scenario2_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario2_NumUnitsCutFirstStage(1,1) + 1.0 Scenario2_NumUnitsCutFirstStage(10,1) + 1.0 Scenario2_NumUnitsCutFirstStage(2,1) + 1.0 Scenario2_NumUnitsCutFirstStage(3,1) + 1.0 Scenario2_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(5,1) + 1.0 Scenario2_NumUnitsCutFirstStage(6,1) + 1.0 Scenario2_NumUnitsCutFirstStage(7,1) + 1.0 Scenario2_NumUnitsCutFirstStage(8,1) + 1.0 Scenario2_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario2_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,2) + 1.0 Scenario2_NumUnitsCutFirstStage(2,2) + 1.0 Scenario2_NumUnitsCutFirstStage(3,2) + 1.0 Scenario2_NumUnitsCutFirstStage(4,2) + 1.0 Scenario2_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(6,2) + 1.0 Scenario2_NumUnitsCutFirstStage(7,2) + 1.0 Scenario2_NumUnitsCutFirstStage(8,2) + 1.0 Scenario2_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario2_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,3) + 1.0 Scenario2_NumUnitsCutFirstStage(3,3) + 1.0 Scenario2_NumUnitsCutFirstStage(4,3) + 1.0 Scenario2_NumUnitsCutFirstStage(5,3) + 1.0 Scenario2_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(7,3) + 1.0 Scenario2_NumUnitsCutFirstStage(8,3) + 1.0 Scenario2_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario2_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,4) + 1.0 Scenario2_NumUnitsCutFirstStage(4,4) + 1.0 Scenario2_NumUnitsCutFirstStage(5,4) + 1.0 Scenario2_NumUnitsCutFirstStage(6,4) + 1.0 Scenario2_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(8,4) + 1.0 Scenario2_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario2_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,5) + 1.0 Scenario2_NumUnitsCutFirstStage(5,5) + 1.0 Scenario2_NumUnitsCutFirstStage(6,5) + 1.0 Scenario2_NumUnitsCutFirstStage(7,5) + 1.0 Scenario2_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario2_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,6) + 1.0 Scenario2_NumUnitsCutFirstStage(6,6) + 1.0 Scenario2_NumUnitsCutFirstStage(7,6) + 1.0 Scenario2_NumUnitsCutFirstStage(8,6) + 1.0 Scenario2_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario2_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,7) + 1.0 Scenario2_NumUnitsCutFirstStage(7,7) + 1.0 Scenario2_NumUnitsCutFirstStage(8,7) + 1.0 Scenario2_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario2_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,8) + 1.0 Scenario2_NumUnitsCutFirstStage(8,8) + 1.0 Scenario2_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario2_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,9) + 1.0 Scenario2_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario2_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario2_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario2_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario2_NumProducedFirstStage(1) - 1.0 Scenario2_NumProducedSecondStage(1) + 1.0 Scenario2_NumUnitsCutFirstStage(1,1) + 1.0 Scenario2_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario2_NumProducedFirstStage(2) - 1.0 Scenario2_NumProducedSecondStage(2) + 1.0 Scenario2_NumUnitsCutFirstStage(2,1) + 1.0 Scenario2_NumUnitsCutFirstStage(2,2) + 1.0 Scenario2_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario2_NumProducedFirstStage(3) - 1.0 Scenario2_NumProducedSecondStage(3) + 1.0 Scenario2_NumUnitsCutFirstStage(3,1) + 1.0 Scenario2_NumUnitsCutFirstStage(3,2) + 1.0 Scenario2_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(3,1) + 1.0 Scenario2_NumUnitsCutSecondStage(3,2) + 1.0 Scenario2_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario2_NumProducedFirstStage(4) - 1.0 Scenario2_NumProducedSecondStage(4) + 1.0 Scenario2_NumUnitsCutFirstStage(4,1) + 1.0 Scenario2_NumUnitsCutFirstStage(4,2) + 1.0 Scenario2_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(4,4) + 1.0 Scenario2_NumUnitsCutSecondStage(4,1) + 1.0 Scenario2_NumUnitsCutSecondStage(4,2) + 1.0 Scenario2_NumUnitsCutSecondStage(4,3) + 1.0 Scenario2_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario2_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario2_NumProducedFirstStage(5) - 1.0 Scenario2_NumProducedSecondStage(5) + 1.0 Scenario2_NumUnitsCutFirstStage(5,1) + 1.0 Scenario2_NumUnitsCutFirstStage(5,2) + 1.0 Scenario2_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(5,4) + 1.0 Scenario2_NumUnitsCutFirstStage(5,5) + 1.0 Scenario2_NumUnitsCutSecondStage(5,1) + 1.0 Scenario2_NumUnitsCutSecondStage(5,2) + 1.0 Scenario2_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(5,4) + 1.0 Scenario2_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario2_NumProducedFirstStage(6) - 1.0 Scenario2_NumProducedSecondStage(6) + 1.0 Scenario2_NumUnitsCutFirstStage(6,1) + 1.0 Scenario2_NumUnitsCutFirstStage(6,2) + 1.0 Scenario2_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(6,4) + 1.0 Scenario2_NumUnitsCutFirstStage(6,5) + 1.0 Scenario2_NumUnitsCutFirstStage(6,6) + 1.0 Scenario2_NumUnitsCutSecondStage(6,1) + 1.0 Scenario2_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(6,3) + 1.0 Scenario2_NumUnitsCutSecondStage(6,4) + 1.0 Scenario2_NumUnitsCutSecondStage(6,5) + 1.0 Scenario2_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario2_NumProducedFirstStage(7) - 1.0 Scenario2_NumProducedSecondStage(7) + 1.0 Scenario2_NumUnitsCutFirstStage(7,1) + 1.0 Scenario2_NumUnitsCutFirstStage(7,2) + 1.0 Scenario2_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(7,4) + 1.0 Scenario2_NumUnitsCutFirstStage(7,5) + 1.0 Scenario2_NumUnitsCutFirstStage(7,6) + 1.0 Scenario2_NumUnitsCutFirstStage(7,7) + 1.0 Scenario2_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(7,2) + 1.0 Scenario2_NumUnitsCutSecondStage(7,3) + 1.0 Scenario2_NumUnitsCutSecondStage(7,4) + 1.0 Scenario2_NumUnitsCutSecondStage(7,5) + 1.0 Scenario2_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario2_NumProducedFirstStage(8) - 1.0 Scenario2_NumProducedSecondStage(8) + 1.0 Scenario2_NumUnitsCutFirstStage(8,1) + 1.0 Scenario2_NumUnitsCutFirstStage(8,2) + 1.0 Scenario2_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(8,4) + 1.0 Scenario2_NumUnitsCutFirstStage(8,5) + 1.0 Scenario2_NumUnitsCutFirstStage(8,6) + 1.0 Scenario2_NumUnitsCutFirstStage(8,7) + 1.0 Scenario2_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(8,1) + 1.0 Scenario2_NumUnitsCutSecondStage(8,2) + 1.0 Scenario2_NumUnitsCutSecondStage(8,3) + 1.0 Scenario2_NumUnitsCutSecondStage(8,4) + 1.0 Scenario2_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(8,6) + 1.0 Scenario2_NumUnitsCutSecondStage(8,7) + 1.0 Scenario2_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario2_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario2_NumProducedFirstStage(9) - 1.0 Scenario2_NumProducedSecondStage(9) + 1.0 Scenario2_NumUnitsCutFirstStage(9,1) + 1.0 Scenario2_NumUnitsCutFirstStage(9,2) + 1.0 Scenario2_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(9,4) + 1.0 Scenario2_NumUnitsCutFirstStage(9,5) + 1.0 Scenario2_NumUnitsCutFirstStage(9,6) + 1.0 Scenario2_NumUnitsCutFirstStage(9,7) + 1.0 Scenario2_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(9,9) + 1.0 Scenario2_NumUnitsCutSecondStage(9,1) + 1.0 Scenario2_NumUnitsCutSecondStage(9,2) + 1.0 Scenario2_NumUnitsCutSecondStage(9,3) + 1.0 Scenario2_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(9,5) + 1.0 Scenario2_NumUnitsCutSecondStage(9,6) + 1.0 Scenario2_NumUnitsCutSecondStage(9,7) + 1.0 Scenario2_NumUnitsCutSecondStage(9,8) + 1.0 Scenario2_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario2_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario2_NumProducedFirstStage(10) - 1.0 Scenario2_NumProducedSecondStage(10) + 1.0 Scenario2_NumUnitsCutFirstStage(10,10) + 1.0 Scenario2_NumUnitsCutFirstStage(10,1) + 1.0 Scenario2_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(10,3) + 1.0 Scenario2_NumUnitsCutFirstStage(10,4) + 1.0 Scenario2_NumUnitsCutFirstStage(10,5) + 1.0 Scenario2_NumUnitsCutFirstStage(10,6) + 1.0 Scenario2_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario2_NumUnitsCutFirstStage(10,8) + 1.0 Scenario2_NumUnitsCutFirstStage(10,9) + 1.0 Scenario2_NumUnitsCutSecondStage(10,10) + 1.0 Scenario2_NumUnitsCutSecondStage(10,1) + 1.0 Scenario2_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(10,3) + 1.0 Scenario2_NumUnitsCutSecondStage(10,4) + 1.0 Scenario2_NumUnitsCutSecondStage(10,5) + 1.0 Scenario2_NumUnitsCutSecondStage(10,6) + 1.0 Scenario2_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(10,8) + 1.0 Scenario2_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario2

\ Begin constraint block for scenario Scenario1
Scenario1_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario1_NumUnitsCutSecondStage(1,1) + 1.0 Scenario1_NumUnitsCutSecondStage(10,1) + 1.0 Scenario1_NumUnitsCutSecondStage(2,1) + 1.0 Scenario1_NumUnitsCutSecondStage(3,1) + 1.0 Scenario1_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(5,1) + 1.0 Scenario1_NumUnitsCutSecondStage(6,1) + 1.0 Scenario1_NumUnitsCutSecondStage(7,1) + 1.0 Scenario1_NumUnitsCutSecondStage(8,1) + 1.0 Scenario1_NumUnitsCutSecondStage(9,1) 
>= 1250.0

Scenario1_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,2) + 1.0 Scenario1_NumUnitsCutSecondStage(2,2) + 1.0 Scenario1_NumUnitsCutSecondStage(3,2) + 1.0 Scenario1_NumUnitsCutSecondStage(4,2) + 1.0 Scenario1_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(6,2) + 1.0 Scenario1_NumUnitsCutSecondStage(7,2) + 1.0 Scenario1_NumUnitsCutSecondStage(8,2) + 1.0 Scenario1_NumUnitsCutSecondStage(9,2) >= 3750.0

Scenario1_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,3) + 1.0 Scenario1_NumUnitsCutSecondStage(3,3) + 1.0 Scenario1_NumUnitsCutSecondStage(4,3) + 1.0 Scenario1_NumUnitsCutSecondStage(5,3) + 1.0 Scenario1_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(7,3) + 1.0 Scenario1_NumUnitsCutSecondStage(8,3) + 1.0 Scenario1_NumUnitsCutSecondStage(9,3) >= 6250.0

Scenario1_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,4) + 1.0 Scenario1_NumUnitsCutSecondStage(4,4) + 1.0 Scenario1_NumUnitsCutSecondStage(5,4) + 1.0 Scenario1_NumUnitsCutSecondStage(6,4) + 1.0 Scenario1_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(8,4) + 1.0 Scenario1_NumUnitsCutSecondStage(9,4) >= 5000.0

Scenario1_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,5) + 1.0 Scenario1_NumUnitsCutSecondStage(5,5) + 1.0 Scenario1_NumUnitsCutSecondStage(6,5) + 1.0 Scenario1_NumUnitsCutSecondStage(7,5) + 1.0 Scenario1_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(9,5) >= 17500.0

Scenario1_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,6) + 1.0 Scenario1_NumUnitsCutSecondStage(6,6) + 1.0 Scenario1_NumUnitsCutSecondStage(7,6) + 1.0 Scenario1_NumUnitsCutSecondStage(8,6) + 1.0 Scenario1_NumUnitsCutSecondStage(9,6) 
>= 12500.0

Scenario1_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,7) + 1.0 Scenario1_NumUnitsCutSecondStage(7,7) + 1.0 Scenario1_NumUnitsCutSecondStage(8,7) + 1.0 Scenario1_NumUnitsCutSecondStage(9,7) >= 7500.0

Scenario1_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,8) + 1.0 Scenario1_NumUnitsCutSecondStage(8,8) + 1.0 Scenario1_NumUnitsCutSecondStage(9,8) >= 6250.0

Scenario1_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,9) + 1.0 Scenario1_NumUnitsCutSecondStage(9,9) >= 6250.0

Scenario1_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,10) >= 2500.0

Scenario1_c_e_ComputeSecondStageCost_:  -0.8416 Scenario1_NumProducedSecondStage(10) - 0.748 Scenario1_NumProducedSecondStage(1) - 0.7584 Scenario1_NumProducedSecondStage(2) - 0.7688 Scenario1_NumProducedSecondStage(3) - 0.7792 Scenario1_NumProducedSecondStage(4) 
- 0.7896 Scenario1_NumProducedSecondStage(5) - 0.8 Scenario1_NumProducedSecondStage(6) - 0.8104 Scenario1_NumProducedSecondStage(7) - 0.8208 Scenario1_NumProducedSecondStage(8) - 0.8312 Scenario1_NumProducedSecondStage(9) 
- 0.008 Scenario1_NumUnitsCutSecondStage(10,1) - 0.008 Scenario1_NumUnitsCutSecondStage(10,2) - 0.008 Scenario1_NumUnitsCutSecondStage(10,3) - 0.008 Scenario1_NumUnitsCutSecondStage(10,4) - 0.008 Scenario1_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario1_NumUnitsCutSecondStage(10,6) - 0.008 Scenario1_NumUnitsCutSecondStage(10,7) - 0.008 Scenario1_NumUnitsCutSecondStage(10,8) - 0.008 Scenario1_NumUnitsCutSecondStage(10,9) - 0.008 Scenario1_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario1_NumUnitsCutSecondStage(3,1) - 0.008 Scenario1_NumUnitsCutSecondStage(3,2) - 0.008 Scenario1_NumUnitsCutSecondStage(4,1) - 0.008 Scenario1_NumUnitsCutSecondStage(4,2) - 0.008 Scenario1_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario1_NumUnitsCutSecondStage(5,1) - 0.008 Scenario1_NumUnitsCutSecondStage(5,2) - 0.008 Scenario1_NumUnitsCutSecondStage(5,3) - 0.008 Scenario1_NumUnitsCutSecondStage(5,4) - 0.008 Scenario1_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario1_NumUnitsCutSecondStage(6,2) - 0.008 Scenario1_NumUnitsCutSecondStage(6,3) - 0.008 Scenario1_NumUnitsCutSecondStage(6,4) - 0.008 Scenario1_NumUnitsCutSecondStage(6,5) - 0.008 Scenario1_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario1_NumUnitsCutSecondStage(7,2) - 0.008 Scenario1_NumUnitsCutSecondStage(7,3) - 0.008 Scenario1_NumUnitsCutSecondStage(7,4) - 0.008 Scenario1_NumUnitsCutSecondStage(7,5) - 0.008 Scenario1_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario1_NumUnitsCutSecondStage(8,1) - 0.008 Scenario1_NumUnitsCutSecondStage(8,2) - 0.008 Scenario1_NumUnitsCutSecondStage(8,3) - 0.008 Scenario1_NumUnitsCutSecondStage(8,4) - 0.008 Scenario1_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario1_NumUnitsCutSecondStage(8,6) - 0.008 Scenario1_NumUnitsCutSecondStage(8,7) - 0.008 Scenario1_NumUnitsCutSecondStage(9,1) - 0.008 Scenario1_NumUnitsCutSecondStage(9,2) - 0.008 Scenario1_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario1_NumUnitsCutSecondStage(9,4) - 0.008 Scenario1_NumUnitsCutSecondStage(9,5) - 0.008 Scenario1_NumUnitsCutSecondStage(9,6) - 0.008 Scenario1_NumUnitsCutSecondStage(9,7) - 0.008 Scenario1_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario1_ProduceSizeSecondStage(10) - 453.0 Scenario1_ProduceSizeSecondStage(1) - 453.0 Scenario1_ProduceSizeSecondStage(2) - 453.0 Scenario1_ProduceSizeSecondStage(3) - 453.0 Scenario1_ProduceSizeSecondStage(4) 
- 453.0 Scenario1_ProduceSizeSecondStage(5) - 453.0 Scenario1_ProduceSizeSecondStage(6) - 453.0 Scenario1_ProduceSizeSecondStage(7) - 453.0 Scenario1_ProduceSizeSecondStage(8) - 453.0 Scenario1_ProduceSizeSecondStage(9) 
+ 1.0 Scenario1_SecondStageCost = 0.0

Scenario1_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario1_NumProducedSecondStage(10) + 1.0 Scenario1_NumProducedSecondStage(1) + 1.0 Scenario1_NumProducedSecondStage(2) + 1.0 Scenario1_NumProducedSecondStage(3) + 1.0 Scenario1_NumProducedSecondStage(4) 
+ 1.0 Scenario1_NumProducedSecondStage(5) + 1.0 Scenario1_NumProducedSecondStage(6) + 1.0 Scenario1_NumProducedSecondStage(7) + 1.0 Scenario1_NumProducedSecondStage(8) + 1.0 Scenario1_NumProducedSecondStage(9) 
<= 200000.0

Scenario1_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario1_NumProducedSecondStage(1) - 200000.0 Scenario1_ProduceSizeSecondStage(1) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario1_NumProducedSecondStage(2) - 200000.0 Scenario1_ProduceSizeSecondStage(2) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario1_NumProducedSecondStage(3) - 200000.0 Scenario1_ProduceSizeSecondStage(3) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario1_NumProducedSecondStage(4) - 200000.0 Scenario1_ProduceSizeSecondStage(4) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario1_NumProducedSecondStage(5) - 200000.0 Scenario1_ProduceSizeSecondStage(5) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario1_NumProducedSecondStage(6) - 200000.0 Scenario1_ProduceSizeSecondStage(6) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario1_NumProducedSecondStage(7) - 200000.0 Scenario1_ProduceSizeSecondStage(7) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario1_NumProducedSecondStage(8) - 200000.0 Scenario1_ProduceSizeSecondStage(8) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario1_NumProducedSecondStage(9) - 200000.0 Scenario1_ProduceSizeSecondStage(9) <= 0.0

Scenario1_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario1_NumProducedSecondStage(10) - 200000.0 Scenario1_ProduceSizeSecondStage(10) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario1_NumProducedFirstStage(1) - 200000.0 Scenario1_ProduceSizeFirstStage(1) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario1_NumProducedFirstStage(2) - 200000.0 Scenario1_ProduceSizeFirstStage(2) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario1_NumProducedFirstStage(3) - 200000.0 Scenario1_ProduceSizeFirstStage(3) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario1_NumProducedFirstStage(4) - 200000.0 Scenario1_ProduceSizeFirstStage(4) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario1_NumProducedFirstStage(5) - 200000.0 Scenario1_ProduceSizeFirstStage(5) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario1_NumProducedFirstStage(6) - 200000.0 Scenario1_ProduceSizeFirstStage(6) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario1_NumProducedFirstStage(7) - 200000.0 Scenario1_ProduceSizeFirstStage(7) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario1_NumProducedFirstStage(8) - 200000.0 Scenario1_ProduceSizeFirstStage(8) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario1_NumProducedFirstStage(9) - 200000.0 Scenario1_ProduceSizeFirstStage(9) <= 0.0

Scenario1_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario1_NumProducedFirstStage(10) - 200000.0 Scenario1_ProduceSizeFirstStage(10) <= 0.0

Scenario1_c_e_ComputeFirstStageCost_:  1.0 Scenario1_FirstStageCost - 0.8416 Scenario1_NumProducedFirstStage(10) - 0.748 Scenario1_NumProducedFirstStage(1) - 0.7584 Scenario1_NumProducedFirstStage(2) - 0.7688 Scenario1_NumProducedFirstStage(3) 
- 0.7792 Scenario1_NumProducedFirstStage(4) - 0.7896 Scenario1_NumProducedFirstStage(5) - 0.8 Scenario1_NumProducedFirstStage(6) - 0.8104 Scenario1_NumProducedFirstStage(7) - 0.8208 Scenario1_NumProducedFirstStage(8) 
- 0.8312 Scenario1_NumProducedFirstStage(9) - 0.008 Scenario1_NumUnitsCutFirstStage(10,1) - 0.008 Scenario1_NumUnitsCutFirstStage(10,2) - 0.008 Scenario1_NumUnitsCutFirstStage(10,3) - 0.008 Scenario1_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario1_NumUnitsCutFirstStage(10,5) - 0.008 Scenario1_NumUnitsCutFirstStage(10,6) - 0.008 Scenario1_NumUnitsCutFirstStage(10,7) - 0.008 Scenario1_NumUnitsCutFirstStage(10,8) - 0.008 Scenario1_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario1_NumUnitsCutFirstStage(2,1) - 0.008 Scenario1_NumUnitsCutFirstStage(3,1) - 0.008 Scenario1_NumUnitsCutFirstStage(3,2) - 0.008 Scenario1_NumUnitsCutFirstStage(4,1) - 0.008 Scenario1_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario1_NumUnitsCutFirstStage(4,3) - 0.008 Scenario1_NumUnitsCutFirstStage(5,1) - 0.008 Scenario1_NumUnitsCutFirstStage(5,2) - 0.008 Scenario1_NumUnitsCutFirstStage(5,3) - 0.008 Scenario1_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario1_NumUnitsCutFirstStage(6,1) - 0.008 Scenario1_NumUnitsCutFirstStage(6,2) - 0.008 Scenario1_NumUnitsCutFirstStage(6,3) - 0.008 Scenario1_NumUnitsCutFirstStage(6,4) - 0.008 Scenario1_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario1_NumUnitsCutFirstStage(7,1) - 0.008 Scenario1_NumUnitsCutFirstStage(7,2) - 0.008 Scenario1_NumUnitsCutFirstStage(7,3) - 0.008 Scenario1_NumUnitsCutFirstStage(7,4) - 0.008 Scenario1_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario1_NumUnitsCutFirstStage(7,6) - 0.008 Scenario1_NumUnitsCutFirstStage(8,1) - 0.008 Scenario1_NumUnitsCutFirstStage(8,2) - 0.008 Scenario1_NumUnitsCutFirstStage(8,3) - 0.008 Scenario1_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario1_NumUnitsCutFirstStage(8,5) - 0.008 Scenario1_NumUnitsCutFirstStage(8,6) - 0.008 Scenario1_NumUnitsCutFirstStage(8,7) - 0.008 Scenario1_NumUnitsCutFirstStage(9,1) - 0.008 Scenario1_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario1_NumUnitsCutFirstStage(9,3) - 0.008 Scenario1_NumUnitsCutFirstStage(9,4) - 0.008 Scenario1_NumUnitsCutFirstStage(9,5) - 0.008 Scenario1_NumUnitsCutFirstStage(9,6) - 0.008 Scenario1_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario1_NumUnitsCutFirstStage(9,8) - 453.0 Scenario1_ProduceSizeFirstStage(10) - 453.0 Scenario1_ProduceSizeFirstStage(1) - 453.0 Scenario1_ProduceSizeFirstStage(2) - 453.0 Scenario1_ProduceSizeFirstStage(3) 
- 453.0 Scenario1_ProduceSizeFirstStage(4) - 453.0 Scenario1_ProduceSizeFirstStage(5) - 453.0 Scenario1_ProduceSizeFirstStage(6) - 453.0 Scenario1_ProduceSizeFirstStage(7) - 453.0 Scenario1_ProduceSizeFirstStage(8) 
- 453.0 Scenario1_ProduceSizeFirstStage(9) = 0.0

Scenario1_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario1_NumProducedFirstStage(1) + 1.0 Scenario1_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario1_NumProducedFirstStage(2) + 1.0 Scenario1_NumUnitsCutFirstStage(2,1) + 1.0 Scenario1_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario1_NumProducedFirstStage(3) + 1.0 Scenario1_NumUnitsCutFirstStage(3,1) + 1.0 Scenario1_NumUnitsCutFirstStage(3,2) + 1.0 Scenario1_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario1_NumProducedFirstStage(4) + 1.0 Scenario1_NumUnitsCutFirstStage(4,1) + 1.0 Scenario1_NumUnitsCutFirstStage(4,2) + 1.0 Scenario1_NumUnitsCutFirstStage(4,3) + 1.0 Scenario1_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario1_NumProducedFirstStage(5) + 1.0 Scenario1_NumUnitsCutFirstStage(5,1) + 1.0 Scenario1_NumUnitsCutFirstStage(5,2) + 1.0 Scenario1_NumUnitsCutFirstStage(5,3) + 1.0 Scenario1_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario1_NumProducedFirstStage(6) + 1.0 Scenario1_NumUnitsCutFirstStage(6,1) + 1.0 Scenario1_NumUnitsCutFirstStage(6,2) + 1.0 Scenario1_NumUnitsCutFirstStage(6,3) + 1.0 Scenario1_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(6,5) + 1.0 Scenario1_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario1_NumProducedFirstStage(7) + 1.0 Scenario1_NumUnitsCutFirstStage(7,1) + 1.0 Scenario1_NumUnitsCutFirstStage(7,2) + 1.0 Scenario1_NumUnitsCutFirstStage(7,3) + 1.0 Scenario1_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(7,5) + 1.0 Scenario1_NumUnitsCutFirstStage(7,6) + 1.0 Scenario1_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario1_NumProducedFirstStage(8) + 1.0 Scenario1_NumUnitsCutFirstStage(8,1) + 1.0 Scenario1_NumUnitsCutFirstStage(8,2) + 1.0 Scenario1_NumUnitsCutFirstStage(8,3) + 1.0 Scenario1_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(8,5) + 1.0 Scenario1_NumUnitsCutFirstStage(8,6) + 1.0 Scenario1_NumUnitsCutFirstStage(8,7) + 1.0 Scenario1_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario1_NumProducedFirstStage(9) + 1.0 Scenario1_NumUnitsCutFirstStage(9,1) + 1.0 Scenario1_NumUnitsCutFirstStage(9,2) + 1.0 Scenario1_NumUnitsCutFirstStage(9,3) + 1.0 Scenario1_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(9,5) + 1.0 Scenario1_NumUnitsCutFirstStage(9,6) + 1.0 Scenario1_NumUnitsCutFirstStage(9,7) + 1.0 Scenario1_NumUnitsCutFirstStage(9,8) + 1.0 Scenario1_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario1_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario1_NumProducedFirstStage(10) + 1.0 Scenario1_NumUnitsCutFirstStage(10,10) + 1.0 Scenario1_NumUnitsCutFirstStage(10,1) + 1.0 Scenario1_NumUnitsCutFirstStage(10,2) + 1.0 Scenario1_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(10,4) + 1.0 Scenario1_NumUnitsCutFirstStage(10,5) + 1.0 Scenario1_NumUnitsCutFirstStage(10,6) + 1.0 Scenario1_NumUnitsCutFirstStage(10,7) + 1.0 Scenario1_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario1_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario1_NumProducedFirstStage(10) + 1.0 Scenario1_NumProducedFirstStage(1) + 1.0 Scenario1_NumProducedFirstStage(2) + 1.0 Scenario1_NumProducedFirstStage(3) + 1.0 Scenario1_NumProducedFirstStage(4) 
+ 1.0 Scenario1_NumProducedFirstStage(5) + 1.0 Scenario1_NumProducedFirstStage(6) + 1.0 Scenario1_NumProducedFirstStage(7) + 1.0 Scenario1_NumProducedFirstStage(8) + 1.0 Scenario1_NumProducedFirstStage(9) 
<= 200000.0

Scenario1_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario1_NumUnitsCutFirstStage(1,1) + 1.0 Scenario1_NumUnitsCutFirstStage(10,1) + 1.0 Scenario1_NumUnitsCutFirstStage(2,1) + 1.0 Scenario1_NumUnitsCutFirstStage(3,1) + 1.0 Scenario1_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(5,1) + 1.0 Scenario1_NumUnitsCutFirstStage(6,1) + 1.0 Scenario1_NumUnitsCutFirstStage(7,1) + 1.0 Scenario1_NumUnitsCutFirstStage(8,1) + 1.0 Scenario1_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario1_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,2) + 1.0 Scenario1_NumUnitsCutFirstStage(2,2) + 1.0 Scenario1_NumUnitsCutFirstStage(3,2) + 1.0 Scenario1_NumUnitsCutFirstStage(4,2) + 1.0 Scenario1_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(6,2) + 1.0 Scenario1_NumUnitsCutFirstStage(7,2) + 1.0 Scenario1_NumUnitsCutFirstStage(8,2) + 1.0 Scenario1_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario1_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,3) + 1.0 Scenario1_NumUnitsCutFirstStage(3,3) + 1.0 Scenario1_NumUnitsCutFirstStage(4,3) + 1.0 Scenario1_NumUnitsCutFirstStage(5,3) + 1.0 Scenario1_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(7,3) + 1.0 Scenario1_NumUnitsCutFirstStage(8,3) + 1.0 Scenario1_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario1_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,4) + 1.0 Scenario1_NumUnitsCutFirstStage(4,4) + 1.0 Scenario1_NumUnitsCutFirstStage(5,4) + 1.0 Scenario1_NumUnitsCutFirstStage(6,4) + 1.0 Scenario1_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(8,4) + 1.0 Scenario1_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario1_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,5) + 1.0 Scenario1_NumUnitsCutFirstStage(5,5) + 1.0 Scenario1_NumUnitsCutFirstStage(6,5) + 1.0 Scenario1_NumUnitsCutFirstStage(7,5) + 1.0 Scenario1_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario1_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,6) + 1.0 Scenario1_NumUnitsCutFirstStage(6,6) + 1.0 Scenario1_NumUnitsCutFirstStage(7,6) + 1.0 Scenario1_NumUnitsCutFirstStage(8,6) + 1.0 Scenario1_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario1_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,7) + 1.0 Scenario1_NumUnitsCutFirstStage(7,7) + 1.0 Scenario1_NumUnitsCutFirstStage(8,7) + 1.0 Scenario1_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario1_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,8) + 1.0 Scenario1_NumUnitsCutFirstStage(8,8) + 1.0 Scenario1_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario1_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,9) + 1.0 Scenario1_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario1_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario1_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario1_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario1_NumProducedFirstStage(1) - 1.0 Scenario1_NumProducedSecondStage(1) + 1.0 Scenario1_NumUnitsCutFirstStage(1,1) + 1.0 Scenario1_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario1_NumProducedFirstStage(2) - 1.0 Scenario1_NumProducedSecondStage(2) + 1.0 Scenario1_NumUnitsCutFirstStage(2,1) + 1.0 Scenario1_NumUnitsCutFirstStage(2,2) + 1.0 Scenario1_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario1_NumProducedFirstStage(3) - 1.0 Scenario1_NumProducedSecondStage(3) + 1.0 Scenario1_NumUnitsCutFirstStage(3,1) + 1.0 Scenario1_NumUnitsCutFirstStage(3,2) + 1.0 Scenario1_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(3,1) + 1.0 Scenario1_NumUnitsCutSecondStage(3,2) + 1.0 Scenario1_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario1_NumProducedFirstStage(4) - 1.0 Scenario1_NumProducedSecondStage(4) + 1.0 Scenario1_NumUnitsCutFirstStage(4,1) + 1.0 Scenario1_NumUnitsCutFirstStage(4,2) + 1.0 Scenario1_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(4,4) + 1.0 Scenario1_NumUnitsCutSecondStage(4,1) + 1.0 Scenario1_NumUnitsCutSecondStage(4,2) + 1.0 Scenario1_NumUnitsCutSecondStage(4,3) + 1.0 Scenario1_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario1_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario1_NumProducedFirstStage(5) - 1.0 Scenario1_NumProducedSecondStage(5) + 1.0 Scenario1_NumUnitsCutFirstStage(5,1) + 1.0 Scenario1_NumUnitsCutFirstStage(5,2) + 1.0 Scenario1_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(5,4) + 1.0 Scenario1_NumUnitsCutFirstStage(5,5) + 1.0 Scenario1_NumUnitsCutSecondStage(5,1) + 1.0 Scenario1_NumUnitsCutSecondStage(5,2) + 1.0 Scenario1_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(5,4) + 1.0 Scenario1_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario1_NumProducedFirstStage(6) - 1.0 Scenario1_NumProducedSecondStage(6) + 1.0 Scenario1_NumUnitsCutFirstStage(6,1) + 1.0 Scenario1_NumUnitsCutFirstStage(6,2) + 1.0 Scenario1_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(6,4) + 1.0 Scenario1_NumUnitsCutFirstStage(6,5) + 1.0 Scenario1_NumUnitsCutFirstStage(6,6) + 1.0 Scenario1_NumUnitsCutSecondStage(6,1) + 1.0 Scenario1_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(6,3) + 1.0 Scenario1_NumUnitsCutSecondStage(6,4) + 1.0 Scenario1_NumUnitsCutSecondStage(6,5) + 1.0 Scenario1_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario1_NumProducedFirstStage(7) - 1.0 Scenario1_NumProducedSecondStage(7) + 1.0 Scenario1_NumUnitsCutFirstStage(7,1) + 1.0 Scenario1_NumUnitsCutFirstStage(7,2) + 1.0 Scenario1_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(7,4) + 1.0 Scenario1_NumUnitsCutFirstStage(7,5) + 1.0 Scenario1_NumUnitsCutFirstStage(7,6) + 1.0 Scenario1_NumUnitsCutFirstStage(7,7) + 1.0 Scenario1_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(7,2) + 1.0 Scenario1_NumUnitsCutSecondStage(7,3) + 1.0 Scenario1_NumUnitsCutSecondStage(7,4) + 1.0 Scenario1_NumUnitsCutSecondStage(7,5) + 1.0 Scenario1_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario1_NumProducedFirstStage(8) - 1.0 Scenario1_NumProducedSecondStage(8) + 1.0 Scenario1_NumUnitsCutFirstStage(8,1) + 1.0 Scenario1_NumUnitsCutFirstStage(8,2) + 1.0 Scenario1_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(8,4) + 1.0 Scenario1_NumUnitsCutFirstStage(8,5) + 1.0 Scenario1_NumUnitsCutFirstStage(8,6) + 1.0 Scenario1_NumUnitsCutFirstStage(8,7) + 1.0 Scenario1_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(8,1) + 1.0 Scenario1_NumUnitsCutSecondStage(8,2) + 1.0 Scenario1_NumUnitsCutSecondStage(8,3) + 1.0 Scenario1_NumUnitsCutSecondStage(8,4) + 1.0 Scenario1_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(8,6) + 1.0 Scenario1_NumUnitsCutSecondStage(8,7) + 1.0 Scenario1_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario1_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario1_NumProducedFirstStage(9) - 1.0 Scenario1_NumProducedSecondStage(9) + 1.0 Scenario1_NumUnitsCutFirstStage(9,1) + 1.0 Scenario1_NumUnitsCutFirstStage(9,2) + 1.0 Scenario1_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(9,4) + 1.0 Scenario1_NumUnitsCutFirstStage(9,5) + 1.0 Scenario1_NumUnitsCutFirstStage(9,6) + 1.0 Scenario1_NumUnitsCutFirstStage(9,7) + 1.0 Scenario1_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(9,9) + 1.0 Scenario1_NumUnitsCutSecondStage(9,1) + 1.0 Scenario1_NumUnitsCutSecondStage(9,2) + 1.0 Scenario1_NumUnitsCutSecondStage(9,3) + 1.0 Scenario1_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(9,5) + 1.0 Scenario1_NumUnitsCutSecondStage(9,6) + 1.0 Scenario1_NumUnitsCutSecondStage(9,7) + 1.0 Scenario1_NumUnitsCutSecondStage(9,8) + 1.0 Scenario1_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario1_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario1_NumProducedFirstStage(10) - 1.0 Scenario1_NumProducedSecondStage(10) + 1.0 Scenario1_NumUnitsCutFirstStage(10,10) + 1.0 Scenario1_NumUnitsCutFirstStage(10,1) + 1.0 Scenario1_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(10,3) + 1.0 Scenario1_NumUnitsCutFirstStage(10,4) + 1.0 Scenario1_NumUnitsCutFirstStage(10,5) + 1.0 Scenario1_NumUnitsCutFirstStage(10,6) + 1.0 Scenario1_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario1_NumUnitsCutFirstStage(10,8) + 1.0 Scenario1_NumUnitsCutFirstStage(10,9) + 1.0 Scenario1_NumUnitsCutSecondStage(10,10) + 1.0 Scenario1_NumUnitsCutSecondStage(10,1) + 1.0 Scenario1_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(10,3) + 1.0 Scenario1_NumUnitsCutSecondStage(10,4) + 1.0 Scenario1_NumUnitsCutSecondStage(10,5) + 1.0 Scenario1_NumUnitsCutSecondStage(10,6) + 1.0 Scenario1_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(10,8) + 1.0 Scenario1_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario1

\ Begin constraint block for scenario Scenario7
Scenario7_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario7_NumUnitsCutSecondStage(1,1) + 1.0 Scenario7_NumUnitsCutSecondStage(10,1) + 1.0 Scenario7_NumUnitsCutSecondStage(2,1) + 1.0 Scenario7_NumUnitsCutSecondStage(3,1) + 1.0 Scenario7_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(5,1) + 1.0 Scenario7_NumUnitsCutSecondStage(6,1) + 1.0 Scenario7_NumUnitsCutSecondStage(7,1) + 1.0 Scenario7_NumUnitsCutSecondStage(8,1) + 1.0 Scenario7_NumUnitsCutSecondStage(9,1) 
>= 3000.0

Scenario7_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,2) + 1.0 Scenario7_NumUnitsCutSecondStage(2,2) + 1.0 Scenario7_NumUnitsCutSecondStage(3,2) + 1.0 Scenario7_NumUnitsCutSecondStage(4,2) + 1.0 Scenario7_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(6,2) + 1.0 Scenario7_NumUnitsCutSecondStage(7,2) + 1.0 Scenario7_NumUnitsCutSecondStage(8,2) + 1.0 Scenario7_NumUnitsCutSecondStage(9,2) >= 9000.0

Scenario7_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,3) + 1.0 Scenario7_NumUnitsCutSecondStage(3,3) + 1.0 Scenario7_NumUnitsCutSecondStage(4,3) + 1.0 Scenario7_NumUnitsCutSecondStage(5,3) + 1.0 Scenario7_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(7,3) + 1.0 Scenario7_NumUnitsCutSecondStage(8,3) + 1.0 Scenario7_NumUnitsCutSecondStage(9,3) >= 15000.0

Scenario7_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,4) + 1.0 Scenario7_NumUnitsCutSecondStage(4,4) + 1.0 Scenario7_NumUnitsCutSecondStage(5,4) + 1.0 Scenario7_NumUnitsCutSecondStage(6,4) + 1.0 Scenario7_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(8,4) + 1.0 Scenario7_NumUnitsCutSecondStage(9,4) >= 12000.0

Scenario7_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,5) + 1.0 Scenario7_NumUnitsCutSecondStage(5,5) + 1.0 Scenario7_NumUnitsCutSecondStage(6,5) + 1.0 Scenario7_NumUnitsCutSecondStage(7,5) + 1.0 Scenario7_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(9,5) >= 42000.0

Scenario7_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,6) + 1.0 Scenario7_NumUnitsCutSecondStage(6,6) + 1.0 Scenario7_NumUnitsCutSecondStage(7,6) + 1.0 Scenario7_NumUnitsCutSecondStage(8,6) + 1.0 Scenario7_NumUnitsCutSecondStage(9,6) 
>= 30000.0

Scenario7_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,7) + 1.0 Scenario7_NumUnitsCutSecondStage(7,7) + 1.0 Scenario7_NumUnitsCutSecondStage(8,7) + 1.0 Scenario7_NumUnitsCutSecondStage(9,7) >= 18000.0

Scenario7_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,8) + 1.0 Scenario7_NumUnitsCutSecondStage(8,8) + 1.0 Scenario7_NumUnitsCutSecondStage(9,8) >= 15000.0

Scenario7_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,9) + 1.0 Scenario7_NumUnitsCutSecondStage(9,9) >= 15000.0

Scenario7_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario7_NumUnitsCutSecondStage(10,10) >= 6000.0

Scenario7_c_e_ComputeSecondStageCost_:  -0.8416 Scenario7_NumProducedSecondStage(10) - 0.748 Scenario7_NumProducedSecondStage(1) - 0.7584 Scenario7_NumProducedSecondStage(2) - 0.7688 Scenario7_NumProducedSecondStage(3) - 0.7792 Scenario7_NumProducedSecondStage(4) 
- 0.7896 Scenario7_NumProducedSecondStage(5) - 0.8 Scenario7_NumProducedSecondStage(6) - 0.8104 Scenario7_NumProducedSecondStage(7) - 0.8208 Scenario7_NumProducedSecondStage(8) - 0.8312 Scenario7_NumProducedSecondStage(9) 
- 0.008 Scenario7_NumUnitsCutSecondStage(10,1) - 0.008 Scenario7_NumUnitsCutSecondStage(10,2) - 0.008 Scenario7_NumUnitsCutSecondStage(10,3) - 0.008 Scenario7_NumUnitsCutSecondStage(10,4) - 0.008 Scenario7_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario7_NumUnitsCutSecondStage(10,6) - 0.008 Scenario7_NumUnitsCutSecondStage(10,7) - 0.008 Scenario7_NumUnitsCutSecondStage(10,8) - 0.008 Scenario7_NumUnitsCutSecondStage(10,9) - 0.008 Scenario7_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario7_NumUnitsCutSecondStage(3,1) - 0.008 Scenario7_NumUnitsCutSecondStage(3,2) - 0.008 Scenario7_NumUnitsCutSecondStage(4,1) - 0.008 Scenario7_NumUnitsCutSecondStage(4,2) - 0.008 Scenario7_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario7_NumUnitsCutSecondStage(5,1) - 0.008 Scenario7_NumUnitsCutSecondStage(5,2) - 0.008 Scenario7_NumUnitsCutSecondStage(5,3) - 0.008 Scenario7_NumUnitsCutSecondStage(5,4) - 0.008 Scenario7_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario7_NumUnitsCutSecondStage(6,2) - 0.008 Scenario7_NumUnitsCutSecondStage(6,3) - 0.008 Scenario7_NumUnitsCutSecondStage(6,4) - 0.008 Scenario7_NumUnitsCutSecondStage(6,5) - 0.008 Scenario7_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario7_NumUnitsCutSecondStage(7,2) - 0.008 Scenario7_NumUnitsCutSecondStage(7,3) - 0.008 Scenario7_NumUnitsCutSecondStage(7,4) - 0.008 Scenario7_NumUnitsCutSecondStage(7,5) - 0.008 Scenario7_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario7_NumUnitsCutSecondStage(8,1) - 0.008 Scenario7_NumUnitsCutSecondStage(8,2) - 0.008 Scenario7_NumUnitsCutSecondStage(8,3) - 0.008 Scenario7_NumUnitsCutSecondStage(8,4) - 0.008 Scenario7_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario7_NumUnitsCutSecondStage(8,6) - 0.008 Scenario7_NumUnitsCutSecondStage(8,7) - 0.008 Scenario7_NumUnitsCutSecondStage(9,1) - 0.008 Scenario7_NumUnitsCutSecondStage(9,2) - 0.008 Scenario7_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario7_NumUnitsCutSecondStage(9,4) - 0.008 Scenario7_NumUnitsCutSecondStage(9,5) - 0.008 Scenario7_NumUnitsCutSecondStage(9,6) - 0.008 Scenario7_NumUnitsCutSecondStage(9,7) - 0.008 Scenario7_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario7_ProduceSizeSecondStage(10) - 453.0 Scenario7_ProduceSizeSecondStage(1) - 453.0 Scenario7_ProduceSizeSecondStage(2) - 453.0 Scenario7_ProduceSizeSecondStage(3) - 453.0 Scenario7_ProduceSizeSecondStage(4) 
- 453.0 Scenario7_ProduceSizeSecondStage(5) - 453.0 Scenario7_ProduceSizeSecondStage(6) - 453.0 Scenario7_ProduceSizeSecondStage(7) - 453.0 Scenario7_ProduceSizeSecondStage(8) - 453.0 Scenario7_ProduceSizeSecondStage(9) 
+ 1.0 Scenario7_SecondStageCost = 0.0

Scenario7_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario7_NumProducedSecondStage(10) + 1.0 Scenario7_NumProducedSecondStage(1) + 1.0 Scenario7_NumProducedSecondStage(2) + 1.0 Scenario7_NumProducedSecondStage(3) + 1.0 Scenario7_NumProducedSecondStage(4) 
+ 1.0 Scenario7_NumProducedSecondStage(5) + 1.0 Scenario7_NumProducedSecondStage(6) + 1.0 Scenario7_NumProducedSecondStage(7) + 1.0 Scenario7_NumProducedSecondStage(8) + 1.0 Scenario7_NumProducedSecondStage(9) 
<= 200000.0

Scenario7_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario7_NumProducedSecondStage(1) - 200000.0 Scenario7_ProduceSizeSecondStage(1) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario7_NumProducedSecondStage(2) - 200000.0 Scenario7_ProduceSizeSecondStage(2) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario7_NumProducedSecondStage(3) - 200000.0 Scenario7_ProduceSizeSecondStage(3) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario7_NumProducedSecondStage(4) - 200000.0 Scenario7_ProduceSizeSecondStage(4) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario7_NumProducedSecondStage(5) - 200000.0 Scenario7_ProduceSizeSecondStage(5) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario7_NumProducedSecondStage(6) - 200000.0 Scenario7_ProduceSizeSecondStage(6) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario7_NumProducedSecondStage(7) - 200000.0 Scenario7_ProduceSizeSecondStage(7) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario7_NumProducedSecondStage(8) - 200000.0 Scenario7_ProduceSizeSecondStage(8) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario7_NumProducedSecondStage(9) - 200000.0 Scenario7_ProduceSizeSecondStage(9) <= 0.0

Scenario7_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario7_NumProducedSecondStage(10) - 200000.0 Scenario7_ProduceSizeSecondStage(10) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario7_NumProducedFirstStage(1) - 200000.0 Scenario7_ProduceSizeFirstStage(1) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario7_NumProducedFirstStage(2) - 200000.0 Scenario7_ProduceSizeFirstStage(2) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario7_NumProducedFirstStage(3) - 200000.0 Scenario7_ProduceSizeFirstStage(3) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario7_NumProducedFirstStage(4) - 200000.0 Scenario7_ProduceSizeFirstStage(4) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario7_NumProducedFirstStage(5) - 200000.0 Scenario7_ProduceSizeFirstStage(5) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario7_NumProducedFirstStage(6) - 200000.0 Scenario7_ProduceSizeFirstStage(6) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario7_NumProducedFirstStage(7) - 200000.0 Scenario7_ProduceSizeFirstStage(7) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario7_NumProducedFirstStage(8) - 200000.0 Scenario7_ProduceSizeFirstStage(8) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario7_NumProducedFirstStage(9) - 200000.0 Scenario7_ProduceSizeFirstStage(9) <= 0.0

Scenario7_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario7_NumProducedFirstStage(10) - 200000.0 Scenario7_ProduceSizeFirstStage(10) <= 0.0

Scenario7_c_e_ComputeFirstStageCost_:  1.0 Scenario7_FirstStageCost - 0.8416 Scenario7_NumProducedFirstStage(10) - 0.748 Scenario7_NumProducedFirstStage(1) - 0.7584 Scenario7_NumProducedFirstStage(2) - 0.7688 Scenario7_NumProducedFirstStage(3) 
- 0.7792 Scenario7_NumProducedFirstStage(4) - 0.7896 Scenario7_NumProducedFirstStage(5) - 0.8 Scenario7_NumProducedFirstStage(6) - 0.8104 Scenario7_NumProducedFirstStage(7) - 0.8208 Scenario7_NumProducedFirstStage(8) 
- 0.8312 Scenario7_NumProducedFirstStage(9) - 0.008 Scenario7_NumUnitsCutFirstStage(10,1) - 0.008 Scenario7_NumUnitsCutFirstStage(10,2) - 0.008 Scenario7_NumUnitsCutFirstStage(10,3) - 0.008 Scenario7_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario7_NumUnitsCutFirstStage(10,5) - 0.008 Scenario7_NumUnitsCutFirstStage(10,6) - 0.008 Scenario7_NumUnitsCutFirstStage(10,7) - 0.008 Scenario7_NumUnitsCutFirstStage(10,8) - 0.008 Scenario7_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario7_NumUnitsCutFirstStage(2,1) - 0.008 Scenario7_NumUnitsCutFirstStage(3,1) - 0.008 Scenario7_NumUnitsCutFirstStage(3,2) - 0.008 Scenario7_NumUnitsCutFirstStage(4,1) - 0.008 Scenario7_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario7_NumUnitsCutFirstStage(4,3) - 0.008 Scenario7_NumUnitsCutFirstStage(5,1) - 0.008 Scenario7_NumUnitsCutFirstStage(5,2) - 0.008 Scenario7_NumUnitsCutFirstStage(5,3) - 0.008 Scenario7_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario7_NumUnitsCutFirstStage(6,1) - 0.008 Scenario7_NumUnitsCutFirstStage(6,2) - 0.008 Scenario7_NumUnitsCutFirstStage(6,3) - 0.008 Scenario7_NumUnitsCutFirstStage(6,4) - 0.008 Scenario7_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario7_NumUnitsCutFirstStage(7,1) - 0.008 Scenario7_NumUnitsCutFirstStage(7,2) - 0.008 Scenario7_NumUnitsCutFirstStage(7,3) - 0.008 Scenario7_NumUnitsCutFirstStage(7,4) - 0.008 Scenario7_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario7_NumUnitsCutFirstStage(7,6) - 0.008 Scenario7_NumUnitsCutFirstStage(8,1) - 0.008 Scenario7_NumUnitsCutFirstStage(8,2) - 0.008 Scenario7_NumUnitsCutFirstStage(8,3) - 0.008 Scenario7_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario7_NumUnitsCutFirstStage(8,5) - 0.008 Scenario7_NumUnitsCutFirstStage(8,6) - 0.008 Scenario7_NumUnitsCutFirstStage(8,7) - 0.008 Scenario7_NumUnitsCutFirstStage(9,1) - 0.008 Scenario7_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario7_NumUnitsCutFirstStage(9,3) - 0.008 Scenario7_NumUnitsCutFirstStage(9,4) - 0.008 Scenario7_NumUnitsCutFirstStage(9,5) - 0.008 Scenario7_NumUnitsCutFirstStage(9,6) - 0.008 Scenario7_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario7_NumUnitsCutFirstStage(9,8) - 453.0 Scenario7_ProduceSizeFirstStage(10) - 453.0 Scenario7_ProduceSizeFirstStage(1) - 453.0 Scenario7_ProduceSizeFirstStage(2) - 453.0 Scenario7_ProduceSizeFirstStage(3) 
- 453.0 Scenario7_ProduceSizeFirstStage(4) - 453.0 Scenario7_ProduceSizeFirstStage(5) - 453.0 Scenario7_ProduceSizeFirstStage(6) - 453.0 Scenario7_ProduceSizeFirstStage(7) - 453.0 Scenario7_ProduceSizeFirstStage(8) 
- 453.0 Scenario7_ProduceSizeFirstStage(9) = 0.0

Scenario7_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario7_NumProducedFirstStage(1) + 1.0 Scenario7_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario7_NumProducedFirstStage(2) + 1.0 Scenario7_NumUnitsCutFirstStage(2,1) + 1.0 Scenario7_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario7_NumProducedFirstStage(3) + 1.0 Scenario7_NumUnitsCutFirstStage(3,1) + 1.0 Scenario7_NumUnitsCutFirstStage(3,2) + 1.0 Scenario7_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario7_NumProducedFirstStage(4) + 1.0 Scenario7_NumUnitsCutFirstStage(4,1) + 1.0 Scenario7_NumUnitsCutFirstStage(4,2) + 1.0 Scenario7_NumUnitsCutFirstStage(4,3) + 1.0 Scenario7_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario7_NumProducedFirstStage(5) + 1.0 Scenario7_NumUnitsCutFirstStage(5,1) + 1.0 Scenario7_NumUnitsCutFirstStage(5,2) + 1.0 Scenario7_NumUnitsCutFirstStage(5,3) + 1.0 Scenario7_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario7_NumProducedFirstStage(6) + 1.0 Scenario7_NumUnitsCutFirstStage(6,1) + 1.0 Scenario7_NumUnitsCutFirstStage(6,2) + 1.0 Scenario7_NumUnitsCutFirstStage(6,3) + 1.0 Scenario7_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(6,5) + 1.0 Scenario7_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario7_NumProducedFirstStage(7) + 1.0 Scenario7_NumUnitsCutFirstStage(7,1) + 1.0 Scenario7_NumUnitsCutFirstStage(7,2) + 1.0 Scenario7_NumUnitsCutFirstStage(7,3) + 1.0 Scenario7_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(7,5) + 1.0 Scenario7_NumUnitsCutFirstStage(7,6) + 1.0 Scenario7_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario7_NumProducedFirstStage(8) + 1.0 Scenario7_NumUnitsCutFirstStage(8,1) + 1.0 Scenario7_NumUnitsCutFirstStage(8,2) + 1.0 Scenario7_NumUnitsCutFirstStage(8,3) + 1.0 Scenario7_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(8,5) + 1.0 Scenario7_NumUnitsCutFirstStage(8,6) + 1.0 Scenario7_NumUnitsCutFirstStage(8,7) + 1.0 Scenario7_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario7_NumProducedFirstStage(9) + 1.0 Scenario7_NumUnitsCutFirstStage(9,1) + 1.0 Scenario7_NumUnitsCutFirstStage(9,2) + 1.0 Scenario7_NumUnitsCutFirstStage(9,3) + 1.0 Scenario7_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(9,5) + 1.0 Scenario7_NumUnitsCutFirstStage(9,6) + 1.0 Scenario7_NumUnitsCutFirstStage(9,7) + 1.0 Scenario7_NumUnitsCutFirstStage(9,8) + 1.0 Scenario7_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario7_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario7_NumProducedFirstStage(10) + 1.0 Scenario7_NumUnitsCutFirstStage(10,10) + 1.0 Scenario7_NumUnitsCutFirstStage(10,1) + 1.0 Scenario7_NumUnitsCutFirstStage(10,2) + 1.0 Scenario7_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(10,4) + 1.0 Scenario7_NumUnitsCutFirstStage(10,5) + 1.0 Scenario7_NumUnitsCutFirstStage(10,6) + 1.0 Scenario7_NumUnitsCutFirstStage(10,7) + 1.0 Scenario7_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario7_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario7_NumProducedFirstStage(10) + 1.0 Scenario7_NumProducedFirstStage(1) + 1.0 Scenario7_NumProducedFirstStage(2) + 1.0 Scenario7_NumProducedFirstStage(3) + 1.0 Scenario7_NumProducedFirstStage(4) 
+ 1.0 Scenario7_NumProducedFirstStage(5) + 1.0 Scenario7_NumProducedFirstStage(6) + 1.0 Scenario7_NumProducedFirstStage(7) + 1.0 Scenario7_NumProducedFirstStage(8) + 1.0 Scenario7_NumProducedFirstStage(9) 
<= 200000.0

Scenario7_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario7_NumUnitsCutFirstStage(1,1) + 1.0 Scenario7_NumUnitsCutFirstStage(10,1) + 1.0 Scenario7_NumUnitsCutFirstStage(2,1) + 1.0 Scenario7_NumUnitsCutFirstStage(3,1) + 1.0 Scenario7_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(5,1) + 1.0 Scenario7_NumUnitsCutFirstStage(6,1) + 1.0 Scenario7_NumUnitsCutFirstStage(7,1) + 1.0 Scenario7_NumUnitsCutFirstStage(8,1) + 1.0 Scenario7_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario7_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,2) + 1.0 Scenario7_NumUnitsCutFirstStage(2,2) + 1.0 Scenario7_NumUnitsCutFirstStage(3,2) + 1.0 Scenario7_NumUnitsCutFirstStage(4,2) + 1.0 Scenario7_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(6,2) + 1.0 Scenario7_NumUnitsCutFirstStage(7,2) + 1.0 Scenario7_NumUnitsCutFirstStage(8,2) + 1.0 Scenario7_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario7_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,3) + 1.0 Scenario7_NumUnitsCutFirstStage(3,3) + 1.0 Scenario7_NumUnitsCutFirstStage(4,3) + 1.0 Scenario7_NumUnitsCutFirstStage(5,3) + 1.0 Scenario7_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(7,3) + 1.0 Scenario7_NumUnitsCutFirstStage(8,3) + 1.0 Scenario7_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario7_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,4) + 1.0 Scenario7_NumUnitsCutFirstStage(4,4) + 1.0 Scenario7_NumUnitsCutFirstStage(5,4) + 1.0 Scenario7_NumUnitsCutFirstStage(6,4) + 1.0 Scenario7_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(8,4) + 1.0 Scenario7_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario7_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,5) + 1.0 Scenario7_NumUnitsCutFirstStage(5,5) + 1.0 Scenario7_NumUnitsCutFirstStage(6,5) + 1.0 Scenario7_NumUnitsCutFirstStage(7,5) + 1.0 Scenario7_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario7_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,6) + 1.0 Scenario7_NumUnitsCutFirstStage(6,6) + 1.0 Scenario7_NumUnitsCutFirstStage(7,6) + 1.0 Scenario7_NumUnitsCutFirstStage(8,6) + 1.0 Scenario7_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario7_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,7) + 1.0 Scenario7_NumUnitsCutFirstStage(7,7) + 1.0 Scenario7_NumUnitsCutFirstStage(8,7) + 1.0 Scenario7_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario7_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,8) + 1.0 Scenario7_NumUnitsCutFirstStage(8,8) + 1.0 Scenario7_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario7_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,9) + 1.0 Scenario7_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario7_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario7_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario7_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario7_NumProducedFirstStage(1) - 1.0 Scenario7_NumProducedSecondStage(1) + 1.0 Scenario7_NumUnitsCutFirstStage(1,1) + 1.0 Scenario7_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario7_NumProducedFirstStage(2) - 1.0 Scenario7_NumProducedSecondStage(2) + 1.0 Scenario7_NumUnitsCutFirstStage(2,1) + 1.0 Scenario7_NumUnitsCutFirstStage(2,2) + 1.0 Scenario7_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario7_NumProducedFirstStage(3) - 1.0 Scenario7_NumProducedSecondStage(3) + 1.0 Scenario7_NumUnitsCutFirstStage(3,1) + 1.0 Scenario7_NumUnitsCutFirstStage(3,2) + 1.0 Scenario7_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(3,1) + 1.0 Scenario7_NumUnitsCutSecondStage(3,2) + 1.0 Scenario7_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario7_NumProducedFirstStage(4) - 1.0 Scenario7_NumProducedSecondStage(4) + 1.0 Scenario7_NumUnitsCutFirstStage(4,1) + 1.0 Scenario7_NumUnitsCutFirstStage(4,2) + 1.0 Scenario7_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(4,4) + 1.0 Scenario7_NumUnitsCutSecondStage(4,1) + 1.0 Scenario7_NumUnitsCutSecondStage(4,2) + 1.0 Scenario7_NumUnitsCutSecondStage(4,3) + 1.0 Scenario7_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario7_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario7_NumProducedFirstStage(5) - 1.0 Scenario7_NumProducedSecondStage(5) + 1.0 Scenario7_NumUnitsCutFirstStage(5,1) + 1.0 Scenario7_NumUnitsCutFirstStage(5,2) + 1.0 Scenario7_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(5,4) + 1.0 Scenario7_NumUnitsCutFirstStage(5,5) + 1.0 Scenario7_NumUnitsCutSecondStage(5,1) + 1.0 Scenario7_NumUnitsCutSecondStage(5,2) + 1.0 Scenario7_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(5,4) + 1.0 Scenario7_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario7_NumProducedFirstStage(6) - 1.0 Scenario7_NumProducedSecondStage(6) + 1.0 Scenario7_NumUnitsCutFirstStage(6,1) + 1.0 Scenario7_NumUnitsCutFirstStage(6,2) + 1.0 Scenario7_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(6,4) + 1.0 Scenario7_NumUnitsCutFirstStage(6,5) + 1.0 Scenario7_NumUnitsCutFirstStage(6,6) + 1.0 Scenario7_NumUnitsCutSecondStage(6,1) + 1.0 Scenario7_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(6,3) + 1.0 Scenario7_NumUnitsCutSecondStage(6,4) + 1.0 Scenario7_NumUnitsCutSecondStage(6,5) + 1.0 Scenario7_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario7_NumProducedFirstStage(7) - 1.0 Scenario7_NumProducedSecondStage(7) + 1.0 Scenario7_NumUnitsCutFirstStage(7,1) + 1.0 Scenario7_NumUnitsCutFirstStage(7,2) + 1.0 Scenario7_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(7,4) + 1.0 Scenario7_NumUnitsCutFirstStage(7,5) + 1.0 Scenario7_NumUnitsCutFirstStage(7,6) + 1.0 Scenario7_NumUnitsCutFirstStage(7,7) + 1.0 Scenario7_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(7,2) + 1.0 Scenario7_NumUnitsCutSecondStage(7,3) + 1.0 Scenario7_NumUnitsCutSecondStage(7,4) + 1.0 Scenario7_NumUnitsCutSecondStage(7,5) + 1.0 Scenario7_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario7_NumProducedFirstStage(8) - 1.0 Scenario7_NumProducedSecondStage(8) + 1.0 Scenario7_NumUnitsCutFirstStage(8,1) + 1.0 Scenario7_NumUnitsCutFirstStage(8,2) + 1.0 Scenario7_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(8,4) + 1.0 Scenario7_NumUnitsCutFirstStage(8,5) + 1.0 Scenario7_NumUnitsCutFirstStage(8,6) + 1.0 Scenario7_NumUnitsCutFirstStage(8,7) + 1.0 Scenario7_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(8,1) + 1.0 Scenario7_NumUnitsCutSecondStage(8,2) + 1.0 Scenario7_NumUnitsCutSecondStage(8,3) + 1.0 Scenario7_NumUnitsCutSecondStage(8,4) + 1.0 Scenario7_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(8,6) + 1.0 Scenario7_NumUnitsCutSecondStage(8,7) + 1.0 Scenario7_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario7_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario7_NumProducedFirstStage(9) - 1.0 Scenario7_NumProducedSecondStage(9) + 1.0 Scenario7_NumUnitsCutFirstStage(9,1) + 1.0 Scenario7_NumUnitsCutFirstStage(9,2) + 1.0 Scenario7_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(9,4) + 1.0 Scenario7_NumUnitsCutFirstStage(9,5) + 1.0 Scenario7_NumUnitsCutFirstStage(9,6) + 1.0 Scenario7_NumUnitsCutFirstStage(9,7) + 1.0 Scenario7_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(9,9) + 1.0 Scenario7_NumUnitsCutSecondStage(9,1) + 1.0 Scenario7_NumUnitsCutSecondStage(9,2) + 1.0 Scenario7_NumUnitsCutSecondStage(9,3) + 1.0 Scenario7_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(9,5) + 1.0 Scenario7_NumUnitsCutSecondStage(9,6) + 1.0 Scenario7_NumUnitsCutSecondStage(9,7) + 1.0 Scenario7_NumUnitsCutSecondStage(9,8) + 1.0 Scenario7_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario7_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario7_NumProducedFirstStage(10) - 1.0 Scenario7_NumProducedSecondStage(10) + 1.0 Scenario7_NumUnitsCutFirstStage(10,10) + 1.0 Scenario7_NumUnitsCutFirstStage(10,1) + 1.0 Scenario7_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(10,3) + 1.0 Scenario7_NumUnitsCutFirstStage(10,4) + 1.0 Scenario7_NumUnitsCutFirstStage(10,5) + 1.0 Scenario7_NumUnitsCutFirstStage(10,6) + 1.0 Scenario7_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario7_NumUnitsCutFirstStage(10,8) + 1.0 Scenario7_NumUnitsCutFirstStage(10,9) + 1.0 Scenario7_NumUnitsCutSecondStage(10,10) + 1.0 Scenario7_NumUnitsCutSecondStage(10,1) + 1.0 Scenario7_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(10,3) + 1.0 Scenario7_NumUnitsCutSecondStage(10,4) + 1.0 Scenario7_NumUnitsCutSecondStage(10,5) + 1.0 Scenario7_NumUnitsCutSecondStage(10,6) + 1.0 Scenario7_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario7_NumUnitsCutSecondStage(10,8) + 1.0 Scenario7_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario7

\ Begin constraint block for scenario Scenario6
Scenario6_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario6_NumUnitsCutSecondStage(1,1) + 1.0 Scenario6_NumUnitsCutSecondStage(10,1) + 1.0 Scenario6_NumUnitsCutSecondStage(2,1) + 1.0 Scenario6_NumUnitsCutSecondStage(3,1) + 1.0 Scenario6_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(5,1) + 1.0 Scenario6_NumUnitsCutSecondStage(6,1) + 1.0 Scenario6_NumUnitsCutSecondStage(7,1) + 1.0 Scenario6_NumUnitsCutSecondStage(8,1) + 1.0 Scenario6_NumUnitsCutSecondStage(9,1) 
>= 2750.0

Scenario6_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,2) + 1.0 Scenario6_NumUnitsCutSecondStage(2,2) + 1.0 Scenario6_NumUnitsCutSecondStage(3,2) + 1.0 Scenario6_NumUnitsCutSecondStage(4,2) + 1.0 Scenario6_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(6,2) + 1.0 Scenario6_NumUnitsCutSecondStage(7,2) + 1.0 Scenario6_NumUnitsCutSecondStage(8,2) + 1.0 Scenario6_NumUnitsCutSecondStage(9,2) >= 8250.0

Scenario6_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,3) + 1.0 Scenario6_NumUnitsCutSecondStage(3,3) + 1.0 Scenario6_NumUnitsCutSecondStage(4,3) + 1.0 Scenario6_NumUnitsCutSecondStage(5,3) + 1.0 Scenario6_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(7,3) + 1.0 Scenario6_NumUnitsCutSecondStage(8,3) + 1.0 Scenario6_NumUnitsCutSecondStage(9,3) >= 13750.0

Scenario6_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,4) + 1.0 Scenario6_NumUnitsCutSecondStage(4,4) + 1.0 Scenario6_NumUnitsCutSecondStage(5,4) + 1.0 Scenario6_NumUnitsCutSecondStage(6,4) + 1.0 Scenario6_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(8,4) + 1.0 Scenario6_NumUnitsCutSecondStage(9,4) >= 11000.0

Scenario6_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,5) + 1.0 Scenario6_NumUnitsCutSecondStage(5,5) + 1.0 Scenario6_NumUnitsCutSecondStage(6,5) + 1.0 Scenario6_NumUnitsCutSecondStage(7,5) + 1.0 Scenario6_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(9,5) >= 38500.0

Scenario6_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,6) + 1.0 Scenario6_NumUnitsCutSecondStage(6,6) + 1.0 Scenario6_NumUnitsCutSecondStage(7,6) + 1.0 Scenario6_NumUnitsCutSecondStage(8,6) + 1.0 Scenario6_NumUnitsCutSecondStage(9,6) 
>= 27500.0

Scenario6_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,7) + 1.0 Scenario6_NumUnitsCutSecondStage(7,7) + 1.0 Scenario6_NumUnitsCutSecondStage(8,7) + 1.0 Scenario6_NumUnitsCutSecondStage(9,7) >= 16500.0

Scenario6_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,8) + 1.0 Scenario6_NumUnitsCutSecondStage(8,8) + 1.0 Scenario6_NumUnitsCutSecondStage(9,8) >= 13750.0

Scenario6_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,9) + 1.0 Scenario6_NumUnitsCutSecondStage(9,9) >= 13750.0

Scenario6_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario6_NumUnitsCutSecondStage(10,10) >= 5500.0

Scenario6_c_e_ComputeSecondStageCost_:  -0.8416 Scenario6_NumProducedSecondStage(10) - 0.748 Scenario6_NumProducedSecondStage(1) - 0.7584 Scenario6_NumProducedSecondStage(2) - 0.7688 Scenario6_NumProducedSecondStage(3) - 0.7792 Scenario6_NumProducedSecondStage(4) 
- 0.7896 Scenario6_NumProducedSecondStage(5) - 0.8 Scenario6_NumProducedSecondStage(6) - 0.8104 Scenario6_NumProducedSecondStage(7) - 0.8208 Scenario6_NumProducedSecondStage(8) - 0.8312 Scenario6_NumProducedSecondStage(9) 
- 0.008 Scenario6_NumUnitsCutSecondStage(10,1) - 0.008 Scenario6_NumUnitsCutSecondStage(10,2) - 0.008 Scenario6_NumUnitsCutSecondStage(10,3) - 0.008 Scenario6_NumUnitsCutSecondStage(10,4) - 0.008 Scenario6_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario6_NumUnitsCutSecondStage(10,6) - 0.008 Scenario6_NumUnitsCutSecondStage(10,7) - 0.008 Scenario6_NumUnitsCutSecondStage(10,8) - 0.008 Scenario6_NumUnitsCutSecondStage(10,9) - 0.008 Scenario6_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario6_NumUnitsCutSecondStage(3,1) - 0.008 Scenario6_NumUnitsCutSecondStage(3,2) - 0.008 Scenario6_NumUnitsCutSecondStage(4,1) - 0.008 Scenario6_NumUnitsCutSecondStage(4,2) - 0.008 Scenario6_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario6_NumUnitsCutSecondStage(5,1) - 0.008 Scenario6_NumUnitsCutSecondStage(5,2) - 0.008 Scenario6_NumUnitsCutSecondStage(5,3) - 0.008 Scenario6_NumUnitsCutSecondStage(5,4) - 0.008 Scenario6_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario6_NumUnitsCutSecondStage(6,2) - 0.008 Scenario6_NumUnitsCutSecondStage(6,3) - 0.008 Scenario6_NumUnitsCutSecondStage(6,4) - 0.008 Scenario6_NumUnitsCutSecondStage(6,5) - 0.008 Scenario6_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario6_NumUnitsCutSecondStage(7,2) - 0.008 Scenario6_NumUnitsCutSecondStage(7,3) - 0.008 Scenario6_NumUnitsCutSecondStage(7,4) - 0.008 Scenario6_NumUnitsCutSecondStage(7,5) - 0.008 Scenario6_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario6_NumUnitsCutSecondStage(8,1) - 0.008 Scenario6_NumUnitsCutSecondStage(8,2) - 0.008 Scenario6_NumUnitsCutSecondStage(8,3) - 0.008 Scenario6_NumUnitsCutSecondStage(8,4) - 0.008 Scenario6_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario6_NumUnitsCutSecondStage(8,6) - 0.008 Scenario6_NumUnitsCutSecondStage(8,7) - 0.008 Scenario6_NumUnitsCutSecondStage(9,1) - 0.008 Scenario6_NumUnitsCutSecondStage(9,2) - 0.008 Scenario6_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario6_NumUnitsCutSecondStage(9,4) - 0.008 Scenario6_NumUnitsCutSecondStage(9,5) - 0.008 Scenario6_NumUnitsCutSecondStage(9,6) - 0.008 Scenario6_NumUnitsCutSecondStage(9,7) - 0.008 Scenario6_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario6_ProduceSizeSecondStage(10) - 453.0 Scenario6_ProduceSizeSecondStage(1) - 453.0 Scenario6_ProduceSizeSecondStage(2) - 453.0 Scenario6_ProduceSizeSecondStage(3) - 453.0 Scenario6_ProduceSizeSecondStage(4) 
- 453.0 Scenario6_ProduceSizeSecondStage(5) - 453.0 Scenario6_ProduceSizeSecondStage(6) - 453.0 Scenario6_ProduceSizeSecondStage(7) - 453.0 Scenario6_ProduceSizeSecondStage(8) - 453.0 Scenario6_ProduceSizeSecondStage(9) 
+ 1.0 Scenario6_SecondStageCost = 0.0

Scenario6_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario6_NumProducedSecondStage(10) + 1.0 Scenario6_NumProducedSecondStage(1) + 1.0 Scenario6_NumProducedSecondStage(2) + 1.0 Scenario6_NumProducedSecondStage(3) + 1.0 Scenario6_NumProducedSecondStage(4) 
+ 1.0 Scenario6_NumProducedSecondStage(5) + 1.0 Scenario6_NumProducedSecondStage(6) + 1.0 Scenario6_NumProducedSecondStage(7) + 1.0 Scenario6_NumProducedSecondStage(8) + 1.0 Scenario6_NumProducedSecondStage(9) 
<= 200000.0

Scenario6_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario6_NumProducedSecondStage(1) - 200000.0 Scenario6_ProduceSizeSecondStage(1) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario6_NumProducedSecondStage(2) - 200000.0 Scenario6_ProduceSizeSecondStage(2) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario6_NumProducedSecondStage(3) - 200000.0 Scenario6_ProduceSizeSecondStage(3) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario6_NumProducedSecondStage(4) - 200000.0 Scenario6_ProduceSizeSecondStage(4) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario6_NumProducedSecondStage(5) - 200000.0 Scenario6_ProduceSizeSecondStage(5) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario6_NumProducedSecondStage(6) - 200000.0 Scenario6_ProduceSizeSecondStage(6) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario6_NumProducedSecondStage(7) - 200000.0 Scenario6_ProduceSizeSecondStage(7) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario6_NumProducedSecondStage(8) - 200000.0 Scenario6_ProduceSizeSecondStage(8) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario6_NumProducedSecondStage(9) - 200000.0 Scenario6_ProduceSizeSecondStage(9) <= 0.0

Scenario6_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario6_NumProducedSecondStage(10) - 200000.0 Scenario6_ProduceSizeSecondStage(10) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario6_NumProducedFirstStage(1) - 200000.0 Scenario6_ProduceSizeFirstStage(1) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario6_NumProducedFirstStage(2) - 200000.0 Scenario6_ProduceSizeFirstStage(2) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario6_NumProducedFirstStage(3) - 200000.0 Scenario6_ProduceSizeFirstStage(3) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario6_NumProducedFirstStage(4) - 200000.0 Scenario6_ProduceSizeFirstStage(4) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario6_NumProducedFirstStage(5) - 200000.0 Scenario6_ProduceSizeFirstStage(5) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario6_NumProducedFirstStage(6) - 200000.0 Scenario6_ProduceSizeFirstStage(6) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario6_NumProducedFirstStage(7) - 200000.0 Scenario6_ProduceSizeFirstStage(7) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario6_NumProducedFirstStage(8) - 200000.0 Scenario6_ProduceSizeFirstStage(8) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario6_NumProducedFirstStage(9) - 200000.0 Scenario6_ProduceSizeFirstStage(9) <= 0.0

Scenario6_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario6_NumProducedFirstStage(10) - 200000.0 Scenario6_ProduceSizeFirstStage(10) <= 0.0

Scenario6_c_e_ComputeFirstStageCost_:  1.0 Scenario6_FirstStageCost - 0.8416 Scenario6_NumProducedFirstStage(10) - 0.748 Scenario6_NumProducedFirstStage(1) - 0.7584 Scenario6_NumProducedFirstStage(2) - 0.7688 Scenario6_NumProducedFirstStage(3) 
- 0.7792 Scenario6_NumProducedFirstStage(4) - 0.7896 Scenario6_NumProducedFirstStage(5) - 0.8 Scenario6_NumProducedFirstStage(6) - 0.8104 Scenario6_NumProducedFirstStage(7) - 0.8208 Scenario6_NumProducedFirstStage(8) 
- 0.8312 Scenario6_NumProducedFirstStage(9) - 0.008 Scenario6_NumUnitsCutFirstStage(10,1) - 0.008 Scenario6_NumUnitsCutFirstStage(10,2) - 0.008 Scenario6_NumUnitsCutFirstStage(10,3) - 0.008 Scenario6_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario6_NumUnitsCutFirstStage(10,5) - 0.008 Scenario6_NumUnitsCutFirstStage(10,6) - 0.008 Scenario6_NumUnitsCutFirstStage(10,7) - 0.008 Scenario6_NumUnitsCutFirstStage(10,8) - 0.008 Scenario6_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario6_NumUnitsCutFirstStage(2,1) - 0.008 Scenario6_NumUnitsCutFirstStage(3,1) - 0.008 Scenario6_NumUnitsCutFirstStage(3,2) - 0.008 Scenario6_NumUnitsCutFirstStage(4,1) - 0.008 Scenario6_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario6_NumUnitsCutFirstStage(4,3) - 0.008 Scenario6_NumUnitsCutFirstStage(5,1) - 0.008 Scenario6_NumUnitsCutFirstStage(5,2) - 0.008 Scenario6_NumUnitsCutFirstStage(5,3) - 0.008 Scenario6_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario6_NumUnitsCutFirstStage(6,1) - 0.008 Scenario6_NumUnitsCutFirstStage(6,2) - 0.008 Scenario6_NumUnitsCutFirstStage(6,3) - 0.008 Scenario6_NumUnitsCutFirstStage(6,4) - 0.008 Scenario6_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario6_NumUnitsCutFirstStage(7,1) - 0.008 Scenario6_NumUnitsCutFirstStage(7,2) - 0.008 Scenario6_NumUnitsCutFirstStage(7,3) - 0.008 Scenario6_NumUnitsCutFirstStage(7,4) - 0.008 Scenario6_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario6_NumUnitsCutFirstStage(7,6) - 0.008 Scenario6_NumUnitsCutFirstStage(8,1) - 0.008 Scenario6_NumUnitsCutFirstStage(8,2) - 0.008 Scenario6_NumUnitsCutFirstStage(8,3) - 0.008 Scenario6_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario6_NumUnitsCutFirstStage(8,5) - 0.008 Scenario6_NumUnitsCutFirstStage(8,6) - 0.008 Scenario6_NumUnitsCutFirstStage(8,7) - 0.008 Scenario6_NumUnitsCutFirstStage(9,1) - 0.008 Scenario6_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario6_NumUnitsCutFirstStage(9,3) - 0.008 Scenario6_NumUnitsCutFirstStage(9,4) - 0.008 Scenario6_NumUnitsCutFirstStage(9,5) - 0.008 Scenario6_NumUnitsCutFirstStage(9,6) - 0.008 Scenario6_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario6_NumUnitsCutFirstStage(9,8) - 453.0 Scenario6_ProduceSizeFirstStage(10) - 453.0 Scenario6_ProduceSizeFirstStage(1) - 453.0 Scenario6_ProduceSizeFirstStage(2) - 453.0 Scenario6_ProduceSizeFirstStage(3) 
- 453.0 Scenario6_ProduceSizeFirstStage(4) - 453.0 Scenario6_ProduceSizeFirstStage(5) - 453.0 Scenario6_ProduceSizeFirstStage(6) - 453.0 Scenario6_ProduceSizeFirstStage(7) - 453.0 Scenario6_ProduceSizeFirstStage(8) 
- 453.0 Scenario6_ProduceSizeFirstStage(9) = 0.0

Scenario6_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario6_NumProducedFirstStage(1) + 1.0 Scenario6_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario6_NumProducedFirstStage(2) + 1.0 Scenario6_NumUnitsCutFirstStage(2,1) + 1.0 Scenario6_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario6_NumProducedFirstStage(3) + 1.0 Scenario6_NumUnitsCutFirstStage(3,1) + 1.0 Scenario6_NumUnitsCutFirstStage(3,2) + 1.0 Scenario6_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario6_NumProducedFirstStage(4) + 1.0 Scenario6_NumUnitsCutFirstStage(4,1) + 1.0 Scenario6_NumUnitsCutFirstStage(4,2) + 1.0 Scenario6_NumUnitsCutFirstStage(4,3) + 1.0 Scenario6_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario6_NumProducedFirstStage(5) + 1.0 Scenario6_NumUnitsCutFirstStage(5,1) + 1.0 Scenario6_NumUnitsCutFirstStage(5,2) + 1.0 Scenario6_NumUnitsCutFirstStage(5,3) + 1.0 Scenario6_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario6_NumProducedFirstStage(6) + 1.0 Scenario6_NumUnitsCutFirstStage(6,1) + 1.0 Scenario6_NumUnitsCutFirstStage(6,2) + 1.0 Scenario6_NumUnitsCutFirstStage(6,3) + 1.0 Scenario6_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(6,5) + 1.0 Scenario6_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario6_NumProducedFirstStage(7) + 1.0 Scenario6_NumUnitsCutFirstStage(7,1) + 1.0 Scenario6_NumUnitsCutFirstStage(7,2) + 1.0 Scenario6_NumUnitsCutFirstStage(7,3) + 1.0 Scenario6_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(7,5) + 1.0 Scenario6_NumUnitsCutFirstStage(7,6) + 1.0 Scenario6_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario6_NumProducedFirstStage(8) + 1.0 Scenario6_NumUnitsCutFirstStage(8,1) + 1.0 Scenario6_NumUnitsCutFirstStage(8,2) + 1.0 Scenario6_NumUnitsCutFirstStage(8,3) + 1.0 Scenario6_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(8,5) + 1.0 Scenario6_NumUnitsCutFirstStage(8,6) + 1.0 Scenario6_NumUnitsCutFirstStage(8,7) + 1.0 Scenario6_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario6_NumProducedFirstStage(9) + 1.0 Scenario6_NumUnitsCutFirstStage(9,1) + 1.0 Scenario6_NumUnitsCutFirstStage(9,2) + 1.0 Scenario6_NumUnitsCutFirstStage(9,3) + 1.0 Scenario6_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(9,5) + 1.0 Scenario6_NumUnitsCutFirstStage(9,6) + 1.0 Scenario6_NumUnitsCutFirstStage(9,7) + 1.0 Scenario6_NumUnitsCutFirstStage(9,8) + 1.0 Scenario6_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario6_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario6_NumProducedFirstStage(10) + 1.0 Scenario6_NumUnitsCutFirstStage(10,10) + 1.0 Scenario6_NumUnitsCutFirstStage(10,1) + 1.0 Scenario6_NumUnitsCutFirstStage(10,2) + 1.0 Scenario6_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(10,4) + 1.0 Scenario6_NumUnitsCutFirstStage(10,5) + 1.0 Scenario6_NumUnitsCutFirstStage(10,6) + 1.0 Scenario6_NumUnitsCutFirstStage(10,7) + 1.0 Scenario6_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario6_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario6_NumProducedFirstStage(10) + 1.0 Scenario6_NumProducedFirstStage(1) + 1.0 Scenario6_NumProducedFirstStage(2) + 1.0 Scenario6_NumProducedFirstStage(3) + 1.0 Scenario6_NumProducedFirstStage(4) 
+ 1.0 Scenario6_NumProducedFirstStage(5) + 1.0 Scenario6_NumProducedFirstStage(6) + 1.0 Scenario6_NumProducedFirstStage(7) + 1.0 Scenario6_NumProducedFirstStage(8) + 1.0 Scenario6_NumProducedFirstStage(9) 
<= 200000.0

Scenario6_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario6_NumUnitsCutFirstStage(1,1) + 1.0 Scenario6_NumUnitsCutFirstStage(10,1) + 1.0 Scenario6_NumUnitsCutFirstStage(2,1) + 1.0 Scenario6_NumUnitsCutFirstStage(3,1) + 1.0 Scenario6_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(5,1) + 1.0 Scenario6_NumUnitsCutFirstStage(6,1) + 1.0 Scenario6_NumUnitsCutFirstStage(7,1) + 1.0 Scenario6_NumUnitsCutFirstStage(8,1) + 1.0 Scenario6_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario6_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,2) + 1.0 Scenario6_NumUnitsCutFirstStage(2,2) + 1.0 Scenario6_NumUnitsCutFirstStage(3,2) + 1.0 Scenario6_NumUnitsCutFirstStage(4,2) + 1.0 Scenario6_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(6,2) + 1.0 Scenario6_NumUnitsCutFirstStage(7,2) + 1.0 Scenario6_NumUnitsCutFirstStage(8,2) + 1.0 Scenario6_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario6_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,3) + 1.0 Scenario6_NumUnitsCutFirstStage(3,3) + 1.0 Scenario6_NumUnitsCutFirstStage(4,3) + 1.0 Scenario6_NumUnitsCutFirstStage(5,3) + 1.0 Scenario6_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(7,3) + 1.0 Scenario6_NumUnitsCutFirstStage(8,3) + 1.0 Scenario6_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario6_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,4) + 1.0 Scenario6_NumUnitsCutFirstStage(4,4) + 1.0 Scenario6_NumUnitsCutFirstStage(5,4) + 1.0 Scenario6_NumUnitsCutFirstStage(6,4) + 1.0 Scenario6_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(8,4) + 1.0 Scenario6_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario6_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,5) + 1.0 Scenario6_NumUnitsCutFirstStage(5,5) + 1.0 Scenario6_NumUnitsCutFirstStage(6,5) + 1.0 Scenario6_NumUnitsCutFirstStage(7,5) + 1.0 Scenario6_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario6_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,6) + 1.0 Scenario6_NumUnitsCutFirstStage(6,6) + 1.0 Scenario6_NumUnitsCutFirstStage(7,6) + 1.0 Scenario6_NumUnitsCutFirstStage(8,6) + 1.0 Scenario6_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario6_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,7) + 1.0 Scenario6_NumUnitsCutFirstStage(7,7) + 1.0 Scenario6_NumUnitsCutFirstStage(8,7) + 1.0 Scenario6_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario6_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,8) + 1.0 Scenario6_NumUnitsCutFirstStage(8,8) + 1.0 Scenario6_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario6_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,9) + 1.0 Scenario6_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario6_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario6_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario6_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario6_NumProducedFirstStage(1) - 1.0 Scenario6_NumProducedSecondStage(1) + 1.0 Scenario6_NumUnitsCutFirstStage(1,1) + 1.0 Scenario6_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario6_NumProducedFirstStage(2) - 1.0 Scenario6_NumProducedSecondStage(2) + 1.0 Scenario6_NumUnitsCutFirstStage(2,1) + 1.0 Scenario6_NumUnitsCutFirstStage(2,2) + 1.0 Scenario6_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario6_NumProducedFirstStage(3) - 1.0 Scenario6_NumProducedSecondStage(3) + 1.0 Scenario6_NumUnitsCutFirstStage(3,1) + 1.0 Scenario6_NumUnitsCutFirstStage(3,2) + 1.0 Scenario6_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(3,1) + 1.0 Scenario6_NumUnitsCutSecondStage(3,2) + 1.0 Scenario6_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario6_NumProducedFirstStage(4) - 1.0 Scenario6_NumProducedSecondStage(4) + 1.0 Scenario6_NumUnitsCutFirstStage(4,1) + 1.0 Scenario6_NumUnitsCutFirstStage(4,2) + 1.0 Scenario6_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(4,4) + 1.0 Scenario6_NumUnitsCutSecondStage(4,1) + 1.0 Scenario6_NumUnitsCutSecondStage(4,2) + 1.0 Scenario6_NumUnitsCutSecondStage(4,3) + 1.0 Scenario6_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario6_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario6_NumProducedFirstStage(5) - 1.0 Scenario6_NumProducedSecondStage(5) + 1.0 Scenario6_NumUnitsCutFirstStage(5,1) + 1.0 Scenario6_NumUnitsCutFirstStage(5,2) + 1.0 Scenario6_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(5,4) + 1.0 Scenario6_NumUnitsCutFirstStage(5,5) + 1.0 Scenario6_NumUnitsCutSecondStage(5,1) + 1.0 Scenario6_NumUnitsCutSecondStage(5,2) + 1.0 Scenario6_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(5,4) + 1.0 Scenario6_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario6_NumProducedFirstStage(6) - 1.0 Scenario6_NumProducedSecondStage(6) + 1.0 Scenario6_NumUnitsCutFirstStage(6,1) + 1.0 Scenario6_NumUnitsCutFirstStage(6,2) + 1.0 Scenario6_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(6,4) + 1.0 Scenario6_NumUnitsCutFirstStage(6,5) + 1.0 Scenario6_NumUnitsCutFirstStage(6,6) + 1.0 Scenario6_NumUnitsCutSecondStage(6,1) + 1.0 Scenario6_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(6,3) + 1.0 Scenario6_NumUnitsCutSecondStage(6,4) + 1.0 Scenario6_NumUnitsCutSecondStage(6,5) + 1.0 Scenario6_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario6_NumProducedFirstStage(7) - 1.0 Scenario6_NumProducedSecondStage(7) + 1.0 Scenario6_NumUnitsCutFirstStage(7,1) + 1.0 Scenario6_NumUnitsCutFirstStage(7,2) + 1.0 Scenario6_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(7,4) + 1.0 Scenario6_NumUnitsCutFirstStage(7,5) + 1.0 Scenario6_NumUnitsCutFirstStage(7,6) + 1.0 Scenario6_NumUnitsCutFirstStage(7,7) + 1.0 Scenario6_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(7,2) + 1.0 Scenario6_NumUnitsCutSecondStage(7,3) + 1.0 Scenario6_NumUnitsCutSecondStage(7,4) + 1.0 Scenario6_NumUnitsCutSecondStage(7,5) + 1.0 Scenario6_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario6_NumProducedFirstStage(8) - 1.0 Scenario6_NumProducedSecondStage(8) + 1.0 Scenario6_NumUnitsCutFirstStage(8,1) + 1.0 Scenario6_NumUnitsCutFirstStage(8,2) + 1.0 Scenario6_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(8,4) + 1.0 Scenario6_NumUnitsCutFirstStage(8,5) + 1.0 Scenario6_NumUnitsCutFirstStage(8,6) + 1.0 Scenario6_NumUnitsCutFirstStage(8,7) + 1.0 Scenario6_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(8,1) + 1.0 Scenario6_NumUnitsCutSecondStage(8,2) + 1.0 Scenario6_NumUnitsCutSecondStage(8,3) + 1.0 Scenario6_NumUnitsCutSecondStage(8,4) + 1.0 Scenario6_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(8,6) + 1.0 Scenario6_NumUnitsCutSecondStage(8,7) + 1.0 Scenario6_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario6_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario6_NumProducedFirstStage(9) - 1.0 Scenario6_NumProducedSecondStage(9) + 1.0 Scenario6_NumUnitsCutFirstStage(9,1) + 1.0 Scenario6_NumUnitsCutFirstStage(9,2) + 1.0 Scenario6_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(9,4) + 1.0 Scenario6_NumUnitsCutFirstStage(9,5) + 1.0 Scenario6_NumUnitsCutFirstStage(9,6) + 1.0 Scenario6_NumUnitsCutFirstStage(9,7) + 1.0 Scenario6_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(9,9) + 1.0 Scenario6_NumUnitsCutSecondStage(9,1) + 1.0 Scenario6_NumUnitsCutSecondStage(9,2) + 1.0 Scenario6_NumUnitsCutSecondStage(9,3) + 1.0 Scenario6_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(9,5) + 1.0 Scenario6_NumUnitsCutSecondStage(9,6) + 1.0 Scenario6_NumUnitsCutSecondStage(9,7) + 1.0 Scenario6_NumUnitsCutSecondStage(9,8) + 1.0 Scenario6_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario6_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario6_NumProducedFirstStage(10) - 1.0 Scenario6_NumProducedSecondStage(10) + 1.0 Scenario6_NumUnitsCutFirstStage(10,10) + 1.0 Scenario6_NumUnitsCutFirstStage(10,1) + 1.0 Scenario6_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(10,3) + 1.0 Scenario6_NumUnitsCutFirstStage(10,4) + 1.0 Scenario6_NumUnitsCutFirstStage(10,5) + 1.0 Scenario6_NumUnitsCutFirstStage(10,6) + 1.0 Scenario6_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario6_NumUnitsCutFirstStage(10,8) + 1.0 Scenario6_NumUnitsCutFirstStage(10,9) + 1.0 Scenario6_NumUnitsCutSecondStage(10,10) + 1.0 Scenario6_NumUnitsCutSecondStage(10,1) + 1.0 Scenario6_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(10,3) + 1.0 Scenario6_NumUnitsCutSecondStage(10,4) + 1.0 Scenario6_NumUnitsCutSecondStage(10,5) + 1.0 Scenario6_NumUnitsCutSecondStage(10,6) + 1.0 Scenario6_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario6_NumUnitsCutSecondStage(10,8) + 1.0 Scenario6_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario6

\ Begin constraint block for scenario Scenario5
Scenario5_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario5_NumUnitsCutSecondStage(1,1) + 1.0 Scenario5_NumUnitsCutSecondStage(10,1) + 1.0 Scenario5_NumUnitsCutSecondStage(2,1) + 1.0 Scenario5_NumUnitsCutSecondStage(3,1) + 1.0 Scenario5_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(5,1) + 1.0 Scenario5_NumUnitsCutSecondStage(6,1) + 1.0 Scenario5_NumUnitsCutSecondStage(7,1) + 1.0 Scenario5_NumUnitsCutSecondStage(8,1) + 1.0 Scenario5_NumUnitsCutSecondStage(9,1) 
>= 2250.0

Scenario5_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,2) + 1.0 Scenario5_NumUnitsCutSecondStage(2,2) + 1.0 Scenario5_NumUnitsCutSecondStage(3,2) + 1.0 Scenario5_NumUnitsCutSecondStage(4,2) + 1.0 Scenario5_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(6,2) + 1.0 Scenario5_NumUnitsCutSecondStage(7,2) + 1.0 Scenario5_NumUnitsCutSecondStage(8,2) + 1.0 Scenario5_NumUnitsCutSecondStage(9,2) >= 6750.0

Scenario5_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,3) + 1.0 Scenario5_NumUnitsCutSecondStage(3,3) + 1.0 Scenario5_NumUnitsCutSecondStage(4,3) + 1.0 Scenario5_NumUnitsCutSecondStage(5,3) + 1.0 Scenario5_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(7,3) + 1.0 Scenario5_NumUnitsCutSecondStage(8,3) + 1.0 Scenario5_NumUnitsCutSecondStage(9,3) >= 11250.0

Scenario5_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,4) + 1.0 Scenario5_NumUnitsCutSecondStage(4,4) + 1.0 Scenario5_NumUnitsCutSecondStage(5,4) + 1.0 Scenario5_NumUnitsCutSecondStage(6,4) + 1.0 Scenario5_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(8,4) + 1.0 Scenario5_NumUnitsCutSecondStage(9,4) >= 9000.0

Scenario5_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,5) + 1.0 Scenario5_NumUnitsCutSecondStage(5,5) + 1.0 Scenario5_NumUnitsCutSecondStage(6,5) + 1.0 Scenario5_NumUnitsCutSecondStage(7,5) + 1.0 Scenario5_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(9,5) >= 31500.0

Scenario5_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,6) + 1.0 Scenario5_NumUnitsCutSecondStage(6,6) + 1.0 Scenario5_NumUnitsCutSecondStage(7,6) + 1.0 Scenario5_NumUnitsCutSecondStage(8,6) + 1.0 Scenario5_NumUnitsCutSecondStage(9,6) 
>= 22500.0

Scenario5_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,7) + 1.0 Scenario5_NumUnitsCutSecondStage(7,7) + 1.0 Scenario5_NumUnitsCutSecondStage(8,7) + 1.0 Scenario5_NumUnitsCutSecondStage(9,7) >= 13500.0

Scenario5_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,8) + 1.0 Scenario5_NumUnitsCutSecondStage(8,8) + 1.0 Scenario5_NumUnitsCutSecondStage(9,8) >= 11250.0

Scenario5_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,9) + 1.0 Scenario5_NumUnitsCutSecondStage(9,9) >= 11250.0

Scenario5_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario5_NumUnitsCutSecondStage(10,10) >= 4500.0

Scenario5_c_e_ComputeSecondStageCost_:  -0.8416 Scenario5_NumProducedSecondStage(10) - 0.748 Scenario5_NumProducedSecondStage(1) - 0.7584 Scenario5_NumProducedSecondStage(2) - 0.7688 Scenario5_NumProducedSecondStage(3) - 0.7792 Scenario5_NumProducedSecondStage(4) 
- 0.7896 Scenario5_NumProducedSecondStage(5) - 0.8 Scenario5_NumProducedSecondStage(6) - 0.8104 Scenario5_NumProducedSecondStage(7) - 0.8208 Scenario5_NumProducedSecondStage(8) - 0.8312 Scenario5_NumProducedSecondStage(9) 
- 0.008 Scenario5_NumUnitsCutSecondStage(10,1) - 0.008 Scenario5_NumUnitsCutSecondStage(10,2) - 0.008 Scenario5_NumUnitsCutSecondStage(10,3) - 0.008 Scenario5_NumUnitsCutSecondStage(10,4) - 0.008 Scenario5_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario5_NumUnitsCutSecondStage(10,6) - 0.008 Scenario5_NumUnitsCutSecondStage(10,7) - 0.008 Scenario5_NumUnitsCutSecondStage(10,8) - 0.008 Scenario5_NumUnitsCutSecondStage(10,9) - 0.008 Scenario5_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario5_NumUnitsCutSecondStage(3,1) - 0.008 Scenario5_NumUnitsCutSecondStage(3,2) - 0.008 Scenario5_NumUnitsCutSecondStage(4,1) - 0.008 Scenario5_NumUnitsCutSecondStage(4,2) - 0.008 Scenario5_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario5_NumUnitsCutSecondStage(5,1) - 0.008 Scenario5_NumUnitsCutSecondStage(5,2) - 0.008 Scenario5_NumUnitsCutSecondStage(5,3) - 0.008 Scenario5_NumUnitsCutSecondStage(5,4) - 0.008 Scenario5_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario5_NumUnitsCutSecondStage(6,2) - 0.008 Scenario5_NumUnitsCutSecondStage(6,3) - 0.008 Scenario5_NumUnitsCutSecondStage(6,4) - 0.008 Scenario5_NumUnitsCutSecondStage(6,5) - 0.008 Scenario5_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario5_NumUnitsCutSecondStage(7,2) - 0.008 Scenario5_NumUnitsCutSecondStage(7,3) - 0.008 Scenario5_NumUnitsCutSecondStage(7,4) - 0.008 Scenario5_NumUnitsCutSecondStage(7,5) - 0.008 Scenario5_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario5_NumUnitsCutSecondStage(8,1) - 0.008 Scenario5_NumUnitsCutSecondStage(8,2) - 0.008 Scenario5_NumUnitsCutSecondStage(8,3) - 0.008 Scenario5_NumUnitsCutSecondStage(8,4) - 0.008 Scenario5_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario5_NumUnitsCutSecondStage(8,6) - 0.008 Scenario5_NumUnitsCutSecondStage(8,7) - 0.008 Scenario5_NumUnitsCutSecondStage(9,1) - 0.008 Scenario5_NumUnitsCutSecondStage(9,2) - 0.008 Scenario5_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario5_NumUnitsCutSecondStage(9,4) - 0.008 Scenario5_NumUnitsCutSecondStage(9,5) - 0.008 Scenario5_NumUnitsCutSecondStage(9,6) - 0.008 Scenario5_NumUnitsCutSecondStage(9,7) - 0.008 Scenario5_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario5_ProduceSizeSecondStage(10) - 453.0 Scenario5_ProduceSizeSecondStage(1) - 453.0 Scenario5_ProduceSizeSecondStage(2) - 453.0 Scenario5_ProduceSizeSecondStage(3) - 453.0 Scenario5_ProduceSizeSecondStage(4) 
- 453.0 Scenario5_ProduceSizeSecondStage(5) - 453.0 Scenario5_ProduceSizeSecondStage(6) - 453.0 Scenario5_ProduceSizeSecondStage(7) - 453.0 Scenario5_ProduceSizeSecondStage(8) - 453.0 Scenario5_ProduceSizeSecondStage(9) 
+ 1.0 Scenario5_SecondStageCost = 0.0

Scenario5_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario5_NumProducedSecondStage(10) + 1.0 Scenario5_NumProducedSecondStage(1) + 1.0 Scenario5_NumProducedSecondStage(2) + 1.0 Scenario5_NumProducedSecondStage(3) + 1.0 Scenario5_NumProducedSecondStage(4) 
+ 1.0 Scenario5_NumProducedSecondStage(5) + 1.0 Scenario5_NumProducedSecondStage(6) + 1.0 Scenario5_NumProducedSecondStage(7) + 1.0 Scenario5_NumProducedSecondStage(8) + 1.0 Scenario5_NumProducedSecondStage(9) 
<= 200000.0

Scenario5_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario5_NumProducedSecondStage(1) - 200000.0 Scenario5_ProduceSizeSecondStage(1) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario5_NumProducedSecondStage(2) - 200000.0 Scenario5_ProduceSizeSecondStage(2) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario5_NumProducedSecondStage(3) - 200000.0 Scenario5_ProduceSizeSecondStage(3) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario5_NumProducedSecondStage(4) - 200000.0 Scenario5_ProduceSizeSecondStage(4) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario5_NumProducedSecondStage(5) - 200000.0 Scenario5_ProduceSizeSecondStage(5) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario5_NumProducedSecondStage(6) - 200000.0 Scenario5_ProduceSizeSecondStage(6) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario5_NumProducedSecondStage(7) - 200000.0 Scenario5_ProduceSizeSecondStage(7) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario5_NumProducedSecondStage(8) - 200000.0 Scenario5_ProduceSizeSecondStage(8) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario5_NumProducedSecondStage(9) - 200000.0 Scenario5_ProduceSizeSecondStage(9) <= 0.0

Scenario5_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario5_NumProducedSecondStage(10) - 200000.0 Scenario5_ProduceSizeSecondStage(10) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario5_NumProducedFirstStage(1) - 200000.0 Scenario5_ProduceSizeFirstStage(1) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario5_NumProducedFirstStage(2) - 200000.0 Scenario5_ProduceSizeFirstStage(2) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario5_NumProducedFirstStage(3) - 200000.0 Scenario5_ProduceSizeFirstStage(3) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario5_NumProducedFirstStage(4) - 200000.0 Scenario5_ProduceSizeFirstStage(4) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario5_NumProducedFirstStage(5) - 200000.0 Scenario5_ProduceSizeFirstStage(5) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario5_NumProducedFirstStage(6) - 200000.0 Scenario5_ProduceSizeFirstStage(6) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario5_NumProducedFirstStage(7) - 200000.0 Scenario5_ProduceSizeFirstStage(7) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario5_NumProducedFirstStage(8) - 200000.0 Scenario5_ProduceSizeFirstStage(8) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario5_NumProducedFirstStage(9) - 200000.0 Scenario5_ProduceSizeFirstStage(9) <= 0.0

Scenario5_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario5_NumProducedFirstStage(10) - 200000.0 Scenario5_ProduceSizeFirstStage(10) <= 0.0

Scenario5_c_e_ComputeFirstStageCost_:  1.0 Scenario5_FirstStageCost - 0.8416 Scenario5_NumProducedFirstStage(10) - 0.748 Scenario5_NumProducedFirstStage(1) - 0.7584 Scenario5_NumProducedFirstStage(2) - 0.7688 Scenario5_NumProducedFirstStage(3) 
- 0.7792 Scenario5_NumProducedFirstStage(4) - 0.7896 Scenario5_NumProducedFirstStage(5) - 0.8 Scenario5_NumProducedFirstStage(6) - 0.8104 Scenario5_NumProducedFirstStage(7) - 0.8208 Scenario5_NumProducedFirstStage(8) 
- 0.8312 Scenario5_NumProducedFirstStage(9) - 0.008 Scenario5_NumUnitsCutFirstStage(10,1) - 0.008 Scenario5_NumUnitsCutFirstStage(10,2) - 0.008 Scenario5_NumUnitsCutFirstStage(10,3) - 0.008 Scenario5_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario5_NumUnitsCutFirstStage(10,5) - 0.008 Scenario5_NumUnitsCutFirstStage(10,6) - 0.008 Scenario5_NumUnitsCutFirstStage(10,7) - 0.008 Scenario5_NumUnitsCutFirstStage(10,8) - 0.008 Scenario5_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario5_NumUnitsCutFirstStage(2,1) - 0.008 Scenario5_NumUnitsCutFirstStage(3,1) - 0.008 Scenario5_NumUnitsCutFirstStage(3,2) - 0.008 Scenario5_NumUnitsCutFirstStage(4,1) - 0.008 Scenario5_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario5_NumUnitsCutFirstStage(4,3) - 0.008 Scenario5_NumUnitsCutFirstStage(5,1) - 0.008 Scenario5_NumUnitsCutFirstStage(5,2) - 0.008 Scenario5_NumUnitsCutFirstStage(5,3) - 0.008 Scenario5_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario5_NumUnitsCutFirstStage(6,1) - 0.008 Scenario5_NumUnitsCutFirstStage(6,2) - 0.008 Scenario5_NumUnitsCutFirstStage(6,3) - 0.008 Scenario5_NumUnitsCutFirstStage(6,4) - 0.008 Scenario5_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario5_NumUnitsCutFirstStage(7,1) - 0.008 Scenario5_NumUnitsCutFirstStage(7,2) - 0.008 Scenario5_NumUnitsCutFirstStage(7,3) - 0.008 Scenario5_NumUnitsCutFirstStage(7,4) - 0.008 Scenario5_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario5_NumUnitsCutFirstStage(7,6) - 0.008 Scenario5_NumUnitsCutFirstStage(8,1) - 0.008 Scenario5_NumUnitsCutFirstStage(8,2) - 0.008 Scenario5_NumUnitsCutFirstStage(8,3) - 0.008 Scenario5_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario5_NumUnitsCutFirstStage(8,5) - 0.008 Scenario5_NumUnitsCutFirstStage(8,6) - 0.008 Scenario5_NumUnitsCutFirstStage(8,7) - 0.008 Scenario5_NumUnitsCutFirstStage(9,1) - 0.008 Scenario5_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario5_NumUnitsCutFirstStage(9,3) - 0.008 Scenario5_NumUnitsCutFirstStage(9,4) - 0.008 Scenario5_NumUnitsCutFirstStage(9,5) - 0.008 Scenario5_NumUnitsCutFirstStage(9,6) - 0.008 Scenario5_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario5_NumUnitsCutFirstStage(9,8) - 453.0 Scenario5_ProduceSizeFirstStage(10) - 453.0 Scenario5_ProduceSizeFirstStage(1) - 453.0 Scenario5_ProduceSizeFirstStage(2) - 453.0 Scenario5_ProduceSizeFirstStage(3) 
- 453.0 Scenario5_ProduceSizeFirstStage(4) - 453.0 Scenario5_ProduceSizeFirstStage(5) - 453.0 Scenario5_ProduceSizeFirstStage(6) - 453.0 Scenario5_ProduceSizeFirstStage(7) - 453.0 Scenario5_ProduceSizeFirstStage(8) 
- 453.0 Scenario5_ProduceSizeFirstStage(9) = 0.0

Scenario5_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario5_NumProducedFirstStage(1) + 1.0 Scenario5_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario5_NumProducedFirstStage(2) + 1.0 Scenario5_NumUnitsCutFirstStage(2,1) + 1.0 Scenario5_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario5_NumProducedFirstStage(3) + 1.0 Scenario5_NumUnitsCutFirstStage(3,1) + 1.0 Scenario5_NumUnitsCutFirstStage(3,2) + 1.0 Scenario5_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario5_NumProducedFirstStage(4) + 1.0 Scenario5_NumUnitsCutFirstStage(4,1) + 1.0 Scenario5_NumUnitsCutFirstStage(4,2) + 1.0 Scenario5_NumUnitsCutFirstStage(4,3) + 1.0 Scenario5_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario5_NumProducedFirstStage(5) + 1.0 Scenario5_NumUnitsCutFirstStage(5,1) + 1.0 Scenario5_NumUnitsCutFirstStage(5,2) + 1.0 Scenario5_NumUnitsCutFirstStage(5,3) + 1.0 Scenario5_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario5_NumProducedFirstStage(6) + 1.0 Scenario5_NumUnitsCutFirstStage(6,1) + 1.0 Scenario5_NumUnitsCutFirstStage(6,2) + 1.0 Scenario5_NumUnitsCutFirstStage(6,3) + 1.0 Scenario5_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(6,5) + 1.0 Scenario5_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario5_NumProducedFirstStage(7) + 1.0 Scenario5_NumUnitsCutFirstStage(7,1) + 1.0 Scenario5_NumUnitsCutFirstStage(7,2) + 1.0 Scenario5_NumUnitsCutFirstStage(7,3) + 1.0 Scenario5_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(7,5) + 1.0 Scenario5_NumUnitsCutFirstStage(7,6) + 1.0 Scenario5_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario5_NumProducedFirstStage(8) + 1.0 Scenario5_NumUnitsCutFirstStage(8,1) + 1.0 Scenario5_NumUnitsCutFirstStage(8,2) + 1.0 Scenario5_NumUnitsCutFirstStage(8,3) + 1.0 Scenario5_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(8,5) + 1.0 Scenario5_NumUnitsCutFirstStage(8,6) + 1.0 Scenario5_NumUnitsCutFirstStage(8,7) + 1.0 Scenario5_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario5_NumProducedFirstStage(9) + 1.0 Scenario5_NumUnitsCutFirstStage(9,1) + 1.0 Scenario5_NumUnitsCutFirstStage(9,2) + 1.0 Scenario5_NumUnitsCutFirstStage(9,3) + 1.0 Scenario5_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(9,5) + 1.0 Scenario5_NumUnitsCutFirstStage(9,6) + 1.0 Scenario5_NumUnitsCutFirstStage(9,7) + 1.0 Scenario5_NumUnitsCutFirstStage(9,8) + 1.0 Scenario5_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario5_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario5_NumProducedFirstStage(10) + 1.0 Scenario5_NumUnitsCutFirstStage(10,10) + 1.0 Scenario5_NumUnitsCutFirstStage(10,1) + 1.0 Scenario5_NumUnitsCutFirstStage(10,2) + 1.0 Scenario5_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(10,4) + 1.0 Scenario5_NumUnitsCutFirstStage(10,5) + 1.0 Scenario5_NumUnitsCutFirstStage(10,6) + 1.0 Scenario5_NumUnitsCutFirstStage(10,7) + 1.0 Scenario5_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario5_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario5_NumProducedFirstStage(10) + 1.0 Scenario5_NumProducedFirstStage(1) + 1.0 Scenario5_NumProducedFirstStage(2) + 1.0 Scenario5_NumProducedFirstStage(3) + 1.0 Scenario5_NumProducedFirstStage(4) 
+ 1.0 Scenario5_NumProducedFirstStage(5) + 1.0 Scenario5_NumProducedFirstStage(6) + 1.0 Scenario5_NumProducedFirstStage(7) + 1.0 Scenario5_NumProducedFirstStage(8) + 1.0 Scenario5_NumProducedFirstStage(9) 
<= 200000.0

Scenario5_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario5_NumUnitsCutFirstStage(1,1) + 1.0 Scenario5_NumUnitsCutFirstStage(10,1) + 1.0 Scenario5_NumUnitsCutFirstStage(2,1) + 1.0 Scenario5_NumUnitsCutFirstStage(3,1) + 1.0 Scenario5_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(5,1) + 1.0 Scenario5_NumUnitsCutFirstStage(6,1) + 1.0 Scenario5_NumUnitsCutFirstStage(7,1) + 1.0 Scenario5_NumUnitsCutFirstStage(8,1) + 1.0 Scenario5_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario5_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,2) + 1.0 Scenario5_NumUnitsCutFirstStage(2,2) + 1.0 Scenario5_NumUnitsCutFirstStage(3,2) + 1.0 Scenario5_NumUnitsCutFirstStage(4,2) + 1.0 Scenario5_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(6,2) + 1.0 Scenario5_NumUnitsCutFirstStage(7,2) + 1.0 Scenario5_NumUnitsCutFirstStage(8,2) + 1.0 Scenario5_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario5_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,3) + 1.0 Scenario5_NumUnitsCutFirstStage(3,3) + 1.0 Scenario5_NumUnitsCutFirstStage(4,3) + 1.0 Scenario5_NumUnitsCutFirstStage(5,3) + 1.0 Scenario5_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(7,3) + 1.0 Scenario5_NumUnitsCutFirstStage(8,3) + 1.0 Scenario5_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario5_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,4) + 1.0 Scenario5_NumUnitsCutFirstStage(4,4) + 1.0 Scenario5_NumUnitsCutFirstStage(5,4) + 1.0 Scenario5_NumUnitsCutFirstStage(6,4) + 1.0 Scenario5_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(8,4) + 1.0 Scenario5_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario5_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,5) + 1.0 Scenario5_NumUnitsCutFirstStage(5,5) + 1.0 Scenario5_NumUnitsCutFirstStage(6,5) + 1.0 Scenario5_NumUnitsCutFirstStage(7,5) + 1.0 Scenario5_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario5_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,6) + 1.0 Scenario5_NumUnitsCutFirstStage(6,6) + 1.0 Scenario5_NumUnitsCutFirstStage(7,6) + 1.0 Scenario5_NumUnitsCutFirstStage(8,6) + 1.0 Scenario5_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario5_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,7) + 1.0 Scenario5_NumUnitsCutFirstStage(7,7) + 1.0 Scenario5_NumUnitsCutFirstStage(8,7) + 1.0 Scenario5_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario5_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,8) + 1.0 Scenario5_NumUnitsCutFirstStage(8,8) + 1.0 Scenario5_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario5_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,9) + 1.0 Scenario5_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario5_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario5_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario5_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario5_NumProducedFirstStage(1) - 1.0 Scenario5_NumProducedSecondStage(1) + 1.0 Scenario5_NumUnitsCutFirstStage(1,1) + 1.0 Scenario5_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario5_NumProducedFirstStage(2) - 1.0 Scenario5_NumProducedSecondStage(2) + 1.0 Scenario5_NumUnitsCutFirstStage(2,1) + 1.0 Scenario5_NumUnitsCutFirstStage(2,2) + 1.0 Scenario5_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario5_NumProducedFirstStage(3) - 1.0 Scenario5_NumProducedSecondStage(3) + 1.0 Scenario5_NumUnitsCutFirstStage(3,1) + 1.0 Scenario5_NumUnitsCutFirstStage(3,2) + 1.0 Scenario5_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(3,1) + 1.0 Scenario5_NumUnitsCutSecondStage(3,2) + 1.0 Scenario5_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario5_NumProducedFirstStage(4) - 1.0 Scenario5_NumProducedSecondStage(4) + 1.0 Scenario5_NumUnitsCutFirstStage(4,1) + 1.0 Scenario5_NumUnitsCutFirstStage(4,2) + 1.0 Scenario5_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(4,4) + 1.0 Scenario5_NumUnitsCutSecondStage(4,1) + 1.0 Scenario5_NumUnitsCutSecondStage(4,2) + 1.0 Scenario5_NumUnitsCutSecondStage(4,3) + 1.0 Scenario5_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario5_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario5_NumProducedFirstStage(5) - 1.0 Scenario5_NumProducedSecondStage(5) + 1.0 Scenario5_NumUnitsCutFirstStage(5,1) + 1.0 Scenario5_NumUnitsCutFirstStage(5,2) + 1.0 Scenario5_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(5,4) + 1.0 Scenario5_NumUnitsCutFirstStage(5,5) + 1.0 Scenario5_NumUnitsCutSecondStage(5,1) + 1.0 Scenario5_NumUnitsCutSecondStage(5,2) + 1.0 Scenario5_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(5,4) + 1.0 Scenario5_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario5_NumProducedFirstStage(6) - 1.0 Scenario5_NumProducedSecondStage(6) + 1.0 Scenario5_NumUnitsCutFirstStage(6,1) + 1.0 Scenario5_NumUnitsCutFirstStage(6,2) + 1.0 Scenario5_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(6,4) + 1.0 Scenario5_NumUnitsCutFirstStage(6,5) + 1.0 Scenario5_NumUnitsCutFirstStage(6,6) + 1.0 Scenario5_NumUnitsCutSecondStage(6,1) + 1.0 Scenario5_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(6,3) + 1.0 Scenario5_NumUnitsCutSecondStage(6,4) + 1.0 Scenario5_NumUnitsCutSecondStage(6,5) + 1.0 Scenario5_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario5_NumProducedFirstStage(7) - 1.0 Scenario5_NumProducedSecondStage(7) + 1.0 Scenario5_NumUnitsCutFirstStage(7,1) + 1.0 Scenario5_NumUnitsCutFirstStage(7,2) + 1.0 Scenario5_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(7,4) + 1.0 Scenario5_NumUnitsCutFirstStage(7,5) + 1.0 Scenario5_NumUnitsCutFirstStage(7,6) + 1.0 Scenario5_NumUnitsCutFirstStage(7,7) + 1.0 Scenario5_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(7,2) + 1.0 Scenario5_NumUnitsCutSecondStage(7,3) + 1.0 Scenario5_NumUnitsCutSecondStage(7,4) + 1.0 Scenario5_NumUnitsCutSecondStage(7,5) + 1.0 Scenario5_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario5_NumProducedFirstStage(8) - 1.0 Scenario5_NumProducedSecondStage(8) + 1.0 Scenario5_NumUnitsCutFirstStage(8,1) + 1.0 Scenario5_NumUnitsCutFirstStage(8,2) + 1.0 Scenario5_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(8,4) + 1.0 Scenario5_NumUnitsCutFirstStage(8,5) + 1.0 Scenario5_NumUnitsCutFirstStage(8,6) + 1.0 Scenario5_NumUnitsCutFirstStage(8,7) + 1.0 Scenario5_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(8,1) + 1.0 Scenario5_NumUnitsCutSecondStage(8,2) + 1.0 Scenario5_NumUnitsCutSecondStage(8,3) + 1.0 Scenario5_NumUnitsCutSecondStage(8,4) + 1.0 Scenario5_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(8,6) + 1.0 Scenario5_NumUnitsCutSecondStage(8,7) + 1.0 Scenario5_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario5_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario5_NumProducedFirstStage(9) - 1.0 Scenario5_NumProducedSecondStage(9) + 1.0 Scenario5_NumUnitsCutFirstStage(9,1) + 1.0 Scenario5_NumUnitsCutFirstStage(9,2) + 1.0 Scenario5_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(9,4) + 1.0 Scenario5_NumUnitsCutFirstStage(9,5) + 1.0 Scenario5_NumUnitsCutFirstStage(9,6) + 1.0 Scenario5_NumUnitsCutFirstStage(9,7) + 1.0 Scenario5_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(9,9) + 1.0 Scenario5_NumUnitsCutSecondStage(9,1) + 1.0 Scenario5_NumUnitsCutSecondStage(9,2) + 1.0 Scenario5_NumUnitsCutSecondStage(9,3) + 1.0 Scenario5_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(9,5) + 1.0 Scenario5_NumUnitsCutSecondStage(9,6) + 1.0 Scenario5_NumUnitsCutSecondStage(9,7) + 1.0 Scenario5_NumUnitsCutSecondStage(9,8) + 1.0 Scenario5_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario5_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario5_NumProducedFirstStage(10) - 1.0 Scenario5_NumProducedSecondStage(10) + 1.0 Scenario5_NumUnitsCutFirstStage(10,10) + 1.0 Scenario5_NumUnitsCutFirstStage(10,1) + 1.0 Scenario5_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(10,3) + 1.0 Scenario5_NumUnitsCutFirstStage(10,4) + 1.0 Scenario5_NumUnitsCutFirstStage(10,5) + 1.0 Scenario5_NumUnitsCutFirstStage(10,6) + 1.0 Scenario5_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario5_NumUnitsCutFirstStage(10,8) + 1.0 Scenario5_NumUnitsCutFirstStage(10,9) + 1.0 Scenario5_NumUnitsCutSecondStage(10,10) + 1.0 Scenario5_NumUnitsCutSecondStage(10,1) + 1.0 Scenario5_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(10,3) + 1.0 Scenario5_NumUnitsCutSecondStage(10,4) + 1.0 Scenario5_NumUnitsCutSecondStage(10,5) + 1.0 Scenario5_NumUnitsCutSecondStage(10,6) + 1.0 Scenario5_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario5_NumUnitsCutSecondStage(10,8) + 1.0 Scenario5_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario5

\ Begin constraint block for scenario Scenario4
Scenario4_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario4_NumUnitsCutSecondStage(1,1) + 1.0 Scenario4_NumUnitsCutSecondStage(10,1) + 1.0 Scenario4_NumUnitsCutSecondStage(2,1) + 1.0 Scenario4_NumUnitsCutSecondStage(3,1) + 1.0 Scenario4_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(5,1) + 1.0 Scenario4_NumUnitsCutSecondStage(6,1) + 1.0 Scenario4_NumUnitsCutSecondStage(7,1) + 1.0 Scenario4_NumUnitsCutSecondStage(8,1) + 1.0 Scenario4_NumUnitsCutSecondStage(9,1) 
>= 2000.0

Scenario4_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,2) + 1.0 Scenario4_NumUnitsCutSecondStage(2,2) + 1.0 Scenario4_NumUnitsCutSecondStage(3,2) + 1.0 Scenario4_NumUnitsCutSecondStage(4,2) + 1.0 Scenario4_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(6,2) + 1.0 Scenario4_NumUnitsCutSecondStage(7,2) + 1.0 Scenario4_NumUnitsCutSecondStage(8,2) + 1.0 Scenario4_NumUnitsCutSecondStage(9,2) >= 6000.0

Scenario4_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,3) + 1.0 Scenario4_NumUnitsCutSecondStage(3,3) + 1.0 Scenario4_NumUnitsCutSecondStage(4,3) + 1.0 Scenario4_NumUnitsCutSecondStage(5,3) + 1.0 Scenario4_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(7,3) + 1.0 Scenario4_NumUnitsCutSecondStage(8,3) + 1.0 Scenario4_NumUnitsCutSecondStage(9,3) >= 10000.0

Scenario4_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,4) + 1.0 Scenario4_NumUnitsCutSecondStage(4,4) + 1.0 Scenario4_NumUnitsCutSecondStage(5,4) + 1.0 Scenario4_NumUnitsCutSecondStage(6,4) + 1.0 Scenario4_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(8,4) + 1.0 Scenario4_NumUnitsCutSecondStage(9,4) >= 8000.0

Scenario4_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,5) + 1.0 Scenario4_NumUnitsCutSecondStage(5,5) + 1.0 Scenario4_NumUnitsCutSecondStage(6,5) + 1.0 Scenario4_NumUnitsCutSecondStage(7,5) + 1.0 Scenario4_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(9,5) >= 28000.0

Scenario4_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,6) + 1.0 Scenario4_NumUnitsCutSecondStage(6,6) + 1.0 Scenario4_NumUnitsCutSecondStage(7,6) + 1.0 Scenario4_NumUnitsCutSecondStage(8,6) + 1.0 Scenario4_NumUnitsCutSecondStage(9,6) 
>= 20000.0

Scenario4_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,7) + 1.0 Scenario4_NumUnitsCutSecondStage(7,7) + 1.0 Scenario4_NumUnitsCutSecondStage(8,7) + 1.0 Scenario4_NumUnitsCutSecondStage(9,7) >= 12000.0

Scenario4_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,8) + 1.0 Scenario4_NumUnitsCutSecondStage(8,8) + 1.0 Scenario4_NumUnitsCutSecondStage(9,8) >= 10000.0

Scenario4_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,9) + 1.0 Scenario4_NumUnitsCutSecondStage(9,9) >= 10000.0

Scenario4_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario4_NumUnitsCutSecondStage(10,10) >= 4000.0

Scenario4_c_e_ComputeSecondStageCost_:  -0.8416 Scenario4_NumProducedSecondStage(10) - 0.748 Scenario4_NumProducedSecondStage(1) - 0.7584 Scenario4_NumProducedSecondStage(2) - 0.7688 Scenario4_NumProducedSecondStage(3) - 0.7792 Scenario4_NumProducedSecondStage(4) 
- 0.7896 Scenario4_NumProducedSecondStage(5) - 0.8 Scenario4_NumProducedSecondStage(6) - 0.8104 Scenario4_NumProducedSecondStage(7) - 0.8208 Scenario4_NumProducedSecondStage(8) - 0.8312 Scenario4_NumProducedSecondStage(9) 
- 0.008 Scenario4_NumUnitsCutSecondStage(10,1) - 0.008 Scenario4_NumUnitsCutSecondStage(10,2) - 0.008 Scenario4_NumUnitsCutSecondStage(10,3) - 0.008 Scenario4_NumUnitsCutSecondStage(10,4) - 0.008 Scenario4_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario4_NumUnitsCutSecondStage(10,6) - 0.008 Scenario4_NumUnitsCutSecondStage(10,7) - 0.008 Scenario4_NumUnitsCutSecondStage(10,8) - 0.008 Scenario4_NumUnitsCutSecondStage(10,9) - 0.008 Scenario4_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario4_NumUnitsCutSecondStage(3,1) - 0.008 Scenario4_NumUnitsCutSecondStage(3,2) - 0.008 Scenario4_NumUnitsCutSecondStage(4,1) - 0.008 Scenario4_NumUnitsCutSecondStage(4,2) - 0.008 Scenario4_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario4_NumUnitsCutSecondStage(5,1) - 0.008 Scenario4_NumUnitsCutSecondStage(5,2) - 0.008 Scenario4_NumUnitsCutSecondStage(5,3) - 0.008 Scenario4_NumUnitsCutSecondStage(5,4) - 0.008 Scenario4_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario4_NumUnitsCutSecondStage(6,2) - 0.008 Scenario4_NumUnitsCutSecondStage(6,3) - 0.008 Scenario4_NumUnitsCutSecondStage(6,4) - 0.008 Scenario4_NumUnitsCutSecondStage(6,5) - 0.008 Scenario4_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario4_NumUnitsCutSecondStage(7,2) - 0.008 Scenario4_NumUnitsCutSecondStage(7,3) - 0.008 Scenario4_NumUnitsCutSecondStage(7,4) - 0.008 Scenario4_NumUnitsCutSecondStage(7,5) - 0.008 Scenario4_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario4_NumUnitsCutSecondStage(8,1) - 0.008 Scenario4_NumUnitsCutSecondStage(8,2) - 0.008 Scenario4_NumUnitsCutSecondStage(8,3) - 0.008 Scenario4_NumUnitsCutSecondStage(8,4) - 0.008 Scenario4_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario4_NumUnitsCutSecondStage(8,6) - 0.008 Scenario4_NumUnitsCutSecondStage(8,7) - 0.008 Scenario4_NumUnitsCutSecondStage(9,1) - 0.008 Scenario4_NumUnitsCutSecondStage(9,2) - 0.008 Scenario4_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario4_NumUnitsCutSecondStage(9,4) - 0.008 Scenario4_NumUnitsCutSecondStage(9,5) - 0.008 Scenario4_NumUnitsCutSecondStage(9,6) - 0.008 Scenario4_NumUnitsCutSecondStage(9,7) - 0.008 Scenario4_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario4_ProduceSizeSecondStage(10) - 453.0 Scenario4_ProduceSizeSecondStage(1) - 453.0 Scenario4_ProduceSizeSecondStage(2) - 453.0 Scenario4_ProduceSizeSecondStage(3) - 453.0 Scenario4_ProduceSizeSecondStage(4) 
- 453.0 Scenario4_ProduceSizeSecondStage(5) - 453.0 Scenario4_ProduceSizeSecondStage(6) - 453.0 Scenario4_ProduceSizeSecondStage(7) - 453.0 Scenario4_ProduceSizeSecondStage(8) - 453.0 Scenario4_ProduceSizeSecondStage(9) 
+ 1.0 Scenario4_SecondStageCost = 0.0

Scenario4_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario4_NumProducedSecondStage(10) + 1.0 Scenario4_NumProducedSecondStage(1) + 1.0 Scenario4_NumProducedSecondStage(2) + 1.0 Scenario4_NumProducedSecondStage(3) + 1.0 Scenario4_NumProducedSecondStage(4) 
+ 1.0 Scenario4_NumProducedSecondStage(5) + 1.0 Scenario4_NumProducedSecondStage(6) + 1.0 Scenario4_NumProducedSecondStage(7) + 1.0 Scenario4_NumProducedSecondStage(8) + 1.0 Scenario4_NumProducedSecondStage(9) 
<= 200000.0

Scenario4_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario4_NumProducedSecondStage(1) - 200000.0 Scenario4_ProduceSizeSecondStage(1) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario4_NumProducedSecondStage(2) - 200000.0 Scenario4_ProduceSizeSecondStage(2) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario4_NumProducedSecondStage(3) - 200000.0 Scenario4_ProduceSizeSecondStage(3) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario4_NumProducedSecondStage(4) - 200000.0 Scenario4_ProduceSizeSecondStage(4) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario4_NumProducedSecondStage(5) - 200000.0 Scenario4_ProduceSizeSecondStage(5) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario4_NumProducedSecondStage(6) - 200000.0 Scenario4_ProduceSizeSecondStage(6) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario4_NumProducedSecondStage(7) - 200000.0 Scenario4_ProduceSizeSecondStage(7) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario4_NumProducedSecondStage(8) - 200000.0 Scenario4_ProduceSizeSecondStage(8) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario4_NumProducedSecondStage(9) - 200000.0 Scenario4_ProduceSizeSecondStage(9) <= 0.0

Scenario4_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario4_NumProducedSecondStage(10) - 200000.0 Scenario4_ProduceSizeSecondStage(10) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario4_NumProducedFirstStage(1) - 200000.0 Scenario4_ProduceSizeFirstStage(1) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario4_NumProducedFirstStage(2) - 200000.0 Scenario4_ProduceSizeFirstStage(2) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario4_NumProducedFirstStage(3) - 200000.0 Scenario4_ProduceSizeFirstStage(3) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario4_NumProducedFirstStage(4) - 200000.0 Scenario4_ProduceSizeFirstStage(4) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario4_NumProducedFirstStage(5) - 200000.0 Scenario4_ProduceSizeFirstStage(5) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario4_NumProducedFirstStage(6) - 200000.0 Scenario4_ProduceSizeFirstStage(6) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario4_NumProducedFirstStage(7) - 200000.0 Scenario4_ProduceSizeFirstStage(7) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario4_NumProducedFirstStage(8) - 200000.0 Scenario4_ProduceSizeFirstStage(8) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario4_NumProducedFirstStage(9) - 200000.0 Scenario4_ProduceSizeFirstStage(9) <= 0.0

Scenario4_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario4_NumProducedFirstStage(10) - 200000.0 Scenario4_ProduceSizeFirstStage(10) <= 0.0

Scenario4_c_e_ComputeFirstStageCost_:  1.0 Scenario4_FirstStageCost - 0.8416 Scenario4_NumProducedFirstStage(10) - 0.748 Scenario4_NumProducedFirstStage(1) - 0.7584 Scenario4_NumProducedFirstStage(2) - 0.7688 Scenario4_NumProducedFirstStage(3) 
- 0.7792 Scenario4_NumProducedFirstStage(4) - 0.7896 Scenario4_NumProducedFirstStage(5) - 0.8 Scenario4_NumProducedFirstStage(6) - 0.8104 Scenario4_NumProducedFirstStage(7) - 0.8208 Scenario4_NumProducedFirstStage(8) 
- 0.8312 Scenario4_NumProducedFirstStage(9) - 0.008 Scenario4_NumUnitsCutFirstStage(10,1) - 0.008 Scenario4_NumUnitsCutFirstStage(10,2) - 0.008 Scenario4_NumUnitsCutFirstStage(10,3) - 0.008 Scenario4_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario4_NumUnitsCutFirstStage(10,5) - 0.008 Scenario4_NumUnitsCutFirstStage(10,6) - 0.008 Scenario4_NumUnitsCutFirstStage(10,7) - 0.008 Scenario4_NumUnitsCutFirstStage(10,8) - 0.008 Scenario4_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario4_NumUnitsCutFirstStage(2,1) - 0.008 Scenario4_NumUnitsCutFirstStage(3,1) - 0.008 Scenario4_NumUnitsCutFirstStage(3,2) - 0.008 Scenario4_NumUnitsCutFirstStage(4,1) - 0.008 Scenario4_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario4_NumUnitsCutFirstStage(4,3) - 0.008 Scenario4_NumUnitsCutFirstStage(5,1) - 0.008 Scenario4_NumUnitsCutFirstStage(5,2) - 0.008 Scenario4_NumUnitsCutFirstStage(5,3) - 0.008 Scenario4_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario4_NumUnitsCutFirstStage(6,1) - 0.008 Scenario4_NumUnitsCutFirstStage(6,2) - 0.008 Scenario4_NumUnitsCutFirstStage(6,3) - 0.008 Scenario4_NumUnitsCutFirstStage(6,4) - 0.008 Scenario4_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario4_NumUnitsCutFirstStage(7,1) - 0.008 Scenario4_NumUnitsCutFirstStage(7,2) - 0.008 Scenario4_NumUnitsCutFirstStage(7,3) - 0.008 Scenario4_NumUnitsCutFirstStage(7,4) - 0.008 Scenario4_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario4_NumUnitsCutFirstStage(7,6) - 0.008 Scenario4_NumUnitsCutFirstStage(8,1) - 0.008 Scenario4_NumUnitsCutFirstStage(8,2) - 0.008 Scenario4_NumUnitsCutFirstStage(8,3) - 0.008 Scenario4_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario4_NumUnitsCutFirstStage(8,5) - 0.008 Scenario4_NumUnitsCutFirstStage(8,6) - 0.008 Scenario4_NumUnitsCutFirstStage(8,7) - 0.008 Scenario4_NumUnitsCutFirstStage(9,1) - 0.008 Scenario4_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario4_NumUnitsCutFirstStage(9,3) - 0.008 Scenario4_NumUnitsCutFirstStage(9,4) - 0.008 Scenario4_NumUnitsCutFirstStage(9,5) - 0.008 Scenario4_NumUnitsCutFirstStage(9,6) - 0.008 Scenario4_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario4_NumUnitsCutFirstStage(9,8) - 453.0 Scenario4_ProduceSizeFirstStage(10) - 453.0 Scenario4_ProduceSizeFirstStage(1) - 453.0 Scenario4_ProduceSizeFirstStage(2) - 453.0 Scenario4_ProduceSizeFirstStage(3) 
- 453.0 Scenario4_ProduceSizeFirstStage(4) - 453.0 Scenario4_ProduceSizeFirstStage(5) - 453.0 Scenario4_ProduceSizeFirstStage(6) - 453.0 Scenario4_ProduceSizeFirstStage(7) - 453.0 Scenario4_ProduceSizeFirstStage(8) 
- 453.0 Scenario4_ProduceSizeFirstStage(9) = 0.0

Scenario4_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario4_NumProducedFirstStage(1) + 1.0 Scenario4_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario4_NumProducedFirstStage(2) + 1.0 Scenario4_NumUnitsCutFirstStage(2,1) + 1.0 Scenario4_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario4_NumProducedFirstStage(3) + 1.0 Scenario4_NumUnitsCutFirstStage(3,1) + 1.0 Scenario4_NumUnitsCutFirstStage(3,2) + 1.0 Scenario4_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario4_NumProducedFirstStage(4) + 1.0 Scenario4_NumUnitsCutFirstStage(4,1) + 1.0 Scenario4_NumUnitsCutFirstStage(4,2) + 1.0 Scenario4_NumUnitsCutFirstStage(4,3) + 1.0 Scenario4_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario4_NumProducedFirstStage(5) + 1.0 Scenario4_NumUnitsCutFirstStage(5,1) + 1.0 Scenario4_NumUnitsCutFirstStage(5,2) + 1.0 Scenario4_NumUnitsCutFirstStage(5,3) + 1.0 Scenario4_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario4_NumProducedFirstStage(6) + 1.0 Scenario4_NumUnitsCutFirstStage(6,1) + 1.0 Scenario4_NumUnitsCutFirstStage(6,2) + 1.0 Scenario4_NumUnitsCutFirstStage(6,3) + 1.0 Scenario4_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(6,5) + 1.0 Scenario4_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario4_NumProducedFirstStage(7) + 1.0 Scenario4_NumUnitsCutFirstStage(7,1) + 1.0 Scenario4_NumUnitsCutFirstStage(7,2) + 1.0 Scenario4_NumUnitsCutFirstStage(7,3) + 1.0 Scenario4_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(7,5) + 1.0 Scenario4_NumUnitsCutFirstStage(7,6) + 1.0 Scenario4_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario4_NumProducedFirstStage(8) + 1.0 Scenario4_NumUnitsCutFirstStage(8,1) + 1.0 Scenario4_NumUnitsCutFirstStage(8,2) + 1.0 Scenario4_NumUnitsCutFirstStage(8,3) + 1.0 Scenario4_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(8,5) + 1.0 Scenario4_NumUnitsCutFirstStage(8,6) + 1.0 Scenario4_NumUnitsCutFirstStage(8,7) + 1.0 Scenario4_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario4_NumProducedFirstStage(9) + 1.0 Scenario4_NumUnitsCutFirstStage(9,1) + 1.0 Scenario4_NumUnitsCutFirstStage(9,2) + 1.0 Scenario4_NumUnitsCutFirstStage(9,3) + 1.0 Scenario4_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(9,5) + 1.0 Scenario4_NumUnitsCutFirstStage(9,6) + 1.0 Scenario4_NumUnitsCutFirstStage(9,7) + 1.0 Scenario4_NumUnitsCutFirstStage(9,8) + 1.0 Scenario4_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario4_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario4_NumProducedFirstStage(10) + 1.0 Scenario4_NumUnitsCutFirstStage(10,10) + 1.0 Scenario4_NumUnitsCutFirstStage(10,1) + 1.0 Scenario4_NumUnitsCutFirstStage(10,2) + 1.0 Scenario4_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(10,4) + 1.0 Scenario4_NumUnitsCutFirstStage(10,5) + 1.0 Scenario4_NumUnitsCutFirstStage(10,6) + 1.0 Scenario4_NumUnitsCutFirstStage(10,7) + 1.0 Scenario4_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario4_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario4_NumProducedFirstStage(10) + 1.0 Scenario4_NumProducedFirstStage(1) + 1.0 Scenario4_NumProducedFirstStage(2) + 1.0 Scenario4_NumProducedFirstStage(3) + 1.0 Scenario4_NumProducedFirstStage(4) 
+ 1.0 Scenario4_NumProducedFirstStage(5) + 1.0 Scenario4_NumProducedFirstStage(6) + 1.0 Scenario4_NumProducedFirstStage(7) + 1.0 Scenario4_NumProducedFirstStage(8) + 1.0 Scenario4_NumProducedFirstStage(9) 
<= 200000.0

Scenario4_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario4_NumUnitsCutFirstStage(1,1) + 1.0 Scenario4_NumUnitsCutFirstStage(10,1) + 1.0 Scenario4_NumUnitsCutFirstStage(2,1) + 1.0 Scenario4_NumUnitsCutFirstStage(3,1) + 1.0 Scenario4_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(5,1) + 1.0 Scenario4_NumUnitsCutFirstStage(6,1) + 1.0 Scenario4_NumUnitsCutFirstStage(7,1) + 1.0 Scenario4_NumUnitsCutFirstStage(8,1) + 1.0 Scenario4_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario4_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,2) + 1.0 Scenario4_NumUnitsCutFirstStage(2,2) + 1.0 Scenario4_NumUnitsCutFirstStage(3,2) + 1.0 Scenario4_NumUnitsCutFirstStage(4,2) + 1.0 Scenario4_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(6,2) + 1.0 Scenario4_NumUnitsCutFirstStage(7,2) + 1.0 Scenario4_NumUnitsCutFirstStage(8,2) + 1.0 Scenario4_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario4_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,3) + 1.0 Scenario4_NumUnitsCutFirstStage(3,3) + 1.0 Scenario4_NumUnitsCutFirstStage(4,3) + 1.0 Scenario4_NumUnitsCutFirstStage(5,3) + 1.0 Scenario4_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(7,3) + 1.0 Scenario4_NumUnitsCutFirstStage(8,3) + 1.0 Scenario4_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario4_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,4) + 1.0 Scenario4_NumUnitsCutFirstStage(4,4) + 1.0 Scenario4_NumUnitsCutFirstStage(5,4) + 1.0 Scenario4_NumUnitsCutFirstStage(6,4) + 1.0 Scenario4_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(8,4) + 1.0 Scenario4_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario4_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,5) + 1.0 Scenario4_NumUnitsCutFirstStage(5,5) + 1.0 Scenario4_NumUnitsCutFirstStage(6,5) + 1.0 Scenario4_NumUnitsCutFirstStage(7,5) + 1.0 Scenario4_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario4_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,6) + 1.0 Scenario4_NumUnitsCutFirstStage(6,6) + 1.0 Scenario4_NumUnitsCutFirstStage(7,6) + 1.0 Scenario4_NumUnitsCutFirstStage(8,6) + 1.0 Scenario4_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario4_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,7) + 1.0 Scenario4_NumUnitsCutFirstStage(7,7) + 1.0 Scenario4_NumUnitsCutFirstStage(8,7) + 1.0 Scenario4_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario4_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,8) + 1.0 Scenario4_NumUnitsCutFirstStage(8,8) + 1.0 Scenario4_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario4_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,9) + 1.0 Scenario4_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario4_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario4_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario4_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario4_NumProducedFirstStage(1) - 1.0 Scenario4_NumProducedSecondStage(1) + 1.0 Scenario4_NumUnitsCutFirstStage(1,1) + 1.0 Scenario4_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario4_NumProducedFirstStage(2) - 1.0 Scenario4_NumProducedSecondStage(2) + 1.0 Scenario4_NumUnitsCutFirstStage(2,1) + 1.0 Scenario4_NumUnitsCutFirstStage(2,2) + 1.0 Scenario4_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario4_NumProducedFirstStage(3) - 1.0 Scenario4_NumProducedSecondStage(3) + 1.0 Scenario4_NumUnitsCutFirstStage(3,1) + 1.0 Scenario4_NumUnitsCutFirstStage(3,2) + 1.0 Scenario4_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(3,1) + 1.0 Scenario4_NumUnitsCutSecondStage(3,2) + 1.0 Scenario4_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario4_NumProducedFirstStage(4) - 1.0 Scenario4_NumProducedSecondStage(4) + 1.0 Scenario4_NumUnitsCutFirstStage(4,1) + 1.0 Scenario4_NumUnitsCutFirstStage(4,2) + 1.0 Scenario4_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(4,4) + 1.0 Scenario4_NumUnitsCutSecondStage(4,1) + 1.0 Scenario4_NumUnitsCutSecondStage(4,2) + 1.0 Scenario4_NumUnitsCutSecondStage(4,3) + 1.0 Scenario4_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario4_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario4_NumProducedFirstStage(5) - 1.0 Scenario4_NumProducedSecondStage(5) + 1.0 Scenario4_NumUnitsCutFirstStage(5,1) + 1.0 Scenario4_NumUnitsCutFirstStage(5,2) + 1.0 Scenario4_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(5,4) + 1.0 Scenario4_NumUnitsCutFirstStage(5,5) + 1.0 Scenario4_NumUnitsCutSecondStage(5,1) + 1.0 Scenario4_NumUnitsCutSecondStage(5,2) + 1.0 Scenario4_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(5,4) + 1.0 Scenario4_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario4_NumProducedFirstStage(6) - 1.0 Scenario4_NumProducedSecondStage(6) + 1.0 Scenario4_NumUnitsCutFirstStage(6,1) + 1.0 Scenario4_NumUnitsCutFirstStage(6,2) + 1.0 Scenario4_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(6,4) + 1.0 Scenario4_NumUnitsCutFirstStage(6,5) + 1.0 Scenario4_NumUnitsCutFirstStage(6,6) + 1.0 Scenario4_NumUnitsCutSecondStage(6,1) + 1.0 Scenario4_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(6,3) + 1.0 Scenario4_NumUnitsCutSecondStage(6,4) + 1.0 Scenario4_NumUnitsCutSecondStage(6,5) + 1.0 Scenario4_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario4_NumProducedFirstStage(7) - 1.0 Scenario4_NumProducedSecondStage(7) + 1.0 Scenario4_NumUnitsCutFirstStage(7,1) + 1.0 Scenario4_NumUnitsCutFirstStage(7,2) + 1.0 Scenario4_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(7,4) + 1.0 Scenario4_NumUnitsCutFirstStage(7,5) + 1.0 Scenario4_NumUnitsCutFirstStage(7,6) + 1.0 Scenario4_NumUnitsCutFirstStage(7,7) + 1.0 Scenario4_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(7,2) + 1.0 Scenario4_NumUnitsCutSecondStage(7,3) + 1.0 Scenario4_NumUnitsCutSecondStage(7,4) + 1.0 Scenario4_NumUnitsCutSecondStage(7,5) + 1.0 Scenario4_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario4_NumProducedFirstStage(8) - 1.0 Scenario4_NumProducedSecondStage(8) + 1.0 Scenario4_NumUnitsCutFirstStage(8,1) + 1.0 Scenario4_NumUnitsCutFirstStage(8,2) + 1.0 Scenario4_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(8,4) + 1.0 Scenario4_NumUnitsCutFirstStage(8,5) + 1.0 Scenario4_NumUnitsCutFirstStage(8,6) + 1.0 Scenario4_NumUnitsCutFirstStage(8,7) + 1.0 Scenario4_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(8,1) + 1.0 Scenario4_NumUnitsCutSecondStage(8,2) + 1.0 Scenario4_NumUnitsCutSecondStage(8,3) + 1.0 Scenario4_NumUnitsCutSecondStage(8,4) + 1.0 Scenario4_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(8,6) + 1.0 Scenario4_NumUnitsCutSecondStage(8,7) + 1.0 Scenario4_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario4_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario4_NumProducedFirstStage(9) - 1.0 Scenario4_NumProducedSecondStage(9) + 1.0 Scenario4_NumUnitsCutFirstStage(9,1) + 1.0 Scenario4_NumUnitsCutFirstStage(9,2) + 1.0 Scenario4_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(9,4) + 1.0 Scenario4_NumUnitsCutFirstStage(9,5) + 1.0 Scenario4_NumUnitsCutFirstStage(9,6) + 1.0 Scenario4_NumUnitsCutFirstStage(9,7) + 1.0 Scenario4_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(9,9) + 1.0 Scenario4_NumUnitsCutSecondStage(9,1) + 1.0 Scenario4_NumUnitsCutSecondStage(9,2) + 1.0 Scenario4_NumUnitsCutSecondStage(9,3) + 1.0 Scenario4_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(9,5) + 1.0 Scenario4_NumUnitsCutSecondStage(9,6) + 1.0 Scenario4_NumUnitsCutSecondStage(9,7) + 1.0 Scenario4_NumUnitsCutSecondStage(9,8) + 1.0 Scenario4_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario4_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario4_NumProducedFirstStage(10) - 1.0 Scenario4_NumProducedSecondStage(10) + 1.0 Scenario4_NumUnitsCutFirstStage(10,10) + 1.0 Scenario4_NumUnitsCutFirstStage(10,1) + 1.0 Scenario4_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(10,3) + 1.0 Scenario4_NumUnitsCutFirstStage(10,4) + 1.0 Scenario4_NumUnitsCutFirstStage(10,5) + 1.0 Scenario4_NumUnitsCutFirstStage(10,6) + 1.0 Scenario4_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario4_NumUnitsCutFirstStage(10,8) + 1.0 Scenario4_NumUnitsCutFirstStage(10,9) + 1.0 Scenario4_NumUnitsCutSecondStage(10,10) + 1.0 Scenario4_NumUnitsCutSecondStage(10,1) + 1.0 Scenario4_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(10,3) + 1.0 Scenario4_NumUnitsCutSecondStage(10,4) + 1.0 Scenario4_NumUnitsCutSecondStage(10,5) + 1.0 Scenario4_NumUnitsCutSecondStage(10,6) + 1.0 Scenario4_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario4_NumUnitsCutSecondStage(10,8) + 1.0 Scenario4_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario4

\ Begin constraint block for scenario Scenario9
Scenario9_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario9_NumUnitsCutSecondStage(1,1) + 1.0 Scenario9_NumUnitsCutSecondStage(10,1) + 1.0 Scenario9_NumUnitsCutSecondStage(2,1) + 1.0 Scenario9_NumUnitsCutSecondStage(3,1) + 1.0 Scenario9_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(5,1) + 1.0 Scenario9_NumUnitsCutSecondStage(6,1) + 1.0 Scenario9_NumUnitsCutSecondStage(7,1) + 1.0 Scenario9_NumUnitsCutSecondStage(8,1) + 1.0 Scenario9_NumUnitsCutSecondStage(9,1) 
>= 3500.0

Scenario9_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,2) + 1.0 Scenario9_NumUnitsCutSecondStage(2,2) + 1.0 Scenario9_NumUnitsCutSecondStage(3,2) + 1.0 Scenario9_NumUnitsCutSecondStage(4,2) + 1.0 Scenario9_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(6,2) + 1.0 Scenario9_NumUnitsCutSecondStage(7,2) + 1.0 Scenario9_NumUnitsCutSecondStage(8,2) + 1.0 Scenario9_NumUnitsCutSecondStage(9,2) >= 10500.0

Scenario9_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,3) + 1.0 Scenario9_NumUnitsCutSecondStage(3,3) + 1.0 Scenario9_NumUnitsCutSecondStage(4,3) + 1.0 Scenario9_NumUnitsCutSecondStage(5,3) + 1.0 Scenario9_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(7,3) + 1.0 Scenario9_NumUnitsCutSecondStage(8,3) + 1.0 Scenario9_NumUnitsCutSecondStage(9,3) >= 17500.0

Scenario9_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,4) + 1.0 Scenario9_NumUnitsCutSecondStage(4,4) + 1.0 Scenario9_NumUnitsCutSecondStage(5,4) + 1.0 Scenario9_NumUnitsCutSecondStage(6,4) + 1.0 Scenario9_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(8,4) + 1.0 Scenario9_NumUnitsCutSecondStage(9,4) >= 14000.0

Scenario9_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,5) + 1.0 Scenario9_NumUnitsCutSecondStage(5,5) + 1.0 Scenario9_NumUnitsCutSecondStage(6,5) + 1.0 Scenario9_NumUnitsCutSecondStage(7,5) + 1.0 Scenario9_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(9,5) >= 49000.0

Scenario9_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,6) + 1.0 Scenario9_NumUnitsCutSecondStage(6,6) + 1.0 Scenario9_NumUnitsCutSecondStage(7,6) + 1.0 Scenario9_NumUnitsCutSecondStage(8,6) + 1.0 Scenario9_NumUnitsCutSecondStage(9,6) 
>= 35000.0

Scenario9_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,7) + 1.0 Scenario9_NumUnitsCutSecondStage(7,7) + 1.0 Scenario9_NumUnitsCutSecondStage(8,7) + 1.0 Scenario9_NumUnitsCutSecondStage(9,7) >= 21000.0

Scenario9_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,8) + 1.0 Scenario9_NumUnitsCutSecondStage(8,8) + 1.0 Scenario9_NumUnitsCutSecondStage(9,8) >= 17500.0

Scenario9_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,9) + 1.0 Scenario9_NumUnitsCutSecondStage(9,9) >= 17500.0

Scenario9_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario9_NumUnitsCutSecondStage(10,10) >= 7000.0

Scenario9_c_e_ComputeSecondStageCost_:  -0.8416 Scenario9_NumProducedSecondStage(10) - 0.748 Scenario9_NumProducedSecondStage(1) - 0.7584 Scenario9_NumProducedSecondStage(2) - 0.7688 Scenario9_NumProducedSecondStage(3) - 0.7792 Scenario9_NumProducedSecondStage(4) 
- 0.7896 Scenario9_NumProducedSecondStage(5) - 0.8 Scenario9_NumProducedSecondStage(6) - 0.8104 Scenario9_NumProducedSecondStage(7) - 0.8208 Scenario9_NumProducedSecondStage(8) - 0.8312 Scenario9_NumProducedSecondStage(9) 
- 0.008 Scenario9_NumUnitsCutSecondStage(10,1) - 0.008 Scenario9_NumUnitsCutSecondStage(10,2) - 0.008 Scenario9_NumUnitsCutSecondStage(10,3) - 0.008 Scenario9_NumUnitsCutSecondStage(10,4) - 0.008 Scenario9_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario9_NumUnitsCutSecondStage(10,6) - 0.008 Scenario9_NumUnitsCutSecondStage(10,7) - 0.008 Scenario9_NumUnitsCutSecondStage(10,8) - 0.008 Scenario9_NumUnitsCutSecondStage(10,9) - 0.008 Scenario9_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario9_NumUnitsCutSecondStage(3,1) - 0.008 Scenario9_NumUnitsCutSecondStage(3,2) - 0.008 Scenario9_NumUnitsCutSecondStage(4,1) - 0.008 Scenario9_NumUnitsCutSecondStage(4,2) - 0.008 Scenario9_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario9_NumUnitsCutSecondStage(5,1) - 0.008 Scenario9_NumUnitsCutSecondStage(5,2) - 0.008 Scenario9_NumUnitsCutSecondStage(5,3) - 0.008 Scenario9_NumUnitsCutSecondStage(5,4) - 0.008 Scenario9_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario9_NumUnitsCutSecondStage(6,2) - 0.008 Scenario9_NumUnitsCutSecondStage(6,3) - 0.008 Scenario9_NumUnitsCutSecondStage(6,4) - 0.008 Scenario9_NumUnitsCutSecondStage(6,5) - 0.008 Scenario9_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario9_NumUnitsCutSecondStage(7,2) - 0.008 Scenario9_NumUnitsCutSecondStage(7,3) - 0.008 Scenario9_NumUnitsCutSecondStage(7,4) - 0.008 Scenario9_NumUnitsCutSecondStage(7,5) - 0.008 Scenario9_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario9_NumUnitsCutSecondStage(8,1) - 0.008 Scenario9_NumUnitsCutSecondStage(8,2) - 0.008 Scenario9_NumUnitsCutSecondStage(8,3) - 0.008 Scenario9_NumUnitsCutSecondStage(8,4) - 0.008 Scenario9_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario9_NumUnitsCutSecondStage(8,6) - 0.008 Scenario9_NumUnitsCutSecondStage(8,7) - 0.008 Scenario9_NumUnitsCutSecondStage(9,1) - 0.008 Scenario9_NumUnitsCutSecondStage(9,2) - 0.008 Scenario9_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario9_NumUnitsCutSecondStage(9,4) - 0.008 Scenario9_NumUnitsCutSecondStage(9,5) - 0.008 Scenario9_NumUnitsCutSecondStage(9,6) - 0.008 Scenario9_NumUnitsCutSecondStage(9,7) - 0.008 Scenario9_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario9_ProduceSizeSecondStage(10) - 453.0 Scenario9_ProduceSizeSecondStage(1) - 453.0 Scenario9_ProduceSizeSecondStage(2) - 453.0 Scenario9_ProduceSizeSecondStage(3) - 453.0 Scenario9_ProduceSizeSecondStage(4) 
- 453.0 Scenario9_ProduceSizeSecondStage(5) - 453.0 Scenario9_ProduceSizeSecondStage(6) - 453.0 Scenario9_ProduceSizeSecondStage(7) - 453.0 Scenario9_ProduceSizeSecondStage(8) - 453.0 Scenario9_ProduceSizeSecondStage(9) 
+ 1.0 Scenario9_SecondStageCost = 0.0

Scenario9_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario9_NumProducedSecondStage(10) + 1.0 Scenario9_NumProducedSecondStage(1) + 1.0 Scenario9_NumProducedSecondStage(2) + 1.0 Scenario9_NumProducedSecondStage(3) + 1.0 Scenario9_NumProducedSecondStage(4) 
+ 1.0 Scenario9_NumProducedSecondStage(5) + 1.0 Scenario9_NumProducedSecondStage(6) + 1.0 Scenario9_NumProducedSecondStage(7) + 1.0 Scenario9_NumProducedSecondStage(8) + 1.0 Scenario9_NumProducedSecondStage(9) 
<= 200000.0

Scenario9_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario9_NumProducedSecondStage(1) - 200000.0 Scenario9_ProduceSizeSecondStage(1) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario9_NumProducedSecondStage(2) - 200000.0 Scenario9_ProduceSizeSecondStage(2) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario9_NumProducedSecondStage(3) - 200000.0 Scenario9_ProduceSizeSecondStage(3) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario9_NumProducedSecondStage(4) - 200000.0 Scenario9_ProduceSizeSecondStage(4) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario9_NumProducedSecondStage(5) - 200000.0 Scenario9_ProduceSizeSecondStage(5) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario9_NumProducedSecondStage(6) - 200000.0 Scenario9_ProduceSizeSecondStage(6) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario9_NumProducedSecondStage(7) - 200000.0 Scenario9_ProduceSizeSecondStage(7) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario9_NumProducedSecondStage(8) - 200000.0 Scenario9_ProduceSizeSecondStage(8) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario9_NumProducedSecondStage(9) - 200000.0 Scenario9_ProduceSizeSecondStage(9) <= 0.0

Scenario9_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario9_NumProducedSecondStage(10) - 200000.0 Scenario9_ProduceSizeSecondStage(10) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario9_NumProducedFirstStage(1) - 200000.0 Scenario9_ProduceSizeFirstStage(1) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario9_NumProducedFirstStage(2) - 200000.0 Scenario9_ProduceSizeFirstStage(2) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario9_NumProducedFirstStage(3) - 200000.0 Scenario9_ProduceSizeFirstStage(3) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario9_NumProducedFirstStage(4) - 200000.0 Scenario9_ProduceSizeFirstStage(4) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario9_NumProducedFirstStage(5) - 200000.0 Scenario9_ProduceSizeFirstStage(5) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario9_NumProducedFirstStage(6) - 200000.0 Scenario9_ProduceSizeFirstStage(6) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario9_NumProducedFirstStage(7) - 200000.0 Scenario9_ProduceSizeFirstStage(7) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario9_NumProducedFirstStage(8) - 200000.0 Scenario9_ProduceSizeFirstStage(8) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario9_NumProducedFirstStage(9) - 200000.0 Scenario9_ProduceSizeFirstStage(9) <= 0.0

Scenario9_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario9_NumProducedFirstStage(10) - 200000.0 Scenario9_ProduceSizeFirstStage(10) <= 0.0

Scenario9_c_e_ComputeFirstStageCost_:  1.0 Scenario9_FirstStageCost - 0.8416 Scenario9_NumProducedFirstStage(10) - 0.748 Scenario9_NumProducedFirstStage(1) - 0.7584 Scenario9_NumProducedFirstStage(2) - 0.7688 Scenario9_NumProducedFirstStage(3) 
- 0.7792 Scenario9_NumProducedFirstStage(4) - 0.7896 Scenario9_NumProducedFirstStage(5) - 0.8 Scenario9_NumProducedFirstStage(6) - 0.8104 Scenario9_NumProducedFirstStage(7) - 0.8208 Scenario9_NumProducedFirstStage(8) 
- 0.8312 Scenario9_NumProducedFirstStage(9) - 0.008 Scenario9_NumUnitsCutFirstStage(10,1) - 0.008 Scenario9_NumUnitsCutFirstStage(10,2) - 0.008 Scenario9_NumUnitsCutFirstStage(10,3) - 0.008 Scenario9_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario9_NumUnitsCutFirstStage(10,5) - 0.008 Scenario9_NumUnitsCutFirstStage(10,6) - 0.008 Scenario9_NumUnitsCutFirstStage(10,7) - 0.008 Scenario9_NumUnitsCutFirstStage(10,8) - 0.008 Scenario9_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario9_NumUnitsCutFirstStage(2,1) - 0.008 Scenario9_NumUnitsCutFirstStage(3,1) - 0.008 Scenario9_NumUnitsCutFirstStage(3,2) - 0.008 Scenario9_NumUnitsCutFirstStage(4,1) - 0.008 Scenario9_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario9_NumUnitsCutFirstStage(4,3) - 0.008 Scenario9_NumUnitsCutFirstStage(5,1) - 0.008 Scenario9_NumUnitsCutFirstStage(5,2) - 0.008 Scenario9_NumUnitsCutFirstStage(5,3) - 0.008 Scenario9_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario9_NumUnitsCutFirstStage(6,1) - 0.008 Scenario9_NumUnitsCutFirstStage(6,2) - 0.008 Scenario9_NumUnitsCutFirstStage(6,3) - 0.008 Scenario9_NumUnitsCutFirstStage(6,4) - 0.008 Scenario9_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario9_NumUnitsCutFirstStage(7,1) - 0.008 Scenario9_NumUnitsCutFirstStage(7,2) - 0.008 Scenario9_NumUnitsCutFirstStage(7,3) - 0.008 Scenario9_NumUnitsCutFirstStage(7,4) - 0.008 Scenario9_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario9_NumUnitsCutFirstStage(7,6) - 0.008 Scenario9_NumUnitsCutFirstStage(8,1) - 0.008 Scenario9_NumUnitsCutFirstStage(8,2) - 0.008 Scenario9_NumUnitsCutFirstStage(8,3) - 0.008 Scenario9_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario9_NumUnitsCutFirstStage(8,5) - 0.008 Scenario9_NumUnitsCutFirstStage(8,6) - 0.008 Scenario9_NumUnitsCutFirstStage(8,7) - 0.008 Scenario9_NumUnitsCutFirstStage(9,1) - 0.008 Scenario9_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario9_NumUnitsCutFirstStage(9,3) - 0.008 Scenario9_NumUnitsCutFirstStage(9,4) - 0.008 Scenario9_NumUnitsCutFirstStage(9,5) - 0.008 Scenario9_NumUnitsCutFirstStage(9,6) - 0.008 Scenario9_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario9_NumUnitsCutFirstStage(9,8) - 453.0 Scenario9_ProduceSizeFirstStage(10) - 453.0 Scenario9_ProduceSizeFirstStage(1) - 453.0 Scenario9_ProduceSizeFirstStage(2) - 453.0 Scenario9_ProduceSizeFirstStage(3) 
- 453.0 Scenario9_ProduceSizeFirstStage(4) - 453.0 Scenario9_ProduceSizeFirstStage(5) - 453.0 Scenario9_ProduceSizeFirstStage(6) - 453.0 Scenario9_ProduceSizeFirstStage(7) - 453.0 Scenario9_ProduceSizeFirstStage(8) 
- 453.0 Scenario9_ProduceSizeFirstStage(9) = 0.0

Scenario9_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario9_NumProducedFirstStage(1) + 1.0 Scenario9_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario9_NumProducedFirstStage(2) + 1.0 Scenario9_NumUnitsCutFirstStage(2,1) + 1.0 Scenario9_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario9_NumProducedFirstStage(3) + 1.0 Scenario9_NumUnitsCutFirstStage(3,1) + 1.0 Scenario9_NumUnitsCutFirstStage(3,2) + 1.0 Scenario9_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario9_NumProducedFirstStage(4) + 1.0 Scenario9_NumUnitsCutFirstStage(4,1) + 1.0 Scenario9_NumUnitsCutFirstStage(4,2) + 1.0 Scenario9_NumUnitsCutFirstStage(4,3) + 1.0 Scenario9_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario9_NumProducedFirstStage(5) + 1.0 Scenario9_NumUnitsCutFirstStage(5,1) + 1.0 Scenario9_NumUnitsCutFirstStage(5,2) + 1.0 Scenario9_NumUnitsCutFirstStage(5,3) + 1.0 Scenario9_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario9_NumProducedFirstStage(6) + 1.0 Scenario9_NumUnitsCutFirstStage(6,1) + 1.0 Scenario9_NumUnitsCutFirstStage(6,2) + 1.0 Scenario9_NumUnitsCutFirstStage(6,3) + 1.0 Scenario9_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(6,5) + 1.0 Scenario9_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario9_NumProducedFirstStage(7) + 1.0 Scenario9_NumUnitsCutFirstStage(7,1) + 1.0 Scenario9_NumUnitsCutFirstStage(7,2) + 1.0 Scenario9_NumUnitsCutFirstStage(7,3) + 1.0 Scenario9_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(7,5) + 1.0 Scenario9_NumUnitsCutFirstStage(7,6) + 1.0 Scenario9_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario9_NumProducedFirstStage(8) + 1.0 Scenario9_NumUnitsCutFirstStage(8,1) + 1.0 Scenario9_NumUnitsCutFirstStage(8,2) + 1.0 Scenario9_NumUnitsCutFirstStage(8,3) + 1.0 Scenario9_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(8,5) + 1.0 Scenario9_NumUnitsCutFirstStage(8,6) + 1.0 Scenario9_NumUnitsCutFirstStage(8,7) + 1.0 Scenario9_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario9_NumProducedFirstStage(9) + 1.0 Scenario9_NumUnitsCutFirstStage(9,1) + 1.0 Scenario9_NumUnitsCutFirstStage(9,2) + 1.0 Scenario9_NumUnitsCutFirstStage(9,3) + 1.0 Scenario9_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(9,5) + 1.0 Scenario9_NumUnitsCutFirstStage(9,6) + 1.0 Scenario9_NumUnitsCutFirstStage(9,7) + 1.0 Scenario9_NumUnitsCutFirstStage(9,8) + 1.0 Scenario9_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario9_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario9_NumProducedFirstStage(10) + 1.0 Scenario9_NumUnitsCutFirstStage(10,10) + 1.0 Scenario9_NumUnitsCutFirstStage(10,1) + 1.0 Scenario9_NumUnitsCutFirstStage(10,2) + 1.0 Scenario9_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(10,4) + 1.0 Scenario9_NumUnitsCutFirstStage(10,5) + 1.0 Scenario9_NumUnitsCutFirstStage(10,6) + 1.0 Scenario9_NumUnitsCutFirstStage(10,7) + 1.0 Scenario9_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario9_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario9_NumProducedFirstStage(10) + 1.0 Scenario9_NumProducedFirstStage(1) + 1.0 Scenario9_NumProducedFirstStage(2) + 1.0 Scenario9_NumProducedFirstStage(3) + 1.0 Scenario9_NumProducedFirstStage(4) 
+ 1.0 Scenario9_NumProducedFirstStage(5) + 1.0 Scenario9_NumProducedFirstStage(6) + 1.0 Scenario9_NumProducedFirstStage(7) + 1.0 Scenario9_NumProducedFirstStage(8) + 1.0 Scenario9_NumProducedFirstStage(9) 
<= 200000.0

Scenario9_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario9_NumUnitsCutFirstStage(1,1) + 1.0 Scenario9_NumUnitsCutFirstStage(10,1) + 1.0 Scenario9_NumUnitsCutFirstStage(2,1) + 1.0 Scenario9_NumUnitsCutFirstStage(3,1) + 1.0 Scenario9_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(5,1) + 1.0 Scenario9_NumUnitsCutFirstStage(6,1) + 1.0 Scenario9_NumUnitsCutFirstStage(7,1) + 1.0 Scenario9_NumUnitsCutFirstStage(8,1) + 1.0 Scenario9_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario9_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,2) + 1.0 Scenario9_NumUnitsCutFirstStage(2,2) + 1.0 Scenario9_NumUnitsCutFirstStage(3,2) + 1.0 Scenario9_NumUnitsCutFirstStage(4,2) + 1.0 Scenario9_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(6,2) + 1.0 Scenario9_NumUnitsCutFirstStage(7,2) + 1.0 Scenario9_NumUnitsCutFirstStage(8,2) + 1.0 Scenario9_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario9_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,3) + 1.0 Scenario9_NumUnitsCutFirstStage(3,3) + 1.0 Scenario9_NumUnitsCutFirstStage(4,3) + 1.0 Scenario9_NumUnitsCutFirstStage(5,3) + 1.0 Scenario9_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(7,3) + 1.0 Scenario9_NumUnitsCutFirstStage(8,3) + 1.0 Scenario9_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario9_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,4) + 1.0 Scenario9_NumUnitsCutFirstStage(4,4) + 1.0 Scenario9_NumUnitsCutFirstStage(5,4) + 1.0 Scenario9_NumUnitsCutFirstStage(6,4) + 1.0 Scenario9_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(8,4) + 1.0 Scenario9_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario9_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,5) + 1.0 Scenario9_NumUnitsCutFirstStage(5,5) + 1.0 Scenario9_NumUnitsCutFirstStage(6,5) + 1.0 Scenario9_NumUnitsCutFirstStage(7,5) + 1.0 Scenario9_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario9_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,6) + 1.0 Scenario9_NumUnitsCutFirstStage(6,6) + 1.0 Scenario9_NumUnitsCutFirstStage(7,6) + 1.0 Scenario9_NumUnitsCutFirstStage(8,6) + 1.0 Scenario9_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario9_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,7) + 1.0 Scenario9_NumUnitsCutFirstStage(7,7) + 1.0 Scenario9_NumUnitsCutFirstStage(8,7) + 1.0 Scenario9_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario9_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,8) + 1.0 Scenario9_NumUnitsCutFirstStage(8,8) + 1.0 Scenario9_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario9_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,9) + 1.0 Scenario9_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario9_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario9_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario9_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario9_NumProducedFirstStage(1) - 1.0 Scenario9_NumProducedSecondStage(1) + 1.0 Scenario9_NumUnitsCutFirstStage(1,1) + 1.0 Scenario9_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario9_NumProducedFirstStage(2) - 1.0 Scenario9_NumProducedSecondStage(2) + 1.0 Scenario9_NumUnitsCutFirstStage(2,1) + 1.0 Scenario9_NumUnitsCutFirstStage(2,2) + 1.0 Scenario9_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario9_NumProducedFirstStage(3) - 1.0 Scenario9_NumProducedSecondStage(3) + 1.0 Scenario9_NumUnitsCutFirstStage(3,1) + 1.0 Scenario9_NumUnitsCutFirstStage(3,2) + 1.0 Scenario9_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(3,1) + 1.0 Scenario9_NumUnitsCutSecondStage(3,2) + 1.0 Scenario9_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario9_NumProducedFirstStage(4) - 1.0 Scenario9_NumProducedSecondStage(4) + 1.0 Scenario9_NumUnitsCutFirstStage(4,1) + 1.0 Scenario9_NumUnitsCutFirstStage(4,2) + 1.0 Scenario9_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(4,4) + 1.0 Scenario9_NumUnitsCutSecondStage(4,1) + 1.0 Scenario9_NumUnitsCutSecondStage(4,2) + 1.0 Scenario9_NumUnitsCutSecondStage(4,3) + 1.0 Scenario9_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario9_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario9_NumProducedFirstStage(5) - 1.0 Scenario9_NumProducedSecondStage(5) + 1.0 Scenario9_NumUnitsCutFirstStage(5,1) + 1.0 Scenario9_NumUnitsCutFirstStage(5,2) + 1.0 Scenario9_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(5,4) + 1.0 Scenario9_NumUnitsCutFirstStage(5,5) + 1.0 Scenario9_NumUnitsCutSecondStage(5,1) + 1.0 Scenario9_NumUnitsCutSecondStage(5,2) + 1.0 Scenario9_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(5,4) + 1.0 Scenario9_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario9_NumProducedFirstStage(6) - 1.0 Scenario9_NumProducedSecondStage(6) + 1.0 Scenario9_NumUnitsCutFirstStage(6,1) + 1.0 Scenario9_NumUnitsCutFirstStage(6,2) + 1.0 Scenario9_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(6,4) + 1.0 Scenario9_NumUnitsCutFirstStage(6,5) + 1.0 Scenario9_NumUnitsCutFirstStage(6,6) + 1.0 Scenario9_NumUnitsCutSecondStage(6,1) + 1.0 Scenario9_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(6,3) + 1.0 Scenario9_NumUnitsCutSecondStage(6,4) + 1.0 Scenario9_NumUnitsCutSecondStage(6,5) + 1.0 Scenario9_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario9_NumProducedFirstStage(7) - 1.0 Scenario9_NumProducedSecondStage(7) + 1.0 Scenario9_NumUnitsCutFirstStage(7,1) + 1.0 Scenario9_NumUnitsCutFirstStage(7,2) + 1.0 Scenario9_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(7,4) + 1.0 Scenario9_NumUnitsCutFirstStage(7,5) + 1.0 Scenario9_NumUnitsCutFirstStage(7,6) + 1.0 Scenario9_NumUnitsCutFirstStage(7,7) + 1.0 Scenario9_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(7,2) + 1.0 Scenario9_NumUnitsCutSecondStage(7,3) + 1.0 Scenario9_NumUnitsCutSecondStage(7,4) + 1.0 Scenario9_NumUnitsCutSecondStage(7,5) + 1.0 Scenario9_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario9_NumProducedFirstStage(8) - 1.0 Scenario9_NumProducedSecondStage(8) + 1.0 Scenario9_NumUnitsCutFirstStage(8,1) + 1.0 Scenario9_NumUnitsCutFirstStage(8,2) + 1.0 Scenario9_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(8,4) + 1.0 Scenario9_NumUnitsCutFirstStage(8,5) + 1.0 Scenario9_NumUnitsCutFirstStage(8,6) + 1.0 Scenario9_NumUnitsCutFirstStage(8,7) + 1.0 Scenario9_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(8,1) + 1.0 Scenario9_NumUnitsCutSecondStage(8,2) + 1.0 Scenario9_NumUnitsCutSecondStage(8,3) + 1.0 Scenario9_NumUnitsCutSecondStage(8,4) + 1.0 Scenario9_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(8,6) + 1.0 Scenario9_NumUnitsCutSecondStage(8,7) + 1.0 Scenario9_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario9_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario9_NumProducedFirstStage(9) - 1.0 Scenario9_NumProducedSecondStage(9) + 1.0 Scenario9_NumUnitsCutFirstStage(9,1) + 1.0 Scenario9_NumUnitsCutFirstStage(9,2) + 1.0 Scenario9_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(9,4) + 1.0 Scenario9_NumUnitsCutFirstStage(9,5) + 1.0 Scenario9_NumUnitsCutFirstStage(9,6) + 1.0 Scenario9_NumUnitsCutFirstStage(9,7) + 1.0 Scenario9_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(9,9) + 1.0 Scenario9_NumUnitsCutSecondStage(9,1) + 1.0 Scenario9_NumUnitsCutSecondStage(9,2) + 1.0 Scenario9_NumUnitsCutSecondStage(9,3) + 1.0 Scenario9_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(9,5) + 1.0 Scenario9_NumUnitsCutSecondStage(9,6) + 1.0 Scenario9_NumUnitsCutSecondStage(9,7) + 1.0 Scenario9_NumUnitsCutSecondStage(9,8) + 1.0 Scenario9_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario9_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario9_NumProducedFirstStage(10) - 1.0 Scenario9_NumProducedSecondStage(10) + 1.0 Scenario9_NumUnitsCutFirstStage(10,10) + 1.0 Scenario9_NumUnitsCutFirstStage(10,1) + 1.0 Scenario9_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(10,3) + 1.0 Scenario9_NumUnitsCutFirstStage(10,4) + 1.0 Scenario9_NumUnitsCutFirstStage(10,5) + 1.0 Scenario9_NumUnitsCutFirstStage(10,6) + 1.0 Scenario9_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario9_NumUnitsCutFirstStage(10,8) + 1.0 Scenario9_NumUnitsCutFirstStage(10,9) + 1.0 Scenario9_NumUnitsCutSecondStage(10,10) + 1.0 Scenario9_NumUnitsCutSecondStage(10,1) + 1.0 Scenario9_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(10,3) + 1.0 Scenario9_NumUnitsCutSecondStage(10,4) + 1.0 Scenario9_NumUnitsCutSecondStage(10,5) + 1.0 Scenario9_NumUnitsCutSecondStage(10,6) + 1.0 Scenario9_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario9_NumUnitsCutSecondStage(10,8) + 1.0 Scenario9_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario9

\ Begin constraint block for scenario Scenario8
Scenario8_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario8_NumUnitsCutSecondStage(1,1) + 1.0 Scenario8_NumUnitsCutSecondStage(10,1) + 1.0 Scenario8_NumUnitsCutSecondStage(2,1) + 1.0 Scenario8_NumUnitsCutSecondStage(3,1) + 1.0 Scenario8_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(5,1) + 1.0 Scenario8_NumUnitsCutSecondStage(6,1) + 1.0 Scenario8_NumUnitsCutSecondStage(7,1) + 1.0 Scenario8_NumUnitsCutSecondStage(8,1) + 1.0 Scenario8_NumUnitsCutSecondStage(9,1) 
>= 3250.0

Scenario8_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,2) + 1.0 Scenario8_NumUnitsCutSecondStage(2,2) + 1.0 Scenario8_NumUnitsCutSecondStage(3,2) + 1.0 Scenario8_NumUnitsCutSecondStage(4,2) + 1.0 Scenario8_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(6,2) + 1.0 Scenario8_NumUnitsCutSecondStage(7,2) + 1.0 Scenario8_NumUnitsCutSecondStage(8,2) + 1.0 Scenario8_NumUnitsCutSecondStage(9,2) >= 9750.0

Scenario8_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,3) + 1.0 Scenario8_NumUnitsCutSecondStage(3,3) + 1.0 Scenario8_NumUnitsCutSecondStage(4,3) + 1.0 Scenario8_NumUnitsCutSecondStage(5,3) + 1.0 Scenario8_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(7,3) + 1.0 Scenario8_NumUnitsCutSecondStage(8,3) + 1.0 Scenario8_NumUnitsCutSecondStage(9,3) >= 16250.0

Scenario8_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,4) + 1.0 Scenario8_NumUnitsCutSecondStage(4,4) + 1.0 Scenario8_NumUnitsCutSecondStage(5,4) + 1.0 Scenario8_NumUnitsCutSecondStage(6,4) + 1.0 Scenario8_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(8,4) + 1.0 Scenario8_NumUnitsCutSecondStage(9,4) >= 13000.0

Scenario8_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,5) + 1.0 Scenario8_NumUnitsCutSecondStage(5,5) + 1.0 Scenario8_NumUnitsCutSecondStage(6,5) + 1.0 Scenario8_NumUnitsCutSecondStage(7,5) + 1.0 Scenario8_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(9,5) >= 45500.0

Scenario8_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,6) + 1.0 Scenario8_NumUnitsCutSecondStage(6,6) + 1.0 Scenario8_NumUnitsCutSecondStage(7,6) + 1.0 Scenario8_NumUnitsCutSecondStage(8,6) + 1.0 Scenario8_NumUnitsCutSecondStage(9,6) 
>= 32500.0

Scenario8_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,7) + 1.0 Scenario8_NumUnitsCutSecondStage(7,7) + 1.0 Scenario8_NumUnitsCutSecondStage(8,7) + 1.0 Scenario8_NumUnitsCutSecondStage(9,7) >= 19500.0

Scenario8_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,8) + 1.0 Scenario8_NumUnitsCutSecondStage(8,8) + 1.0 Scenario8_NumUnitsCutSecondStage(9,8) >= 16250.0

Scenario8_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,9) + 1.0 Scenario8_NumUnitsCutSecondStage(9,9) >= 16250.0

Scenario8_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario8_NumUnitsCutSecondStage(10,10) >= 6500.0

Scenario8_c_e_ComputeSecondStageCost_:  -0.8416 Scenario8_NumProducedSecondStage(10) - 0.748 Scenario8_NumProducedSecondStage(1) - 0.7584 Scenario8_NumProducedSecondStage(2) - 0.7688 Scenario8_NumProducedSecondStage(3) - 0.7792 Scenario8_NumProducedSecondStage(4) 
- 0.7896 Scenario8_NumProducedSecondStage(5) - 0.8 Scenario8_NumProducedSecondStage(6) - 0.8104 Scenario8_NumProducedSecondStage(7) - 0.8208 Scenario8_NumProducedSecondStage(8) - 0.8312 Scenario8_NumProducedSecondStage(9) 
- 0.008 Scenario8_NumUnitsCutSecondStage(10,1) - 0.008 Scenario8_NumUnitsCutSecondStage(10,2) - 0.008 Scenario8_NumUnitsCutSecondStage(10,3) - 0.008 Scenario8_NumUnitsCutSecondStage(10,4) - 0.008 Scenario8_NumUnitsCutSecondStage(10,5) 
- 0.008 Scenario8_NumUnitsCutSecondStage(10,6) - 0.008 Scenario8_NumUnitsCutSecondStage(10,7) - 0.008 Scenario8_NumUnitsCutSecondStage(10,8) - 0.008 Scenario8_NumUnitsCutSecondStage(10,9) - 0.008 Scenario8_NumUnitsCutSecondStage(2,1) 
- 0.008 Scenario8_NumUnitsCutSecondStage(3,1) - 0.008 Scenario8_NumUnitsCutSecondStage(3,2) - 0.008 Scenario8_NumUnitsCutSecondStage(4,1) - 0.008 Scenario8_NumUnitsCutSecondStage(4,2) - 0.008 Scenario8_NumUnitsCutSecondStage(4,3) 
- 0.008 Scenario8_NumUnitsCutSecondStage(5,1) - 0.008 Scenario8_NumUnitsCutSecondStage(5,2) - 0.008 Scenario8_NumUnitsCutSecondStage(5,3) - 0.008 Scenario8_NumUnitsCutSecondStage(5,4) - 0.008 Scenario8_NumUnitsCutSecondStage(6,1) 
- 0.008 Scenario8_NumUnitsCutSecondStage(6,2) - 0.008 Scenario8_NumUnitsCutSecondStage(6,3) - 0.008 Scenario8_NumUnitsCutSecondStage(6,4) - 0.008 Scenario8_NumUnitsCutSecondStage(6,5) - 0.008 Scenario8_NumUnitsCutSecondStage(7,1) 
- 0.008 Scenario8_NumUnitsCutSecondStage(7,2) - 0.008 Scenario8_NumUnitsCutSecondStage(7,3) - 0.008 Scenario8_NumUnitsCutSecondStage(7,4) - 0.008 Scenario8_NumUnitsCutSecondStage(7,5) - 0.008 Scenario8_NumUnitsCutSecondStage(7,6) 
- 0.008 Scenario8_NumUnitsCutSecondStage(8,1) - 0.008 Scenario8_NumUnitsCutSecondStage(8,2) - 0.008 Scenario8_NumUnitsCutSecondStage(8,3) - 0.008 Scenario8_NumUnitsCutSecondStage(8,4) - 0.008 Scenario8_NumUnitsCutSecondStage(8,5) 
- 0.008 Scenario8_NumUnitsCutSecondStage(8,6) - 0.008 Scenario8_NumUnitsCutSecondStage(8,7) - 0.008 Scenario8_NumUnitsCutSecondStage(9,1) - 0.008 Scenario8_NumUnitsCutSecondStage(9,2) - 0.008 Scenario8_NumUnitsCutSecondStage(9,3) 
- 0.008 Scenario8_NumUnitsCutSecondStage(9,4) - 0.008 Scenario8_NumUnitsCutSecondStage(9,5) - 0.008 Scenario8_NumUnitsCutSecondStage(9,6) - 0.008 Scenario8_NumUnitsCutSecondStage(9,7) - 0.008 Scenario8_NumUnitsCutSecondStage(9,8) 
- 453.0 Scenario8_ProduceSizeSecondStage(10) - 453.0 Scenario8_ProduceSizeSecondStage(1) - 453.0 Scenario8_ProduceSizeSecondStage(2) - 453.0 Scenario8_ProduceSizeSecondStage(3) - 453.0 Scenario8_ProduceSizeSecondStage(4) 
- 453.0 Scenario8_ProduceSizeSecondStage(5) - 453.0 Scenario8_ProduceSizeSecondStage(6) - 453.0 Scenario8_ProduceSizeSecondStage(7) - 453.0 Scenario8_ProduceSizeSecondStage(8) - 453.0 Scenario8_ProduceSizeSecondStage(9) 
+ 1.0 Scenario8_SecondStageCost = 0.0

Scenario8_c_u_EnforceCapacityLimitSecondStage_:  1.0 Scenario8_NumProducedSecondStage(10) + 1.0 Scenario8_NumProducedSecondStage(1) + 1.0 Scenario8_NumProducedSecondStage(2) + 1.0 Scenario8_NumProducedSecondStage(3) + 1.0 Scenario8_NumProducedSecondStage(4) 
+ 1.0 Scenario8_NumProducedSecondStage(5) + 1.0 Scenario8_NumProducedSecondStage(6) + 1.0 Scenario8_NumProducedSecondStage(7) + 1.0 Scenario8_NumProducedSecondStage(8) + 1.0 Scenario8_NumProducedSecondStage(9) 
<= 200000.0

Scenario8_c_u_EnforceProductionBinarySecondStage(1)_:  1.0 Scenario8_NumProducedSecondStage(1) - 200000.0 Scenario8_ProduceSizeSecondStage(1) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(2)_:  1.0 Scenario8_NumProducedSecondStage(2) - 200000.0 Scenario8_ProduceSizeSecondStage(2) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(3)_:  1.0 Scenario8_NumProducedSecondStage(3) - 200000.0 Scenario8_ProduceSizeSecondStage(3) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(4)_:  1.0 Scenario8_NumProducedSecondStage(4) - 200000.0 Scenario8_ProduceSizeSecondStage(4) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(5)_:  1.0 Scenario8_NumProducedSecondStage(5) - 200000.0 Scenario8_ProduceSizeSecondStage(5) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(6)_:  1.0 Scenario8_NumProducedSecondStage(6) - 200000.0 Scenario8_ProduceSizeSecondStage(6) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(7)_:  1.0 Scenario8_NumProducedSecondStage(7) - 200000.0 Scenario8_ProduceSizeSecondStage(7) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(8)_:  1.0 Scenario8_NumProducedSecondStage(8) - 200000.0 Scenario8_ProduceSizeSecondStage(8) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(9)_:  1.0 Scenario8_NumProducedSecondStage(9) - 200000.0 Scenario8_ProduceSizeSecondStage(9) <= 0.0

Scenario8_c_u_EnforceProductionBinarySecondStage(10)_:  1.0 Scenario8_NumProducedSecondStage(10) - 200000.0 Scenario8_ProduceSizeSecondStage(10) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(1)_:  1.0 Scenario8_NumProducedFirstStage(1) - 200000.0 Scenario8_ProduceSizeFirstStage(1) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(2)_:  1.0 Scenario8_NumProducedFirstStage(2) - 200000.0 Scenario8_ProduceSizeFirstStage(2) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(3)_:  1.0 Scenario8_NumProducedFirstStage(3) - 200000.0 Scenario8_ProduceSizeFirstStage(3) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(4)_:  1.0 Scenario8_NumProducedFirstStage(4) - 200000.0 Scenario8_ProduceSizeFirstStage(4) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(5)_:  1.0 Scenario8_NumProducedFirstStage(5) - 200000.0 Scenario8_ProduceSizeFirstStage(5) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(6)_:  1.0 Scenario8_NumProducedFirstStage(6) - 200000.0 Scenario8_ProduceSizeFirstStage(6) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(7)_:  1.0 Scenario8_NumProducedFirstStage(7) - 200000.0 Scenario8_ProduceSizeFirstStage(7) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(8)_:  1.0 Scenario8_NumProducedFirstStage(8) - 200000.0 Scenario8_ProduceSizeFirstStage(8) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(9)_:  1.0 Scenario8_NumProducedFirstStage(9) - 200000.0 Scenario8_ProduceSizeFirstStage(9) <= 0.0

Scenario8_c_u_EnforceProductionBinaryFirstStage(10)_:  1.0 Scenario8_NumProducedFirstStage(10) - 200000.0 Scenario8_ProduceSizeFirstStage(10) <= 0.0

Scenario8_c_e_ComputeFirstStageCost_:  1.0 Scenario8_FirstStageCost - 0.8416 Scenario8_NumProducedFirstStage(10) - 0.748 Scenario8_NumProducedFirstStage(1) - 0.7584 Scenario8_NumProducedFirstStage(2) - 0.7688 Scenario8_NumProducedFirstStage(3) 
- 0.7792 Scenario8_NumProducedFirstStage(4) - 0.7896 Scenario8_NumProducedFirstStage(5) - 0.8 Scenario8_NumProducedFirstStage(6) - 0.8104 Scenario8_NumProducedFirstStage(7) - 0.8208 Scenario8_NumProducedFirstStage(8) 
- 0.8312 Scenario8_NumProducedFirstStage(9) - 0.008 Scenario8_NumUnitsCutFirstStage(10,1) - 0.008 Scenario8_NumUnitsCutFirstStage(10,2) - 0.008 Scenario8_NumUnitsCutFirstStage(10,3) - 0.008 Scenario8_NumUnitsCutFirstStage(10,4) 
- 0.008 Scenario8_NumUnitsCutFirstStage(10,5) - 0.008 Scenario8_NumUnitsCutFirstStage(10,6) - 0.008 Scenario8_NumUnitsCutFirstStage(10,7) - 0.008 Scenario8_NumUnitsCutFirstStage(10,8) - 0.008 Scenario8_NumUnitsCutFirstStage(10,9) 
- 0.008 Scenario8_NumUnitsCutFirstStage(2,1) - 0.008 Scenario8_NumUnitsCutFirstStage(3,1) - 0.008 Scenario8_NumUnitsCutFirstStage(3,2) - 0.008 Scenario8_NumUnitsCutFirstStage(4,1) - 0.008 Scenario8_NumUnitsCutFirstStage(4,2) 
- 0.008 Scenario8_NumUnitsCutFirstStage(4,3) - 0.008 Scenario8_NumUnitsCutFirstStage(5,1) - 0.008 Scenario8_NumUnitsCutFirstStage(5,2) - 0.008 Scenario8_NumUnitsCutFirstStage(5,3) - 0.008 Scenario8_NumUnitsCutFirstStage(5,4) 
- 0.008 Scenario8_NumUnitsCutFirstStage(6,1) - 0.008 Scenario8_NumUnitsCutFirstStage(6,2) - 0.008 Scenario8_NumUnitsCutFirstStage(6,3) - 0.008 Scenario8_NumUnitsCutFirstStage(6,4) - 0.008 Scenario8_NumUnitsCutFirstStage(6,5) 
- 0.008 Scenario8_NumUnitsCutFirstStage(7,1) - 0.008 Scenario8_NumUnitsCutFirstStage(7,2) - 0.008 Scenario8_NumUnitsCutFirstStage(7,3) - 0.008 Scenario8_NumUnitsCutFirstStage(7,4) - 0.008 Scenario8_NumUnitsCutFirstStage(7,5) 
- 0.008 Scenario8_NumUnitsCutFirstStage(7,6) - 0.008 Scenario8_NumUnitsCutFirstStage(8,1) - 0.008 Scenario8_NumUnitsCutFirstStage(8,2) - 0.008 Scenario8_NumUnitsCutFirstStage(8,3) - 0.008 Scenario8_NumUnitsCutFirstStage(8,4) 
- 0.008 Scenario8_NumUnitsCutFirstStage(8,5) - 0.008 Scenario8_NumUnitsCutFirstStage(8,6) - 0.008 Scenario8_NumUnitsCutFirstStage(8,7) - 0.008 Scenario8_NumUnitsCutFirstStage(9,1) - 0.008 Scenario8_NumUnitsCutFirstStage(9,2) 
- 0.008 Scenario8_NumUnitsCutFirstStage(9,3) - 0.008 Scenario8_NumUnitsCutFirstStage(9,4) - 0.008 Scenario8_NumUnitsCutFirstStage(9,5) - 0.008 Scenario8_NumUnitsCutFirstStage(9,6) - 0.008 Scenario8_NumUnitsCutFirstStage(9,7) 
- 0.008 Scenario8_NumUnitsCutFirstStage(9,8) - 453.0 Scenario8_ProduceSizeFirstStage(10) - 453.0 Scenario8_ProduceSizeFirstStage(1) - 453.0 Scenario8_ProduceSizeFirstStage(2) - 453.0 Scenario8_ProduceSizeFirstStage(3) 
- 453.0 Scenario8_ProduceSizeFirstStage(4) - 453.0 Scenario8_ProduceSizeFirstStage(5) - 453.0 Scenario8_ProduceSizeFirstStage(6) - 453.0 Scenario8_ProduceSizeFirstStage(7) - 453.0 Scenario8_ProduceSizeFirstStage(8) 
- 453.0 Scenario8_ProduceSizeFirstStage(9) = 0.0

Scenario8_c_u_EnforceInventoryFirstStage(1)_:  -1.0 Scenario8_NumProducedFirstStage(1) + 1.0 Scenario8_NumUnitsCutFirstStage(1,1) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(2)_:  -1.0 Scenario8_NumProducedFirstStage(2) + 1.0 Scenario8_NumUnitsCutFirstStage(2,1) + 1.0 Scenario8_NumUnitsCutFirstStage(2,2) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(3)_:  -1.0 Scenario8_NumProducedFirstStage(3) + 1.0 Scenario8_NumUnitsCutFirstStage(3,1) + 1.0 Scenario8_NumUnitsCutFirstStage(3,2) + 1.0 Scenario8_NumUnitsCutFirstStage(3,3) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(4)_:  -1.0 Scenario8_NumProducedFirstStage(4) + 1.0 Scenario8_NumUnitsCutFirstStage(4,1) + 1.0 Scenario8_NumUnitsCutFirstStage(4,2) + 1.0 Scenario8_NumUnitsCutFirstStage(4,3) + 1.0 Scenario8_NumUnitsCutFirstStage(4,4) 
<= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(5)_:  -1.0 Scenario8_NumProducedFirstStage(5) + 1.0 Scenario8_NumUnitsCutFirstStage(5,1) + 1.0 Scenario8_NumUnitsCutFirstStage(5,2) + 1.0 Scenario8_NumUnitsCutFirstStage(5,3) + 1.0 Scenario8_NumUnitsCutFirstStage(5,4) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(5,5) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(6)_:  -1.0 Scenario8_NumProducedFirstStage(6) + 1.0 Scenario8_NumUnitsCutFirstStage(6,1) + 1.0 Scenario8_NumUnitsCutFirstStage(6,2) + 1.0 Scenario8_NumUnitsCutFirstStage(6,3) + 1.0 Scenario8_NumUnitsCutFirstStage(6,4) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(6,5) + 1.0 Scenario8_NumUnitsCutFirstStage(6,6) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(7)_:  -1.0 Scenario8_NumProducedFirstStage(7) + 1.0 Scenario8_NumUnitsCutFirstStage(7,1) + 1.0 Scenario8_NumUnitsCutFirstStage(7,2) + 1.0 Scenario8_NumUnitsCutFirstStage(7,3) + 1.0 Scenario8_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(7,5) + 1.0 Scenario8_NumUnitsCutFirstStage(7,6) + 1.0 Scenario8_NumUnitsCutFirstStage(7,7) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(8)_:  -1.0 Scenario8_NumProducedFirstStage(8) + 1.0 Scenario8_NumUnitsCutFirstStage(8,1) + 1.0 Scenario8_NumUnitsCutFirstStage(8,2) + 1.0 Scenario8_NumUnitsCutFirstStage(8,3) + 1.0 Scenario8_NumUnitsCutFirstStage(8,4) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(8,5) + 1.0 Scenario8_NumUnitsCutFirstStage(8,6) + 1.0 Scenario8_NumUnitsCutFirstStage(8,7) + 1.0 Scenario8_NumUnitsCutFirstStage(8,8) <= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(9)_:  -1.0 Scenario8_NumProducedFirstStage(9) + 1.0 Scenario8_NumUnitsCutFirstStage(9,1) + 1.0 Scenario8_NumUnitsCutFirstStage(9,2) + 1.0 Scenario8_NumUnitsCutFirstStage(9,3) + 1.0 Scenario8_NumUnitsCutFirstStage(9,4) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(9,5) + 1.0 Scenario8_NumUnitsCutFirstStage(9,6) + 1.0 Scenario8_NumUnitsCutFirstStage(9,7) + 1.0 Scenario8_NumUnitsCutFirstStage(9,8) + 1.0 Scenario8_NumUnitsCutFirstStage(9,9) 
<= 0.0

Scenario8_c_u_EnforceInventoryFirstStage(10)_:  -1.0 Scenario8_NumProducedFirstStage(10) + 1.0 Scenario8_NumUnitsCutFirstStage(10,10) + 1.0 Scenario8_NumUnitsCutFirstStage(10,1) + 1.0 Scenario8_NumUnitsCutFirstStage(10,2) + 1.0 Scenario8_NumUnitsCutFirstStage(10,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(10,4) + 1.0 Scenario8_NumUnitsCutFirstStage(10,5) + 1.0 Scenario8_NumUnitsCutFirstStage(10,6) + 1.0 Scenario8_NumUnitsCutFirstStage(10,7) + 1.0 Scenario8_NumUnitsCutFirstStage(10,8) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(10,9) <= 0.0

Scenario8_c_u_EnforceCapacityLimitFirstStage_:  1.0 Scenario8_NumProducedFirstStage(10) + 1.0 Scenario8_NumProducedFirstStage(1) + 1.0 Scenario8_NumProducedFirstStage(2) + 1.0 Scenario8_NumProducedFirstStage(3) + 1.0 Scenario8_NumProducedFirstStage(4) 
+ 1.0 Scenario8_NumProducedFirstStage(5) + 1.0 Scenario8_NumProducedFirstStage(6) + 1.0 Scenario8_NumProducedFirstStage(7) + 1.0 Scenario8_NumProducedFirstStage(8) + 1.0 Scenario8_NumProducedFirstStage(9) 
<= 200000.0

Scenario8_c_l_DemandSatisfiedFirstStage(1)_:  1.0 Scenario8_NumUnitsCutFirstStage(1,1) + 1.0 Scenario8_NumUnitsCutFirstStage(10,1) + 1.0 Scenario8_NumUnitsCutFirstStage(2,1) + 1.0 Scenario8_NumUnitsCutFirstStage(3,1) + 1.0 Scenario8_NumUnitsCutFirstStage(4,1) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(5,1) + 1.0 Scenario8_NumUnitsCutFirstStage(6,1) + 1.0 Scenario8_NumUnitsCutFirstStage(7,1) + 1.0 Scenario8_NumUnitsCutFirstStage(8,1) + 1.0 Scenario8_NumUnitsCutFirstStage(9,1) 
>= 2500.0

Scenario8_c_l_DemandSatisfiedFirstStage(2)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,2) + 1.0 Scenario8_NumUnitsCutFirstStage(2,2) + 1.0 Scenario8_NumUnitsCutFirstStage(3,2) + 1.0 Scenario8_NumUnitsCutFirstStage(4,2) + 1.0 Scenario8_NumUnitsCutFirstStage(5,2) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(6,2) + 1.0 Scenario8_NumUnitsCutFirstStage(7,2) + 1.0 Scenario8_NumUnitsCutFirstStage(8,2) + 1.0 Scenario8_NumUnitsCutFirstStage(9,2) >= 7500.0

Scenario8_c_l_DemandSatisfiedFirstStage(3)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,3) + 1.0 Scenario8_NumUnitsCutFirstStage(3,3) + 1.0 Scenario8_NumUnitsCutFirstStage(4,3) + 1.0 Scenario8_NumUnitsCutFirstStage(5,3) + 1.0 Scenario8_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(7,3) + 1.0 Scenario8_NumUnitsCutFirstStage(8,3) + 1.0 Scenario8_NumUnitsCutFirstStage(9,3) >= 12500.0

Scenario8_c_l_DemandSatisfiedFirstStage(4)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,4) + 1.0 Scenario8_NumUnitsCutFirstStage(4,4) + 1.0 Scenario8_NumUnitsCutFirstStage(5,4) + 1.0 Scenario8_NumUnitsCutFirstStage(6,4) + 1.0 Scenario8_NumUnitsCutFirstStage(7,4) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(8,4) + 1.0 Scenario8_NumUnitsCutFirstStage(9,4) >= 10000.0

Scenario8_c_l_DemandSatisfiedFirstStage(5)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,5) + 1.0 Scenario8_NumUnitsCutFirstStage(5,5) + 1.0 Scenario8_NumUnitsCutFirstStage(6,5) + 1.0 Scenario8_NumUnitsCutFirstStage(7,5) + 1.0 Scenario8_NumUnitsCutFirstStage(8,5) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(9,5) >= 35000.0

Scenario8_c_l_DemandSatisfiedFirstStage(6)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,6) + 1.0 Scenario8_NumUnitsCutFirstStage(6,6) + 1.0 Scenario8_NumUnitsCutFirstStage(7,6) + 1.0 Scenario8_NumUnitsCutFirstStage(8,6) + 1.0 Scenario8_NumUnitsCutFirstStage(9,6) 
>= 25000.0

Scenario8_c_l_DemandSatisfiedFirstStage(7)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,7) + 1.0 Scenario8_NumUnitsCutFirstStage(7,7) + 1.0 Scenario8_NumUnitsCutFirstStage(8,7) + 1.0 Scenario8_NumUnitsCutFirstStage(9,7) >= 15000.0

Scenario8_c_l_DemandSatisfiedFirstStage(8)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,8) + 1.0 Scenario8_NumUnitsCutFirstStage(8,8) + 1.0 Scenario8_NumUnitsCutFirstStage(9,8) >= 12500.0

Scenario8_c_l_DemandSatisfiedFirstStage(9)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,9) + 1.0 Scenario8_NumUnitsCutFirstStage(9,9) >= 12500.0

Scenario8_c_l_DemandSatisfiedFirstStage(10)_:  1.0 Scenario8_NumUnitsCutFirstStage(10,10) >= 5000.0

Scenario8_c_u_EnforceInventorySecondStage(1)_:  -1.0 Scenario8_NumProducedFirstStage(1) - 1.0 Scenario8_NumProducedSecondStage(1) + 1.0 Scenario8_NumUnitsCutFirstStage(1,1) + 1.0 Scenario8_NumUnitsCutSecondStage(1,1) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(2)_:  -1.0 Scenario8_NumProducedFirstStage(2) - 1.0 Scenario8_NumProducedSecondStage(2) + 1.0 Scenario8_NumUnitsCutFirstStage(2,1) + 1.0 Scenario8_NumUnitsCutFirstStage(2,2) + 1.0 Scenario8_NumUnitsCutSecondStage(2,1) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(2,2) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(3)_:  -1.0 Scenario8_NumProducedFirstStage(3) - 1.0 Scenario8_NumProducedSecondStage(3) + 1.0 Scenario8_NumUnitsCutFirstStage(3,1) + 1.0 Scenario8_NumUnitsCutFirstStage(3,2) + 1.0 Scenario8_NumUnitsCutFirstStage(3,3) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(3,1) + 1.0 Scenario8_NumUnitsCutSecondStage(3,2) + 1.0 Scenario8_NumUnitsCutSecondStage(3,3) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(4)_:  -1.0 Scenario8_NumProducedFirstStage(4) - 1.0 Scenario8_NumProducedSecondStage(4) + 1.0 Scenario8_NumUnitsCutFirstStage(4,1) + 1.0 Scenario8_NumUnitsCutFirstStage(4,2) + 1.0 Scenario8_NumUnitsCutFirstStage(4,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(4,4) + 1.0 Scenario8_NumUnitsCutSecondStage(4,1) + 1.0 Scenario8_NumUnitsCutSecondStage(4,2) + 1.0 Scenario8_NumUnitsCutSecondStage(4,3) + 1.0 Scenario8_NumUnitsCutSecondStage(4,4) 
<= 0.0

Scenario8_c_u_EnforceInventorySecondStage(5)_:  -1.0 Scenario8_NumProducedFirstStage(5) - 1.0 Scenario8_NumProducedSecondStage(5) + 1.0 Scenario8_NumUnitsCutFirstStage(5,1) + 1.0 Scenario8_NumUnitsCutFirstStage(5,2) + 1.0 Scenario8_NumUnitsCutFirstStage(5,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(5,4) + 1.0 Scenario8_NumUnitsCutFirstStage(5,5) + 1.0 Scenario8_NumUnitsCutSecondStage(5,1) + 1.0 Scenario8_NumUnitsCutSecondStage(5,2) + 1.0 Scenario8_NumUnitsCutSecondStage(5,3) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(5,4) + 1.0 Scenario8_NumUnitsCutSecondStage(5,5) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(6)_:  -1.0 Scenario8_NumProducedFirstStage(6) - 1.0 Scenario8_NumProducedSecondStage(6) + 1.0 Scenario8_NumUnitsCutFirstStage(6,1) + 1.0 Scenario8_NumUnitsCutFirstStage(6,2) + 1.0 Scenario8_NumUnitsCutFirstStage(6,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(6,4) + 1.0 Scenario8_NumUnitsCutFirstStage(6,5) + 1.0 Scenario8_NumUnitsCutFirstStage(6,6) + 1.0 Scenario8_NumUnitsCutSecondStage(6,1) + 1.0 Scenario8_NumUnitsCutSecondStage(6,2) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(6,3) + 1.0 Scenario8_NumUnitsCutSecondStage(6,4) + 1.0 Scenario8_NumUnitsCutSecondStage(6,5) + 1.0 Scenario8_NumUnitsCutSecondStage(6,6) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(7)_:  -1.0 Scenario8_NumProducedFirstStage(7) - 1.0 Scenario8_NumProducedSecondStage(7) + 1.0 Scenario8_NumUnitsCutFirstStage(7,1) + 1.0 Scenario8_NumUnitsCutFirstStage(7,2) + 1.0 Scenario8_NumUnitsCutFirstStage(7,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(7,4) + 1.0 Scenario8_NumUnitsCutFirstStage(7,5) + 1.0 Scenario8_NumUnitsCutFirstStage(7,6) + 1.0 Scenario8_NumUnitsCutFirstStage(7,7) + 1.0 Scenario8_NumUnitsCutSecondStage(7,1) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(7,2) + 1.0 Scenario8_NumUnitsCutSecondStage(7,3) + 1.0 Scenario8_NumUnitsCutSecondStage(7,4) + 1.0 Scenario8_NumUnitsCutSecondStage(7,5) + 1.0 Scenario8_NumUnitsCutSecondStage(7,6) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(7,7) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(8)_:  -1.0 Scenario8_NumProducedFirstStage(8) - 1.0 Scenario8_NumProducedSecondStage(8) + 1.0 Scenario8_NumUnitsCutFirstStage(8,1) + 1.0 Scenario8_NumUnitsCutFirstStage(8,2) + 1.0 Scenario8_NumUnitsCutFirstStage(8,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(8,4) + 1.0 Scenario8_NumUnitsCutFirstStage(8,5) + 1.0 Scenario8_NumUnitsCutFirstStage(8,6) + 1.0 Scenario8_NumUnitsCutFirstStage(8,7) + 1.0 Scenario8_NumUnitsCutFirstStage(8,8) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(8,1) + 1.0 Scenario8_NumUnitsCutSecondStage(8,2) + 1.0 Scenario8_NumUnitsCutSecondStage(8,3) + 1.0 Scenario8_NumUnitsCutSecondStage(8,4) + 1.0 Scenario8_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(8,6) + 1.0 Scenario8_NumUnitsCutSecondStage(8,7) + 1.0 Scenario8_NumUnitsCutSecondStage(8,8) <= 0.0

Scenario8_c_u_EnforceInventorySecondStage(9)_:  -1.0 Scenario8_NumProducedFirstStage(9) - 1.0 Scenario8_NumProducedSecondStage(9) + 1.0 Scenario8_NumUnitsCutFirstStage(9,1) + 1.0 Scenario8_NumUnitsCutFirstStage(9,2) + 1.0 Scenario8_NumUnitsCutFirstStage(9,3) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(9,4) + 1.0 Scenario8_NumUnitsCutFirstStage(9,5) + 1.0 Scenario8_NumUnitsCutFirstStage(9,6) + 1.0 Scenario8_NumUnitsCutFirstStage(9,7) + 1.0 Scenario8_NumUnitsCutFirstStage(9,8) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(9,9) + 1.0 Scenario8_NumUnitsCutSecondStage(9,1) + 1.0 Scenario8_NumUnitsCutSecondStage(9,2) + 1.0 Scenario8_NumUnitsCutSecondStage(9,3) + 1.0 Scenario8_NumUnitsCutSecondStage(9,4) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(9,5) + 1.0 Scenario8_NumUnitsCutSecondStage(9,6) + 1.0 Scenario8_NumUnitsCutSecondStage(9,7) + 1.0 Scenario8_NumUnitsCutSecondStage(9,8) + 1.0 Scenario8_NumUnitsCutSecondStage(9,9) 
<= 0.0

Scenario8_c_u_EnforceInventorySecondStage(10)_:  -1.0 Scenario8_NumProducedFirstStage(10) - 1.0 Scenario8_NumProducedSecondStage(10) + 1.0 Scenario8_NumUnitsCutFirstStage(10,10) + 1.0 Scenario8_NumUnitsCutFirstStage(10,1) + 1.0 Scenario8_NumUnitsCutFirstStage(10,2) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(10,3) + 1.0 Scenario8_NumUnitsCutFirstStage(10,4) + 1.0 Scenario8_NumUnitsCutFirstStage(10,5) + 1.0 Scenario8_NumUnitsCutFirstStage(10,6) + 1.0 Scenario8_NumUnitsCutFirstStage(10,7) 
+ 1.0 Scenario8_NumUnitsCutFirstStage(10,8) + 1.0 Scenario8_NumUnitsCutFirstStage(10,9) + 1.0 Scenario8_NumUnitsCutSecondStage(10,10) + 1.0 Scenario8_NumUnitsCutSecondStage(10,1) + 1.0 Scenario8_NumUnitsCutSecondStage(10,2) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(10,3) + 1.0 Scenario8_NumUnitsCutSecondStage(10,4) + 1.0 Scenario8_NumUnitsCutSecondStage(10,5) + 1.0 Scenario8_NumUnitsCutSecondStage(10,6) + 1.0 Scenario8_NumUnitsCutSecondStage(10,7) 
+ 1.0 Scenario8_NumUnitsCutSecondStage(10,8) + 1.0 Scenario8_NumUnitsCutSecondStage(10,9) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario Scenario8

bounds

\ Begin variable bounds block for master
     -inf <=  MASTER_EXPECTED_COST_Scenario1Node  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(1)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(2)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(3)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(4)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(5)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(6)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(7)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(8)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(9)  <= +inf
    0 <=  MASTER_RootNode_NumProducedFirstStage(10)  <= +inf
    0 <=  MASTER_Scenario2Node_SecondStageCost  <= +inf
    0 <=  MASTER_Scenario8Node_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario8Node  <= +inf
    0 <=  MASTER_Scenario9Node_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario2Node  <= +inf
    0 <=  MASTER_Scenario7Node_SecondStageCost  <= +inf
    0 <=  MASTER_RootNode_FirstStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario9Node  <= +inf
    0 <=  MASTER_Scenario3Node_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario6Node  <= +inf
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(1)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(2)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(3)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(4)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(5)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(6)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(7)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(8)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(9)  <= 1 
    0 <=  MASTER_RootNode_ProduceSizeFirstStage(10)  <= 1 
     -inf <=  MASTER_EXPECTED_COST_RootNode  <= +inf
    0 <=  MASTER_Scenario1Node_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario10Node  <= +inf
    0 <=  MASTER_Scenario10Node_SecondStageCost  <= +inf
    0 <=  MASTER_Scenario6Node_SecondStageCost  <= +inf
    0 <=  MASTER_Scenario5Node_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario4Node  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario3Node  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  MASTER_RootNode_NumUnitsCutFirstStage(10,2)  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario7Node  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario5Node  <= +inf
    0 <=  MASTER_Scenario4Node_SecondStageCost  <= +inf
\ End variable bounds block for master

\ Begin variable bounds block for scenario Scenario10
    0 <=  Scenario10_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario10_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario10_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario10_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario10_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario10_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario10_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario10_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario10_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario10_FirstStageCost  <= +inf
    0 <=  Scenario10_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario10

\ Begin variable bounds block for scenario Scenario3
    0 <=  Scenario3_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario3_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario3_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario3_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario3_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario3_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario3_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario3_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario3_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario3_FirstStageCost  <= +inf
    0 <=  Scenario3_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario3

\ Begin variable bounds block for scenario Scenario2
    0 <=  Scenario2_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario2_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario2_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario2_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario2_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario2_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario2_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario2_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario2_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario2_FirstStageCost  <= +inf
    0 <=  Scenario2_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario2

\ Begin variable bounds block for scenario Scenario1
    0 <=  Scenario1_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario1_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario1_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario1_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario1_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario1_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario1_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario1_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario1_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario1_FirstStageCost  <= +inf
    0 <=  Scenario1_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario1

\ Begin variable bounds block for scenario Scenario7
    0 <=  Scenario7_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario7_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario7_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario7_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario7_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario7_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario7_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario7_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario7_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario7_FirstStageCost  <= +inf
    0 <=  Scenario7_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario7

\ Begin variable bounds block for scenario Scenario6
    0 <=  Scenario6_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario6_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario6_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario6_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario6_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario6_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario6_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario6_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario6_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario6_FirstStageCost  <= +inf
    0 <=  Scenario6_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario6

\ Begin variable bounds block for scenario Scenario5
    0 <=  Scenario5_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario5_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario5_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario5_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario5_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario5_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario5_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario5_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario5_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario5_FirstStageCost  <= +inf
    0 <=  Scenario5_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario5

\ Begin variable bounds block for scenario Scenario4
    0 <=  Scenario4_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario4_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario4_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario4_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario4_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario4_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario4_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario4_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario4_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario4_FirstStageCost  <= +inf
    0 <=  Scenario4_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario4

\ Begin variable bounds block for scenario Scenario9
    0 <=  Scenario9_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario9_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario9_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario9_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario9_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario9_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario9_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario9_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario9_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario9_FirstStageCost  <= +inf
    0 <=  Scenario9_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario9

\ Begin variable bounds block for scenario Scenario8
    0 <=  Scenario8_ProduceSizeSecondStage(1)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(2)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(3)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(4)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(5)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(6)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(7)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(8)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(9)  <= 1 
    0 <=  Scenario8_ProduceSizeSecondStage(10)  <= 1 
    0 <=  Scenario8_NumProducedFirstStage(1)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(2)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(3)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(4)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(5)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(6)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(7)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(8)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(9)  <= +inf
    0 <=  Scenario8_NumProducedFirstStage(10)  <= +inf
    0 <=  Scenario8_ProduceSizeFirstStage(1)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(2)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(3)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(4)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(5)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(6)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(7)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(8)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(9)  <= 1 
    0 <=  Scenario8_ProduceSizeFirstStage(10)  <= 1 
    0 <=  Scenario8_NumUnitsCutSecondStage(7,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(6,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,8)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(5,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(2,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(6,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(5,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(7,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,8)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(3,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(7,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(4,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(6,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(2,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(5,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(4,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(1,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,9)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(6,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(3,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(7,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(7,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(7,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(4,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(6,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(5,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,10)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(6,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(3,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,9)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(7,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,8)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(4,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(9,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(10,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(5,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutSecondStage(8,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(6,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,8)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(5,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(2,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(6,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(5,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,8)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(3,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(4,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(6,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(2,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(5,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(4,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(1,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,9)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(6,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(3,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(4,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,6)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(6,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(5,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,10)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(6,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(3,1)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,9)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(7,4)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,7)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,8)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(4,3)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(9,5)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(10,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(5,2)  <= +inf
    0 <=  Scenario8_NumUnitsCutFirstStage(8,4)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(1)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(2)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(3)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(4)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(5)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(6)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(7)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(8)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(9)  <= +inf
    0 <=  Scenario8_NumProducedSecondStage(10)  <= +inf
    0 <=  Scenario8_FirstStageCost  <= +inf
    0 <=  Scenario8_SecondStageCost  <= +inf
\ End variable bounds block for scenario Scenario8

integer

\ Begin integer variable block for master
    MASTER_RootNode_NumProducedFirstStage(1)
    MASTER_RootNode_NumProducedFirstStage(2)
    MASTER_RootNode_NumProducedFirstStage(3)
    MASTER_RootNode_NumProducedFirstStage(4)
    MASTER_RootNode_NumProducedFirstStage(5)
    MASTER_RootNode_NumProducedFirstStage(6)
    MASTER_RootNode_NumProducedFirstStage(7)
    MASTER_RootNode_NumProducedFirstStage(8)
    MASTER_RootNode_NumProducedFirstStage(9)
    MASTER_RootNode_NumProducedFirstStage(10)
    MASTER_RootNode_NumUnitsCutFirstStage(7,3)
    MASTER_RootNode_NumUnitsCutFirstStage(9,1)
    MASTER_RootNode_NumUnitsCutFirstStage(6,6)
    MASTER_RootNode_NumUnitsCutFirstStage(9,8)
    MASTER_RootNode_NumUnitsCutFirstStage(10,6)
    MASTER_RootNode_NumUnitsCutFirstStage(3,2)
    MASTER_RootNode_NumUnitsCutFirstStage(2,1)
    MASTER_RootNode_NumUnitsCutFirstStage(6,2)
    MASTER_RootNode_NumUnitsCutFirstStage(9,4)
    MASTER_RootNode_NumUnitsCutFirstStage(5,1)
    MASTER_RootNode_NumUnitsCutFirstStage(10,3)
    MASTER_RootNode_NumUnitsCutFirstStage(7,2)
    MASTER_RootNode_NumUnitsCutFirstStage(8,2)
    MASTER_RootNode_NumUnitsCutFirstStage(10,8)
    MASTER_RootNode_NumUnitsCutFirstStage(8,7)
    MASTER_RootNode_NumUnitsCutFirstStage(3,3)
    MASTER_RootNode_NumUnitsCutFirstStage(8,1)
    MASTER_RootNode_NumUnitsCutFirstStage(8,5)
    MASTER_RootNode_NumUnitsCutFirstStage(7,6)
    MASTER_RootNode_NumUnitsCutFirstStage(4,4)
    MASTER_RootNode_NumUnitsCutFirstStage(6,3)
    MASTER_RootNode_NumUnitsCutFirstStage(2,2)
    MASTER_RootNode_NumUnitsCutFirstStage(8,6)
    MASTER_RootNode_NumUnitsCutFirstStage(4,1)
    MASTER_RootNode_NumUnitsCutFirstStage(1,1)
    MASTER_RootNode_NumUnitsCutFirstStage(10,9)
    MASTER_RootNode_NumUnitsCutFirstStage(9,7)
    MASTER_RootNode_NumUnitsCutFirstStage(6,4)
    MASTER_RootNode_NumUnitsCutFirstStage(5,4)
    MASTER_RootNode_NumUnitsCutFirstStage(10,4)
    MASTER_RootNode_NumUnitsCutFirstStage(7,1)
    MASTER_RootNode_NumUnitsCutFirstStage(9,3)
    MASTER_RootNode_NumUnitsCutFirstStage(5,5)
    MASTER_RootNode_NumUnitsCutFirstStage(7,7)
    MASTER_RootNode_NumUnitsCutFirstStage(7,5)
    MASTER_RootNode_NumUnitsCutFirstStage(10,1)
    MASTER_RootNode_NumUnitsCutFirstStage(4,2)
    MASTER_RootNode_NumUnitsCutFirstStage(9,6)
    MASTER_RootNode_NumUnitsCutFirstStage(6,5)
    MASTER_RootNode_NumUnitsCutFirstStage(5,3)
    MASTER_RootNode_NumUnitsCutFirstStage(8,3)
    MASTER_RootNode_NumUnitsCutFirstStage(10,10)
    MASTER_RootNode_NumUnitsCutFirstStage(9,2)
    MASTER_RootNode_NumUnitsCutFirstStage(10,5)
    MASTER_RootNode_NumUnitsCutFirstStage(3,1)
    MASTER_RootNode_NumUnitsCutFirstStage(9,9)
    MASTER_RootNode_NumUnitsCutFirstStage(7,4)
    MASTER_RootNode_NumUnitsCutFirstStage(10,7)
    MASTER_RootNode_NumUnitsCutFirstStage(8,8)
    MASTER_RootNode_NumUnitsCutFirstStage(4,3)
    MASTER_RootNode_NumUnitsCutFirstStage(9,5)
    MASTER_RootNode_NumUnitsCutFirstStage(8,4)
    MASTER_RootNode_NumUnitsCutFirstStage(5,2)
    MASTER_RootNode_NumUnitsCutFirstStage(6,1)
    MASTER_RootNode_NumUnitsCutFirstStage(10,2)
\ End integer variable block for master

\ Begin integer variable block for scenario Scenario10
    Scenario10_NumProducedFirstStage(1)
    Scenario10_NumProducedFirstStage(2)
    Scenario10_NumProducedFirstStage(3)
    Scenario10_NumProducedFirstStage(4)
    Scenario10_NumProducedFirstStage(5)
    Scenario10_NumProducedFirstStage(6)
    Scenario10_NumProducedFirstStage(7)
    Scenario10_NumProducedFirstStage(8)
    Scenario10_NumProducedFirstStage(9)
    Scenario10_NumProducedFirstStage(10)
    Scenario10_NumUnitsCutSecondStage(7,3)
    Scenario10_NumUnitsCutSecondStage(9,1)
    Scenario10_NumUnitsCutSecondStage(6,6)
    Scenario10_NumUnitsCutSecondStage(9,8)
    Scenario10_NumUnitsCutSecondStage(10,6)
    Scenario10_NumUnitsCutSecondStage(5,4)
    Scenario10_NumUnitsCutSecondStage(2,1)
    Scenario10_NumUnitsCutSecondStage(6,2)
    Scenario10_NumUnitsCutSecondStage(9,4)
    Scenario10_NumUnitsCutSecondStage(5,1)
    Scenario10_NumUnitsCutSecondStage(8,5)
    Scenario10_NumUnitsCutSecondStage(7,2)
    Scenario10_NumUnitsCutSecondStage(10,4)
    Scenario10_NumUnitsCutSecondStage(10,8)
    Scenario10_NumUnitsCutSecondStage(3,3)
    Scenario10_NumUnitsCutSecondStage(8,1)
    Scenario10_NumUnitsCutSecondStage(10,3)
    Scenario10_NumUnitsCutSecondStage(7,6)
    Scenario10_NumUnitsCutSecondStage(4,4)
    Scenario10_NumUnitsCutSecondStage(6,3)
    Scenario10_NumUnitsCutSecondStage(2,2)
    Scenario10_NumUnitsCutSecondStage(10,1)
    Scenario10_NumUnitsCutSecondStage(8,6)
    Scenario10_NumUnitsCutSecondStage(5,3)
    Scenario10_NumUnitsCutSecondStage(4,1)
    Scenario10_NumUnitsCutSecondStage(1,1)
    Scenario10_NumUnitsCutSecondStage(10,9)
    Scenario10_NumUnitsCutSecondStage(9,7)
    Scenario10_NumUnitsCutSecondStage(6,4)
    Scenario10_NumUnitsCutSecondStage(3,2)
    Scenario10_NumUnitsCutSecondStage(8,2)
    Scenario10_NumUnitsCutSecondStage(7,1)
    Scenario10_NumUnitsCutSecondStage(9,3)
    Scenario10_NumUnitsCutSecondStage(8,3)
    Scenario10_NumUnitsCutSecondStage(7,7)
    Scenario10_NumUnitsCutSecondStage(7,5)
    Scenario10_NumUnitsCutSecondStage(8,7)
    Scenario10_NumUnitsCutSecondStage(4,2)
    Scenario10_NumUnitsCutSecondStage(9,6)
    Scenario10_NumUnitsCutSecondStage(6,5)
    Scenario10_NumUnitsCutSecondStage(5,5)
    Scenario10_NumUnitsCutSecondStage(10,5)
    Scenario10_NumUnitsCutSecondStage(10,10)
    Scenario10_NumUnitsCutSecondStage(9,2)
    Scenario10_NumUnitsCutSecondStage(6,1)
    Scenario10_NumUnitsCutSecondStage(3,1)
    Scenario10_NumUnitsCutSecondStage(9,9)
    Scenario10_NumUnitsCutSecondStage(7,4)
    Scenario10_NumUnitsCutSecondStage(10,7)
    Scenario10_NumUnitsCutSecondStage(8,8)
    Scenario10_NumUnitsCutSecondStage(4,3)
    Scenario10_NumUnitsCutSecondStage(9,5)
    Scenario10_NumUnitsCutSecondStage(10,2)
    Scenario10_NumUnitsCutSecondStage(5,2)
    Scenario10_NumUnitsCutSecondStage(8,4)
    Scenario10_NumUnitsCutFirstStage(7,3)
    Scenario10_NumUnitsCutFirstStage(9,1)
    Scenario10_NumUnitsCutFirstStage(6,6)
    Scenario10_NumUnitsCutFirstStage(9,8)
    Scenario10_NumUnitsCutFirstStage(10,6)
    Scenario10_NumUnitsCutFirstStage(5,4)
    Scenario10_NumUnitsCutFirstStage(2,1)
    Scenario10_NumUnitsCutFirstStage(6,2)
    Scenario10_NumUnitsCutFirstStage(9,4)
    Scenario10_NumUnitsCutFirstStage(5,1)
    Scenario10_NumUnitsCutFirstStage(8,5)
    Scenario10_NumUnitsCutFirstStage(7,2)
    Scenario10_NumUnitsCutFirstStage(10,4)
    Scenario10_NumUnitsCutFirstStage(10,8)
    Scenario10_NumUnitsCutFirstStage(3,3)
    Scenario10_NumUnitsCutFirstStage(8,1)
    Scenario10_NumUnitsCutFirstStage(10,3)
    Scenario10_NumUnitsCutFirstStage(7,6)
    Scenario10_NumUnitsCutFirstStage(4,4)
    Scenario10_NumUnitsCutFirstStage(6,3)
    Scenario10_NumUnitsCutFirstStage(2,2)
    Scenario10_NumUnitsCutFirstStage(10,1)
    Scenario10_NumUnitsCutFirstStage(8,6)
    Scenario10_NumUnitsCutFirstStage(5,3)
    Scenario10_NumUnitsCutFirstStage(4,1)
    Scenario10_NumUnitsCutFirstStage(1,1)
    Scenario10_NumUnitsCutFirstStage(10,9)
    Scenario10_NumUnitsCutFirstStage(9,7)
    Scenario10_NumUnitsCutFirstStage(6,4)
    Scenario10_NumUnitsCutFirstStage(3,2)
    Scenario10_NumUnitsCutFirstStage(8,2)
    Scenario10_NumUnitsCutFirstStage(7,1)
    Scenario10_NumUnitsCutFirstStage(9,3)
    Scenario10_NumUnitsCutFirstStage(8,3)
    Scenario10_NumUnitsCutFirstStage(7,7)
    Scenario10_NumUnitsCutFirstStage(7,5)
    Scenario10_NumUnitsCutFirstStage(8,7)
    Scenario10_NumUnitsCutFirstStage(4,2)
    Scenario10_NumUnitsCutFirstStage(9,6)
    Scenario10_NumUnitsCutFirstStage(6,5)
    Scenario10_NumUnitsCutFirstStage(5,5)
    Scenario10_NumUnitsCutFirstStage(10,5)
    Scenario10_NumUnitsCutFirstStage(10,10)
    Scenario10_NumUnitsCutFirstStage(9,2)
    Scenario10_NumUnitsCutFirstStage(6,1)
    Scenario10_NumUnitsCutFirstStage(3,1)
    Scenario10_NumUnitsCutFirstStage(9,9)
    Scenario10_NumUnitsCutFirstStage(7,4)
    Scenario10_NumUnitsCutFirstStage(10,7)
    Scenario10_NumUnitsCutFirstStage(8,8)
    Scenario10_NumUnitsCutFirstStage(4,3)
    Scenario10_NumUnitsCutFirstStage(9,5)
    Scenario10_NumUnitsCutFirstStage(10,2)
    Scenario10_NumUnitsCutFirstStage(5,2)
    Scenario10_NumUnitsCutFirstStage(8,4)
    Scenario10_NumProducedSecondStage(1)
    Scenario10_NumProducedSecondStage(2)
    Scenario10_NumProducedSecondStage(3)
    Scenario10_NumProducedSecondStage(4)
    Scenario10_NumProducedSecondStage(5)
    Scenario10_NumProducedSecondStage(6)
    Scenario10_NumProducedSecondStage(7)
    Scenario10_NumProducedSecondStage(8)
    Scenario10_NumProducedSecondStage(9)
    Scenario10_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario10

\ Begin integer variable block for scenario Scenario3
    Scenario3_NumProducedFirstStage(1)
    Scenario3_NumProducedFirstStage(2)
    Scenario3_NumProducedFirstStage(3)
    Scenario3_NumProducedFirstStage(4)
    Scenario3_NumProducedFirstStage(5)
    Scenario3_NumProducedFirstStage(6)
    Scenario3_NumProducedFirstStage(7)
    Scenario3_NumProducedFirstStage(8)
    Scenario3_NumProducedFirstStage(9)
    Scenario3_NumProducedFirstStage(10)
    Scenario3_NumUnitsCutSecondStage(7,3)
    Scenario3_NumUnitsCutSecondStage(9,1)
    Scenario3_NumUnitsCutSecondStage(6,6)
    Scenario3_NumUnitsCutSecondStage(9,8)
    Scenario3_NumUnitsCutSecondStage(10,6)
    Scenario3_NumUnitsCutSecondStage(5,4)
    Scenario3_NumUnitsCutSecondStage(2,1)
    Scenario3_NumUnitsCutSecondStage(6,2)
    Scenario3_NumUnitsCutSecondStage(9,4)
    Scenario3_NumUnitsCutSecondStage(5,1)
    Scenario3_NumUnitsCutSecondStage(8,5)
    Scenario3_NumUnitsCutSecondStage(7,2)
    Scenario3_NumUnitsCutSecondStage(10,4)
    Scenario3_NumUnitsCutSecondStage(10,8)
    Scenario3_NumUnitsCutSecondStage(3,3)
    Scenario3_NumUnitsCutSecondStage(8,1)
    Scenario3_NumUnitsCutSecondStage(10,3)
    Scenario3_NumUnitsCutSecondStage(7,6)
    Scenario3_NumUnitsCutSecondStage(4,4)
    Scenario3_NumUnitsCutSecondStage(6,3)
    Scenario3_NumUnitsCutSecondStage(2,2)
    Scenario3_NumUnitsCutSecondStage(10,1)
    Scenario3_NumUnitsCutSecondStage(8,6)
    Scenario3_NumUnitsCutSecondStage(5,3)
    Scenario3_NumUnitsCutSecondStage(4,1)
    Scenario3_NumUnitsCutSecondStage(1,1)
    Scenario3_NumUnitsCutSecondStage(10,9)
    Scenario3_NumUnitsCutSecondStage(9,7)
    Scenario3_NumUnitsCutSecondStage(6,4)
    Scenario3_NumUnitsCutSecondStage(3,2)
    Scenario3_NumUnitsCutSecondStage(8,2)
    Scenario3_NumUnitsCutSecondStage(7,1)
    Scenario3_NumUnitsCutSecondStage(9,3)
    Scenario3_NumUnitsCutSecondStage(8,3)
    Scenario3_NumUnitsCutSecondStage(7,7)
    Scenario3_NumUnitsCutSecondStage(7,5)
    Scenario3_NumUnitsCutSecondStage(8,7)
    Scenario3_NumUnitsCutSecondStage(4,2)
    Scenario3_NumUnitsCutSecondStage(9,6)
    Scenario3_NumUnitsCutSecondStage(6,5)
    Scenario3_NumUnitsCutSecondStage(5,5)
    Scenario3_NumUnitsCutSecondStage(10,5)
    Scenario3_NumUnitsCutSecondStage(10,10)
    Scenario3_NumUnitsCutSecondStage(9,2)
    Scenario3_NumUnitsCutSecondStage(6,1)
    Scenario3_NumUnitsCutSecondStage(3,1)
    Scenario3_NumUnitsCutSecondStage(9,9)
    Scenario3_NumUnitsCutSecondStage(7,4)
    Scenario3_NumUnitsCutSecondStage(10,7)
    Scenario3_NumUnitsCutSecondStage(8,8)
    Scenario3_NumUnitsCutSecondStage(4,3)
    Scenario3_NumUnitsCutSecondStage(9,5)
    Scenario3_NumUnitsCutSecondStage(10,2)
    Scenario3_NumUnitsCutSecondStage(5,2)
    Scenario3_NumUnitsCutSecondStage(8,4)
    Scenario3_NumUnitsCutFirstStage(7,3)
    Scenario3_NumUnitsCutFirstStage(9,1)
    Scenario3_NumUnitsCutFirstStage(6,6)
    Scenario3_NumUnitsCutFirstStage(9,8)
    Scenario3_NumUnitsCutFirstStage(10,6)
    Scenario3_NumUnitsCutFirstStage(5,4)
    Scenario3_NumUnitsCutFirstStage(2,1)
    Scenario3_NumUnitsCutFirstStage(6,2)
    Scenario3_NumUnitsCutFirstStage(9,4)
    Scenario3_NumUnitsCutFirstStage(5,1)
    Scenario3_NumUnitsCutFirstStage(8,5)
    Scenario3_NumUnitsCutFirstStage(7,2)
    Scenario3_NumUnitsCutFirstStage(10,4)
    Scenario3_NumUnitsCutFirstStage(10,8)
    Scenario3_NumUnitsCutFirstStage(3,3)
    Scenario3_NumUnitsCutFirstStage(8,1)
    Scenario3_NumUnitsCutFirstStage(10,3)
    Scenario3_NumUnitsCutFirstStage(7,6)
    Scenario3_NumUnitsCutFirstStage(4,4)
    Scenario3_NumUnitsCutFirstStage(6,3)
    Scenario3_NumUnitsCutFirstStage(2,2)
    Scenario3_NumUnitsCutFirstStage(10,1)
    Scenario3_NumUnitsCutFirstStage(8,6)
    Scenario3_NumUnitsCutFirstStage(5,3)
    Scenario3_NumUnitsCutFirstStage(4,1)
    Scenario3_NumUnitsCutFirstStage(1,1)
    Scenario3_NumUnitsCutFirstStage(10,9)
    Scenario3_NumUnitsCutFirstStage(9,7)
    Scenario3_NumUnitsCutFirstStage(6,4)
    Scenario3_NumUnitsCutFirstStage(3,2)
    Scenario3_NumUnitsCutFirstStage(8,2)
    Scenario3_NumUnitsCutFirstStage(7,1)
    Scenario3_NumUnitsCutFirstStage(9,3)
    Scenario3_NumUnitsCutFirstStage(8,3)
    Scenario3_NumUnitsCutFirstStage(7,7)
    Scenario3_NumUnitsCutFirstStage(7,5)
    Scenario3_NumUnitsCutFirstStage(8,7)
    Scenario3_NumUnitsCutFirstStage(4,2)
    Scenario3_NumUnitsCutFirstStage(9,6)
    Scenario3_NumUnitsCutFirstStage(6,5)
    Scenario3_NumUnitsCutFirstStage(5,5)
    Scenario3_NumUnitsCutFirstStage(10,5)
    Scenario3_NumUnitsCutFirstStage(10,10)
    Scenario3_NumUnitsCutFirstStage(9,2)
    Scenario3_NumUnitsCutFirstStage(6,1)
    Scenario3_NumUnitsCutFirstStage(3,1)
    Scenario3_NumUnitsCutFirstStage(9,9)
    Scenario3_NumUnitsCutFirstStage(7,4)
    Scenario3_NumUnitsCutFirstStage(10,7)
    Scenario3_NumUnitsCutFirstStage(8,8)
    Scenario3_NumUnitsCutFirstStage(4,3)
    Scenario3_NumUnitsCutFirstStage(9,5)
    Scenario3_NumUnitsCutFirstStage(10,2)
    Scenario3_NumUnitsCutFirstStage(5,2)
    Scenario3_NumUnitsCutFirstStage(8,4)
    Scenario3_NumProducedSecondStage(1)
    Scenario3_NumProducedSecondStage(2)
    Scenario3_NumProducedSecondStage(3)
    Scenario3_NumProducedSecondStage(4)
    Scenario3_NumProducedSecondStage(5)
    Scenario3_NumProducedSecondStage(6)
    Scenario3_NumProducedSecondStage(7)
    Scenario3_NumProducedSecondStage(8)
    Scenario3_NumProducedSecondStage(9)
    Scenario3_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario3

\ Begin integer variable block for scenario Scenario2
    Scenario2_NumProducedFirstStage(1)
    Scenario2_NumProducedFirstStage(2)
    Scenario2_NumProducedFirstStage(3)
    Scenario2_NumProducedFirstStage(4)
    Scenario2_NumProducedFirstStage(5)
    Scenario2_NumProducedFirstStage(6)
    Scenario2_NumProducedFirstStage(7)
    Scenario2_NumProducedFirstStage(8)
    Scenario2_NumProducedFirstStage(9)
    Scenario2_NumProducedFirstStage(10)
    Scenario2_NumUnitsCutSecondStage(7,3)
    Scenario2_NumUnitsCutSecondStage(9,1)
    Scenario2_NumUnitsCutSecondStage(6,6)
    Scenario2_NumUnitsCutSecondStage(9,8)
    Scenario2_NumUnitsCutSecondStage(10,6)
    Scenario2_NumUnitsCutSecondStage(5,4)
    Scenario2_NumUnitsCutSecondStage(2,1)
    Scenario2_NumUnitsCutSecondStage(6,2)
    Scenario2_NumUnitsCutSecondStage(9,4)
    Scenario2_NumUnitsCutSecondStage(5,1)
    Scenario2_NumUnitsCutSecondStage(8,5)
    Scenario2_NumUnitsCutSecondStage(7,2)
    Scenario2_NumUnitsCutSecondStage(10,4)
    Scenario2_NumUnitsCutSecondStage(10,8)
    Scenario2_NumUnitsCutSecondStage(3,3)
    Scenario2_NumUnitsCutSecondStage(8,1)
    Scenario2_NumUnitsCutSecondStage(10,3)
    Scenario2_NumUnitsCutSecondStage(7,6)
    Scenario2_NumUnitsCutSecondStage(4,4)
    Scenario2_NumUnitsCutSecondStage(6,3)
    Scenario2_NumUnitsCutSecondStage(2,2)
    Scenario2_NumUnitsCutSecondStage(10,1)
    Scenario2_NumUnitsCutSecondStage(8,6)
    Scenario2_NumUnitsCutSecondStage(5,3)
    Scenario2_NumUnitsCutSecondStage(4,1)
    Scenario2_NumUnitsCutSecondStage(1,1)
    Scenario2_NumUnitsCutSecondStage(10,9)
    Scenario2_NumUnitsCutSecondStage(9,7)
    Scenario2_NumUnitsCutSecondStage(6,4)
    Scenario2_NumUnitsCutSecondStage(3,2)
    Scenario2_NumUnitsCutSecondStage(8,2)
    Scenario2_NumUnitsCutSecondStage(7,1)
    Scenario2_NumUnitsCutSecondStage(9,3)
    Scenario2_NumUnitsCutSecondStage(8,3)
    Scenario2_NumUnitsCutSecondStage(7,7)
    Scenario2_NumUnitsCutSecondStage(7,5)
    Scenario2_NumUnitsCutSecondStage(8,7)
    Scenario2_NumUnitsCutSecondStage(4,2)
    Scenario2_NumUnitsCutSecondStage(9,6)
    Scenario2_NumUnitsCutSecondStage(6,5)
    Scenario2_NumUnitsCutSecondStage(5,5)
    Scenario2_NumUnitsCutSecondStage(10,5)
    Scenario2_NumUnitsCutSecondStage(10,10)
    Scenario2_NumUnitsCutSecondStage(9,2)
    Scenario2_NumUnitsCutSecondStage(6,1)
    Scenario2_NumUnitsCutSecondStage(3,1)
    Scenario2_NumUnitsCutSecondStage(9,9)
    Scenario2_NumUnitsCutSecondStage(7,4)
    Scenario2_NumUnitsCutSecondStage(10,7)
    Scenario2_NumUnitsCutSecondStage(8,8)
    Scenario2_NumUnitsCutSecondStage(4,3)
    Scenario2_NumUnitsCutSecondStage(9,5)
    Scenario2_NumUnitsCutSecondStage(10,2)
    Scenario2_NumUnitsCutSecondStage(5,2)
    Scenario2_NumUnitsCutSecondStage(8,4)
    Scenario2_NumUnitsCutFirstStage(7,3)
    Scenario2_NumUnitsCutFirstStage(9,1)
    Scenario2_NumUnitsCutFirstStage(6,6)
    Scenario2_NumUnitsCutFirstStage(9,8)
    Scenario2_NumUnitsCutFirstStage(10,6)
    Scenario2_NumUnitsCutFirstStage(5,4)
    Scenario2_NumUnitsCutFirstStage(2,1)
    Scenario2_NumUnitsCutFirstStage(6,2)
    Scenario2_NumUnitsCutFirstStage(9,4)
    Scenario2_NumUnitsCutFirstStage(5,1)
    Scenario2_NumUnitsCutFirstStage(8,5)
    Scenario2_NumUnitsCutFirstStage(7,2)
    Scenario2_NumUnitsCutFirstStage(10,4)
    Scenario2_NumUnitsCutFirstStage(10,8)
    Scenario2_NumUnitsCutFirstStage(3,3)
    Scenario2_NumUnitsCutFirstStage(8,1)
    Scenario2_NumUnitsCutFirstStage(10,3)
    Scenario2_NumUnitsCutFirstStage(7,6)
    Scenario2_NumUnitsCutFirstStage(4,4)
    Scenario2_NumUnitsCutFirstStage(6,3)
    Scenario2_NumUnitsCutFirstStage(2,2)
    Scenario2_NumUnitsCutFirstStage(10,1)
    Scenario2_NumUnitsCutFirstStage(8,6)
    Scenario2_NumUnitsCutFirstStage(5,3)
    Scenario2_NumUnitsCutFirstStage(4,1)
    Scenario2_NumUnitsCutFirstStage(1,1)
    Scenario2_NumUnitsCutFirstStage(10,9)
    Scenario2_NumUnitsCutFirstStage(9,7)
    Scenario2_NumUnitsCutFirstStage(6,4)
    Scenario2_NumUnitsCutFirstStage(3,2)
    Scenario2_NumUnitsCutFirstStage(8,2)
    Scenario2_NumUnitsCutFirstStage(7,1)
    Scenario2_NumUnitsCutFirstStage(9,3)
    Scenario2_NumUnitsCutFirstStage(8,3)
    Scenario2_NumUnitsCutFirstStage(7,7)
    Scenario2_NumUnitsCutFirstStage(7,5)
    Scenario2_NumUnitsCutFirstStage(8,7)
    Scenario2_NumUnitsCutFirstStage(4,2)
    Scenario2_NumUnitsCutFirstStage(9,6)
    Scenario2_NumUnitsCutFirstStage(6,5)
    Scenario2_NumUnitsCutFirstStage(5,5)
    Scenario2_NumUnitsCutFirstStage(10,5)
    Scenario2_NumUnitsCutFirstStage(10,10)
    Scenario2_NumUnitsCutFirstStage(9,2)
    Scenario2_NumUnitsCutFirstStage(6,1)
    Scenario2_NumUnitsCutFirstStage(3,1)
    Scenario2_NumUnitsCutFirstStage(9,9)
    Scenario2_NumUnitsCutFirstStage(7,4)
    Scenario2_NumUnitsCutFirstStage(10,7)
    Scenario2_NumUnitsCutFirstStage(8,8)
    Scenario2_NumUnitsCutFirstStage(4,3)
    Scenario2_NumUnitsCutFirstStage(9,5)
    Scenario2_NumUnitsCutFirstStage(10,2)
    Scenario2_NumUnitsCutFirstStage(5,2)
    Scenario2_NumUnitsCutFirstStage(8,4)
    Scenario2_NumProducedSecondStage(1)
    Scenario2_NumProducedSecondStage(2)
    Scenario2_NumProducedSecondStage(3)
    Scenario2_NumProducedSecondStage(4)
    Scenario2_NumProducedSecondStage(5)
    Scenario2_NumProducedSecondStage(6)
    Scenario2_NumProducedSecondStage(7)
    Scenario2_NumProducedSecondStage(8)
    Scenario2_NumProducedSecondStage(9)
    Scenario2_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario2

\ Begin integer variable block for scenario Scenario1
    Scenario1_NumProducedFirstStage(1)
    Scenario1_NumProducedFirstStage(2)
    Scenario1_NumProducedFirstStage(3)
    Scenario1_NumProducedFirstStage(4)
    Scenario1_NumProducedFirstStage(5)
    Scenario1_NumProducedFirstStage(6)
    Scenario1_NumProducedFirstStage(7)
    Scenario1_NumProducedFirstStage(8)
    Scenario1_NumProducedFirstStage(9)
    Scenario1_NumProducedFirstStage(10)
    Scenario1_NumUnitsCutSecondStage(7,3)
    Scenario1_NumUnitsCutSecondStage(9,1)
    Scenario1_NumUnitsCutSecondStage(6,6)
    Scenario1_NumUnitsCutSecondStage(9,8)
    Scenario1_NumUnitsCutSecondStage(10,6)
    Scenario1_NumUnitsCutSecondStage(5,4)
    Scenario1_NumUnitsCutSecondStage(2,1)
    Scenario1_NumUnitsCutSecondStage(6,2)
    Scenario1_NumUnitsCutSecondStage(9,4)
    Scenario1_NumUnitsCutSecondStage(5,1)
    Scenario1_NumUnitsCutSecondStage(8,5)
    Scenario1_NumUnitsCutSecondStage(7,2)
    Scenario1_NumUnitsCutSecondStage(10,4)
    Scenario1_NumUnitsCutSecondStage(10,8)
    Scenario1_NumUnitsCutSecondStage(3,3)
    Scenario1_NumUnitsCutSecondStage(8,1)
    Scenario1_NumUnitsCutSecondStage(10,3)
    Scenario1_NumUnitsCutSecondStage(7,6)
    Scenario1_NumUnitsCutSecondStage(4,4)
    Scenario1_NumUnitsCutSecondStage(6,3)
    Scenario1_NumUnitsCutSecondStage(2,2)
    Scenario1_NumUnitsCutSecondStage(10,1)
    Scenario1_NumUnitsCutSecondStage(8,6)
    Scenario1_NumUnitsCutSecondStage(5,3)
    Scenario1_NumUnitsCutSecondStage(4,1)
    Scenario1_NumUnitsCutSecondStage(1,1)
    Scenario1_NumUnitsCutSecondStage(10,9)
    Scenario1_NumUnitsCutSecondStage(9,7)
    Scenario1_NumUnitsCutSecondStage(6,4)
    Scenario1_NumUnitsCutSecondStage(3,2)
    Scenario1_NumUnitsCutSecondStage(8,2)
    Scenario1_NumUnitsCutSecondStage(7,1)
    Scenario1_NumUnitsCutSecondStage(9,3)
    Scenario1_NumUnitsCutSecondStage(8,3)
    Scenario1_NumUnitsCutSecondStage(7,7)
    Scenario1_NumUnitsCutSecondStage(7,5)
    Scenario1_NumUnitsCutSecondStage(8,7)
    Scenario1_NumUnitsCutSecondStage(4,2)
    Scenario1_NumUnitsCutSecondStage(9,6)
    Scenario1_NumUnitsCutSecondStage(6,5)
    Scenario1_NumUnitsCutSecondStage(5,5)
    Scenario1_NumUnitsCutSecondStage(10,5)
    Scenario1_NumUnitsCutSecondStage(10,10)
    Scenario1_NumUnitsCutSecondStage(9,2)
    Scenario1_NumUnitsCutSecondStage(6,1)
    Scenario1_NumUnitsCutSecondStage(3,1)
    Scenario1_NumUnitsCutSecondStage(9,9)
    Scenario1_NumUnitsCutSecondStage(7,4)
    Scenario1_NumUnitsCutSecondStage(10,7)
    Scenario1_NumUnitsCutSecondStage(8,8)
    Scenario1_NumUnitsCutSecondStage(4,3)
    Scenario1_NumUnitsCutSecondStage(9,5)
    Scenario1_NumUnitsCutSecondStage(10,2)
    Scenario1_NumUnitsCutSecondStage(5,2)
    Scenario1_NumUnitsCutSecondStage(8,4)
    Scenario1_NumUnitsCutFirstStage(7,3)
    Scenario1_NumUnitsCutFirstStage(9,1)
    Scenario1_NumUnitsCutFirstStage(6,6)
    Scenario1_NumUnitsCutFirstStage(9,8)
    Scenario1_NumUnitsCutFirstStage(10,6)
    Scenario1_NumUnitsCutFirstStage(5,4)
    Scenario1_NumUnitsCutFirstStage(2,1)
    Scenario1_NumUnitsCutFirstStage(6,2)
    Scenario1_NumUnitsCutFirstStage(9,4)
    Scenario1_NumUnitsCutFirstStage(5,1)
    Scenario1_NumUnitsCutFirstStage(8,5)
    Scenario1_NumUnitsCutFirstStage(7,2)
    Scenario1_NumUnitsCutFirstStage(10,4)
    Scenario1_NumUnitsCutFirstStage(10,8)
    Scenario1_NumUnitsCutFirstStage(3,3)
    Scenario1_NumUnitsCutFirstStage(8,1)
    Scenario1_NumUnitsCutFirstStage(10,3)
    Scenario1_NumUnitsCutFirstStage(7,6)
    Scenario1_NumUnitsCutFirstStage(4,4)
    Scenario1_NumUnitsCutFirstStage(6,3)
    Scenario1_NumUnitsCutFirstStage(2,2)
    Scenario1_NumUnitsCutFirstStage(10,1)
    Scenario1_NumUnitsCutFirstStage(8,6)
    Scenario1_NumUnitsCutFirstStage(5,3)
    Scenario1_NumUnitsCutFirstStage(4,1)
    Scenario1_NumUnitsCutFirstStage(1,1)
    Scenario1_NumUnitsCutFirstStage(10,9)
    Scenario1_NumUnitsCutFirstStage(9,7)
    Scenario1_NumUnitsCutFirstStage(6,4)
    Scenario1_NumUnitsCutFirstStage(3,2)
    Scenario1_NumUnitsCutFirstStage(8,2)
    Scenario1_NumUnitsCutFirstStage(7,1)
    Scenario1_NumUnitsCutFirstStage(9,3)
    Scenario1_NumUnitsCutFirstStage(8,3)
    Scenario1_NumUnitsCutFirstStage(7,7)
    Scenario1_NumUnitsCutFirstStage(7,5)
    Scenario1_NumUnitsCutFirstStage(8,7)
    Scenario1_NumUnitsCutFirstStage(4,2)
    Scenario1_NumUnitsCutFirstStage(9,6)
    Scenario1_NumUnitsCutFirstStage(6,5)
    Scenario1_NumUnitsCutFirstStage(5,5)
    Scenario1_NumUnitsCutFirstStage(10,5)
    Scenario1_NumUnitsCutFirstStage(10,10)
    Scenario1_NumUnitsCutFirstStage(9,2)
    Scenario1_NumUnitsCutFirstStage(6,1)
    Scenario1_NumUnitsCutFirstStage(3,1)
    Scenario1_NumUnitsCutFirstStage(9,9)
    Scenario1_NumUnitsCutFirstStage(7,4)
    Scenario1_NumUnitsCutFirstStage(10,7)
    Scenario1_NumUnitsCutFirstStage(8,8)
    Scenario1_NumUnitsCutFirstStage(4,3)
    Scenario1_NumUnitsCutFirstStage(9,5)
    Scenario1_NumUnitsCutFirstStage(10,2)
    Scenario1_NumUnitsCutFirstStage(5,2)
    Scenario1_NumUnitsCutFirstStage(8,4)
    Scenario1_NumProducedSecondStage(1)
    Scenario1_NumProducedSecondStage(2)
    Scenario1_NumProducedSecondStage(3)
    Scenario1_NumProducedSecondStage(4)
    Scenario1_NumProducedSecondStage(5)
    Scenario1_NumProducedSecondStage(6)
    Scenario1_NumProducedSecondStage(7)
    Scenario1_NumProducedSecondStage(8)
    Scenario1_NumProducedSecondStage(9)
    Scenario1_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario1

\ Begin integer variable block for scenario Scenario7
    Scenario7_NumProducedFirstStage(1)
    Scenario7_NumProducedFirstStage(2)
    Scenario7_NumProducedFirstStage(3)
    Scenario7_NumProducedFirstStage(4)
    Scenario7_NumProducedFirstStage(5)
    Scenario7_NumProducedFirstStage(6)
    Scenario7_NumProducedFirstStage(7)
    Scenario7_NumProducedFirstStage(8)
    Scenario7_NumProducedFirstStage(9)
    Scenario7_NumProducedFirstStage(10)
    Scenario7_NumUnitsCutSecondStage(7,3)
    Scenario7_NumUnitsCutSecondStage(9,1)
    Scenario7_NumUnitsCutSecondStage(6,6)
    Scenario7_NumUnitsCutSecondStage(9,8)
    Scenario7_NumUnitsCutSecondStage(10,6)
    Scenario7_NumUnitsCutSecondStage(5,4)
    Scenario7_NumUnitsCutSecondStage(2,1)
    Scenario7_NumUnitsCutSecondStage(6,2)
    Scenario7_NumUnitsCutSecondStage(9,4)
    Scenario7_NumUnitsCutSecondStage(5,1)
    Scenario7_NumUnitsCutSecondStage(8,5)
    Scenario7_NumUnitsCutSecondStage(7,2)
    Scenario7_NumUnitsCutSecondStage(10,4)
    Scenario7_NumUnitsCutSecondStage(10,8)
    Scenario7_NumUnitsCutSecondStage(3,3)
    Scenario7_NumUnitsCutSecondStage(8,1)
    Scenario7_NumUnitsCutSecondStage(10,3)
    Scenario7_NumUnitsCutSecondStage(7,6)
    Scenario7_NumUnitsCutSecondStage(4,4)
    Scenario7_NumUnitsCutSecondStage(6,3)
    Scenario7_NumUnitsCutSecondStage(2,2)
    Scenario7_NumUnitsCutSecondStage(10,1)
    Scenario7_NumUnitsCutSecondStage(8,6)
    Scenario7_NumUnitsCutSecondStage(5,3)
    Scenario7_NumUnitsCutSecondStage(4,1)
    Scenario7_NumUnitsCutSecondStage(1,1)
    Scenario7_NumUnitsCutSecondStage(10,9)
    Scenario7_NumUnitsCutSecondStage(9,7)
    Scenario7_NumUnitsCutSecondStage(6,4)
    Scenario7_NumUnitsCutSecondStage(3,2)
    Scenario7_NumUnitsCutSecondStage(8,2)
    Scenario7_NumUnitsCutSecondStage(7,1)
    Scenario7_NumUnitsCutSecondStage(9,3)
    Scenario7_NumUnitsCutSecondStage(8,3)
    Scenario7_NumUnitsCutSecondStage(7,7)
    Scenario7_NumUnitsCutSecondStage(7,5)
    Scenario7_NumUnitsCutSecondStage(8,7)
    Scenario7_NumUnitsCutSecondStage(4,2)
    Scenario7_NumUnitsCutSecondStage(9,6)
    Scenario7_NumUnitsCutSecondStage(6,5)
    Scenario7_NumUnitsCutSecondStage(5,5)
    Scenario7_NumUnitsCutSecondStage(10,5)
    Scenario7_NumUnitsCutSecondStage(10,10)
    Scenario7_NumUnitsCutSecondStage(9,2)
    Scenario7_NumUnitsCutSecondStage(6,1)
    Scenario7_NumUnitsCutSecondStage(3,1)
    Scenario7_NumUnitsCutSecondStage(9,9)
    Scenario7_NumUnitsCutSecondStage(7,4)
    Scenario7_NumUnitsCutSecondStage(10,7)
    Scenario7_NumUnitsCutSecondStage(8,8)
    Scenario7_NumUnitsCutSecondStage(4,3)
    Scenario7_NumUnitsCutSecondStage(9,5)
    Scenario7_NumUnitsCutSecondStage(10,2)
    Scenario7_NumUnitsCutSecondStage(5,2)
    Scenario7_NumUnitsCutSecondStage(8,4)
    Scenario7_NumUnitsCutFirstStage(7,3)
    Scenario7_NumUnitsCutFirstStage(9,1)
    Scenario7_NumUnitsCutFirstStage(6,6)
    Scenario7_NumUnitsCutFirstStage(9,8)
    Scenario7_NumUnitsCutFirstStage(10,6)
    Scenario7_NumUnitsCutFirstStage(5,4)
    Scenario7_NumUnitsCutFirstStage(2,1)
    Scenario7_NumUnitsCutFirstStage(6,2)
    Scenario7_NumUnitsCutFirstStage(9,4)
    Scenario7_NumUnitsCutFirstStage(5,1)
    Scenario7_NumUnitsCutFirstStage(8,5)
    Scenario7_NumUnitsCutFirstStage(7,2)
    Scenario7_NumUnitsCutFirstStage(10,4)
    Scenario7_NumUnitsCutFirstStage(10,8)
    Scenario7_NumUnitsCutFirstStage(3,3)
    Scenario7_NumUnitsCutFirstStage(8,1)
    Scenario7_NumUnitsCutFirstStage(10,3)
    Scenario7_NumUnitsCutFirstStage(7,6)
    Scenario7_NumUnitsCutFirstStage(4,4)
    Scenario7_NumUnitsCutFirstStage(6,3)
    Scenario7_NumUnitsCutFirstStage(2,2)
    Scenario7_NumUnitsCutFirstStage(10,1)
    Scenario7_NumUnitsCutFirstStage(8,6)
    Scenario7_NumUnitsCutFirstStage(5,3)
    Scenario7_NumUnitsCutFirstStage(4,1)
    Scenario7_NumUnitsCutFirstStage(1,1)
    Scenario7_NumUnitsCutFirstStage(10,9)
    Scenario7_NumUnitsCutFirstStage(9,7)
    Scenario7_NumUnitsCutFirstStage(6,4)
    Scenario7_NumUnitsCutFirstStage(3,2)
    Scenario7_NumUnitsCutFirstStage(8,2)
    Scenario7_NumUnitsCutFirstStage(7,1)
    Scenario7_NumUnitsCutFirstStage(9,3)
    Scenario7_NumUnitsCutFirstStage(8,3)
    Scenario7_NumUnitsCutFirstStage(7,7)
    Scenario7_NumUnitsCutFirstStage(7,5)
    Scenario7_NumUnitsCutFirstStage(8,7)
    Scenario7_NumUnitsCutFirstStage(4,2)
    Scenario7_NumUnitsCutFirstStage(9,6)
    Scenario7_NumUnitsCutFirstStage(6,5)
    Scenario7_NumUnitsCutFirstStage(5,5)
    Scenario7_NumUnitsCutFirstStage(10,5)
    Scenario7_NumUnitsCutFirstStage(10,10)
    Scenario7_NumUnitsCutFirstStage(9,2)
    Scenario7_NumUnitsCutFirstStage(6,1)
    Scenario7_NumUnitsCutFirstStage(3,1)
    Scenario7_NumUnitsCutFirstStage(9,9)
    Scenario7_NumUnitsCutFirstStage(7,4)
    Scenario7_NumUnitsCutFirstStage(10,7)
    Scenario7_NumUnitsCutFirstStage(8,8)
    Scenario7_NumUnitsCutFirstStage(4,3)
    Scenario7_NumUnitsCutFirstStage(9,5)
    Scenario7_NumUnitsCutFirstStage(10,2)
    Scenario7_NumUnitsCutFirstStage(5,2)
    Scenario7_NumUnitsCutFirstStage(8,4)
    Scenario7_NumProducedSecondStage(1)
    Scenario7_NumProducedSecondStage(2)
    Scenario7_NumProducedSecondStage(3)
    Scenario7_NumProducedSecondStage(4)
    Scenario7_NumProducedSecondStage(5)
    Scenario7_NumProducedSecondStage(6)
    Scenario7_NumProducedSecondStage(7)
    Scenario7_NumProducedSecondStage(8)
    Scenario7_NumProducedSecondStage(9)
    Scenario7_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario7

\ Begin integer variable block for scenario Scenario6
    Scenario6_NumProducedFirstStage(1)
    Scenario6_NumProducedFirstStage(2)
    Scenario6_NumProducedFirstStage(3)
    Scenario6_NumProducedFirstStage(4)
    Scenario6_NumProducedFirstStage(5)
    Scenario6_NumProducedFirstStage(6)
    Scenario6_NumProducedFirstStage(7)
    Scenario6_NumProducedFirstStage(8)
    Scenario6_NumProducedFirstStage(9)
    Scenario6_NumProducedFirstStage(10)
    Scenario6_NumUnitsCutSecondStage(7,3)
    Scenario6_NumUnitsCutSecondStage(9,1)
    Scenario6_NumUnitsCutSecondStage(6,6)
    Scenario6_NumUnitsCutSecondStage(9,8)
    Scenario6_NumUnitsCutSecondStage(10,6)
    Scenario6_NumUnitsCutSecondStage(5,4)
    Scenario6_NumUnitsCutSecondStage(2,1)
    Scenario6_NumUnitsCutSecondStage(6,2)
    Scenario6_NumUnitsCutSecondStage(9,4)
    Scenario6_NumUnitsCutSecondStage(5,1)
    Scenario6_NumUnitsCutSecondStage(8,5)
    Scenario6_NumUnitsCutSecondStage(7,2)
    Scenario6_NumUnitsCutSecondStage(10,4)
    Scenario6_NumUnitsCutSecondStage(10,8)
    Scenario6_NumUnitsCutSecondStage(3,3)
    Scenario6_NumUnitsCutSecondStage(8,1)
    Scenario6_NumUnitsCutSecondStage(10,3)
    Scenario6_NumUnitsCutSecondStage(7,6)
    Scenario6_NumUnitsCutSecondStage(4,4)
    Scenario6_NumUnitsCutSecondStage(6,3)
    Scenario6_NumUnitsCutSecondStage(2,2)
    Scenario6_NumUnitsCutSecondStage(10,1)
    Scenario6_NumUnitsCutSecondStage(8,6)
    Scenario6_NumUnitsCutSecondStage(5,3)
    Scenario6_NumUnitsCutSecondStage(4,1)
    Scenario6_NumUnitsCutSecondStage(1,1)
    Scenario6_NumUnitsCutSecondStage(10,9)
    Scenario6_NumUnitsCutSecondStage(9,7)
    Scenario6_NumUnitsCutSecondStage(6,4)
    Scenario6_NumUnitsCutSecondStage(3,2)
    Scenario6_NumUnitsCutSecondStage(8,2)
    Scenario6_NumUnitsCutSecondStage(7,1)
    Scenario6_NumUnitsCutSecondStage(9,3)
    Scenario6_NumUnitsCutSecondStage(8,3)
    Scenario6_NumUnitsCutSecondStage(7,7)
    Scenario6_NumUnitsCutSecondStage(7,5)
    Scenario6_NumUnitsCutSecondStage(8,7)
    Scenario6_NumUnitsCutSecondStage(4,2)
    Scenario6_NumUnitsCutSecondStage(9,6)
    Scenario6_NumUnitsCutSecondStage(6,5)
    Scenario6_NumUnitsCutSecondStage(5,5)
    Scenario6_NumUnitsCutSecondStage(10,5)
    Scenario6_NumUnitsCutSecondStage(10,10)
    Scenario6_NumUnitsCutSecondStage(9,2)
    Scenario6_NumUnitsCutSecondStage(6,1)
    Scenario6_NumUnitsCutSecondStage(3,1)
    Scenario6_NumUnitsCutSecondStage(9,9)
    Scenario6_NumUnitsCutSecondStage(7,4)
    Scenario6_NumUnitsCutSecondStage(10,7)
    Scenario6_NumUnitsCutSecondStage(8,8)
    Scenario6_NumUnitsCutSecondStage(4,3)
    Scenario6_NumUnitsCutSecondStage(9,5)
    Scenario6_NumUnitsCutSecondStage(10,2)
    Scenario6_NumUnitsCutSecondStage(5,2)
    Scenario6_NumUnitsCutSecondStage(8,4)
    Scenario6_NumUnitsCutFirstStage(7,3)
    Scenario6_NumUnitsCutFirstStage(9,1)
    Scenario6_NumUnitsCutFirstStage(6,6)
    Scenario6_NumUnitsCutFirstStage(9,8)
    Scenario6_NumUnitsCutFirstStage(10,6)
    Scenario6_NumUnitsCutFirstStage(5,4)
    Scenario6_NumUnitsCutFirstStage(2,1)
    Scenario6_NumUnitsCutFirstStage(6,2)
    Scenario6_NumUnitsCutFirstStage(9,4)
    Scenario6_NumUnitsCutFirstStage(5,1)
    Scenario6_NumUnitsCutFirstStage(8,5)
    Scenario6_NumUnitsCutFirstStage(7,2)
    Scenario6_NumUnitsCutFirstStage(10,4)
    Scenario6_NumUnitsCutFirstStage(10,8)
    Scenario6_NumUnitsCutFirstStage(3,3)
    Scenario6_NumUnitsCutFirstStage(8,1)
    Scenario6_NumUnitsCutFirstStage(10,3)
    Scenario6_NumUnitsCutFirstStage(7,6)
    Scenario6_NumUnitsCutFirstStage(4,4)
    Scenario6_NumUnitsCutFirstStage(6,3)
    Scenario6_NumUnitsCutFirstStage(2,2)
    Scenario6_NumUnitsCutFirstStage(10,1)
    Scenario6_NumUnitsCutFirstStage(8,6)
    Scenario6_NumUnitsCutFirstStage(5,3)
    Scenario6_NumUnitsCutFirstStage(4,1)
    Scenario6_NumUnitsCutFirstStage(1,1)
    Scenario6_NumUnitsCutFirstStage(10,9)
    Scenario6_NumUnitsCutFirstStage(9,7)
    Scenario6_NumUnitsCutFirstStage(6,4)
    Scenario6_NumUnitsCutFirstStage(3,2)
    Scenario6_NumUnitsCutFirstStage(8,2)
    Scenario6_NumUnitsCutFirstStage(7,1)
    Scenario6_NumUnitsCutFirstStage(9,3)
    Scenario6_NumUnitsCutFirstStage(8,3)
    Scenario6_NumUnitsCutFirstStage(7,7)
    Scenario6_NumUnitsCutFirstStage(7,5)
    Scenario6_NumUnitsCutFirstStage(8,7)
    Scenario6_NumUnitsCutFirstStage(4,2)
    Scenario6_NumUnitsCutFirstStage(9,6)
    Scenario6_NumUnitsCutFirstStage(6,5)
    Scenario6_NumUnitsCutFirstStage(5,5)
    Scenario6_NumUnitsCutFirstStage(10,5)
    Scenario6_NumUnitsCutFirstStage(10,10)
    Scenario6_NumUnitsCutFirstStage(9,2)
    Scenario6_NumUnitsCutFirstStage(6,1)
    Scenario6_NumUnitsCutFirstStage(3,1)
    Scenario6_NumUnitsCutFirstStage(9,9)
    Scenario6_NumUnitsCutFirstStage(7,4)
    Scenario6_NumUnitsCutFirstStage(10,7)
    Scenario6_NumUnitsCutFirstStage(8,8)
    Scenario6_NumUnitsCutFirstStage(4,3)
    Scenario6_NumUnitsCutFirstStage(9,5)
    Scenario6_NumUnitsCutFirstStage(10,2)
    Scenario6_NumUnitsCutFirstStage(5,2)
    Scenario6_NumUnitsCutFirstStage(8,4)
    Scenario6_NumProducedSecondStage(1)
    Scenario6_NumProducedSecondStage(2)
    Scenario6_NumProducedSecondStage(3)
    Scenario6_NumProducedSecondStage(4)
    Scenario6_NumProducedSecondStage(5)
    Scenario6_NumProducedSecondStage(6)
    Scenario6_NumProducedSecondStage(7)
    Scenario6_NumProducedSecondStage(8)
    Scenario6_NumProducedSecondStage(9)
    Scenario6_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario6

\ Begin integer variable block for scenario Scenario5
    Scenario5_NumProducedFirstStage(1)
    Scenario5_NumProducedFirstStage(2)
    Scenario5_NumProducedFirstStage(3)
    Scenario5_NumProducedFirstStage(4)
    Scenario5_NumProducedFirstStage(5)
    Scenario5_NumProducedFirstStage(6)
    Scenario5_NumProducedFirstStage(7)
    Scenario5_NumProducedFirstStage(8)
    Scenario5_NumProducedFirstStage(9)
    Scenario5_NumProducedFirstStage(10)
    Scenario5_NumUnitsCutSecondStage(7,3)
    Scenario5_NumUnitsCutSecondStage(9,1)
    Scenario5_NumUnitsCutSecondStage(6,6)
    Scenario5_NumUnitsCutSecondStage(9,8)
    Scenario5_NumUnitsCutSecondStage(10,6)
    Scenario5_NumUnitsCutSecondStage(5,4)
    Scenario5_NumUnitsCutSecondStage(2,1)
    Scenario5_NumUnitsCutSecondStage(6,2)
    Scenario5_NumUnitsCutSecondStage(9,4)
    Scenario5_NumUnitsCutSecondStage(5,1)
    Scenario5_NumUnitsCutSecondStage(8,5)
    Scenario5_NumUnitsCutSecondStage(7,2)
    Scenario5_NumUnitsCutSecondStage(10,4)
    Scenario5_NumUnitsCutSecondStage(10,8)
    Scenario5_NumUnitsCutSecondStage(3,3)
    Scenario5_NumUnitsCutSecondStage(8,1)
    Scenario5_NumUnitsCutSecondStage(10,3)
    Scenario5_NumUnitsCutSecondStage(7,6)
    Scenario5_NumUnitsCutSecondStage(4,4)
    Scenario5_NumUnitsCutSecondStage(6,3)
    Scenario5_NumUnitsCutSecondStage(2,2)
    Scenario5_NumUnitsCutSecondStage(10,1)
    Scenario5_NumUnitsCutSecondStage(8,6)
    Scenario5_NumUnitsCutSecondStage(5,3)
    Scenario5_NumUnitsCutSecondStage(4,1)
    Scenario5_NumUnitsCutSecondStage(1,1)
    Scenario5_NumUnitsCutSecondStage(10,9)
    Scenario5_NumUnitsCutSecondStage(9,7)
    Scenario5_NumUnitsCutSecondStage(6,4)
    Scenario5_NumUnitsCutSecondStage(3,2)
    Scenario5_NumUnitsCutSecondStage(8,2)
    Scenario5_NumUnitsCutSecondStage(7,1)
    Scenario5_NumUnitsCutSecondStage(9,3)
    Scenario5_NumUnitsCutSecondStage(8,3)
    Scenario5_NumUnitsCutSecondStage(7,7)
    Scenario5_NumUnitsCutSecondStage(7,5)
    Scenario5_NumUnitsCutSecondStage(8,7)
    Scenario5_NumUnitsCutSecondStage(4,2)
    Scenario5_NumUnitsCutSecondStage(9,6)
    Scenario5_NumUnitsCutSecondStage(6,5)
    Scenario5_NumUnitsCutSecondStage(5,5)
    Scenario5_NumUnitsCutSecondStage(10,5)
    Scenario5_NumUnitsCutSecondStage(10,10)
    Scenario5_NumUnitsCutSecondStage(9,2)
    Scenario5_NumUnitsCutSecondStage(6,1)
    Scenario5_NumUnitsCutSecondStage(3,1)
    Scenario5_NumUnitsCutSecondStage(9,9)
    Scenario5_NumUnitsCutSecondStage(7,4)
    Scenario5_NumUnitsCutSecondStage(10,7)
    Scenario5_NumUnitsCutSecondStage(8,8)
    Scenario5_NumUnitsCutSecondStage(4,3)
    Scenario5_NumUnitsCutSecondStage(9,5)
    Scenario5_NumUnitsCutSecondStage(10,2)
    Scenario5_NumUnitsCutSecondStage(5,2)
    Scenario5_NumUnitsCutSecondStage(8,4)
    Scenario5_NumUnitsCutFirstStage(7,3)
    Scenario5_NumUnitsCutFirstStage(9,1)
    Scenario5_NumUnitsCutFirstStage(6,6)
    Scenario5_NumUnitsCutFirstStage(9,8)
    Scenario5_NumUnitsCutFirstStage(10,6)
    Scenario5_NumUnitsCutFirstStage(5,4)
    Scenario5_NumUnitsCutFirstStage(2,1)
    Scenario5_NumUnitsCutFirstStage(6,2)
    Scenario5_NumUnitsCutFirstStage(9,4)
    Scenario5_NumUnitsCutFirstStage(5,1)
    Scenario5_NumUnitsCutFirstStage(8,5)
    Scenario5_NumUnitsCutFirstStage(7,2)
    Scenario5_NumUnitsCutFirstStage(10,4)
    Scenario5_NumUnitsCutFirstStage(10,8)
    Scenario5_NumUnitsCutFirstStage(3,3)
    Scenario5_NumUnitsCutFirstStage(8,1)
    Scenario5_NumUnitsCutFirstStage(10,3)
    Scenario5_NumUnitsCutFirstStage(7,6)
    Scenario5_NumUnitsCutFirstStage(4,4)
    Scenario5_NumUnitsCutFirstStage(6,3)
    Scenario5_NumUnitsCutFirstStage(2,2)
    Scenario5_NumUnitsCutFirstStage(10,1)
    Scenario5_NumUnitsCutFirstStage(8,6)
    Scenario5_NumUnitsCutFirstStage(5,3)
    Scenario5_NumUnitsCutFirstStage(4,1)
    Scenario5_NumUnitsCutFirstStage(1,1)
    Scenario5_NumUnitsCutFirstStage(10,9)
    Scenario5_NumUnitsCutFirstStage(9,7)
    Scenario5_NumUnitsCutFirstStage(6,4)
    Scenario5_NumUnitsCutFirstStage(3,2)
    Scenario5_NumUnitsCutFirstStage(8,2)
    Scenario5_NumUnitsCutFirstStage(7,1)
    Scenario5_NumUnitsCutFirstStage(9,3)
    Scenario5_NumUnitsCutFirstStage(8,3)
    Scenario5_NumUnitsCutFirstStage(7,7)
    Scenario5_NumUnitsCutFirstStage(7,5)
    Scenario5_NumUnitsCutFirstStage(8,7)
    Scenario5_NumUnitsCutFirstStage(4,2)
    Scenario5_NumUnitsCutFirstStage(9,6)
    Scenario5_NumUnitsCutFirstStage(6,5)
    Scenario5_NumUnitsCutFirstStage(5,5)
    Scenario5_NumUnitsCutFirstStage(10,5)
    Scenario5_NumUnitsCutFirstStage(10,10)
    Scenario5_NumUnitsCutFirstStage(9,2)
    Scenario5_NumUnitsCutFirstStage(6,1)
    Scenario5_NumUnitsCutFirstStage(3,1)
    Scenario5_NumUnitsCutFirstStage(9,9)
    Scenario5_NumUnitsCutFirstStage(7,4)
    Scenario5_NumUnitsCutFirstStage(10,7)
    Scenario5_NumUnitsCutFirstStage(8,8)
    Scenario5_NumUnitsCutFirstStage(4,3)
    Scenario5_NumUnitsCutFirstStage(9,5)
    Scenario5_NumUnitsCutFirstStage(10,2)
    Scenario5_NumUnitsCutFirstStage(5,2)
    Scenario5_NumUnitsCutFirstStage(8,4)
    Scenario5_NumProducedSecondStage(1)
    Scenario5_NumProducedSecondStage(2)
    Scenario5_NumProducedSecondStage(3)
    Scenario5_NumProducedSecondStage(4)
    Scenario5_NumProducedSecondStage(5)
    Scenario5_NumProducedSecondStage(6)
    Scenario5_NumProducedSecondStage(7)
    Scenario5_NumProducedSecondStage(8)
    Scenario5_NumProducedSecondStage(9)
    Scenario5_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario5

\ Begin integer variable block for scenario Scenario4
    Scenario4_NumProducedFirstStage(1)
    Scenario4_NumProducedFirstStage(2)
    Scenario4_NumProducedFirstStage(3)
    Scenario4_NumProducedFirstStage(4)
    Scenario4_NumProducedFirstStage(5)
    Scenario4_NumProducedFirstStage(6)
    Scenario4_NumProducedFirstStage(7)
    Scenario4_NumProducedFirstStage(8)
    Scenario4_NumProducedFirstStage(9)
    Scenario4_NumProducedFirstStage(10)
    Scenario4_NumUnitsCutSecondStage(7,3)
    Scenario4_NumUnitsCutSecondStage(9,1)
    Scenario4_NumUnitsCutSecondStage(6,6)
    Scenario4_NumUnitsCutSecondStage(9,8)
    Scenario4_NumUnitsCutSecondStage(10,6)
    Scenario4_NumUnitsCutSecondStage(5,4)
    Scenario4_NumUnitsCutSecondStage(2,1)
    Scenario4_NumUnitsCutSecondStage(6,2)
    Scenario4_NumUnitsCutSecondStage(9,4)
    Scenario4_NumUnitsCutSecondStage(5,1)
    Scenario4_NumUnitsCutSecondStage(8,5)
    Scenario4_NumUnitsCutSecondStage(7,2)
    Scenario4_NumUnitsCutSecondStage(10,4)
    Scenario4_NumUnitsCutSecondStage(10,8)
    Scenario4_NumUnitsCutSecondStage(3,3)
    Scenario4_NumUnitsCutSecondStage(8,1)
    Scenario4_NumUnitsCutSecondStage(10,3)
    Scenario4_NumUnitsCutSecondStage(7,6)
    Scenario4_NumUnitsCutSecondStage(4,4)
    Scenario4_NumUnitsCutSecondStage(6,3)
    Scenario4_NumUnitsCutSecondStage(2,2)
    Scenario4_NumUnitsCutSecondStage(10,1)
    Scenario4_NumUnitsCutSecondStage(8,6)
    Scenario4_NumUnitsCutSecondStage(5,3)
    Scenario4_NumUnitsCutSecondStage(4,1)
    Scenario4_NumUnitsCutSecondStage(1,1)
    Scenario4_NumUnitsCutSecondStage(10,9)
    Scenario4_NumUnitsCutSecondStage(9,7)
    Scenario4_NumUnitsCutSecondStage(6,4)
    Scenario4_NumUnitsCutSecondStage(3,2)
    Scenario4_NumUnitsCutSecondStage(8,2)
    Scenario4_NumUnitsCutSecondStage(7,1)
    Scenario4_NumUnitsCutSecondStage(9,3)
    Scenario4_NumUnitsCutSecondStage(8,3)
    Scenario4_NumUnitsCutSecondStage(7,7)
    Scenario4_NumUnitsCutSecondStage(7,5)
    Scenario4_NumUnitsCutSecondStage(8,7)
    Scenario4_NumUnitsCutSecondStage(4,2)
    Scenario4_NumUnitsCutSecondStage(9,6)
    Scenario4_NumUnitsCutSecondStage(6,5)
    Scenario4_NumUnitsCutSecondStage(5,5)
    Scenario4_NumUnitsCutSecondStage(10,5)
    Scenario4_NumUnitsCutSecondStage(10,10)
    Scenario4_NumUnitsCutSecondStage(9,2)
    Scenario4_NumUnitsCutSecondStage(6,1)
    Scenario4_NumUnitsCutSecondStage(3,1)
    Scenario4_NumUnitsCutSecondStage(9,9)
    Scenario4_NumUnitsCutSecondStage(7,4)
    Scenario4_NumUnitsCutSecondStage(10,7)
    Scenario4_NumUnitsCutSecondStage(8,8)
    Scenario4_NumUnitsCutSecondStage(4,3)
    Scenario4_NumUnitsCutSecondStage(9,5)
    Scenario4_NumUnitsCutSecondStage(10,2)
    Scenario4_NumUnitsCutSecondStage(5,2)
    Scenario4_NumUnitsCutSecondStage(8,4)
    Scenario4_NumUnitsCutFirstStage(7,3)
    Scenario4_NumUnitsCutFirstStage(9,1)
    Scenario4_NumUnitsCutFirstStage(6,6)
    Scenario4_NumUnitsCutFirstStage(9,8)
    Scenario4_NumUnitsCutFirstStage(10,6)
    Scenario4_NumUnitsCutFirstStage(5,4)
    Scenario4_NumUnitsCutFirstStage(2,1)
    Scenario4_NumUnitsCutFirstStage(6,2)
    Scenario4_NumUnitsCutFirstStage(9,4)
    Scenario4_NumUnitsCutFirstStage(5,1)
    Scenario4_NumUnitsCutFirstStage(8,5)
    Scenario4_NumUnitsCutFirstStage(7,2)
    Scenario4_NumUnitsCutFirstStage(10,4)
    Scenario4_NumUnitsCutFirstStage(10,8)
    Scenario4_NumUnitsCutFirstStage(3,3)
    Scenario4_NumUnitsCutFirstStage(8,1)
    Scenario4_NumUnitsCutFirstStage(10,3)
    Scenario4_NumUnitsCutFirstStage(7,6)
    Scenario4_NumUnitsCutFirstStage(4,4)
    Scenario4_NumUnitsCutFirstStage(6,3)
    Scenario4_NumUnitsCutFirstStage(2,2)
    Scenario4_NumUnitsCutFirstStage(10,1)
    Scenario4_NumUnitsCutFirstStage(8,6)
    Scenario4_NumUnitsCutFirstStage(5,3)
    Scenario4_NumUnitsCutFirstStage(4,1)
    Scenario4_NumUnitsCutFirstStage(1,1)
    Scenario4_NumUnitsCutFirstStage(10,9)
    Scenario4_NumUnitsCutFirstStage(9,7)
    Scenario4_NumUnitsCutFirstStage(6,4)
    Scenario4_NumUnitsCutFirstStage(3,2)
    Scenario4_NumUnitsCutFirstStage(8,2)
    Scenario4_NumUnitsCutFirstStage(7,1)
    Scenario4_NumUnitsCutFirstStage(9,3)
    Scenario4_NumUnitsCutFirstStage(8,3)
    Scenario4_NumUnitsCutFirstStage(7,7)
    Scenario4_NumUnitsCutFirstStage(7,5)
    Scenario4_NumUnitsCutFirstStage(8,7)
    Scenario4_NumUnitsCutFirstStage(4,2)
    Scenario4_NumUnitsCutFirstStage(9,6)
    Scenario4_NumUnitsCutFirstStage(6,5)
    Scenario4_NumUnitsCutFirstStage(5,5)
    Scenario4_NumUnitsCutFirstStage(10,5)
    Scenario4_NumUnitsCutFirstStage(10,10)
    Scenario4_NumUnitsCutFirstStage(9,2)
    Scenario4_NumUnitsCutFirstStage(6,1)
    Scenario4_NumUnitsCutFirstStage(3,1)
    Scenario4_NumUnitsCutFirstStage(9,9)
    Scenario4_NumUnitsCutFirstStage(7,4)
    Scenario4_NumUnitsCutFirstStage(10,7)
    Scenario4_NumUnitsCutFirstStage(8,8)
    Scenario4_NumUnitsCutFirstStage(4,3)
    Scenario4_NumUnitsCutFirstStage(9,5)
    Scenario4_NumUnitsCutFirstStage(10,2)
    Scenario4_NumUnitsCutFirstStage(5,2)
    Scenario4_NumUnitsCutFirstStage(8,4)
    Scenario4_NumProducedSecondStage(1)
    Scenario4_NumProducedSecondStage(2)
    Scenario4_NumProducedSecondStage(3)
    Scenario4_NumProducedSecondStage(4)
    Scenario4_NumProducedSecondStage(5)
    Scenario4_NumProducedSecondStage(6)
    Scenario4_NumProducedSecondStage(7)
    Scenario4_NumProducedSecondStage(8)
    Scenario4_NumProducedSecondStage(9)
    Scenario4_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario4

\ Begin integer variable block for scenario Scenario9
    Scenario9_NumProducedFirstStage(1)
    Scenario9_NumProducedFirstStage(2)
    Scenario9_NumProducedFirstStage(3)
    Scenario9_NumProducedFirstStage(4)
    Scenario9_NumProducedFirstStage(5)
    Scenario9_NumProducedFirstStage(6)
    Scenario9_NumProducedFirstStage(7)
    Scenario9_NumProducedFirstStage(8)
    Scenario9_NumProducedFirstStage(9)
    Scenario9_NumProducedFirstStage(10)
    Scenario9_NumUnitsCutSecondStage(7,3)
    Scenario9_NumUnitsCutSecondStage(9,1)
    Scenario9_NumUnitsCutSecondStage(6,6)
    Scenario9_NumUnitsCutSecondStage(9,8)
    Scenario9_NumUnitsCutSecondStage(10,6)
    Scenario9_NumUnitsCutSecondStage(5,4)
    Scenario9_NumUnitsCutSecondStage(2,1)
    Scenario9_NumUnitsCutSecondStage(6,2)
    Scenario9_NumUnitsCutSecondStage(9,4)
    Scenario9_NumUnitsCutSecondStage(5,1)
    Scenario9_NumUnitsCutSecondStage(8,5)
    Scenario9_NumUnitsCutSecondStage(7,2)
    Scenario9_NumUnitsCutSecondStage(10,4)
    Scenario9_NumUnitsCutSecondStage(10,8)
    Scenario9_NumUnitsCutSecondStage(3,3)
    Scenario9_NumUnitsCutSecondStage(8,1)
    Scenario9_NumUnitsCutSecondStage(10,3)
    Scenario9_NumUnitsCutSecondStage(7,6)
    Scenario9_NumUnitsCutSecondStage(4,4)
    Scenario9_NumUnitsCutSecondStage(6,3)
    Scenario9_NumUnitsCutSecondStage(2,2)
    Scenario9_NumUnitsCutSecondStage(10,1)
    Scenario9_NumUnitsCutSecondStage(8,6)
    Scenario9_NumUnitsCutSecondStage(5,3)
    Scenario9_NumUnitsCutSecondStage(4,1)
    Scenario9_NumUnitsCutSecondStage(1,1)
    Scenario9_NumUnitsCutSecondStage(10,9)
    Scenario9_NumUnitsCutSecondStage(9,7)
    Scenario9_NumUnitsCutSecondStage(6,4)
    Scenario9_NumUnitsCutSecondStage(3,2)
    Scenario9_NumUnitsCutSecondStage(8,2)
    Scenario9_NumUnitsCutSecondStage(7,1)
    Scenario9_NumUnitsCutSecondStage(9,3)
    Scenario9_NumUnitsCutSecondStage(8,3)
    Scenario9_NumUnitsCutSecondStage(7,7)
    Scenario9_NumUnitsCutSecondStage(7,5)
    Scenario9_NumUnitsCutSecondStage(8,7)
    Scenario9_NumUnitsCutSecondStage(4,2)
    Scenario9_NumUnitsCutSecondStage(9,6)
    Scenario9_NumUnitsCutSecondStage(6,5)
    Scenario9_NumUnitsCutSecondStage(5,5)
    Scenario9_NumUnitsCutSecondStage(10,5)
    Scenario9_NumUnitsCutSecondStage(10,10)
    Scenario9_NumUnitsCutSecondStage(9,2)
    Scenario9_NumUnitsCutSecondStage(6,1)
    Scenario9_NumUnitsCutSecondStage(3,1)
    Scenario9_NumUnitsCutSecondStage(9,9)
    Scenario9_NumUnitsCutSecondStage(7,4)
    Scenario9_NumUnitsCutSecondStage(10,7)
    Scenario9_NumUnitsCutSecondStage(8,8)
    Scenario9_NumUnitsCutSecondStage(4,3)
    Scenario9_NumUnitsCutSecondStage(9,5)
    Scenario9_NumUnitsCutSecondStage(10,2)
    Scenario9_NumUnitsCutSecondStage(5,2)
    Scenario9_NumUnitsCutSecondStage(8,4)
    Scenario9_NumUnitsCutFirstStage(7,3)
    Scenario9_NumUnitsCutFirstStage(9,1)
    Scenario9_NumUnitsCutFirstStage(6,6)
    Scenario9_NumUnitsCutFirstStage(9,8)
    Scenario9_NumUnitsCutFirstStage(10,6)
    Scenario9_NumUnitsCutFirstStage(5,4)
    Scenario9_NumUnitsCutFirstStage(2,1)
    Scenario9_NumUnitsCutFirstStage(6,2)
    Scenario9_NumUnitsCutFirstStage(9,4)
    Scenario9_NumUnitsCutFirstStage(5,1)
    Scenario9_NumUnitsCutFirstStage(8,5)
    Scenario9_NumUnitsCutFirstStage(7,2)
    Scenario9_NumUnitsCutFirstStage(10,4)
    Scenario9_NumUnitsCutFirstStage(10,8)
    Scenario9_NumUnitsCutFirstStage(3,3)
    Scenario9_NumUnitsCutFirstStage(8,1)
    Scenario9_NumUnitsCutFirstStage(10,3)
    Scenario9_NumUnitsCutFirstStage(7,6)
    Scenario9_NumUnitsCutFirstStage(4,4)
    Scenario9_NumUnitsCutFirstStage(6,3)
    Scenario9_NumUnitsCutFirstStage(2,2)
    Scenario9_NumUnitsCutFirstStage(10,1)
    Scenario9_NumUnitsCutFirstStage(8,6)
    Scenario9_NumUnitsCutFirstStage(5,3)
    Scenario9_NumUnitsCutFirstStage(4,1)
    Scenario9_NumUnitsCutFirstStage(1,1)
    Scenario9_NumUnitsCutFirstStage(10,9)
    Scenario9_NumUnitsCutFirstStage(9,7)
    Scenario9_NumUnitsCutFirstStage(6,4)
    Scenario9_NumUnitsCutFirstStage(3,2)
    Scenario9_NumUnitsCutFirstStage(8,2)
    Scenario9_NumUnitsCutFirstStage(7,1)
    Scenario9_NumUnitsCutFirstStage(9,3)
    Scenario9_NumUnitsCutFirstStage(8,3)
    Scenario9_NumUnitsCutFirstStage(7,7)
    Scenario9_NumUnitsCutFirstStage(7,5)
    Scenario9_NumUnitsCutFirstStage(8,7)
    Scenario9_NumUnitsCutFirstStage(4,2)
    Scenario9_NumUnitsCutFirstStage(9,6)
    Scenario9_NumUnitsCutFirstStage(6,5)
    Scenario9_NumUnitsCutFirstStage(5,5)
    Scenario9_NumUnitsCutFirstStage(10,5)
    Scenario9_NumUnitsCutFirstStage(10,10)
    Scenario9_NumUnitsCutFirstStage(9,2)
    Scenario9_NumUnitsCutFirstStage(6,1)
    Scenario9_NumUnitsCutFirstStage(3,1)
    Scenario9_NumUnitsCutFirstStage(9,9)
    Scenario9_NumUnitsCutFirstStage(7,4)
    Scenario9_NumUnitsCutFirstStage(10,7)
    Scenario9_NumUnitsCutFirstStage(8,8)
    Scenario9_NumUnitsCutFirstStage(4,3)
    Scenario9_NumUnitsCutFirstStage(9,5)
    Scenario9_NumUnitsCutFirstStage(10,2)
    Scenario9_NumUnitsCutFirstStage(5,2)
    Scenario9_NumUnitsCutFirstStage(8,4)
    Scenario9_NumProducedSecondStage(1)
    Scenario9_NumProducedSecondStage(2)
    Scenario9_NumProducedSecondStage(3)
    Scenario9_NumProducedSecondStage(4)
    Scenario9_NumProducedSecondStage(5)
    Scenario9_NumProducedSecondStage(6)
    Scenario9_NumProducedSecondStage(7)
    Scenario9_NumProducedSecondStage(8)
    Scenario9_NumProducedSecondStage(9)
    Scenario9_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario9

\ Begin integer variable block for scenario Scenario8
    Scenario8_NumProducedFirstStage(1)
    Scenario8_NumProducedFirstStage(2)
    Scenario8_NumProducedFirstStage(3)
    Scenario8_NumProducedFirstStage(4)
    Scenario8_NumProducedFirstStage(5)
    Scenario8_NumProducedFirstStage(6)
    Scenario8_NumProducedFirstStage(7)
    Scenario8_NumProducedFirstStage(8)
    Scenario8_NumProducedFirstStage(9)
    Scenario8_NumProducedFirstStage(10)
    Scenario8_NumUnitsCutSecondStage(7,3)
    Scenario8_NumUnitsCutSecondStage(9,1)
    Scenario8_NumUnitsCutSecondStage(6,6)
    Scenario8_NumUnitsCutSecondStage(9,8)
    Scenario8_NumUnitsCutSecondStage(10,6)
    Scenario8_NumUnitsCutSecondStage(5,4)
    Scenario8_NumUnitsCutSecondStage(2,1)
    Scenario8_NumUnitsCutSecondStage(6,2)
    Scenario8_NumUnitsCutSecondStage(9,4)
    Scenario8_NumUnitsCutSecondStage(5,1)
    Scenario8_NumUnitsCutSecondStage(8,5)
    Scenario8_NumUnitsCutSecondStage(7,2)
    Scenario8_NumUnitsCutSecondStage(10,4)
    Scenario8_NumUnitsCutSecondStage(10,8)
    Scenario8_NumUnitsCutSecondStage(3,3)
    Scenario8_NumUnitsCutSecondStage(8,1)
    Scenario8_NumUnitsCutSecondStage(10,3)
    Scenario8_NumUnitsCutSecondStage(7,6)
    Scenario8_NumUnitsCutSecondStage(4,4)
    Scenario8_NumUnitsCutSecondStage(6,3)
    Scenario8_NumUnitsCutSecondStage(2,2)
    Scenario8_NumUnitsCutSecondStage(10,1)
    Scenario8_NumUnitsCutSecondStage(8,6)
    Scenario8_NumUnitsCutSecondStage(5,3)
    Scenario8_NumUnitsCutSecondStage(4,1)
    Scenario8_NumUnitsCutSecondStage(1,1)
    Scenario8_NumUnitsCutSecondStage(10,9)
    Scenario8_NumUnitsCutSecondStage(9,7)
    Scenario8_NumUnitsCutSecondStage(6,4)
    Scenario8_NumUnitsCutSecondStage(3,2)
    Scenario8_NumUnitsCutSecondStage(8,2)
    Scenario8_NumUnitsCutSecondStage(7,1)
    Scenario8_NumUnitsCutSecondStage(9,3)
    Scenario8_NumUnitsCutSecondStage(8,3)
    Scenario8_NumUnitsCutSecondStage(7,7)
    Scenario8_NumUnitsCutSecondStage(7,5)
    Scenario8_NumUnitsCutSecondStage(8,7)
    Scenario8_NumUnitsCutSecondStage(4,2)
    Scenario8_NumUnitsCutSecondStage(9,6)
    Scenario8_NumUnitsCutSecondStage(6,5)
    Scenario8_NumUnitsCutSecondStage(5,5)
    Scenario8_NumUnitsCutSecondStage(10,5)
    Scenario8_NumUnitsCutSecondStage(10,10)
    Scenario8_NumUnitsCutSecondStage(9,2)
    Scenario8_NumUnitsCutSecondStage(6,1)
    Scenario8_NumUnitsCutSecondStage(3,1)
    Scenario8_NumUnitsCutSecondStage(9,9)
    Scenario8_NumUnitsCutSecondStage(7,4)
    Scenario8_NumUnitsCutSecondStage(10,7)
    Scenario8_NumUnitsCutSecondStage(8,8)
    Scenario8_NumUnitsCutSecondStage(4,3)
    Scenario8_NumUnitsCutSecondStage(9,5)
    Scenario8_NumUnitsCutSecondStage(10,2)
    Scenario8_NumUnitsCutSecondStage(5,2)
    Scenario8_NumUnitsCutSecondStage(8,4)
    Scenario8_NumUnitsCutFirstStage(7,3)
    Scenario8_NumUnitsCutFirstStage(9,1)
    Scenario8_NumUnitsCutFirstStage(6,6)
    Scenario8_NumUnitsCutFirstStage(9,8)
    Scenario8_NumUnitsCutFirstStage(10,6)
    Scenario8_NumUnitsCutFirstStage(5,4)
    Scenario8_NumUnitsCutFirstStage(2,1)
    Scenario8_NumUnitsCutFirstStage(6,2)
    Scenario8_NumUnitsCutFirstStage(9,4)
    Scenario8_NumUnitsCutFirstStage(5,1)
    Scenario8_NumUnitsCutFirstStage(8,5)
    Scenario8_NumUnitsCutFirstStage(7,2)
    Scenario8_NumUnitsCutFirstStage(10,4)
    Scenario8_NumUnitsCutFirstStage(10,8)
    Scenario8_NumUnitsCutFirstStage(3,3)
    Scenario8_NumUnitsCutFirstStage(8,1)
    Scenario8_NumUnitsCutFirstStage(10,3)
    Scenario8_NumUnitsCutFirstStage(7,6)
    Scenario8_NumUnitsCutFirstStage(4,4)
    Scenario8_NumUnitsCutFirstStage(6,3)
    Scenario8_NumUnitsCutFirstStage(2,2)
    Scenario8_NumUnitsCutFirstStage(10,1)
    Scenario8_NumUnitsCutFirstStage(8,6)
    Scenario8_NumUnitsCutFirstStage(5,3)
    Scenario8_NumUnitsCutFirstStage(4,1)
    Scenario8_NumUnitsCutFirstStage(1,1)
    Scenario8_NumUnitsCutFirstStage(10,9)
    Scenario8_NumUnitsCutFirstStage(9,7)
    Scenario8_NumUnitsCutFirstStage(6,4)
    Scenario8_NumUnitsCutFirstStage(3,2)
    Scenario8_NumUnitsCutFirstStage(8,2)
    Scenario8_NumUnitsCutFirstStage(7,1)
    Scenario8_NumUnitsCutFirstStage(9,3)
    Scenario8_NumUnitsCutFirstStage(8,3)
    Scenario8_NumUnitsCutFirstStage(7,7)
    Scenario8_NumUnitsCutFirstStage(7,5)
    Scenario8_NumUnitsCutFirstStage(8,7)
    Scenario8_NumUnitsCutFirstStage(4,2)
    Scenario8_NumUnitsCutFirstStage(9,6)
    Scenario8_NumUnitsCutFirstStage(6,5)
    Scenario8_NumUnitsCutFirstStage(5,5)
    Scenario8_NumUnitsCutFirstStage(10,5)
    Scenario8_NumUnitsCutFirstStage(10,10)
    Scenario8_NumUnitsCutFirstStage(9,2)
    Scenario8_NumUnitsCutFirstStage(6,1)
    Scenario8_NumUnitsCutFirstStage(3,1)
    Scenario8_NumUnitsCutFirstStage(9,9)
    Scenario8_NumUnitsCutFirstStage(7,4)
    Scenario8_NumUnitsCutFirstStage(10,7)
    Scenario8_NumUnitsCutFirstStage(8,8)
    Scenario8_NumUnitsCutFirstStage(4,3)
    Scenario8_NumUnitsCutFirstStage(9,5)
    Scenario8_NumUnitsCutFirstStage(10,2)
    Scenario8_NumUnitsCutFirstStage(5,2)
    Scenario8_NumUnitsCutFirstStage(8,4)
    Scenario8_NumProducedSecondStage(1)
    Scenario8_NumProducedSecondStage(2)
    Scenario8_NumProducedSecondStage(3)
    Scenario8_NumProducedSecondStage(4)
    Scenario8_NumProducedSecondStage(5)
    Scenario8_NumProducedSecondStage(6)
    Scenario8_NumProducedSecondStage(7)
    Scenario8_NumProducedSecondStage(8)
    Scenario8_NumProducedSecondStage(9)
    Scenario8_NumProducedSecondStage(10)
\ End integer variable block for scenario Scenario8

binary

\ Begin binary variable block for master
    MASTER_RootNode_ProduceSizeFirstStage(1)
    MASTER_RootNode_ProduceSizeFirstStage(2)
    MASTER_RootNode_ProduceSizeFirstStage(3)
    MASTER_RootNode_ProduceSizeFirstStage(4)
    MASTER_RootNode_ProduceSizeFirstStage(5)
    MASTER_RootNode_ProduceSizeFirstStage(6)
    MASTER_RootNode_ProduceSizeFirstStage(7)
    MASTER_RootNode_ProduceSizeFirstStage(8)
    MASTER_RootNode_ProduceSizeFirstStage(9)
    MASTER_RootNode_ProduceSizeFirstStage(10)
\ End binary variable block for master

\ Begin binary variable block for scenario Scenario10
    Scenario10_ProduceSizeSecondStage(1)
    Scenario10_ProduceSizeSecondStage(2)
    Scenario10_ProduceSizeSecondStage(3)
    Scenario10_ProduceSizeSecondStage(4)
    Scenario10_ProduceSizeSecondStage(5)
    Scenario10_ProduceSizeSecondStage(6)
    Scenario10_ProduceSizeSecondStage(7)
    Scenario10_ProduceSizeSecondStage(8)
    Scenario10_ProduceSizeSecondStage(9)
    Scenario10_ProduceSizeSecondStage(10)
    Scenario10_ProduceSizeFirstStage(1)
    Scenario10_ProduceSizeFirstStage(2)
    Scenario10_ProduceSizeFirstStage(3)
    Scenario10_ProduceSizeFirstStage(4)
    Scenario10_ProduceSizeFirstStage(5)
    Scenario10_ProduceSizeFirstStage(6)
    Scenario10_ProduceSizeFirstStage(7)
    Scenario10_ProduceSizeFirstStage(8)
    Scenario10_ProduceSizeFirstStage(9)
    Scenario10_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario10

\ Begin binary variable block for scenario Scenario3
    Scenario3_ProduceSizeSecondStage(1)
    Scenario3_ProduceSizeSecondStage(2)
    Scenario3_ProduceSizeSecondStage(3)
    Scenario3_ProduceSizeSecondStage(4)
    Scenario3_ProduceSizeSecondStage(5)
    Scenario3_ProduceSizeSecondStage(6)
    Scenario3_ProduceSizeSecondStage(7)
    Scenario3_ProduceSizeSecondStage(8)
    Scenario3_ProduceSizeSecondStage(9)
    Scenario3_ProduceSizeSecondStage(10)
    Scenario3_ProduceSizeFirstStage(1)
    Scenario3_ProduceSizeFirstStage(2)
    Scenario3_ProduceSizeFirstStage(3)
    Scenario3_ProduceSizeFirstStage(4)
    Scenario3_ProduceSizeFirstStage(5)
    Scenario3_ProduceSizeFirstStage(6)
    Scenario3_ProduceSizeFirstStage(7)
    Scenario3_ProduceSizeFirstStage(8)
    Scenario3_ProduceSizeFirstStage(9)
    Scenario3_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario3

\ Begin binary variable block for scenario Scenario2
    Scenario2_ProduceSizeSecondStage(1)
    Scenario2_ProduceSizeSecondStage(2)
    Scenario2_ProduceSizeSecondStage(3)
    Scenario2_ProduceSizeSecondStage(4)
    Scenario2_ProduceSizeSecondStage(5)
    Scenario2_ProduceSizeSecondStage(6)
    Scenario2_ProduceSizeSecondStage(7)
    Scenario2_ProduceSizeSecondStage(8)
    Scenario2_ProduceSizeSecondStage(9)
    Scenario2_ProduceSizeSecondStage(10)
    Scenario2_ProduceSizeFirstStage(1)
    Scenario2_ProduceSizeFirstStage(2)
    Scenario2_ProduceSizeFirstStage(3)
    Scenario2_ProduceSizeFirstStage(4)
    Scenario2_ProduceSizeFirstStage(5)
    Scenario2_ProduceSizeFirstStage(6)
    Scenario2_ProduceSizeFirstStage(7)
    Scenario2_ProduceSizeFirstStage(8)
    Scenario2_ProduceSizeFirstStage(9)
    Scenario2_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario2

\ Begin binary variable block for scenario Scenario1
    Scenario1_ProduceSizeSecondStage(1)
    Scenario1_ProduceSizeSecondStage(2)
    Scenario1_ProduceSizeSecondStage(3)
    Scenario1_ProduceSizeSecondStage(4)
    Scenario1_ProduceSizeSecondStage(5)
    Scenario1_ProduceSizeSecondStage(6)
    Scenario1_ProduceSizeSecondStage(7)
    Scenario1_ProduceSizeSecondStage(8)
    Scenario1_ProduceSizeSecondStage(9)
    Scenario1_ProduceSizeSecondStage(10)
    Scenario1_ProduceSizeFirstStage(1)
    Scenario1_ProduceSizeFirstStage(2)
    Scenario1_ProduceSizeFirstStage(3)
    Scenario1_ProduceSizeFirstStage(4)
    Scenario1_ProduceSizeFirstStage(5)
    Scenario1_ProduceSizeFirstStage(6)
    Scenario1_ProduceSizeFirstStage(7)
    Scenario1_ProduceSizeFirstStage(8)
    Scenario1_ProduceSizeFirstStage(9)
    Scenario1_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario1

\ Begin binary variable block for scenario Scenario7
    Scenario7_ProduceSizeSecondStage(1)
    Scenario7_ProduceSizeSecondStage(2)
    Scenario7_ProduceSizeSecondStage(3)
    Scenario7_ProduceSizeSecondStage(4)
    Scenario7_ProduceSizeSecondStage(5)
    Scenario7_ProduceSizeSecondStage(6)
    Scenario7_ProduceSizeSecondStage(7)
    Scenario7_ProduceSizeSecondStage(8)
    Scenario7_ProduceSizeSecondStage(9)
    Scenario7_ProduceSizeSecondStage(10)
    Scenario7_ProduceSizeFirstStage(1)
    Scenario7_ProduceSizeFirstStage(2)
    Scenario7_ProduceSizeFirstStage(3)
    Scenario7_ProduceSizeFirstStage(4)
    Scenario7_ProduceSizeFirstStage(5)
    Scenario7_ProduceSizeFirstStage(6)
    Scenario7_ProduceSizeFirstStage(7)
    Scenario7_ProduceSizeFirstStage(8)
    Scenario7_ProduceSizeFirstStage(9)
    Scenario7_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario7

\ Begin binary variable block for scenario Scenario6
    Scenario6_ProduceSizeSecondStage(1)
    Scenario6_ProduceSizeSecondStage(2)
    Scenario6_ProduceSizeSecondStage(3)
    Scenario6_ProduceSizeSecondStage(4)
    Scenario6_ProduceSizeSecondStage(5)
    Scenario6_ProduceSizeSecondStage(6)
    Scenario6_ProduceSizeSecondStage(7)
    Scenario6_ProduceSizeSecondStage(8)
    Scenario6_ProduceSizeSecondStage(9)
    Scenario6_ProduceSizeSecondStage(10)
    Scenario6_ProduceSizeFirstStage(1)
    Scenario6_ProduceSizeFirstStage(2)
    Scenario6_ProduceSizeFirstStage(3)
    Scenario6_ProduceSizeFirstStage(4)
    Scenario6_ProduceSizeFirstStage(5)
    Scenario6_ProduceSizeFirstStage(6)
    Scenario6_ProduceSizeFirstStage(7)
    Scenario6_ProduceSizeFirstStage(8)
    Scenario6_ProduceSizeFirstStage(9)
    Scenario6_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario6

\ Begin binary variable block for scenario Scenario5
    Scenario5_ProduceSizeSecondStage(1)
    Scenario5_ProduceSizeSecondStage(2)
    Scenario5_ProduceSizeSecondStage(3)
    Scenario5_ProduceSizeSecondStage(4)
    Scenario5_ProduceSizeSecondStage(5)
    Scenario5_ProduceSizeSecondStage(6)
    Scenario5_ProduceSizeSecondStage(7)
    Scenario5_ProduceSizeSecondStage(8)
    Scenario5_ProduceSizeSecondStage(9)
    Scenario5_ProduceSizeSecondStage(10)
    Scenario5_ProduceSizeFirstStage(1)
    Scenario5_ProduceSizeFirstStage(2)
    Scenario5_ProduceSizeFirstStage(3)
    Scenario5_ProduceSizeFirstStage(4)
    Scenario5_ProduceSizeFirstStage(5)
    Scenario5_ProduceSizeFirstStage(6)
    Scenario5_ProduceSizeFirstStage(7)
    Scenario5_ProduceSizeFirstStage(8)
    Scenario5_ProduceSizeFirstStage(9)
    Scenario5_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario5

\ Begin binary variable block for scenario Scenario4
    Scenario4_ProduceSizeSecondStage(1)
    Scenario4_ProduceSizeSecondStage(2)
    Scenario4_ProduceSizeSecondStage(3)
    Scenario4_ProduceSizeSecondStage(4)
    Scenario4_ProduceSizeSecondStage(5)
    Scenario4_ProduceSizeSecondStage(6)
    Scenario4_ProduceSizeSecondStage(7)
    Scenario4_ProduceSizeSecondStage(8)
    Scenario4_ProduceSizeSecondStage(9)
    Scenario4_ProduceSizeSecondStage(10)
    Scenario4_ProduceSizeFirstStage(1)
    Scenario4_ProduceSizeFirstStage(2)
    Scenario4_ProduceSizeFirstStage(3)
    Scenario4_ProduceSizeFirstStage(4)
    Scenario4_ProduceSizeFirstStage(5)
    Scenario4_ProduceSizeFirstStage(6)
    Scenario4_ProduceSizeFirstStage(7)
    Scenario4_ProduceSizeFirstStage(8)
    Scenario4_ProduceSizeFirstStage(9)
    Scenario4_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario4

\ Begin binary variable block for scenario Scenario9
    Scenario9_ProduceSizeSecondStage(1)
    Scenario9_ProduceSizeSecondStage(2)
    Scenario9_ProduceSizeSecondStage(3)
    Scenario9_ProduceSizeSecondStage(4)
    Scenario9_ProduceSizeSecondStage(5)
    Scenario9_ProduceSizeSecondStage(6)
    Scenario9_ProduceSizeSecondStage(7)
    Scenario9_ProduceSizeSecondStage(8)
    Scenario9_ProduceSizeSecondStage(9)
    Scenario9_ProduceSizeSecondStage(10)
    Scenario9_ProduceSizeFirstStage(1)
    Scenario9_ProduceSizeFirstStage(2)
    Scenario9_ProduceSizeFirstStage(3)
    Scenario9_ProduceSizeFirstStage(4)
    Scenario9_ProduceSizeFirstStage(5)
    Scenario9_ProduceSizeFirstStage(6)
    Scenario9_ProduceSizeFirstStage(7)
    Scenario9_ProduceSizeFirstStage(8)
    Scenario9_ProduceSizeFirstStage(9)
    Scenario9_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario9

\ Begin binary variable block for scenario Scenario8
    Scenario8_ProduceSizeSecondStage(1)
    Scenario8_ProduceSizeSecondStage(2)
    Scenario8_ProduceSizeSecondStage(3)
    Scenario8_ProduceSizeSecondStage(4)
    Scenario8_ProduceSizeSecondStage(5)
    Scenario8_ProduceSizeSecondStage(6)
    Scenario8_ProduceSizeSecondStage(7)
    Scenario8_ProduceSizeSecondStage(8)
    Scenario8_ProduceSizeSecondStage(9)
    Scenario8_ProduceSizeSecondStage(10)
    Scenario8_ProduceSizeFirstStage(1)
    Scenario8_ProduceSizeFirstStage(2)
    Scenario8_ProduceSizeFirstStage(3)
    Scenario8_ProduceSizeFirstStage(4)
    Scenario8_ProduceSizeFirstStage(5)
    Scenario8_ProduceSizeFirstStage(6)
    Scenario8_ProduceSizeFirstStage(7)
    Scenario8_ProduceSizeFirstStage(8)
    Scenario8_ProduceSizeFirstStage(9)
    Scenario8_ProduceSizeFirstStage(10)
\ End integer binary block for scenario Scenario8

end
