\ Begin objective block for master
min 
obj:  1.0 MASTER_EXPECTED_COST_RootNode 
\ End objective block for master

s.t.

\ Begin constraint block for master
MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario3_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario3_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario1_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario1_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario2_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario2_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario1_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario1_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario1_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario1_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario2_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_Scenario2Node_SecondStageCost - 1.0 Scenario2_SecondStageCost >= 0.0

MASTER_c_u_Scenario2_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_Scenario2Node_SecondStageCost - 1.0 Scenario2_SecondStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario3_FirstStageCost >= 0.0

MASTER_c_u_Scenario3_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario3_FirstStageCost <= 0.0

MASTER_c_l_Scenario2_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario2_FirstStageCost >= 0.0

MASTER_c_u_Scenario2_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario2_FirstStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario3_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario3_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario1_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario1_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario1_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario1_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario2_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario2_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario3_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario3_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario2_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario2_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario3_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_Scenario3Node_SecondStageCost - 1.0 Scenario3_SecondStageCost >= 0.0

MASTER_c_u_Scenario3_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_Scenario3Node_SecondStageCost - 1.0 Scenario3_SecondStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario3_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario3_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario1_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario1_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario2_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario2_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario1_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario1_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario2_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario2_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario2_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario2_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario3_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario3_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario3_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario3_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario3_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario3_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario2_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario2_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario3_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario3_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario3_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario3_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario1_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario1_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario3_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario3_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario1_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario1_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario2_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario2_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario1_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario1_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario2_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario2_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario2_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario2_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario3_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario3_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario2_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario2_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario2_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario2_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario2_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario2_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario2_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario2_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario2_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario2_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario2_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario2_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario2_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario2_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario1_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario1_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario1_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario1_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario1_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario1_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario1_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario1_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario1_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario1_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario1_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario1_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario1_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario1_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario2_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario2_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario2_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario2_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario2_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario2_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario2_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario2_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario1_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario1_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario2_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario2_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario1_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario1_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario1_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario1_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario3_NumUnitsCutFirstStage(9,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,4) - 1.0 Scenario3_NumUnitsCutFirstStage(9,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario1_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario1_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario3_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario3_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario3_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario3_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario3_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario3_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario3_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario3_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario3_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario3_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario3_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario3_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_COST_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario3Node - 1.0 MASTER_Scenario3Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario3Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario3Node - 1.0 MASTER_Scenario3Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario1_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario1_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario2_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario2_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario3_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario3_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario3_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario3_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario2_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario2_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario3_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario3_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario3_NumUnitsCutFirstStage(7,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,3) - 1.0 Scenario3_NumUnitsCutFirstStage(7,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario3_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario3_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario3_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario3_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario2_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario2_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario1_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario1_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_COST_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario1Node - 1.0 MASTER_Scenario1Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario1Node - 1.0 MASTER_Scenario1Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario1_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario1_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario2_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario2_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario2_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario2_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario2_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario2_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario1_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario1_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario2_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario2_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario2_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario2_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario2_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario2_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario3_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario3_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario2_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario2_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario2_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario2_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario3_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario3_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario1_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario1_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario1_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario1_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario1_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario1_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario1_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario1_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario1_NumUnitsCutFirstStage(10,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,3) - 1.0 Scenario1_NumUnitsCutFirstStage(10,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario3_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario3_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario2_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario2_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario2_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario2_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario3_NumUnitsCutFirstStage(3,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,2) - 1.0 Scenario3_NumUnitsCutFirstStage(3,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario2_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario2_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario2_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario2_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario2_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario2_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario2_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario2_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario2_NumUnitsCutFirstStage(4,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,4) - 1.0 Scenario2_NumUnitsCutFirstStage(4,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario2_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario2_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario2_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario1_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario1_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario1_NumUnitsCutFirstStage(7,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,5) - 1.0 Scenario1_NumUnitsCutFirstStage(7,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario3_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario3_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario2_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario2_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario2_NumUnitsCutFirstStage(5,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,4) - 1.0 Scenario2_NumUnitsCutFirstStage(5,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario3_ProduceSizeFirstStage(9) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_9_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(9) - 1.0 Scenario3_ProduceSizeFirstStage(9) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario3_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario3_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario1_NumUnitsCutFirstStage(8,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,5) - 1.0 Scenario1_NumUnitsCutFirstStage(8,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario3_ProduceSizeFirstStage(5) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_5_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(5) - 1.0 Scenario3_ProduceSizeFirstStage(5) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario3_ProduceSizeFirstStage(4) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_4_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(4) - 1.0 Scenario3_ProduceSizeFirstStage(4) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario3_ProduceSizeFirstStage(7) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_7_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(7) - 1.0 Scenario3_ProduceSizeFirstStage(7) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario3_ProduceSizeFirstStage(6) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_6_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(6) - 1.0 Scenario3_ProduceSizeFirstStage(6) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario3_ProduceSizeFirstStage(1) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_1_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(1) - 1.0 Scenario3_ProduceSizeFirstStage(1) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario3_ProduceSizeFirstStage(3) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_3_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(3) - 1.0 Scenario3_ProduceSizeFirstStage(3) <= 0.0

MASTER_c_l_Scenario3_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario3_ProduceSizeFirstStage(2) >= 0.0

MASTER_c_u_Scenario3_RootNode_ProduceSizeFirstStage_2_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(2) - 1.0 Scenario3_ProduceSizeFirstStage(2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario2_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario2_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario1_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario1_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario2_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario2_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario2_NumUnitsCutFirstStage(10,4) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,4) - 1.0 Scenario2_NumUnitsCutFirstStage(10,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario1_ProduceSizeFirstStage(10) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_10_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(10) - 1.0 Scenario1_ProduceSizeFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario1_NumUnitsCutFirstStage(6,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,6) - 1.0 Scenario1_NumUnitsCutFirstStage(6,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario3_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario3_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario3_NumUnitsCutFirstStage(3,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,3) - 1.0 Scenario3_NumUnitsCutFirstStage(3,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario1_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario1_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario3_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario3_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario3_NumUnitsCutFirstStage(4,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,2) - 1.0 Scenario3_NumUnitsCutFirstStage(4,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario1_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario1_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario2_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario2_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario2_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario2_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario2_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario2_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario3_NumUnitsCutFirstStage(8,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,3) - 1.0 Scenario3_NumUnitsCutFirstStage(8,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario3_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario3_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario2_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario2_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario1_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario1_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario1_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario1_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario1_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario1_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario2_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario2_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario1_NumProducedFirstStage(4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_4_:  1.0 MASTER_RootNode_NumProducedFirstStage(4) - 1.0 Scenario1_NumProducedFirstStage(4) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario1_NumProducedFirstStage(5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_5_:  1.0 MASTER_RootNode_NumProducedFirstStage(5) - 1.0 Scenario1_NumProducedFirstStage(5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario1_NumProducedFirstStage(6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_6_:  1.0 MASTER_RootNode_NumProducedFirstStage(6) - 1.0 Scenario1_NumProducedFirstStage(6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario1_NumProducedFirstStage(7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_7_:  1.0 MASTER_RootNode_NumProducedFirstStage(7) - 1.0 Scenario1_NumProducedFirstStage(7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario1_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario1_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario2_NumProducedFirstStage(10) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumProducedFirstStage_10_:  1.0 MASTER_RootNode_NumProducedFirstStage(10) - 1.0 Scenario2_NumProducedFirstStage(10) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario1_NumProducedFirstStage(2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_2_:  1.0 MASTER_RootNode_NumProducedFirstStage(2) - 1.0 Scenario1_NumProducedFirstStage(2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario1_NumProducedFirstStage(3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_3_:  1.0 MASTER_RootNode_NumProducedFirstStage(3) - 1.0 Scenario1_NumProducedFirstStage(3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario2_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario2_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario1_NumProducedFirstStage(8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_8_:  1.0 MASTER_RootNode_NumProducedFirstStage(8) - 1.0 Scenario1_NumProducedFirstStage(8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario3_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario3_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario2_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario2_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario3_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario3_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_COST_RootNode_FirstStageCost_None_:  1.0 MASTER_EXPECTED_COST_RootNode - 0.333333333 MASTER_EXPECTED_COST_Scenario1Node - 0.333333333 MASTER_EXPECTED_COST_Scenario2Node - 0.333333334 MASTER_EXPECTED_COST_Scenario3Node - 1.0 MASTER_RootNode_FirstStageCost 
>= 0.0

MASTER_c_u_COST_RootNode_FirstStageCost_None_:  1.0 MASTER_EXPECTED_COST_RootNode - 0.333333333 MASTER_EXPECTED_COST_Scenario1Node - 0.333333333 MASTER_EXPECTED_COST_Scenario2Node - 0.333333334 MASTER_EXPECTED_COST_Scenario3Node - 1.0 MASTER_RootNode_FirstStageCost 
<= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario3_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario3_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario3_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario3_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario1_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario1_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario3_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario3_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario2_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario2_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario1_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario1_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario1_NumUnitsCutFirstStage(2,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,2) - 1.0 Scenario1_NumUnitsCutFirstStage(2,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario3_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario3_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario2_NumUnitsCutFirstStage(7,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,6) - 1.0 Scenario2_NumUnitsCutFirstStage(7,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario3_NumUnitsCutFirstStage(4,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(4,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,1) - 1.0 Scenario3_NumUnitsCutFirstStage(4,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario3_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario3_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario1_NumUnitsCutFirstStage(8,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,7) - 1.0 Scenario1_NumUnitsCutFirstStage(8,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario1_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario1_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario1_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario1_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario2_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario2_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario2_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario2_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario3_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario3_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario3_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario3_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario2_NumUnitsCutFirstStage(10,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,6) - 1.0 Scenario2_NumUnitsCutFirstStage(10,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario1_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario1_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario1_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_Scenario1Node_SecondStageCost - 1.0 Scenario1_SecondStageCost >= 0.0

MASTER_c_u_Scenario1_Scenario1Node_SecondStageCost_None_:  1.0 MASTER_Scenario1Node_SecondStageCost - 1.0 Scenario1_SecondStageCost <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario3_NumUnitsCutFirstStage(5,3) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,3) - 1.0 Scenario3_NumUnitsCutFirstStage(5,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario1_NumUnitsCutFirstStage(9,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,2) - 1.0 Scenario1_NumUnitsCutFirstStage(9,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario3_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario3_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario3_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario3_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario2_NumUnitsCutFirstStage(1,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(1,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(1,1) - 1.0 Scenario2_NumUnitsCutFirstStage(1,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario1_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario1_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario3_NumUnitsCutFirstStage(10,8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,8) - 1.0 Scenario3_NumUnitsCutFirstStage(10,8) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario2_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario2_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario3_NumUnitsCutFirstStage(5,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,2) - 1.0 Scenario3_NumUnitsCutFirstStage(5,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario1_NumUnitsCutFirstStage(6,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,4) - 1.0 Scenario1_NumUnitsCutFirstStage(6,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario3_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario3_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario1_NumUnitsCutFirstStage(10,10) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_10)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,10) - 1.0 Scenario1_NumUnitsCutFirstStage(10,10) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario2_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario2_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario2_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario2_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario3_NumUnitsCutFirstStage(6,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,2) - 1.0 Scenario3_NumUnitsCutFirstStage(6,2) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario3_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario3_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario2_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario2_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario3_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario3_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario3_NumUnitsCutFirstStage(7,7) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,7) - 1.0 Scenario3_NumUnitsCutFirstStage(7,7) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario2_NumUnitsCutFirstStage(10,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,1) - 1.0 Scenario2_NumUnitsCutFirstStage(10,1) <= 0.0

MASTER_c_l_COST_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario2Node - 1.0 MASTER_Scenario2Node_SecondStageCost >= 0.0

MASTER_c_u_COST_Scenario2Node_SecondStageCost_None_:  1.0 MASTER_EXPECTED_COST_Scenario2Node - 1.0 MASTER_Scenario2Node_SecondStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario1_NumUnitsCutFirstStage(6,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,1) - 1.0 Scenario1_NumUnitsCutFirstStage(6,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario1_NumUnitsCutFirstStage(2,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(2,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(2,1) - 1.0 Scenario1_NumUnitsCutFirstStage(2,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario2_NumUnitsCutFirstStage(7,1) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,1) - 1.0 Scenario2_NumUnitsCutFirstStage(7,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario1_NumProducedFirstStage(1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_1_:  1.0 MASTER_RootNode_NumProducedFirstStage(1) - 1.0 Scenario1_NumProducedFirstStage(1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario2_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario2_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario3_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario3_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario1_NumUnitsCutFirstStage(8,6) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,6) - 1.0 Scenario1_NumUnitsCutFirstStage(8,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario1_FirstStageCost >= 0.0

MASTER_c_u_Scenario1_RootNode_FirstStageCost_None_:  1.0 MASTER_RootNode_FirstStageCost - 1.0 Scenario1_FirstStageCost <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario1_NumUnitsCutFirstStage(9,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,3) - 1.0 Scenario1_NumUnitsCutFirstStage(9,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario2_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario2_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario2_NumUnitsCutFirstStage(7,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(7,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,2) - 1.0 Scenario2_NumUnitsCutFirstStage(7,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario1_NumUnitsCutFirstStage(9,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,7) - 1.0 Scenario1_NumUnitsCutFirstStage(9,7) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario1_NumUnitsCutFirstStage(6,3) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(6,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,3) - 1.0 Scenario1_NumUnitsCutFirstStage(6,3) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario1_NumProducedFirstStage(9) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumProducedFirstStage_9_:  1.0 MASTER_RootNode_NumProducedFirstStage(9) - 1.0 Scenario1_NumProducedFirstStage(9) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario1_NumUnitsCutFirstStage(10,7) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(10,_7)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,7) - 1.0 Scenario1_NumUnitsCutFirstStage(10,7) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario3_NumUnitsCutFirstStage(3,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(3,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(3,1) - 1.0 Scenario3_NumUnitsCutFirstStage(3,1) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario3_NumUnitsCutFirstStage(6,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(6,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(6,5) - 1.0 Scenario3_NumUnitsCutFirstStage(6,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario3_NumUnitsCutFirstStage(5,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,1) - 1.0 Scenario3_NumUnitsCutFirstStage(5,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario2_NumUnitsCutFirstStage(10,5) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,5) - 1.0 Scenario2_NumUnitsCutFirstStage(10,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario1_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario1_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario2_NumUnitsCutFirstStage(9,8) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,8) - 1.0 Scenario2_NumUnitsCutFirstStage(9,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario3_NumUnitsCutFirstStage(9,9) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,9) - 1.0 Scenario3_NumUnitsCutFirstStage(9,9) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario3_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario3_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario3_NumUnitsCutFirstStage(8,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(8,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,4) - 1.0 Scenario3_NumUnitsCutFirstStage(8,4) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario2_NumUnitsCutFirstStage(9,6) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(9,_6)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,6) - 1.0 Scenario2_NumUnitsCutFirstStage(9,6) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario1_NumUnitsCutFirstStage(8,8) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_8)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,8) - 1.0 Scenario1_NumUnitsCutFirstStage(8,8) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario3_NumUnitsCutFirstStage(9,1) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(9,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,1) - 1.0 Scenario3_NumUnitsCutFirstStage(9,1) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario2_NumUnitsCutFirstStage(4,3) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(4,_3)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(4,3) - 1.0 Scenario2_NumUnitsCutFirstStage(4,3) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario2_NumUnitsCutFirstStage(8,2) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(8,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,2) - 1.0 Scenario2_NumUnitsCutFirstStage(8,2) <= 0.0

MASTER_c_l_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario2_NumUnitsCutFirstStage(10,9) >= 0.0

MASTER_c_u_Scenario2_RootNode_NumUnitsCutFirstStage_(10,_9)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,9) - 1.0 Scenario2_NumUnitsCutFirstStage(10,9) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario3_NumUnitsCutFirstStage(10,2) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(10,_2)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(10,2) - 1.0 Scenario3_NumUnitsCutFirstStage(10,2) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario1_NumUnitsCutFirstStage(9,5) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(9,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(9,5) - 1.0 Scenario1_NumUnitsCutFirstStage(9,5) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario3_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario3_NumUnitsCutFirstStage(7,4) <= 0.0

MASTER_c_l_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario3_NumUnitsCutFirstStage(5,5) >= 0.0

MASTER_c_u_Scenario3_RootNode_NumUnitsCutFirstStage_(5,_5)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(5,5) - 1.0 Scenario3_NumUnitsCutFirstStage(5,5) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario1_NumUnitsCutFirstStage(8,1) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(8,_1)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(8,1) - 1.0 Scenario1_NumUnitsCutFirstStage(8,1) <= 0.0

MASTER_c_l_Scenario1_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario1_ProduceSizeFirstStage(8) >= 0.0

MASTER_c_u_Scenario1_RootNode_ProduceSizeFirstStage_8_:  1.0 MASTER_RootNode_ProduceSizeFirstStage(8) - 1.0 Scenario1_ProduceSizeFirstStage(8) <= 0.0

MASTER_c_l_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario1_NumUnitsCutFirstStage(7,4) >= 0.0

MASTER_c_u_Scenario1_RootNode_NumUnitsCutFirstStage_(7,_4)_:  1.0 MASTER_RootNode_NumUnitsCutFirstStage(7,4) - 1.0 Scenario1_NumUnitsCutFirstStage(7,4) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for master 
\ Begin constraint block for scenario Scenario3
Scenario3_c_l_DemandSatisfiedSecondStage(1)_:  1.0 Scenario3_NumUnitsCutSecondStage(1,1) + 1.0 Scenario3_NumUnitsCutSecondStage(10,1) + 1.0 Scenario3_NumUnitsCutSecondStage(2,1) + 1.0 Scenario3_NumUnitsCutSecondStage(3,1) + 1.0 Scenario3_NumUnitsCutSecondStage(4,1) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(5,1) + 1.0 Scenario3_NumUnitsCutSecondStage(6,1) + 1.0 Scenario3_NumUnitsCutSecondStage(7,1) + 1.0 Scenario3_NumUnitsCutSecondStage(8,1) + 1.0 Scenario3_NumUnitsCutSecondStage(9,1) 
>= 3250.0

Scenario3_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,2) + 1.0 Scenario3_NumUnitsCutSecondStage(2,2) + 1.0 Scenario3_NumUnitsCutSecondStage(3,2) + 1.0 Scenario3_NumUnitsCutSecondStage(4,2) + 1.0 Scenario3_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(6,2) + 1.0 Scenario3_NumUnitsCutSecondStage(7,2) + 1.0 Scenario3_NumUnitsCutSecondStage(8,2) + 1.0 Scenario3_NumUnitsCutSecondStage(9,2) >= 9750.0

Scenario3_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,3) + 1.0 Scenario3_NumUnitsCutSecondStage(3,3) + 1.0 Scenario3_NumUnitsCutSecondStage(4,3) + 1.0 Scenario3_NumUnitsCutSecondStage(5,3) + 1.0 Scenario3_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(7,3) + 1.0 Scenario3_NumUnitsCutSecondStage(8,3) + 1.0 Scenario3_NumUnitsCutSecondStage(9,3) >= 16250.0

Scenario3_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,4) + 1.0 Scenario3_NumUnitsCutSecondStage(4,4) + 1.0 Scenario3_NumUnitsCutSecondStage(5,4) + 1.0 Scenario3_NumUnitsCutSecondStage(6,4) + 1.0 Scenario3_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(8,4) + 1.0 Scenario3_NumUnitsCutSecondStage(9,4) >= 13000.0

Scenario3_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,5) + 1.0 Scenario3_NumUnitsCutSecondStage(5,5) + 1.0 Scenario3_NumUnitsCutSecondStage(6,5) + 1.0 Scenario3_NumUnitsCutSecondStage(7,5) + 1.0 Scenario3_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario3_NumUnitsCutSecondStage(9,5) >= 45500.0

Scenario3_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,6) + 1.0 Scenario3_NumUnitsCutSecondStage(6,6) + 1.0 Scenario3_NumUnitsCutSecondStage(7,6) + 1.0 Scenario3_NumUnitsCutSecondStage(8,6) + 1.0 Scenario3_NumUnitsCutSecondStage(9,6) 
>= 32500.0

Scenario3_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,7) + 1.0 Scenario3_NumUnitsCutSecondStage(7,7) + 1.0 Scenario3_NumUnitsCutSecondStage(8,7) + 1.0 Scenario3_NumUnitsCutSecondStage(9,7) >= 19500.0

Scenario3_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,8) + 1.0 Scenario3_NumUnitsCutSecondStage(8,8) + 1.0 Scenario3_NumUnitsCutSecondStage(9,8) >= 16250.0

Scenario3_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,9) + 1.0 Scenario3_NumUnitsCutSecondStage(9,9) >= 16250.0

Scenario3_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario3_NumUnitsCutSecondStage(10,10) >= 6500.0

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
>= 2500.0

Scenario2_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,2) + 1.0 Scenario2_NumUnitsCutSecondStage(2,2) + 1.0 Scenario2_NumUnitsCutSecondStage(3,2) + 1.0 Scenario2_NumUnitsCutSecondStage(4,2) + 1.0 Scenario2_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(6,2) + 1.0 Scenario2_NumUnitsCutSecondStage(7,2) + 1.0 Scenario2_NumUnitsCutSecondStage(8,2) + 1.0 Scenario2_NumUnitsCutSecondStage(9,2) >= 7500.0

Scenario2_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,3) + 1.0 Scenario2_NumUnitsCutSecondStage(3,3) + 1.0 Scenario2_NumUnitsCutSecondStage(4,3) + 1.0 Scenario2_NumUnitsCutSecondStage(5,3) + 1.0 Scenario2_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(7,3) + 1.0 Scenario2_NumUnitsCutSecondStage(8,3) + 1.0 Scenario2_NumUnitsCutSecondStage(9,3) >= 12500.0

Scenario2_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,4) + 1.0 Scenario2_NumUnitsCutSecondStage(4,4) + 1.0 Scenario2_NumUnitsCutSecondStage(5,4) + 1.0 Scenario2_NumUnitsCutSecondStage(6,4) + 1.0 Scenario2_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(8,4) + 1.0 Scenario2_NumUnitsCutSecondStage(9,4) >= 10000.0

Scenario2_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,5) + 1.0 Scenario2_NumUnitsCutSecondStage(5,5) + 1.0 Scenario2_NumUnitsCutSecondStage(6,5) + 1.0 Scenario2_NumUnitsCutSecondStage(7,5) + 1.0 Scenario2_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario2_NumUnitsCutSecondStage(9,5) >= 35000.0

Scenario2_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,6) + 1.0 Scenario2_NumUnitsCutSecondStage(6,6) + 1.0 Scenario2_NumUnitsCutSecondStage(7,6) + 1.0 Scenario2_NumUnitsCutSecondStage(8,6) + 1.0 Scenario2_NumUnitsCutSecondStage(9,6) 
>= 25000.0

Scenario2_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,7) + 1.0 Scenario2_NumUnitsCutSecondStage(7,7) + 1.0 Scenario2_NumUnitsCutSecondStage(8,7) + 1.0 Scenario2_NumUnitsCutSecondStage(9,7) >= 15000.0

Scenario2_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,8) + 1.0 Scenario2_NumUnitsCutSecondStage(8,8) + 1.0 Scenario2_NumUnitsCutSecondStage(9,8) >= 12500.0

Scenario2_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,9) + 1.0 Scenario2_NumUnitsCutSecondStage(9,9) >= 12500.0

Scenario2_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario2_NumUnitsCutSecondStage(10,10) >= 5000.0

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
>= 1750.0

Scenario1_c_l_DemandSatisfiedSecondStage(2)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,2) + 1.0 Scenario1_NumUnitsCutSecondStage(2,2) + 1.0 Scenario1_NumUnitsCutSecondStage(3,2) + 1.0 Scenario1_NumUnitsCutSecondStage(4,2) + 1.0 Scenario1_NumUnitsCutSecondStage(5,2) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(6,2) + 1.0 Scenario1_NumUnitsCutSecondStage(7,2) + 1.0 Scenario1_NumUnitsCutSecondStage(8,2) + 1.0 Scenario1_NumUnitsCutSecondStage(9,2) >= 5250.0

Scenario1_c_l_DemandSatisfiedSecondStage(3)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,3) + 1.0 Scenario1_NumUnitsCutSecondStage(3,3) + 1.0 Scenario1_NumUnitsCutSecondStage(4,3) + 1.0 Scenario1_NumUnitsCutSecondStage(5,3) + 1.0 Scenario1_NumUnitsCutSecondStage(6,3) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(7,3) + 1.0 Scenario1_NumUnitsCutSecondStage(8,3) + 1.0 Scenario1_NumUnitsCutSecondStage(9,3) >= 8750.0

Scenario1_c_l_DemandSatisfiedSecondStage(4)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,4) + 1.0 Scenario1_NumUnitsCutSecondStage(4,4) + 1.0 Scenario1_NumUnitsCutSecondStage(5,4) + 1.0 Scenario1_NumUnitsCutSecondStage(6,4) + 1.0 Scenario1_NumUnitsCutSecondStage(7,4) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(8,4) + 1.0 Scenario1_NumUnitsCutSecondStage(9,4) >= 7000.0

Scenario1_c_l_DemandSatisfiedSecondStage(5)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,5) + 1.0 Scenario1_NumUnitsCutSecondStage(5,5) + 1.0 Scenario1_NumUnitsCutSecondStage(6,5) + 1.0 Scenario1_NumUnitsCutSecondStage(7,5) + 1.0 Scenario1_NumUnitsCutSecondStage(8,5) 
+ 1.0 Scenario1_NumUnitsCutSecondStage(9,5) >= 24500.0

Scenario1_c_l_DemandSatisfiedSecondStage(6)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,6) + 1.0 Scenario1_NumUnitsCutSecondStage(6,6) + 1.0 Scenario1_NumUnitsCutSecondStage(7,6) + 1.0 Scenario1_NumUnitsCutSecondStage(8,6) + 1.0 Scenario1_NumUnitsCutSecondStage(9,6) 
>= 17500.0

Scenario1_c_l_DemandSatisfiedSecondStage(7)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,7) + 1.0 Scenario1_NumUnitsCutSecondStage(7,7) + 1.0 Scenario1_NumUnitsCutSecondStage(8,7) + 1.0 Scenario1_NumUnitsCutSecondStage(9,7) >= 10500.0

Scenario1_c_l_DemandSatisfiedSecondStage(8)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,8) + 1.0 Scenario1_NumUnitsCutSecondStage(8,8) + 1.0 Scenario1_NumUnitsCutSecondStage(9,8) >= 8750.0

Scenario1_c_l_DemandSatisfiedSecondStage(9)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,9) + 1.0 Scenario1_NumUnitsCutSecondStage(9,9) >= 8750.0

Scenario1_c_l_DemandSatisfiedSecondStage(10)_:  1.0 Scenario1_NumUnitsCutSecondStage(10,10) >= 3500.0

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

bounds

\ Begin variable bounds block for master
    0 <=  MASTER_Scenario3Node_SecondStageCost  <= +inf
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
    0 <=  MASTER_RootNode_FirstStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_Scenario2Node  <= +inf
\ End variable bounds block for master

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

end
