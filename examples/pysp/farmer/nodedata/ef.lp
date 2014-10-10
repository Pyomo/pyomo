\ Begin objective block for master
min 
obj:  1.0 MASTER_EXPECTED_COST_RootNode 
\ End objective block for master

s.t.

\ Begin constraint block for master
MASTER_c_l_AverageScenario_AverageNode_SecondStageCost_None(None)_:  -1.0 AverageScenario_SecondStageCost + 1.0 MASTER_AverageNode_SecondStageCost >= 0.0

MASTER_c_u_AverageScenario_AverageNode_SecondStageCost_None(None)_:  -1.0 AverageScenario_SecondStageCost + 1.0 MASTER_AverageNode_SecondStageCost <= 0.0

MASTER_c_l_COST_BelowAverageNode_SecondStageCost_None(None)_:  -1.0 MASTER_BelowAverageNode_SecondStageCost + 1.0 MASTER_EXPECTED_COST_BelowAverageNode >= 0.0

MASTER_c_u_COST_BelowAverageNode_SecondStageCost_None(None)_:  -1.0 MASTER_BelowAverageNode_SecondStageCost + 1.0 MASTER_EXPECTED_COST_BelowAverageNode <= 0.0

MASTER_c_l_AboveAverageScenario_RootNode_DevotedAcreage_CORN(None)_:  -1.0 AboveAverageScenario_DevotedAcreage(CORN) + 1.0 MASTER_RootNode_DevotedAcreage(CORN) >= 0.0

MASTER_c_u_AboveAverageScenario_RootNode_DevotedAcreage_CORN(None)_:  -1.0 AboveAverageScenario_DevotedAcreage(CORN) + 1.0 MASTER_RootNode_DevotedAcreage(CORN) <= 0.0

MASTER_c_l_AboveAverageScenario_RootNode_DevotedAcreage_SUGAR_BEETS(None)_:  -1.0 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 MASTER_RootNode_DevotedAcreage(SUGAR_BEETS) >= 0.0

MASTER_c_u_AboveAverageScenario_RootNode_DevotedAcreage_SUGAR_BEETS(None)_:  -1.0 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 MASTER_RootNode_DevotedAcreage(SUGAR_BEETS) <= 0.0

MASTER_c_l_COST_AverageNode_SecondStageCost_None(None)_:  -1.0 MASTER_AverageNode_SecondStageCost + 1.0 MASTER_EXPECTED_COST_AverageNode >= 0.0

MASTER_c_u_COST_AverageNode_SecondStageCost_None(None)_:  -1.0 MASTER_AverageNode_SecondStageCost + 1.0 MASTER_EXPECTED_COST_AverageNode <= 0.0

MASTER_c_l_AboveAverageScenario_RootNode_DevotedAcreage_WHEAT(None)_:  -1.0 AboveAverageScenario_DevotedAcreage(WHEAT) + 1.0 MASTER_RootNode_DevotedAcreage(WHEAT) >= 0.0

MASTER_c_u_AboveAverageScenario_RootNode_DevotedAcreage_WHEAT(None)_:  -1.0 AboveAverageScenario_DevotedAcreage(WHEAT) + 1.0 MASTER_RootNode_DevotedAcreage(WHEAT) <= 0.0

MASTER_c_l_AboveAverageScenario_AboveAverageNode_SecondStageCost_None(None)_:  -1.0 AboveAverageScenario_SecondStageCost + 1.0 MASTER_AboveAverageNode_SecondStageCost >= 0.0

MASTER_c_u_AboveAverageScenario_AboveAverageNode_SecondStageCost_None(None)_:  -1.0 AboveAverageScenario_SecondStageCost + 1.0 MASTER_AboveAverageNode_SecondStageCost <= 0.0

MASTER_c_l_BelowAverageScenario_BelowAverageNode_SecondStageCost_None(None)_:  -1.0 BelowAverageScenario_SecondStageCost + 1.0 MASTER_BelowAverageNode_SecondStageCost >= 0.0

MASTER_c_u_BelowAverageScenario_BelowAverageNode_SecondStageCost_None(None)_:  -1.0 BelowAverageScenario_SecondStageCost + 1.0 MASTER_BelowAverageNode_SecondStageCost <= 0.0

MASTER_c_l_AverageScenario_RootNode_DevotedAcreage_WHEAT(None)_:  -1.0 AverageScenario_DevotedAcreage(WHEAT) + 1.0 MASTER_RootNode_DevotedAcreage(WHEAT) >= 0.0

MASTER_c_u_AverageScenario_RootNode_DevotedAcreage_WHEAT(None)_:  -1.0 AverageScenario_DevotedAcreage(WHEAT) + 1.0 MASTER_RootNode_DevotedAcreage(WHEAT) <= 0.0

MASTER_c_l_COST_AboveAverageNode_SecondStageCost_None(None)_:  -1.0 MASTER_AboveAverageNode_SecondStageCost + 1.0 MASTER_EXPECTED_COST_AboveAverageNode >= 0.0

MASTER_c_u_COST_AboveAverageNode_SecondStageCost_None(None)_:  -1.0 MASTER_AboveAverageNode_SecondStageCost + 1.0 MASTER_EXPECTED_COST_AboveAverageNode <= 0.0

MASTER_c_l_BelowAverageScenario_RootNode_FirstStageCost_None(None)_:  -1.0 BelowAverageScenario_FirstStageCost + 1.0 MASTER_RootNode_FirstStageCost >= 0.0

MASTER_c_u_BelowAverageScenario_RootNode_FirstStageCost_None(None)_:  -1.0 BelowAverageScenario_FirstStageCost + 1.0 MASTER_RootNode_FirstStageCost <= 0.0

MASTER_c_l_COST_RootNode_FirstStageCost_None(None)_:  -0.33333333 MASTER_EXPECTED_COST_AboveAverageNode - 0.33333334 MASTER_EXPECTED_COST_AverageNode - 0.33333333 MASTER_EXPECTED_COST_BelowAverageNode + 1.0 MASTER_EXPECTED_COST_RootNode - 1.0 MASTER_RootNode_FirstStageCost 
>= 0.0

MASTER_c_u_COST_RootNode_FirstStageCost_None(None)_:  -0.33333333 MASTER_EXPECTED_COST_AboveAverageNode - 0.33333334 MASTER_EXPECTED_COST_AverageNode - 0.33333333 MASTER_EXPECTED_COST_BelowAverageNode + 1.0 MASTER_EXPECTED_COST_RootNode - 1.0 MASTER_RootNode_FirstStageCost 
<= 0.0

MASTER_c_l_BelowAverageScenario_RootNode_DevotedAcreage_WHEAT(None)_:  -1.0 BelowAverageScenario_DevotedAcreage(WHEAT) + 1.0 MASTER_RootNode_DevotedAcreage(WHEAT) >= 0.0

MASTER_c_u_BelowAverageScenario_RootNode_DevotedAcreage_WHEAT(None)_:  -1.0 BelowAverageScenario_DevotedAcreage(WHEAT) + 1.0 MASTER_RootNode_DevotedAcreage(WHEAT) <= 0.0

MASTER_c_l_AverageScenario_RootNode_DevotedAcreage_CORN(None)_:  -1.0 AverageScenario_DevotedAcreage(CORN) + 1.0 MASTER_RootNode_DevotedAcreage(CORN) >= 0.0

MASTER_c_u_AverageScenario_RootNode_DevotedAcreage_CORN(None)_:  -1.0 AverageScenario_DevotedAcreage(CORN) + 1.0 MASTER_RootNode_DevotedAcreage(CORN) <= 0.0

MASTER_c_l_BelowAverageScenario_RootNode_DevotedAcreage_SUGAR_BEETS(None)_:  -1.0 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 MASTER_RootNode_DevotedAcreage(SUGAR_BEETS) >= 0.0

MASTER_c_u_BelowAverageScenario_RootNode_DevotedAcreage_SUGAR_BEETS(None)_:  -1.0 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 MASTER_RootNode_DevotedAcreage(SUGAR_BEETS) <= 0.0

MASTER_c_l_AverageScenario_RootNode_DevotedAcreage_SUGAR_BEETS(None)_:  -1.0 AverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 MASTER_RootNode_DevotedAcreage(SUGAR_BEETS) >= 0.0

MASTER_c_u_AverageScenario_RootNode_DevotedAcreage_SUGAR_BEETS(None)_:  -1.0 AverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 MASTER_RootNode_DevotedAcreage(SUGAR_BEETS) <= 0.0

MASTER_c_l_AverageScenario_RootNode_FirstStageCost_None(None)_:  -1.0 AverageScenario_FirstStageCost + 1.0 MASTER_RootNode_FirstStageCost >= 0.0

MASTER_c_u_AverageScenario_RootNode_FirstStageCost_None(None)_:  -1.0 AverageScenario_FirstStageCost + 1.0 MASTER_RootNode_FirstStageCost <= 0.0

MASTER_c_l_AboveAverageScenario_RootNode_FirstStageCost_None(None)_:  -1.0 AboveAverageScenario_FirstStageCost + 1.0 MASTER_RootNode_FirstStageCost >= 0.0

MASTER_c_u_AboveAverageScenario_RootNode_FirstStageCost_None(None)_:  -1.0 AboveAverageScenario_FirstStageCost + 1.0 MASTER_RootNode_FirstStageCost <= 0.0

MASTER_c_l_BelowAverageScenario_RootNode_DevotedAcreage_CORN(None)_:  -1.0 BelowAverageScenario_DevotedAcreage(CORN) + 1.0 MASTER_RootNode_DevotedAcreage(CORN) >= 0.0

MASTER_c_u_BelowAverageScenario_RootNode_DevotedAcreage_CORN(None)_:  -1.0 BelowAverageScenario_DevotedAcreage(CORN) + 1.0 MASTER_RootNode_DevotedAcreage(CORN) <= 0.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for master 
\ Begin constraint block for scenario AverageScenario
AverageScenario_c_e_ComputeSecondStageCost(None)_:  -210.0 AverageScenario_QuantityPurchased(CORN) - 100000.0 AverageScenario_QuantityPurchased(SUGAR_BEETS) - 238.0 AverageScenario_QuantityPurchased(WHEAT) + 150.0 AverageScenario_QuantitySubQuotaSold(CORN) + 36.0 AverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) 
+ 170.0 AverageScenario_QuantitySubQuotaSold(WHEAT) + 10.0 AverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) + 1.0 AverageScenario_SecondStageCost = 0.0

AverageScenario_c_u_LimitAmountSold(CORN)_:  -3.0 AverageScenario_DevotedAcreage(CORN) + 1.0 AverageScenario_QuantitySubQuotaSold(CORN) + 1.0 AverageScenario_QuantitySuperQuotaSold(CORN) <= 0.0

AverageScenario_c_u_LimitAmountSold(SUGAR_BEETS)_:  -20.0 AverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 AverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) + 1.0 AverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) <= 0.0

AverageScenario_c_u_LimitAmountSold(WHEAT)_:  -2.5 AverageScenario_DevotedAcreage(WHEAT) + 1.0 AverageScenario_QuantitySubQuotaSold(WHEAT) + 1.0 AverageScenario_QuantitySuperQuotaSold(WHEAT) <= 0.0

AverageScenario_c_u_ConstrainTotalAcreage(None)_:  1.0 AverageScenario_DevotedAcreage(CORN) + 1.0 AverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 AverageScenario_DevotedAcreage(WHEAT) <= 500.0

AverageScenario_c_e_ComputeFirstStageCost(None)_:  -230.0 AverageScenario_DevotedAcreage(CORN) - 260.0 AverageScenario_DevotedAcreage(SUGAR_BEETS) - 150.0 AverageScenario_DevotedAcreage(WHEAT) + 1.0 AverageScenario_FirstStageCost = 0.0

AverageScenario_c_l_EnforceQuotas(CORN)_:  1.0 AverageScenario_QuantitySubQuotaSold(CORN) >= 0.0

AverageScenario_c_u_EnforceQuotas(CORN)_:  1.0 AverageScenario_QuantitySubQuotaSold(CORN) <= 100000.0

AverageScenario_c_l_EnforceQuotas(SUGAR_BEETS)_:  1.0 AverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) >= 0.0

AverageScenario_c_u_EnforceQuotas(SUGAR_BEETS)_:  1.0 AverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) <= 6000.0

AverageScenario_c_l_EnforceQuotas(WHEAT)_:  1.0 AverageScenario_QuantitySubQuotaSold(WHEAT) >= 0.0

AverageScenario_c_u_EnforceQuotas(WHEAT)_:  1.0 AverageScenario_QuantitySubQuotaSold(WHEAT) <= 100000.0

AverageScenario_c_l_EnforceCattleFeedRequirement(CORN)_:  3.0 AverageScenario_DevotedAcreage(CORN) + 1.0 AverageScenario_QuantityPurchased(CORN) - 1.0 AverageScenario_QuantitySubQuotaSold(CORN) - 1.0 AverageScenario_QuantitySuperQuotaSold(CORN) >= 240.0

AverageScenario_c_l_EnforceCattleFeedRequirement(SUGAR_BEETS)_:  20.0 AverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 AverageScenario_QuantityPurchased(SUGAR_BEETS) - 1.0 AverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) - 1.0 AverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) >= 0.0

AverageScenario_c_l_EnforceCattleFeedRequirement(WHEAT)_:  2.5 AverageScenario_DevotedAcreage(WHEAT) + 1.0 AverageScenario_QuantityPurchased(WHEAT) - 1.0 AverageScenario_QuantitySubQuotaSold(WHEAT) - 1.0 AverageScenario_QuantitySuperQuotaSold(WHEAT) >= 200.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario AverageScenario

\ Begin constraint block for scenario BelowAverageScenario
BelowAverageScenario_c_e_ComputeSecondStageCost(None)_:  -210.0 BelowAverageScenario_QuantityPurchased(CORN) - 100000.0 BelowAverageScenario_QuantityPurchased(SUGAR_BEETS) - 238.0 BelowAverageScenario_QuantityPurchased(WHEAT) + 150.0 BelowAverageScenario_QuantitySubQuotaSold(CORN) + 36.0 BelowAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) 
+ 170.0 BelowAverageScenario_QuantitySubQuotaSold(WHEAT) + 10.0 BelowAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) + 1.0 BelowAverageScenario_SecondStageCost = 0.0

BelowAverageScenario_c_u_LimitAmountSold(CORN)_:  -2.4 BelowAverageScenario_DevotedAcreage(CORN) + 1.0 BelowAverageScenario_QuantitySubQuotaSold(CORN) + 1.0 BelowAverageScenario_QuantitySuperQuotaSold(CORN) <= 0.0

BelowAverageScenario_c_u_LimitAmountSold(SUGAR_BEETS)_:  -16.0 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 BelowAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) + 1.0 BelowAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) <= 0.0

BelowAverageScenario_c_u_LimitAmountSold(WHEAT)_:  -2.0 BelowAverageScenario_DevotedAcreage(WHEAT) + 1.0 BelowAverageScenario_QuantitySubQuotaSold(WHEAT) + 1.0 BelowAverageScenario_QuantitySuperQuotaSold(WHEAT) <= 0.0

BelowAverageScenario_c_u_ConstrainTotalAcreage(None)_:  1.0 BelowAverageScenario_DevotedAcreage(CORN) + 1.0 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 BelowAverageScenario_DevotedAcreage(WHEAT) <= 500.0

BelowAverageScenario_c_e_ComputeFirstStageCost(None)_:  -230.0 BelowAverageScenario_DevotedAcreage(CORN) - 260.0 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) - 150.0 BelowAverageScenario_DevotedAcreage(WHEAT) + 1.0 BelowAverageScenario_FirstStageCost = 0.0

BelowAverageScenario_c_l_EnforceQuotas(CORN)_:  1.0 BelowAverageScenario_QuantitySubQuotaSold(CORN) >= 0.0

BelowAverageScenario_c_u_EnforceQuotas(CORN)_:  1.0 BelowAverageScenario_QuantitySubQuotaSold(CORN) <= 100000.0

BelowAverageScenario_c_l_EnforceQuotas(SUGAR_BEETS)_:  1.0 BelowAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) >= 0.0

BelowAverageScenario_c_u_EnforceQuotas(SUGAR_BEETS)_:  1.0 BelowAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) <= 6000.0

BelowAverageScenario_c_l_EnforceQuotas(WHEAT)_:  1.0 BelowAverageScenario_QuantitySubQuotaSold(WHEAT) >= 0.0

BelowAverageScenario_c_u_EnforceQuotas(WHEAT)_:  1.0 BelowAverageScenario_QuantitySubQuotaSold(WHEAT) <= 100000.0

BelowAverageScenario_c_l_EnforceCattleFeedRequirement(CORN)_:  2.4 BelowAverageScenario_DevotedAcreage(CORN) + 1.0 BelowAverageScenario_QuantityPurchased(CORN) - 1.0 BelowAverageScenario_QuantitySubQuotaSold(CORN) - 1.0 BelowAverageScenario_QuantitySuperQuotaSold(CORN) >= 240.0

BelowAverageScenario_c_l_EnforceCattleFeedRequirement(SUGAR_BEETS)_:  16.0 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 BelowAverageScenario_QuantityPurchased(SUGAR_BEETS) - 1.0 BelowAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) - 1.0 BelowAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) >= 0.0

BelowAverageScenario_c_l_EnforceCattleFeedRequirement(WHEAT)_:  2.0 BelowAverageScenario_DevotedAcreage(WHEAT) + 1.0 BelowAverageScenario_QuantityPurchased(WHEAT) - 1.0 BelowAverageScenario_QuantitySubQuotaSold(WHEAT) - 1.0 BelowAverageScenario_QuantitySuperQuotaSold(WHEAT) >= 200.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario BelowAverageScenario

\ Begin constraint block for scenario AboveAverageScenario
AboveAverageScenario_c_e_ComputeSecondStageCost(None)_:  -210.0 AboveAverageScenario_QuantityPurchased(CORN) - 100000.0 AboveAverageScenario_QuantityPurchased(SUGAR_BEETS) - 238.0 AboveAverageScenario_QuantityPurchased(WHEAT) + 150.0 AboveAverageScenario_QuantitySubQuotaSold(CORN) + 36.0 AboveAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) 
+ 170.0 AboveAverageScenario_QuantitySubQuotaSold(WHEAT) + 10.0 AboveAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) + 1.0 AboveAverageScenario_SecondStageCost = 0.0

AboveAverageScenario_c_u_LimitAmountSold(CORN)_:  -3.6 AboveAverageScenario_DevotedAcreage(CORN) + 1.0 AboveAverageScenario_QuantitySubQuotaSold(CORN) + 1.0 AboveAverageScenario_QuantitySuperQuotaSold(CORN) <= 0.0

AboveAverageScenario_c_u_LimitAmountSold(SUGAR_BEETS)_:  -24.0 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 AboveAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) + 1.0 AboveAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) <= 0.0

AboveAverageScenario_c_u_LimitAmountSold(WHEAT)_:  -3.0 AboveAverageScenario_DevotedAcreage(WHEAT) + 1.0 AboveAverageScenario_QuantitySubQuotaSold(WHEAT) + 1.0 AboveAverageScenario_QuantitySuperQuotaSold(WHEAT) <= 0.0

AboveAverageScenario_c_u_ConstrainTotalAcreage(None)_:  1.0 AboveAverageScenario_DevotedAcreage(CORN) + 1.0 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 AboveAverageScenario_DevotedAcreage(WHEAT) <= 500.0

AboveAverageScenario_c_e_ComputeFirstStageCost(None)_:  -230.0 AboveAverageScenario_DevotedAcreage(CORN) - 260.0 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) - 150.0 AboveAverageScenario_DevotedAcreage(WHEAT) + 1.0 AboveAverageScenario_FirstStageCost = 0.0

AboveAverageScenario_c_l_EnforceQuotas(CORN)_:  1.0 AboveAverageScenario_QuantitySubQuotaSold(CORN) >= 0.0

AboveAverageScenario_c_u_EnforceQuotas(CORN)_:  1.0 AboveAverageScenario_QuantitySubQuotaSold(CORN) <= 100000.0

AboveAverageScenario_c_l_EnforceQuotas(SUGAR_BEETS)_:  1.0 AboveAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) >= 0.0

AboveAverageScenario_c_u_EnforceQuotas(SUGAR_BEETS)_:  1.0 AboveAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) <= 6000.0

AboveAverageScenario_c_l_EnforceQuotas(WHEAT)_:  1.0 AboveAverageScenario_QuantitySubQuotaSold(WHEAT) >= 0.0

AboveAverageScenario_c_u_EnforceQuotas(WHEAT)_:  1.0 AboveAverageScenario_QuantitySubQuotaSold(WHEAT) <= 100000.0

AboveAverageScenario_c_l_EnforceCattleFeedRequirement(CORN)_:  3.6 AboveAverageScenario_DevotedAcreage(CORN) + 1.0 AboveAverageScenario_QuantityPurchased(CORN) - 1.0 AboveAverageScenario_QuantitySubQuotaSold(CORN) - 1.0 AboveAverageScenario_QuantitySuperQuotaSold(CORN) >= 240.0

AboveAverageScenario_c_l_EnforceCattleFeedRequirement(SUGAR_BEETS)_:  24.0 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) + 1.0 AboveAverageScenario_QuantityPurchased(SUGAR_BEETS) - 1.0 AboveAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS) - 1.0 AboveAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS) >= 0.0

AboveAverageScenario_c_l_EnforceCattleFeedRequirement(WHEAT)_:  3.0 AboveAverageScenario_DevotedAcreage(WHEAT) + 1.0 AboveAverageScenario_QuantityPurchased(WHEAT) - 1.0 AboveAverageScenario_QuantitySubQuotaSold(WHEAT) - 1.0 AboveAverageScenario_QuantitySuperQuotaSold(WHEAT) >= 200.0

c_e_ONE_VAR_CONSTANT: ONE_VAR_CONSTANT = 1.0

\ End constraint block for scenario AboveAverageScenario

bounds

\ Begin variable bounds block for master
    0.0 <=  MASTER_RootNode_DevotedAcreage(CORN)  <= +inf
    0.0 <=  MASTER_RootNode_DevotedAcreage(SUGAR_BEETS)  <= +inf
    0.0 <=  MASTER_RootNode_DevotedAcreage(WHEAT)  <= +inf
     -inf <=  MASTER_AverageNode_SecondStageCost  <= +inf
     -inf <=  MASTER_BelowAverageNode_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_AboveAverageNode  <= +inf
     -inf <=  MASTER_AboveAverageNode_SecondStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_RootNode  <= +inf
     -inf <=  MASTER_EXPECTED_COST_BelowAverageNode  <= +inf
     -inf <=  MASTER_RootNode_FirstStageCost  <= +inf
     -inf <=  MASTER_EXPECTED_COST_AverageNode  <= +inf
\ End variable bounds block for master

\ Begin variable bounds block for scenario AverageScenario
    0.0 <=  AverageScenario_QuantitySubQuotaSold(CORN)  <= +inf
    0.0 <=  AverageScenario_QuantitySubQuotaSold(SUGAR_BEETS)  <= +inf
    0.0 <=  AverageScenario_QuantitySubQuotaSold(WHEAT)  <= +inf
    0.0 <=  AverageScenario_DevotedAcreage(CORN)  <= +inf
    0.0 <=  AverageScenario_DevotedAcreage(SUGAR_BEETS)  <= +inf
    0.0 <=  AverageScenario_DevotedAcreage(WHEAT)  <= +inf
     -inf <=  AverageScenario_FirstStageCost  <= +inf
     -inf <=  AverageScenario_SecondStageCost  <= +inf
    0.0 <=  AverageScenario_QuantityPurchased(CORN)  <= +inf
    0.0 <=  AverageScenario_QuantityPurchased(SUGAR_BEETS)  <= +inf
    0.0 <=  AverageScenario_QuantityPurchased(WHEAT)  <= +inf
    0.0 <=  AverageScenario_QuantitySuperQuotaSold(CORN)  <= +inf
    0.0 <=  AverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS)  <= +inf
    0.0 <=  AverageScenario_QuantitySuperQuotaSold(WHEAT)  <= +inf
\ End variable bounds block for scenario AverageScenario

\ Begin variable bounds block for scenario BelowAverageScenario
    0.0 <=  BelowAverageScenario_QuantitySubQuotaSold(CORN)  <= +inf
    0.0 <=  BelowAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS)  <= +inf
    0.0 <=  BelowAverageScenario_QuantitySubQuotaSold(WHEAT)  <= +inf
    0.0 <=  BelowAverageScenario_DevotedAcreage(CORN)  <= +inf
    0.0 <=  BelowAverageScenario_DevotedAcreage(SUGAR_BEETS)  <= +inf
    0.0 <=  BelowAverageScenario_DevotedAcreage(WHEAT)  <= +inf
     -inf <=  BelowAverageScenario_FirstStageCost  <= +inf
     -inf <=  BelowAverageScenario_SecondStageCost  <= +inf
    0.0 <=  BelowAverageScenario_QuantityPurchased(CORN)  <= +inf
    0.0 <=  BelowAverageScenario_QuantityPurchased(SUGAR_BEETS)  <= +inf
    0.0 <=  BelowAverageScenario_QuantityPurchased(WHEAT)  <= +inf
    0.0 <=  BelowAverageScenario_QuantitySuperQuotaSold(CORN)  <= +inf
    0.0 <=  BelowAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS)  <= +inf
    0.0 <=  BelowAverageScenario_QuantitySuperQuotaSold(WHEAT)  <= +inf
\ End variable bounds block for scenario BelowAverageScenario

\ Begin variable bounds block for scenario AboveAverageScenario
    0.0 <=  AboveAverageScenario_QuantitySubQuotaSold(CORN)  <= +inf
    0.0 <=  AboveAverageScenario_QuantitySubQuotaSold(SUGAR_BEETS)  <= +inf
    0.0 <=  AboveAverageScenario_QuantitySubQuotaSold(WHEAT)  <= +inf
    0.0 <=  AboveAverageScenario_DevotedAcreage(CORN)  <= +inf
    0.0 <=  AboveAverageScenario_DevotedAcreage(SUGAR_BEETS)  <= +inf
    0.0 <=  AboveAverageScenario_DevotedAcreage(WHEAT)  <= +inf
     -inf <=  AboveAverageScenario_FirstStageCost  <= +inf
     -inf <=  AboveAverageScenario_SecondStageCost  <= +inf
    0.0 <=  AboveAverageScenario_QuantityPurchased(CORN)  <= +inf
    0.0 <=  AboveAverageScenario_QuantityPurchased(SUGAR_BEETS)  <= +inf
    0.0 <=  AboveAverageScenario_QuantityPurchased(WHEAT)  <= +inf
    0.0 <=  AboveAverageScenario_QuantitySuperQuotaSold(CORN)  <= +inf
    0.0 <=  AboveAverageScenario_QuantitySuperQuotaSold(SUGAR_BEETS)  <= +inf
    0.0 <=  AboveAverageScenario_QuantitySuperQuotaSold(WHEAT)  <= +inf
\ End variable bounds block for scenario AboveAverageScenario

end
