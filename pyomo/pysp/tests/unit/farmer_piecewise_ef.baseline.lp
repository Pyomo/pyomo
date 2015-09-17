\* Source Pyomo model name=MASTER *\

min 
MASTER:
+0.33333332999999998 AboveAverageScenario_FirstStageCost
+0.33333332999999998 AboveAverageScenario_SecondStageCost
+0.33333333999999998 AverageScenario_FirstStageCost
+0.33333333999999998 AverageScenario_SecondStageCost
+0.33333332999999998 BelowAverageScenario_FirstStageCost
+0.33333332999999998 BelowAverageScenario_SecondStageCost

s.t.

c_e_MASTER_BLEND_CONSTRAINT_RootNode(1)_:
-1 BelowAverageScenario_DevotedAcreage(CORN)
+1 MASTER_BLEND_VAR_RootNode(1)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(2)_:
-1 AverageScenario_DevotedAcreage(CORN)
+1 MASTER_BLEND_VAR_RootNode(1)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(3)_:
-1 AboveAverageScenario_DevotedAcreage(CORN)
+1 MASTER_BLEND_VAR_RootNode(1)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(4)_:
-1 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 MASTER_BLEND_VAR_RootNode(2)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(5)_:
-1 AverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 MASTER_BLEND_VAR_RootNode(2)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(6)_:
-1 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 MASTER_BLEND_VAR_RootNode(2)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(7)_:
-1 BelowAverageScenario_DevotedAcreage(WHEAT)
+1 MASTER_BLEND_VAR_RootNode(3)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(8)_:
-1 AverageScenario_DevotedAcreage(WHEAT)
+1 MASTER_BLEND_VAR_RootNode(3)
= 0

c_e_MASTER_BLEND_CONSTRAINT_RootNode(9)_:
-1 AboveAverageScenario_DevotedAcreage(WHEAT)
+1 MASTER_BLEND_VAR_RootNode(3)
= 0

c_u_BelowAverageScenario_ConstrainTotalAcreage_:
+1 BelowAverageScenario_DevotedAcreage(CORN)
+1 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 BelowAverageScenario_DevotedAcreage(WHEAT)
<= 500

c_l_BelowAverageScenario_EnforceCattleFeedRequirement(CORN)_:
+2.3999999999999999 BelowAverageScenario_DevotedAcreage(CORN)
+1 BelowAverageScenario_QuantityPurchased(CORN)
-1 BelowAverageScenario_QuantitySold(CORN)
>= 240

c_l_BelowAverageScenario_EnforceCattleFeedRequirement(SUGAR_BEETS)_:
+16 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 BelowAverageScenario_QuantityPurchased(SUGAR_BEETS)
-1 BelowAverageScenario_QuantitySold(SUGAR_BEETS)
>= 0

c_l_BelowAverageScenario_EnforceCattleFeedRequirement(WHEAT)_:
+2 BelowAverageScenario_DevotedAcreage(WHEAT)
+1 BelowAverageScenario_QuantityPurchased(WHEAT)
-1 BelowAverageScenario_QuantitySold(WHEAT)
>= 200

c_u_BelowAverageScenario_LimitAmountSold(CORN)_:
-2.3999999999999999 BelowAverageScenario_DevotedAcreage(CORN)
+1 BelowAverageScenario_QuantitySold(CORN)
<= 0

c_u_BelowAverageScenario_LimitAmountSold(SUGAR_BEETS)_:
-16 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 BelowAverageScenario_QuantitySold(SUGAR_BEETS)
<= 0

c_u_BelowAverageScenario_LimitAmountSold(WHEAT)_:
-2 BelowAverageScenario_DevotedAcreage(WHEAT)
+1 BelowAverageScenario_QuantitySold(WHEAT)
<= 0

c_e_BelowAverageScenario_ComputeFirstStageCost_:
-230 BelowAverageScenario_DevotedAcreage(CORN)
-260 BelowAverageScenario_DevotedAcreage(SUGAR_BEETS)
-150 BelowAverageScenario_DevotedAcreage(WHEAT)
+1 BelowAverageScenario_FirstStageCost
= 0

c_e_BelowAverageScenario_ComputeSecondStageCost_:
+1 BelowAverageScenario_Profit(CORN)
+1 BelowAverageScenario_Profit(SUGAR_BEETS)
+1 BelowAverageScenario_Profit(WHEAT)
-210 BelowAverageScenario_QuantityPurchased(CORN)
-100000 BelowAverageScenario_QuantityPurchased(SUGAR_BEETS)
-238 BelowAverageScenario_QuantityPurchased(WHEAT)
+1 BelowAverageScenario_SecondStageCost
= 0

c_e_BelowAverageScenario_ComputeProfits(CORN)_SOS2_constraint(1)_:
-4000 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(1)
-8000 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(2)
+1 BelowAverageScenario_QuantitySold(CORN)
= 0

c_u_BelowAverageScenario_ComputeProfits(CORN)_SOS2_constraint(2)_:
-3212000 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(1)
-12824000 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(2)
+1 BelowAverageScenario_Profit(CORN)
<= 0

c_e_BelowAverageScenario_ComputeProfits(CORN)_SOS2_constraint(3)_:
+1 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(0)
+1 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(1)
+1 BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(2)
= 1

c_e_BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(1)_:
-4000 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
-8000 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
+1 BelowAverageScenario_QuantitySold(SUGAR_BEETS)
= 0

c_u_BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(2)_:
-804000 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
-3208000 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
+1 BelowAverageScenario_Profit(SUGAR_BEETS)
<= 0

c_e_BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(3)_:
+1 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0)
+1 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
+1 BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
= 1

c_e_BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(1)_:
-4000 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
-8000 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
+1 BelowAverageScenario_QuantitySold(WHEAT)
= 0

c_u_BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(2)_:
-8004000 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
-32008000 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
+1 BelowAverageScenario_Profit(WHEAT)
<= 0

c_e_BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(3)_:
+1 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(0)
+1 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
+1 BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
= 1

c_u_AverageScenario_ConstrainTotalAcreage_:
+1 AverageScenario_DevotedAcreage(CORN)
+1 AverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 AverageScenario_DevotedAcreage(WHEAT)
<= 500

c_l_AverageScenario_EnforceCattleFeedRequirement(CORN)_:
+3 AverageScenario_DevotedAcreage(CORN)
+1 AverageScenario_QuantityPurchased(CORN)
-1 AverageScenario_QuantitySold(CORN)
>= 240

c_l_AverageScenario_EnforceCattleFeedRequirement(SUGAR_BEETS)_:
+20 AverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 AverageScenario_QuantityPurchased(SUGAR_BEETS)
-1 AverageScenario_QuantitySold(SUGAR_BEETS)
>= 0

c_l_AverageScenario_EnforceCattleFeedRequirement(WHEAT)_:
+2.5 AverageScenario_DevotedAcreage(WHEAT)
+1 AverageScenario_QuantityPurchased(WHEAT)
-1 AverageScenario_QuantitySold(WHEAT)
>= 200

c_u_AverageScenario_LimitAmountSold(CORN)_:
-3 AverageScenario_DevotedAcreage(CORN)
+1 AverageScenario_QuantitySold(CORN)
<= 0

c_u_AverageScenario_LimitAmountSold(SUGAR_BEETS)_:
-20 AverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 AverageScenario_QuantitySold(SUGAR_BEETS)
<= 0

c_u_AverageScenario_LimitAmountSold(WHEAT)_:
-2.5 AverageScenario_DevotedAcreage(WHEAT)
+1 AverageScenario_QuantitySold(WHEAT)
<= 0

c_e_AverageScenario_ComputeFirstStageCost_:
-230 AverageScenario_DevotedAcreage(CORN)
-260 AverageScenario_DevotedAcreage(SUGAR_BEETS)
-150 AverageScenario_DevotedAcreage(WHEAT)
+1 AverageScenario_FirstStageCost
= 0

c_e_AverageScenario_ComputeSecondStageCost_:
+1 AverageScenario_Profit(CORN)
+1 AverageScenario_Profit(SUGAR_BEETS)
+1 AverageScenario_Profit(WHEAT)
-210 AverageScenario_QuantityPurchased(CORN)
-100000 AverageScenario_QuantityPurchased(SUGAR_BEETS)
-238 AverageScenario_QuantityPurchased(WHEAT)
+1 AverageScenario_SecondStageCost
= 0

c_e_AverageScenario_ComputeProfits(CORN)_SOS2_constraint(1)_:
-5000 AverageScenario_ComputeProfits(CORN)_SOS2_y(1)
-10000 AverageScenario_ComputeProfits(CORN)_SOS2_y(2)
+1 AverageScenario_QuantitySold(CORN)
= 0

c_u_AverageScenario_ComputeProfits(CORN)_SOS2_constraint(2)_:
-5015000 AverageScenario_ComputeProfits(CORN)_SOS2_y(1)
-20030000 AverageScenario_ComputeProfits(CORN)_SOS2_y(2)
+1 AverageScenario_Profit(CORN)
<= 0

c_e_AverageScenario_ComputeProfits(CORN)_SOS2_constraint(3)_:
+1 AverageScenario_ComputeProfits(CORN)_SOS2_y(0)
+1 AverageScenario_ComputeProfits(CORN)_SOS2_y(1)
+1 AverageScenario_ComputeProfits(CORN)_SOS2_y(2)
= 1

c_e_AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(1)_:
-5000 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
-10000 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
+1 AverageScenario_QuantitySold(SUGAR_BEETS)
= 0

c_u_AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(2)_:
-1255000 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
-5010000 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
+1 AverageScenario_Profit(SUGAR_BEETS)
<= 0

c_e_AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(3)_:
+1 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0)
+1 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
+1 AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
= 1

c_e_AverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(1)_:
-5000 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
-10000 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
+1 AverageScenario_QuantitySold(WHEAT)
= 0

c_u_AverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(2)_:
-12505000 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
-50010000 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
+1 AverageScenario_Profit(WHEAT)
<= 0

c_e_AverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(3)_:
+1 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(0)
+1 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
+1 AverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
= 1

c_u_AboveAverageScenario_ConstrainTotalAcreage_:
+1 AboveAverageScenario_DevotedAcreage(CORN)
+1 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 AboveAverageScenario_DevotedAcreage(WHEAT)
<= 500

c_l_AboveAverageScenario_EnforceCattleFeedRequirement(CORN)_:
+3.6000000000000001 AboveAverageScenario_DevotedAcreage(CORN)
+1 AboveAverageScenario_QuantityPurchased(CORN)
-1 AboveAverageScenario_QuantitySold(CORN)
>= 240

c_l_AboveAverageScenario_EnforceCattleFeedRequirement(SUGAR_BEETS)_:
+24 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 AboveAverageScenario_QuantityPurchased(SUGAR_BEETS)
-1 AboveAverageScenario_QuantitySold(SUGAR_BEETS)
>= 0

c_l_AboveAverageScenario_EnforceCattleFeedRequirement(WHEAT)_:
+3 AboveAverageScenario_DevotedAcreage(WHEAT)
+1 AboveAverageScenario_QuantityPurchased(WHEAT)
-1 AboveAverageScenario_QuantitySold(WHEAT)
>= 200

c_u_AboveAverageScenario_LimitAmountSold(CORN)_:
-3.6000000000000001 AboveAverageScenario_DevotedAcreage(CORN)
+1 AboveAverageScenario_QuantitySold(CORN)
<= 0

c_u_AboveAverageScenario_LimitAmountSold(SUGAR_BEETS)_:
-24 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS)
+1 AboveAverageScenario_QuantitySold(SUGAR_BEETS)
<= 0

c_u_AboveAverageScenario_LimitAmountSold(WHEAT)_:
-3 AboveAverageScenario_DevotedAcreage(WHEAT)
+1 AboveAverageScenario_QuantitySold(WHEAT)
<= 0

c_e_AboveAverageScenario_ComputeFirstStageCost_:
-230 AboveAverageScenario_DevotedAcreage(CORN)
-260 AboveAverageScenario_DevotedAcreage(SUGAR_BEETS)
-150 AboveAverageScenario_DevotedAcreage(WHEAT)
+1 AboveAverageScenario_FirstStageCost
= 0

c_e_AboveAverageScenario_ComputeSecondStageCost_:
+1 AboveAverageScenario_Profit(CORN)
+1 AboveAverageScenario_Profit(SUGAR_BEETS)
+1 AboveAverageScenario_Profit(WHEAT)
-210 AboveAverageScenario_QuantityPurchased(CORN)
-100000 AboveAverageScenario_QuantityPurchased(SUGAR_BEETS)
-238 AboveAverageScenario_QuantityPurchased(WHEAT)
+1 AboveAverageScenario_SecondStageCost
= 0

c_e_AboveAverageScenario_ComputeProfits(CORN)_SOS2_constraint(1)_:
-6000 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(1)
-12000 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(2)
+1 AboveAverageScenario_QuantitySold(CORN)
= 0

c_u_AboveAverageScenario_ComputeProfits(CORN)_SOS2_constraint(2)_:
-7218000 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(1)
-28836000 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(2)
+1 AboveAverageScenario_Profit(CORN)
<= 0

c_e_AboveAverageScenario_ComputeProfits(CORN)_SOS2_constraint(3)_:
+1 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(0)
+1 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(1)
+1 AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(2)
= 1

c_e_AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(1)_:
-6000 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
-12000 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
+1 AboveAverageScenario_QuantitySold(SUGAR_BEETS)
= 0

c_u_AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(2)_:
-1806000 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
-7212000 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
+1 AboveAverageScenario_Profit(SUGAR_BEETS)
<= 0

c_e_AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_constraint(3)_:
+1 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0)
+1 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1)
+1 AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2)
= 1

c_e_AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(1)_:
-6000 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
-12000 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
+1 AboveAverageScenario_QuantitySold(WHEAT)
= 0

c_u_AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(2)_:
-18006000 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
-72012000 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
+1 AboveAverageScenario_Profit(WHEAT)
<= 0

c_e_AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_constraint(3)_:
+1 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(0)
+1 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1)
+1 AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2)
= 1

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds 
    -inf <= MASTER_BLEND_VAR_RootNode(1) <= +inf
    -inf <= MASTER_BLEND_VAR_RootNode(2) <= +inf
    -inf <= MASTER_BLEND_VAR_RootNode(3) <= +inf
   0 <= BelowAverageScenario_DevotedAcreage(CORN) <= 500
   0 <= BelowAverageScenario_DevotedAcreage(SUGAR_BEETS) <= 500
   0 <= BelowAverageScenario_DevotedAcreage(WHEAT) <= 500
   0 <= BelowAverageScenario_QuantityPurchased(CORN) <= +inf
   0 <= BelowAverageScenario_QuantityPurchased(SUGAR_BEETS) <= +inf
   0 <= BelowAverageScenario_QuantityPurchased(WHEAT) <= +inf
   0 <= BelowAverageScenario_QuantitySold(CORN) <= 8000
   0 <= BelowAverageScenario_QuantitySold(SUGAR_BEETS) <= 8000
   0 <= BelowAverageScenario_QuantitySold(WHEAT) <= 8000
   0 <= BelowAverageScenario_Profit(CORN) <= +inf
   0 <= BelowAverageScenario_Profit(SUGAR_BEETS) <= +inf
   0 <= BelowAverageScenario_Profit(WHEAT) <= +inf
    -inf <= BelowAverageScenario_FirstStageCost <= +inf
    -inf <= BelowAverageScenario_SecondStageCost <= +inf
   0 <= BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(0) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(1) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(2) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(0) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1) <= +inf
   0 <= BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2) <= +inf
   0 <= AverageScenario_DevotedAcreage(CORN) <= 500
   0 <= AverageScenario_DevotedAcreage(SUGAR_BEETS) <= 500
   0 <= AverageScenario_DevotedAcreage(WHEAT) <= 500
   0 <= AverageScenario_QuantityPurchased(CORN) <= +inf
   0 <= AverageScenario_QuantityPurchased(SUGAR_BEETS) <= +inf
   0 <= AverageScenario_QuantityPurchased(WHEAT) <= +inf
   0 <= AverageScenario_QuantitySold(CORN) <= 10000
   0 <= AverageScenario_QuantitySold(SUGAR_BEETS) <= 10000
   0 <= AverageScenario_QuantitySold(WHEAT) <= 10000
   0 <= AverageScenario_Profit(CORN) <= +inf
   0 <= AverageScenario_Profit(SUGAR_BEETS) <= +inf
   0 <= AverageScenario_Profit(WHEAT) <= +inf
    -inf <= AverageScenario_FirstStageCost <= +inf
    -inf <= AverageScenario_SecondStageCost <= +inf
   0 <= AverageScenario_ComputeProfits(CORN)_SOS2_y(0) <= +inf
   0 <= AverageScenario_ComputeProfits(CORN)_SOS2_y(1) <= +inf
   0 <= AverageScenario_ComputeProfits(CORN)_SOS2_y(2) <= +inf
   0 <= AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0) <= +inf
   0 <= AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1) <= +inf
   0 <= AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2) <= +inf
   0 <= AverageScenario_ComputeProfits(WHEAT)_SOS2_y(0) <= +inf
   0 <= AverageScenario_ComputeProfits(WHEAT)_SOS2_y(1) <= +inf
   0 <= AverageScenario_ComputeProfits(WHEAT)_SOS2_y(2) <= +inf
   0 <= AboveAverageScenario_DevotedAcreage(CORN) <= 500
   0 <= AboveAverageScenario_DevotedAcreage(SUGAR_BEETS) <= 500
   0 <= AboveAverageScenario_DevotedAcreage(WHEAT) <= 500
   0 <= AboveAverageScenario_QuantityPurchased(CORN) <= +inf
   0 <= AboveAverageScenario_QuantityPurchased(SUGAR_BEETS) <= +inf
   0 <= AboveAverageScenario_QuantityPurchased(WHEAT) <= +inf
   0 <= AboveAverageScenario_QuantitySold(CORN) <= 12000
   0 <= AboveAverageScenario_QuantitySold(SUGAR_BEETS) <= 12000
   0 <= AboveAverageScenario_QuantitySold(WHEAT) <= 12000
   0 <= AboveAverageScenario_Profit(CORN) <= +inf
   0 <= AboveAverageScenario_Profit(SUGAR_BEETS) <= +inf
   0 <= AboveAverageScenario_Profit(WHEAT) <= +inf
    -inf <= AboveAverageScenario_FirstStageCost <= +inf
    -inf <= AboveAverageScenario_SecondStageCost <= +inf
   0 <= AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(0) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(1) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(2) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(0) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1) <= +inf
   0 <= AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2) <= +inf
SOS
BelowAverageScenario_ComputeProfits(CORN)_SOS2_sosconstraint: S2::
BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(0):1
BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(1):2
BelowAverageScenario_ComputeProfits(CORN)_SOS2_y(2):3
BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_sosconstraint: S2::
BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0):1
BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1):2
BelowAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2):3
BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_sosconstraint: S2::
BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(0):1
BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1):2
BelowAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2):3
AverageScenario_ComputeProfits(CORN)_SOS2_sosconstraint: S2::
AverageScenario_ComputeProfits(CORN)_SOS2_y(0):1
AverageScenario_ComputeProfits(CORN)_SOS2_y(1):2
AverageScenario_ComputeProfits(CORN)_SOS2_y(2):3
AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_sosconstraint: S2::
AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0):1
AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1):2
AverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2):3
AverageScenario_ComputeProfits(WHEAT)_SOS2_sosconstraint: S2::
AverageScenario_ComputeProfits(WHEAT)_SOS2_y(0):1
AverageScenario_ComputeProfits(WHEAT)_SOS2_y(1):2
AverageScenario_ComputeProfits(WHEAT)_SOS2_y(2):3
AboveAverageScenario_ComputeProfits(CORN)_SOS2_sosconstraint: S2::
AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(0):1
AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(1):2
AboveAverageScenario_ComputeProfits(CORN)_SOS2_y(2):3
AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_sosconstraint: S2::
AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(0):1
AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(1):2
AboveAverageScenario_ComputeProfits(SUGAR_BEETS)_SOS2_y(2):3
AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_sosconstraint: S2::
AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(0):1
AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(1):2
AboveAverageScenario_ComputeProfits(WHEAT)_SOS2_y(2):3
end 
