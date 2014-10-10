Model source: Jorjani, Scott, and Woodruff. 1996. Selection of Optimal Subset of Sizes. Technical Report. 

Described also in: Progressive Hedging and Tabu Search Applied to Mixed-Integer (0,1) Multistage Stochastic Programming. 
                   Journal of Heuristics, Volume 2, 1996, pp. 111-128.

With CPLEX 11.21, the following are the EF optimal objectives and solution times:

SIZES3:  Optimal objective value: 224275.3334       Achieved after 1 second, 6K branch-and-bound nodes.
SIZES10: Objective value with gap=0.12%: 224406.64  Achieved after 10+ minutes and 2.5M branch-and-bound nodes.

SIZES3 EF (OPTIMAL) SOLUTION (non-zero elements):

ProduceSizeFirstStage(3)                  1.000000
ProduceSizeFirstStage(5)                  1.000000
ProduceSizeFirstStage(6)                  1.000000
ProduceSizeFirstStage(8)                  1.000000
ProduceSizeFirstStage(10)                 1.000000

NumProducedFirstStage(3)              38250.000000
NumProducedFirstStage(5)              45000.000000
NumProducedFirstStage(6)              50000.000000
NumProducedFirstStage(8)              42750.000000
NumProducedFirstStage(10)             24000.000000

NumUnitsCutFirstStage(3,3)            12500.000000
NumUnitsCutFirstStage(3,1)             2500.000000
NumUnitsCutFirstStage(8,7)            15000.000000
NumUnitsCutFirstStage(3,2)             7500.000000
NumUnitsCutFirstStage(5,4)            10000.000000
NumUnitsCutFirstStage(6,6)            25000.000000
NumUnitsCutFirstStage(5,5)            35000.000000
NumUnitsCutFirstStage(10,10)           5000.000000
NumUnitsCutFirstStage(10,9)           12500.000000
NumUnitsCutFirstStage(8,8)            12500.000000

SIZES10 EF (SUB-OPTIMAL) SOLUTION (non-zero elements):

ProduceSizeFirstStage(3)                  1.000000
ProduceSizeFirstStage(5)                  1.000000
ProduceSizeFirstStage(6)                  1.000000
ProduceSizeFirstStage(8)                  1.000000
ProduceSizeFirstStage(10)                 1.000000

NumProducedFirstStage(3)              33750.000000
NumProducedFirstStage(5)              45000.000000
NumProducedFirstStage(6)              52500.000000
NumProducedFirstStage(8)              43750.000000
NumProducedFirstStage(10)             25000.000000

NumUnitsCutFirstStage(8,8)            12500.000000
NumUnitsCutFirstStage(5,4)            10000.000000
NumUnitsCutFirstStage(3,2)             7500.000000
NumUnitsCutFirstStage(6,6)            25000.000000
NumUnitsCutFirstStage(10,10)           5000.000000
NumUnitsCutFirstStage(9,7)               -0.000000
NumUnitsCutFirstStage(3,1)             2500.000000
NumUnitsCutFirstStage(3,3)            12500.000000
NumUnitsCutFirstStage(8,7)            15000.000000
NumUnitsCutFirstStage(6,5)                0.000000
NumUnitsCutFirstStage(5,5)            35000.000000
NumUnitsCutFirstStage(10,9)           12500.000000

Baseline PH results (no plug-ins or tweaks, other than reasonable rho values):

runph --model-directory=models --instance-directory=SIZESX --max-iterations=200 --rho-cfgfile=config/rhosetter.cfg --scenario-solver-options="mip_tolerances_integrality=1e-7"

NOTE: The big-M formulation requires the mip tolerances to be tightened; otherwise, with a production capacity of 200K and a default (CPLEX)
      integrality tolerance of 1e-5, you can end up producing quantity of certain sizes when production of that size is disabled!

NOTE: Small rhos in general are required to obtain good solutions. The rule specified in the rhosetter.cfg file is cost-proportional
      with a factor equal to 0.001. Larger values accelerate convergence significantly, as the cost of solution quality.

SIZES3:  Converges in 131 iterations, with an objective equal to 224480.8819
SIZES10: Doesn't converge after 200 iterations, but is very close. Minor discrepancies in the 
         NumProducedFirstStage and NumUnitsCutFirstStage variables. Objective value equals
         224591.7098, but this doesn't make much sense because the solution isn't fully converged.
         But: The max-min difference in first-stage costs is only 1.2272, so this solution is very
         good.

The SIZES example is a great example of where fixing is needed - PH gets close to a high-quality solution quickly, but takes a long
time to "polish" the solution to complete agreement. Thus, acceleration can be achieved by using variable fixing and slamming; both
techniques are described in the PySP documentation. To use these techniques, simply add the following command-line arguments to the
runph invocation:

--enable-ww-extensions --ww-extension-cfgfile=config/wwph.cfg --ww-extension-suffixfile=config/wwph.suffixes 

With these techniques, runph yields the following performance:

SIZES3:  Converges in 113 iterations with an objective value=224553.5936
SIZES10: Converges in 125 iterations with an objective value=224702.6184

In the case of SIZES3, the acceleration comes at a slight increase in solution cost. For SIZES10, it makes convergence possible. 
In both cases, the solutions are very high-quality relative to the EF objective function values. 
