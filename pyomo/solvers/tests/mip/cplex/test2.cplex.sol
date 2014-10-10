<?xml version = "1.0" standalone="yes"?>
<?xml-stylesheet href="http://www.ilog.com/products/cplex/xmlv1.1/solution.xsl" type="text/xsl"?>
<CPLEXSolution version="1.1">
 <header
   problemName="test2.lp"
   solutionName="incumbent"
   solutionIndex="-1"
   objectiveValue="2"
   solutionTypeValue="3"
   solutionTypeString="primal"
   solutionStatusValue="101"
   solutionStatusString="integer optimal solution"
   solutionMethodString="mip"
   primalFeasible="1"
   dualFeasible="1"
   MIPNodes="0"
   MIPIterations="0"/>
 <quality
   epInt="1e-05"
   epRHS="1e-06"
   maxIntInfeas="0"
   maxPrimalInfeas="0"
   maxX="1"
   maxSlack="4"/>
 <linearConstraints>
  <constraint name="C1" index="0" slack="4"/>
  <constraint name="c2" index="1" slack="1"/>
 </linearConstraints>
 <variables>
  <variable name="x1" index="0" value="0"/>
  <variable name="x2" index="1" value="1"/>
 </variables>
</CPLEXSolution>
