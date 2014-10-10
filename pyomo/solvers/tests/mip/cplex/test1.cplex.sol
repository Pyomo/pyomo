<?xml version = "1.0" standalone="yes"?>
<?xml-stylesheet href="http://www.ilog.com/products/cplex/xmlv1.1/solution.xsl" type="text/xsl"?>
<CPLEXSolution version="1.1">
 <header
   problemName="test1.mps"
   objectiveValue="3"
   solutionTypeValue="1"
   solutionTypeString="basic"
   solutionStatusValue="1"
   solutionStatusString="optimal"
   solutionMethodString="dual"
   primalFeasible="1"
   dualFeasible="1"
   simplexIterations="0"/>
 <quality
   epRHS="1e-06"
   epOpt="1e-06"
   maxPrimalInfeas="0"
   maxDualInfeas="0"
   maxPrimalResidual="0"
   maxDualResidual="0"
   maxX="1"
   maxPi="1"
   maxSlack="0"
   maxRedCost="2"
   kappa="1"/>
 <linearConstraints>
  <constraint name="NODEA1" index="0" status="LL" slack="0" dual="1"/>
  <constraint name="NODEA2" index="1" status="LL" slack="0" dual="1"/>
  <constraint name="NODEA3" index="2" status="LL" slack="0" dual="1"/>
 </linearConstraints>
 <variables>
  <variable name="X11" index="0" status="BS" value="1" reducedCost="0"/>
  <variable name="X12" index="1" status="LL" value="0" reducedCost="1"/>
  <variable name="X13" index="2" status="LL" value="0" reducedCost="2"/>
  <variable name="X21" index="3" status="LL" value="0" reducedCost="1"/>
  <variable name="X22" index="4" status="LL" value="0" reducedCost="2"/>
  <variable name="X23" index="5" status="BS" value="1" reducedCost="0"/>
  <variable name="X31" index="6" status="LL" value="0" reducedCost="2"/>
  <variable name="X32" index="7" status="BS" value="1" reducedCost="0"/>
  <variable name="X33" index="8" status="LL" value="0" reducedCost="1"/>
 </variables>
</CPLEXSolution>
