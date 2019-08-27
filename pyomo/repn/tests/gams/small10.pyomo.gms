$offdigit

EQUATIONS
	c1
	c2
	c3
	c4
	c5
	c6
	c7
	c8
	c9
	c10
	c11
	c12
	c13
	c14
	c15;

VARIABLES
	GAMS_OBJECTIVE
	x1;

c1.. x1 =e= 0 ;
c2.. 0*x1 + x1 =e= 0 ;
c3.. x1 =e= 0 ;
c4.. x1 =e= 0 ;
c5.. 0*power(x1, 2) + x1 =e= 0 ;
c6.. 0*x1*1 + x1 =e= 0 ;
c7.. 0*power(x1, 2) + x1 =e= 0 ;
c8.. 0*x1*1 + x1 =e= 0 ;
c9.. 0*x1 =e= 0 ;
c10.. 0*power(x1, 2) =e= 0 ;
c11.. 0*x1*1 =e= 0 ;
c12.. 0*power(x1, 2) =e= 0 ;
c13.. 0*x1*1 =e= 0 ;
c14.. 0*x1 =e= 0 ;
c15.. GAMS_OBJECTIVE =e= x1 + 0*x1 + 0*x1 + x1*x1*0 + x1*x1*0 + 0*power(x1, 2) ;


MODEL GAMS_MODEL /all/ ;
SOLVE GAMS_MODEL USING nlp minimizing GAMS_OBJECTIVE;

Scalars MODELSTAT 'model status', SOLVESTAT 'solve status';
MODELSTAT = GAMS_MODEL.modelstat;
SOLVESTAT = GAMS_MODEL.solvestat;

Scalar OBJEST 'best objective', OBJVAL 'objective value';
OBJEST = GAMS_MODEL.objest;
OBJVAL = GAMS_MODEL.objval;

Scalar NUMVAR 'number of variables';
NUMVAR = GAMS_MODEL.numvar

Scalar NUMEQU 'number of equations';
NUMEQU = GAMS_MODEL.numequ

Scalar NUMDVAR 'number of discrete variables';
NUMDVAR = GAMS_MODEL.numdvar

Scalar NUMNZ 'number of nonzeros';
NUMNZ = GAMS_MODEL.numnz

Scalar ETSOLVE 'time to execute solve statement';
ETSOLVE = GAMS_MODEL.etsolve

