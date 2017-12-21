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
	c15
	c16
	c17
	c18
	c19
	c20;

VARIABLES
	GAMS_OBJECTIVE
	x1;

c1.. log(x1) =e= 0.0 ;
c2.. log10(x1) =e= 0.0 ;
c3.. sin(x1) =e= 0.0 ;
c4.. cos(x1) =e= 0.0 ;
c5.. tan(x1) =e= 0.0 ;
c6.. sinh(x1) =e= 0.0 ;
c7.. cosh(x1) =e= 0.0 ;
c8.. tanh(x1) =e= 0.0 ;
c9.. asin(x1) =e= 0.0 ;
c10.. acos(x1) =e= 0.0 ;
c11.. atan(x1) =e= 0.0 ;
c12.. asinh(x1) =e= 0.0 ;
c13.. acosh(x1) =e= 0.0 ;
c14.. atanh(x1) =e= 0.0 ;
c15.. exp(x1) =e= 0.0 ;
c16.. sqrt(x1) =e= 0.0 ;
c17.. ceil(x1) =e= 0.0 ;
c18.. floor(x1) =e= 0.0 ;
c19.. abs(x1) =e= 0.0 ;
c20.. GAMS_OBJECTIVE =e= x1 ;


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

