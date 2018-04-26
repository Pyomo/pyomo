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
	x1
	x2;

c1.. log(x1) =e= 0.0 ;
c2.. log10(x1) =e= 0.0 ;
c3.. sin(x2) =e= 0.0 ;
c4.. cos(x2) =e= 1.0 ;
c5.. tan(x2) =e= 0.0 ;
c6.. sinh(x2) =e= 0.0 ;
c7.. cosh(x2) =e= 1.0 ;
c8.. tanh(x2) =e= 0.0 ;
c9.. asin(x2) =e= 0.0 ;
c10.. acos(x2) =e= 1.5707963267948966 ;
c11.. atan(x2) =e= 0.0 ;
c12.. asinh(x2) =e= 0.0 ;
c13.. acosh((7.3890560989306495 + x1)*0.18393972058572117) =e= 0.0 ;
c14.. atanh(x2) =e= 0.0 ;
c15.. exp(x2) =e= 1.0 ;
c16.. sqrt(x1) =e= 1.0 ;
c17.. ceil(x1) =e= 1.0 ;
c18.. floor(x1) =e= 1.0 ;
c19.. abs(x1) =e= 1.0 ;
c20.. GAMS_OBJECTIVE =e= x1 + x2 ;

x1.l = 1;
x2.l = 0;

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

