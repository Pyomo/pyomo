var ONE := 1;
var ZERO := 0;
param pi = 4 * atan(1);
param e = exp(1);

minimize obj: ONE+ZERO;

s.t.
     c_log: log(ONE) = 0;
     c_log10: log10(ONE) = 0;
     c_sin: sin(ZERO) = 0;
     c_cos: cos(ZERO) = 1;
     c_tan: tan(ZERO) = 0;
     c_sinh: sinh(ZERO) = 0;
     c_cosh: cosh(ZERO) = 1;
     c_tanh: tanh(ZERO) = 0;
     c_asin: asin(ZERO) = 0;
     c_acos: acos(ZERO) = pi/2;
     c_atan: atan(ZERO) = 0;
     c_asinh: asinh(ZERO) = 0;
     c_acosh: acosh((e^2 + ONE)/(2*e)) = 0;
     c_atanh: atanh(ZERO) = 0;
     c_exp: exp(ZERO) = 1;
     c_sqrt: sqrt(ONE) = 1;
     c_ceil: ceil(ONE) = 1;
     c_floor: floor(ONE) = 1;
     c_abs: abs(ONE) = 1;

option substout 0;
option presolve 0;
option auxfiles 'rc';
write gsmall14.ampl;
