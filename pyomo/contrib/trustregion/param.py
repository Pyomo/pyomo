# param.py
# This file is modified to provide TRF algorithmic parameters

# Replace current use of constants with Config Blocks
# from pyutilib.misc.config import ConfigBlock, ConfigValue

# CONFIG = ConfigBlock('Trust Region')

# CONFIG.declare('trust radius', ConfigValue(
#     default=1,
#     domain=_positiveFloat,
#     description='short description',
#     doc='long documentation'))
# CONFIG.declare('sample region', ConfigValue(
#     default=True,
#     domain=bool,
#     description='short description',
#     doc='long documentation'))


# Initialization
TRUST_RADIUS = 1
SAMPLEREGION_YN = True

if SAMPLEREGION_YN:
    SAMPLE_RADIUS = 0.1
else:
    SAMPLE_RADIUS = TRUST_RADIUS/2

RADIUS_MAX = 1000*TRUST_RADIUS

# Termination tolerances
EP_I = 1e-5
EP_DELT = 1e-5
EP_CHI = 1e-3
DELTMIN = 1e-6  # DELTMIN <= EP_DELT
MAXIT = 20

# Compatibility Check Parameters
KAPPA_DELTA = 0.8
KAPPA_MU = 1.0
MU = 0.5
EP_COMPAT = EP_I  # Suggested value: EP_COMPAT = EP_I
COMPAT_PENALTY = 0.

# Criticality Check Parameters
CRITICALITY_CHECK = 0.1

# Trust region update parameters
GAMMA_C = 0.5
GAMMA_E = 2.5

# Switching Condition
GAMMA_S = 2.0
KAPPA_THETA = 0.1
THETA_MIN = 1e-4

# Filter
GAMMA_F = 0.01
GAMMA_THETA = 0.01
THETA_MAX = 50

# Ratio test parameters (for theta steps)
ETA1 = 0.05
ETA2 = 0.2

# Output level (replace with real printlevels!!!)
PRINT_VARS = False

# Sample Radius reset parameter
SR_ADJUST = 0.5

# Default romtype
# 0 = Linear
# 1 = Quadratic
DEFAULT_ROMTYPE = 1


