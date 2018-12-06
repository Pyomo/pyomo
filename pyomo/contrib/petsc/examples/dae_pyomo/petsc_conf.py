import warnings

def get_conf():
    """Parses various PETSc configuration/include files to get data types.

    precision, indices, complexscalars = get_conf()

    Output:
      precision: 'single', 'double', 'longlong' indicates precision of PetscScalar
      indices: '32', '64' indicates bit-size of PetscInt
      complex: True/False indicates whether PetscScalar is complex or not.
    """

    import sys, os
    precision = None
    indices = None
    complexscalars = None

    if 'PETSC_DIR' in os.environ:
        petscdir = os.environ['PETSC_DIR']
    else:
        warnings.warn('PETSC_DIR env not set - unable to locate PETSc installation, using defaults')
        return None, None, None

    if os.path.isfile(os.path.join(petscdir,'lib','petsc','conf','petscrules')):
        # found prefix install
        petscvariables = os.path.join(petscdir,'lib','petsc','conf','petscvariables')
        petscconfinclude = os.path.join(petscdir,'include','petscconf.h')
    else:
        if 'PETSC_ARCH' in os.environ:
            petscarch = os.environ['PETSC_ARCH']
            if os.path.isfile(os.path.join(petscdir,petscarch,'lib','petsc','conf','petscrules')):
                # found legacy install
                petscvariables = os.path.join(petscdir,petscarch,'lib','petsc','conf','petscvariables')
                petscconfinclude = os.path.join(petscdir,petscarch,'include','petscconf.h')
            else:
                warnings.warn('Unable to locate PETSc installation in specified PETSC_DIR/PETSC_ARCH, using defaults')
                return None, None, None
        else:
            warnings.warn('PETSC_ARCH env not set or incorrect PETSC_DIR is given - unable to locate PETSc installation, using defaults')
            return None, None, None

    try:
        fid = open(petscvariables, 'r')
    except IOError:
        warnings.warn('Nonexistent or invalid PETSc installation, using defaults')
        return None, None, None
    else:
        for line in fid:
            if line.startswith('PETSC_PRECISION'):
                precision = line.strip().split('=')[1].strip('\n').strip()

        fid.close()

    try:
        fid = open(petscconfinclude, 'r')
    except IOError:
        warnings.warn('Nonexistent or invalid PETSc installation, using defaults')
        return None, None, None
    else:
        for line in fid:
            if line.startswith('#define PETSC_USE_64BIT_INDICES 1'):
                indices = '64bit'
            elif line.startswith('#define PETSC_USE_COMPLEX 1'):
                complexscalars = True

        if indices is None:
            indices = '32bit'
        if complexscalars is None:
            complexscalars = False
        fid.close()

    return precision, indices, complexscalars
