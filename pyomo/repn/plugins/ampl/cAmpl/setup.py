from distutils.core import setup, Extension

import os
import fnmatch

sourceFiles = []
for root, dirnames, filenames in os.walk('.'):
    for filename in fnmatch.filter(filenames, "*.c"):
        sourceFiles.append(os.path.join(root, filename))

# Enable for debug only
eca = ['-Werror', '-O3']
cAmpl = Extension('cAmpl', sources = sourceFiles, extra_compile_args = eca)

setup( name = 'cAmpl',
        version = '0.1',
        description = 'C AMPL representation generator',
        ext_modules = [cAmpl] )
