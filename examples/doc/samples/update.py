#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#!/usr/bin/env python
#
# This is a Python script that regenerates the top-level TRAC.txt file, which
# is loaded by the Pyomo Trac wiki
#

import glob
import os.path


def get_title(fname):
    INPUT = open(fname, 'r')
    for line in INPUT:
        sline = line.strip()
        # print sline
        # print sline[0:2]
        # print '.%s.' % sline[-2:]
        if sline[0:2] == '= ' and sline[-2:] == ' =':
            tmp = sline[2:-2]
            tmp.strip()
            return tmp
    return fname


OUTPUT = open('TRAC.txt', 'w')
print >> OUTPUT, """{{{
#!comment
;
; Trac examples generated automatically by the update.py script
;
}}}

= Pyomo Gallery =

Click on a link below for case studies, Pyomo script examples, and comparisons between Pyomo and other modeling languages.


"""

print >> OUTPUT, '== Case Studies =='
print >> OUTPUT, ''
print >> OUTPUT, """The following links provide case studies that illustrate the use of Pyomo to formulate and analyze optimization models."""
print >> OUTPUT, ''

for Dir in glob.glob('case_studies/*'):
    dir = os.path.basename(Dir)
    fname = 'case_studies/%s/README.txt' % dir
    if os.path.exists(fname):
        print >> OUTPUT, " * [wiki:Documentation/PyomoGallery/CaseStudies/%s %s]" % (
            dir,
            get_title(fname),
        )
        print >> OUTPUT, "{{{\n#!comment\n[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/case_studies/%s/README.txt, text/x-trac-wiki)]]\n}}}" % dir


print >> OUTPUT, ''
print >> OUTPUT, '== Pyomo Scripts =='
print >> OUTPUT, ''
print >> OUTPUT, """The following links describe examples that show how to execute Pyomo functionality with Python scripts."""
print >> OUTPUT, ''

for Dir in glob.glob('scripts/*'):
    dir = os.path.basename(Dir)
    fname = 'scripts/%s/README.txt' % dir
    if os.path.exists(fname):
        print >> OUTPUT, " * [wiki:Documentation/PyomoGallery/Scripts/%s %s]" % (
            dir,
            get_title(fname),
        )
        print >> OUTPUT, "{{{\n#!comment\n[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/%s/README.txt, text/x-trac-wiki)]]\n}}}" % dir

print >> OUTPUT, ''
print >> OUTPUT, '== Modeling Comparisons =='
print >> OUTPUT, ''
print >> OUTPUT, """The following links provide documentation of optimization models that can be used to compare and contrast Pyomo with other optimization modeling tools. Note that the list of [wiki:Documentation/RelatedProjects related projects] summarizes Python software frameworks that provide optimization functionality that is similar to Pyomo."""
print >> OUTPUT, ''

for Dir in glob.glob('comparisons/*'):
    dir = os.path.basename(Dir)
    fname = 'comparisons/%s/README.txt' % dir
    if os.path.exists(fname):
        print >> OUTPUT, " * [wiki:Documentation/PyomoGallery/ModelingComparisons/%s %s]" % (
            dir,
            get_title(fname),
        )
        print >> OUTPUT, "{{{\n#!comment\n[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/comparisons/%s/README.txt, text/x-trac-wiki)]]\n}}}" % dir
