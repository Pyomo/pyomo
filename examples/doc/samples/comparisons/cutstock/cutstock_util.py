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


def getCutCount():
    cutCount = 0
    fout1 = open('WidthDemand.csv', 'r')
    for eachline in fout1:
        cutCount += 1
    fout1.close()
    return cutCount


def getPatCount():
    patCount = 0
    fout2 = open('Waste.csv', 'r')
    for eachline in fout2:
        patCount += 1
    fout2.close()
    return patCount


def getPriceSheetData():
    return 28


def getSheetsAvail():
    return 2000


def getCuts():
    cutcount = getCutCount()
    Cuts = range(cutcount)
    for i in range(cutcount):
        nstr = str(i + 1)
        Cuts[i] = 'w' + nstr
    return Cuts


def getPatterns():
    patcount = getPatCount()
    Patterns = range(patcount)
    for j in range(patcount):
        pstr = str(j + 1)
        Patterns[j] = 'P' + pstr
    return Patterns


def getCutDemand():
    i = 0
    cutcount = getCutCount()
    CutDemand = range(cutcount)
    fout1 = open('WidthDemand.csv', 'r')
    for eachline in fout1.readlines():
        str = eachline.rstrip("\n")
        CutDemand[i] = int(str)
        i += 1
    fout1.close()
    return CutDemand


def getCutsInPattern():
    cutcount = getCutCount()
    patcount = getPatCount()
    CutsInPattern = [[0 for col in range(patcount)] for row in range(cutcount)]
    fout2 = open('Patterns.csv', 'r')
    for eachline in fout2.readlines():
        str = eachline
        lstr = str.split(",")
        pstr = lstr[0]
        wstr = lstr[1]
        cstr = lstr[2]
        pstr = pstr.replace("P", "")
        wstr = wstr.replace("w", "")
        cstr = cstr.rstrip("\n")
        p = int(pstr)
        w = int(wstr)
        c = int(cstr)
        CutsInPattern[w - 1][p - 1] = c
    fout2.close()
    return CutsInPattern
