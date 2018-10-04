""" SPECIAL COPY by DLW, Sept 1, 2018
Delete this file if you see it after Oct 1, 2018
It is here because I am waiting for the release of IDAES before
a PR merging the (minor) changes to make it work with parmest.
Between now and then, you will need to copy this down into idaes
if you want to try MEA with parmest (i.e., replace mea_estimate_pysp.py
in examples/contrib/projects/mea_simple/parameter_estimate/)
"""

##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
# 
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes".
##############################################################################
"""
This demonstrates a paramter estimation using process models. This is currently
a monolithic problem, but the idea is to use this as a starting point to solve
the same problem with PySP for decomposition, which will allow bigger problems
to be solved.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import #disable implicit relative imports

#
import os
import sys
import csv
import re
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Pyomo Imports
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.pysp.scenariotree.tree_structure_model \
        import CreateConcreteTwoStageScenarioTreeModel
#
from idaes.contrib.projects.mea_simple.flowsheet.units import Column
from idaes.contrib.projects.mea_simple.flowsheet.units.transport \
    import TransportParams
from idaes.contrib.projects.mea_simple.flowsheet.properties.prop \
    import PropertyParams
import idaes.core.util.misc as model_util
from idaes.core.util.misc import hhmmss
from idaes.core.util.model_serializer import StoreSpec
from idaes.core import UnitBlock, FlowsheetBlock
from idaes.core.util.model_serializer import to_json, from_json

case_char = "K" # leading character for the cases of interest

initial_01 = {
    "Keq":{
        "a1":{"value":198.9, "ub":200, "lb":198, "fixed":True},
        "b1":{"value":-1986.0, "ub":-1900, "lb":-1987, "fixed":True},
        "c1":{"value":-32.71, "ub":-31, "lb":-33, "fixed":True},
        "a2":{"value":167.0, "ub":168, "lb":165, "fixed":True},
        "b2":{"value":-2102.0, "ub":-2000, "lb":-2200, "fixed":True},
        "c2":{"value":-26.51, "ub":-26, "lb":-27, "fixed":True}},
    "kin":{
        "A_H2O":{"value":4.55, "ub":4.6, "lb":4.0, "fixed":False},
        "E_H2O":{"value":3287, "ub":4100, "lb":3280, "fixed":False},
        "A_MEA":{"value":4610, "ub":4800, "lb":4500, "fixed":False},
        "E_MEA":{"value":4412, "ub":4500, "lb":4000, "fixed":False}},
    "nrtl":{
        "A_co2_h2o":{"value":69.38507, "ub":71, "lb":66, "fixed":True}},
    "kv":{
        "Cv":{"value":0.357, "ub":0.50, "lb":0.33, "fixed":False},
        "a":{"value":0.75, "ub":0.80, "lb":0.22, "fixed":False},
        "b":{"value":1.0/3.0, "ub":0.4, "lb":0.25, "fixed":False}},
    "kl":{
        "Cl":{"value":0.5, "ub":0.55, "lb":0.45, "fixed":False}},
    "ae":{
        "X":{"value":157.035790903, "ub":160, "lb":15, "fixed":False},
        "a":{"value":8.62068965517, "ub":8.7, "lb":6, "fixed":False},
        "b":{"value":1.33333333333, "ub":1.4, "lb":-0.5, "fixed":False}}
}

init_set = initial_01

def unfix(blk, init=init_set):
    def fix(c, fixed):
        if fixed:
            c.fix()
        else:
            c.unfix()
    fix(blk.pp.Keq_a[1], init["Keq"]["a1"]["fixed"])
    fix(blk.pp.Keq_b[1], init["Keq"]["b1"]["fixed"])
    fix(blk.pp.Keq_c[1], init["Keq"]["c1"]["fixed"])
    fix(blk.pp.Keq_a[2], init["Keq"]["a2"]["fixed"])
    fix(blk.pp.Keq_b[2], init["Keq"]["b2"]["fixed"])
    fix(blk.pp.Keq_c[2], init["Keq"]["c2"]["fixed"])
    fix(blk.pp.nrtl_parA["CO2","H2O"], init["nrtl"]["A_co2_h2o"]["fixed"])
    fix(blk.tp.Cv, init["kv"]["Cv"]["fixed"])
    fix(blk.tp.kv_para, init["kv"]["a"]["fixed"])
    fix(blk.tp.kv_parb, init["kv"]["b"]["fixed"])
    fix(blk.tp.Cl, init["kl"]["Cl"]["fixed"])
    fix(blk.tp.ae_parX, init["ae"]["X"]["fixed"])
    fix(blk.tp.ae_para, init["ae"]["a"]["fixed"])
    fix(blk.tp.ae_parb, init["ae"]["b"]["fixed"])
    fix(blk.tp.A_H2O, init["kin"]["A_H2O"]["fixed"])
    fix(blk.tp.E_H2O, init["kin"]["E_H2O"]["fixed"])
    fix(blk.tp.A_MEA, init["kin"]["A_MEA"]["fixed"])
    fix(blk.tp.E_MEA, init["kin"]["E_MEA"]["fixed"])

def set_bounds(blk, init=init_set):
    blk.pp.Keq_a[1].setlb(init["Keq"]["a1"]["lb"])
    blk.pp.Keq_a[1].setub(init["Keq"]["a1"]["ub"])
    blk.pp.Keq_a[1].value = init["Keq"]["a1"]["value"]

    blk.pp.Keq_a[2].setlb(init["Keq"]["a2"]["lb"])
    blk.pp.Keq_a[2].setub(init["Keq"]["a2"]["ub"])
    blk.pp.Keq_a[2].value = init["Keq"]["a2"]["value"]

    blk.pp.Keq_b[1].setlb(init["Keq"]["b1"]["lb"])
    blk.pp.Keq_b[1].setub(init["Keq"]["b1"]["ub"])
    blk.pp.Keq_b[1].value = init["Keq"]["b1"]["value"]

    blk.pp.Keq_b[2].setlb(init["Keq"]["b2"]["lb"])
    blk.pp.Keq_b[2].setub(init["Keq"]["b2"]["ub"])
    blk.pp.Keq_b[2].value = init["Keq"]["b2"]["value"]

    blk.pp.Keq_c[1].setlb(init["Keq"]["c1"]["lb"])
    blk.pp.Keq_c[1].setub(init["Keq"]["c1"]["ub"])
    blk.pp.Keq_c[1].value = init["Keq"]["c1"]["value"]

    blk.pp.Keq_c[2].setlb(init["Keq"]["c2"]["lb"])
    blk.pp.Keq_c[2].setub(init["Keq"]["c2"]["ub"])
    blk.pp.Keq_c[2].value = init["Keq"]["c2"]["value"]

    blk.pp.nrtl_parA["CO2","H2O"].setlb(init["nrtl"]["A_co2_h2o"]["lb"])
    blk.pp.nrtl_parA["CO2","H2O"].setub(init["nrtl"]["A_co2_h2o"]["ub"])
    blk.pp.nrtl_parA["CO2","H2O"].value = init["nrtl"]["A_co2_h2o"]["value"]

    blk.tp.Cl.setlb(init["kl"]["Cl"]["lb"])
    blk.tp.Cl.setub(init["kl"]["Cl"]["ub"])
    blk.tp.Cl.value = init["kl"]["Cl"]["value"]

    blk.tp.Cv.setlb(init["kv"]["Cv"]["lb"])
    blk.tp.Cv.setub(init["kv"]["Cv"]["ub"])
    blk.tp.Cv.value = init["kv"]["Cv"]["value"]

    blk.tp.kv_para.setlb(init["kv"]["a"]["lb"])
    blk.tp.kv_para.setub(init["kv"]["a"]["ub"])
    blk.tp.kv_para.value = init["kv"]["a"]["value"]

    blk.tp.kv_parb.setlb(init["kv"]["b"]["lb"])
    blk.tp.kv_parb.setub(init["kv"]["b"]["ub"])
    blk.tp.kv_parb.value = init["kv"]["b"]["value"]

    blk.tp.ae_parX.setlb(init["ae"]["X"]["lb"])
    blk.tp.ae_parX.setub(init["ae"]["X"]["ub"])
    blk.tp.ae_parX.value = init["ae"]["X"]["value"]

    blk.tp.ae_para.setlb(init["ae"]["a"]["lb"])
    blk.tp.ae_para.setub(init["ae"]["a"]["ub"])
    blk.tp.ae_para.value = init["ae"]["a"]["value"]

    blk.tp.ae_parb.setlb(init["ae"]["b"]["lb"])
    blk.tp.ae_parb.setub(init["ae"]["b"]["ub"])
    blk.tp.ae_parb.value = init["ae"]["b"]["value"]

    blk.tp.A_H2O.setlb(init["kin"]["A_H2O"]["lb"])
    blk.tp.A_H2O.setub(init["kin"]["A_H2O"]["ub"])
    blk.tp.A_H2O.value = init["kin"]["A_H2O"]["value"]

    blk.tp.E_H2O.setlb(init["kin"]["E_H2O"]["lb"])
    blk.tp.E_H2O.setub(init["kin"]["E_H2O"]["ub"])
    blk.tp.E_H2O.value = init["kin"]["E_H2O"]["value"]

    blk.tp.A_MEA.setlb(init["kin"]["A_MEA"]["lb"])
    blk.tp.A_MEA.setub(init["kin"]["A_MEA"]["ub"])
    blk.tp.A_MEA.value = init["kin"]["A_MEA"]["value"]

    blk.tp.E_MEA.setlb(init["kin"]["E_MEA"]["lb"])
    blk.tp.E_MEA.setub(init["kin"]["E_MEA"]["ub"])
    blk.tp.E_MEA.value = init["kin"]["E_MEA"]["value"]


"""
Case data dict
==============
Cases "K??" are NCCC and Cases "W???" are WWC bench scale
For NCCC we are looking at loading for WWC we are looking at CO2 flux
between gas and liquid phases.

Keys:
    FL = liquid inlet flow (mol/s)
    TL = liquid inlet temperature (K)
    x = liquid inlet composition dict (mol i/mol total)
    FG = gas inlet flow (mol/s)
    TG = gas inlet temperature (K)
    y = gas inlet composition dict (mol i/mol total)
    P = Column pressure (Pa)
    nb = Number of beds. Intercolers may be present between beds.
    ICT = Temperature of feed to bed after i.  If None use outlet T (K)
    ICF = Intercooler flow (Kg/s) don't forget to convert to mol/s
    rl = Rich loading, for NCCC (mole CO2/mol MEA)
    flux = CO2 flux vapor to liquid for WWC (mol/s/m2)
"""
case_data = {
    "W001":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":7.94373684131938E-05, "N2":0.9999163817, "H2O":4.1809141270272E-06, "MEA":0.0},
        "FG":0.0076557029, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.00000323},
    "W002":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0001212465, "N2":0.9998745726, "H2O":4.18091412720224E-06, "MEA":0.0},
        "FG":0.0076557029, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.0000242},
    "W003":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0001630557, "N2":0.9998327634, "H2O":4.18091412671113E-06, "MEA":0.0},
        "FG":0.0076557029, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.0000355},
    "W004":{"P":239182.142857143, "TL":313.15, "TG":313.15,"nb":1,
        "y":{"CO2":0.000484986, "N2":0.9994982903, "H2O":1.67236565082539E-05, "MEA":0.0},
        "FG":0.0076557029, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.0000444},
    "W005":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0002006839, "N2":0.9997909543, "H2O":8.36182825434046E-06, "MEA":0.0},
        "FG":0.0076557029, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.0000496},
    "W006":{"P":411503.571428572, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0014313363, "N2":0.9985419325, "H2O":2.67312382290371E-05, "MEA":0.0},
        "FG":0.013171339, "FL":2.42020,
        "x":{"MEA":0.10674, "CO2":0.04611, "H2O":0.84715}, "flux":0.0000704},
    "W007":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0007274791, "N2":0.9992474355, "H2O":2.50854847619922E-05, "MEA":0.0},
        "FG":0.0076557029, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.000105},
    "W008":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.000974153, "N2":0.9989965806, "H2O":2.92663988892755E-05, "MEA":0.0},
        "FG":0.0076557029, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.000145},
    "W009":{"P":411503.571428572, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0019100685, "N2":0.9980559099, "H2O":3.40215759280071E-05, "MEA":0.0},
        "FG":0.013171339, "FL":2.42020,
        "x":{"MEA":0.10674, "CO2":0.04611, "H2O":0.84715}, "flux":0.000164},
    "W010":{"P":514896.428571429, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0095708568, "N2":0.9902893092, "H2O":0.0001398339},
        "FG":0.0164807206, "FL":2.43673,
        "x":{"MEA":0.10602, "CO2":0.05258, "H2O":0.84140}, "flux":0.000187},
    "W011":{"P":239182.142857143, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.001216646, "N2":0.9987457258, "H2O":3.76282271430137E-05, "MEA":0.0},
        "FG":0.0076557029, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.000194},
    "W012":{"P":411503.571428572, "TL":313.15, "TG":313.15, "nb":1,
        "y":{"CO2":0.0023863705, "N2":0.9975698874, "H2O":0.000043742},
        "FG":0.013171339, "FL":2.42020,
        "x":{"MEA":0.10674, "CO2":0.04611, "H2O":0.84715}, "flux":0.00027},
    "W013":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0004599006, "N2":0.9994982903, "H2O":4.1809141270315E-05, "MEA":0.0},
        "FG":0.0071961079, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.0000121},
    "W014":{"P":411503.571428572, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0086706416, "N2":0.9908870779, "H2O":0.0004422805},
        "FG":0.0123806237, "FL":2.42020,
        "x":{"MEA":0.10674, "CO2":0.04611, "H2O":0.84715}, "flux":0.0000469},
    "W015":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0006898508, "N2":0.9992474355, "H2O":6.2713711905306E-05, "MEA":0.0},
        "FG":0.0071961079, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.0000888},
    "W016":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0009198011, "N2":0.9989965806, "H2O":8.3618282540408E-05, "MEA":0.0},
        "FG":0.0071961079, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.000161},
    "W017":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0034492542, "N2":0.9962371773, "H2O":0.0003135686, "MEA":0.0},
        "FG":0.0071961079, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.000182},
    "W018":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0011497514, "N2":0.9987457258, "H2O":0.0001045229, "MEA":0.0},
        "FG":0.0071961079, "FL":2.45027,
        "x":{"MEA":0.10883, "CO2":0.02743, "H2O":0.86374}, "flux":0.000242},
    "W019":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0045990055, "N2":0.994982903, "H2O":0.0004180914, "MEA":0.0},
        "FG":0.0071961079, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.000484},
    "W020":{"P":239182.142857143, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0057487569, "N2":0.9937286288, "H2O":0.0005226143, "MEA":0.0},
        "FG":0.0071961079, "FL":2.47667,
        "x":{"MEA":0.10767, "CO2":0.03779, "H2O":0.85454}, "flux":0.000787},
    "W021":{"P":411503.571428572, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0115624756, "N2":0.9878494372, "H2O":0.0005880872, "MEA":0.0},
        "FG":0.0123806237, "FL":2.42020,
        "x":{"MEA":0.10674, "CO2":0.04611, "H2O":0.84715}, "flux":0.000798},
    "W022":{"P":411503.571428572, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0144518794, "N2":0.9848117965, "H2O":0.0007363241, "MEA":0.0},
        "FG":0.0123806237, "FL":2.42020,
        "x":{"MEA":0.10674, "CO2":0.04611, "H2O":0.84715}, "flux":0.00152},
    "W023":{"P":514896.428571429, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0448070694, "N2":0.9533886843, "H2O":0.0018042463, "MEA":0.0},
        "FG":0.0154913332, "FL":2.43673,
        "x":{"MEA":0.10602, "CO2":0.05258, "H2O":0.84140}, "flux":0.00187},
    "W024":{"P":514896.428571429, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0597421118, "N2":0.937851579, "H2O":0.0024063092, "MEA":0.0},
        "FG":0.0154913332, "FL":2.43673,
        "x":{"MEA":0.10602, "CO2":0.05258, "H2O":0.84140}, "flux":0.00431},
    "W025":{"P":514896.428571429, "TL":333.15, "TG":333.15, "nb":1,
        "y":{"CO2":0.0746790964, "N2":0.9223144738, "H2O":0.0030064299, "MEA":0.0},
        "FG":0.0154913332, "FL":2.43673,
        "x":{"MEA":0.10602, "CO2":0.05258, "H2O":0.84140}, "flux":0.00675},
    "W026":{"P":273646.428571429, "TL":353.15, "TG":353.15, "nb":1,
        "y":{"CO2":0.0045313948, "N2":0.994518474, "H2O":0.0009501312, "MEA":0.0},
        "FG":0.0077667497, "FL":2.14842,
        "x":{"MEA":0.10861, "CO2":0.02943, "H2O":0.86196}, "flux":0.000317},
    "W027":{"P":308110.714285714, "TL":353.15, "TG":353.15, "nb":1,
        "y":{"CO2":0.0164778431, "N2":0.9805264805, "H2O":0.0029956764, "MEA":0.0},
        "FG":0.0087449297, "FL":2.09306,
        "x":{"MEA":0.10750, "CO2":0.03934, "H2O":0.85316}, "flux":0.000627},
    "W028":{"P":308110.714285714, "TL":353.15, "TG":353.15, "nb":1,
        "y":{"CO2":0.0219726212, "N2":0.9740353073, "H2O":0.0039920715, "MEA":0.0},
        "FG":0.0087449297, "FL":2.09306,
        "x":{"MEA":0.10750, "CO2":0.03934, "H2O":0.85316}, "flux":0.00266},
    "W029":{"P":308110.714285714, "TL":353.15, "TG":353.15, "nb":1,
        "y":{"CO2":0.0274641537, "N2":0.9675441342, "H2O":0.0049917122, "MEA":0.0},
        "FG":0.0087449297, "FL":2.09306,
        "x":{"MEA":0.10750, "CO2":0.03934, "H2O":0.85316}, "flux":0.00439},
    "W030":{"P":377039.285714286, "TL":373.15, "TG":373.15, "nb":1,
        "y":{"CO2":0.0155156246, "N2":0.9787820519, "H2O":0.0057023236, "MEA":0.0},
        "FG":0.0101277248, "FL":2.14842,
        "x":{"MEA":0.10861, "CO2":0.02943, "H2O":0.86196}, "flux":0.00064},
    "W031":{"P":377039.285714286, "TL":373.15, "TG":373.15, "nb":1,
        "y":{"CO2":0.0232734368, "N2":0.9681730778, "H2O":0.0085534853, "MEA":0.0},
        "FG":0.0101277248, "FL":2.14842,
        "x":{"MEA":0.10861, "CO2":0.02943, "H2O":0.86196}, "flux":0.00397},
    "W032":{"P":480432.142857143, "TL":373.15, "TG":373.15, "nb":1,
        "y":{"CO2":0.0492743884, "N2":0.937556218, "H2O":0.0131693936, "MEA":0.0},
        "FG":0.0129049802, "FL":2.17059,
        "x":{"MEA":0.10750, "CO2":0.03934, "H2O":0.85316}, "flux":0.00422},
    "W033":{"P":480432.142857143, "TL":373.15, "TG":373.15, "nb":1,
        "y":{"CO2":0.0615924651, "N2":0.9219452725, "H2O":0.0164622624, "MEA":0.0},
        "FG":0.0129049802, "FL":2.17059,
        "x":{"MEA":0.10750, "CO2":0.03934, "H2O":0.85316}, "flux":0.00784},
    "W034":{"P":377039.285714286, "TL":373.15, "TG":373.15, "nb":1,
        "y":{"CO2":0.0310312491, "N2":0.9575641038, "H2O":0.0114046471, "MEA":0.0},
        "FG":0.0101277248, "FL":2.14842,
        "x":{"MEA":0.10861, "CO2":0.02943, "H2O":0.86196}, "flux":0.00871},
    "W035":{"P":377039.285714286, "TL":373.15, "TG":373.15, "nb":1,
        "y":{"CO2":0.0387890614, "N2":0.9469551297, "H2O":0.0142558089, "MEA":0.0},
        "FG":0.0101277248, "FL":2.14842,
        "x":{"MEA":0.10861, "CO2":0.02943, "H2O":0.86196}, "flux":0.0125},
    "K01":{"FL":81.6443360212, "TL":314.12, "P":108820, "FG":21.5713, "TG":315.63,
        "x":{"CO2":0.0159, "H2O":0.8746, "MEA":0.1095},
        "y":{"CO2":0.1233, "H2O":0.0804, "N2":0.7963, "MEA":0.0},
        "nb":3, "rl":0.384, "cap":99.91, "ICT":{1:313.28, 2:316.47},
        "ICF":{1:2.05,2:1.67}},
    "K02":{"FL":138.271464853, "TL":313.67, "P":107060, "FG":21.6263, "TG":318.09,
        "x":{"CO2":0.0283, "H2O":0.8570, "MEA":0.1147},
        "y":{"CO2":0.1217, "H2O":0.0917, "N2":0.7866, "MEA":0.0},
        "nb":3, "rl":0.385, "cap":99.49, "ICT":{1:313.24, 2:316.47},
        "ICF":{1:3.43,2:3.02}},
    "K03":{"FL":38.4087338035, "TL":318.83, "P":107780, "FG":21.6717, "TG":316.82,
        "x":{"CO2":0.0099, "H2O":0.8813, "MEA":0.1088},
        "y":{"CO2":0.1142, "H2O":0.0870, "N2":0.7988, "MEA":0.0},
        "nb":3, "rl":0.475, "cap":83.3, "ICT":{1:316.52, 2:316.41},
        "ICF":{1:1.07,2:0.36}},
    "K04":{"FL":37.9150926551, "TL":319.87, "P":107650, "FG":21.6150, "TG":317.88,
        "x":{"CO2":0.0096, "H2O":0.8745, "MEA":0.1159},
        "y":{"CO2":0.1221, "H2O":0.0908, "N2":0.7871, "MEA":0.0},
        "nb":3, "rl":0.47, "cap":77.73, "ICT":{1:316.49, 2:316.48},
        "ICF":{1:1.07,2:0.36}},
    "K05":{"FL":81.3433391761, "TL":314.72, "P":106940, "FG":21.7824, "TG":316.93,
        "x":{"CO2":0.0123, "H2O":0.8740, "MEA":0.1137},
        "y":{"CO2":0.1006, "H2O":0.0889, "N2":0.8104, "MEA":0.0},
        "nb":3, "rl":0.295, "cap":99.53, "ICT":{1:314.48, 2:316.51},
        "ICF":{1:2.05,2:1.61}},
    "K06":{"FL":79.3865721793, "TL":314.02, "P":107100, "FG":21.7566, "TG":315.33,
        "x":{"CO2":0.0386, "H2O":0.8503, "MEA":0.1111},
        "y":{"CO2":0.1001, "H2O":0.0821, "N2":0.8178, "MEA":0.0},
        "nb":3, "rl":0.469, "cap":58.87, "ICT":{1:316.54, 2:316.41},
        "ICF":{1:2.10,2:1.60}},
    "K07":{"FL":139.4691795238, "TL":313.77, "P":107300, "FG":21.7955, "TG":317.87,
        "x":{"CO2":0.0408, "H2O":0.8569, "MEA":0.1023},
        "y":{"CO2":0.1003, "H2O":0.0927, "N2":0.8070, "MEA":0.0},
        "nb":3, "rl":0.471, "cap":54.62, "ICT":{1:316.58, 2:316.39},
        "ICF":{1:3.44,2:2.98}},
    "K08":{"FL":141.2727047864, "TL":313.72, "P":107260, "FG":21.7501, "TG":315.62,
        "x":{"CO2":0.0159, "H2O":0.8805, "MEA":0.1035},
        "y":{"CO2":0.1007, "H2O":0.0823, "N2":0.8170, "MEA":0.0},
        "nb":3, "rl":0.275, "cap":98.06, "ICT":{1:313.14, 2:313.52},
        "ICF":{1:3.23,2:2.82}},
    "K09":{"FL":37.2882165821, "TL":315.81, "P":107490, "FG":21.7872, "TG":318.02,
        "x":{"CO2":0.0273, "H2O":0.8584, "MEA":0.1143},
        "y":{"CO2":0.1007, "H2O":0.0936, "N2":0.8057, "MEA":0.0},
        "nb":3, "rl":0.474, "cap":55.12, "ICT":{1:316.46, 2:316.51},
        "ICF":{1:1.07,2:0.25}},
    "K10":{"FL":38.4507579495, "TL":321.74, "P":107710, "FG":21.7464, "TG":315.3,
        "x":{"CO2":0.0068, "H2O":0.8831, "MEA":0.1101},
        "y":{"CO2":0.1007, "H2O":0.0817, "N2":0.8176, "MEA":0.0},
        "nb":3, "rl":0.477, "cap":98.41, "ICT":{1:316.47, 2:316.48},
        "ICF":{1:1.08,2:0.17}},
    "K11":{"FL":81.895012708, "TL":313.98, "P":107200, "FG":21.7237, "TG":317.55,
        "x":{"CO2":0.0172, "H2O":0.8757, "MEA":0.1071},
        "y":{"CO2":0.1083, "H2O":0.0910, "N2":0.8007, "MEA":0.0},
        "nb":3, "rl":0.378, "cap":99.75, "ICT":{1:313.38, 2:316.47},
        "ICF":{1:2.08,2:1.56}},
    "K12":{"FL":81.9029670173, "TL":314.07, "P":106370, "FG":21.8508, "TG":316.39,
        "x":{"CO2":0.0171, "H2O":0.8758, "MEA":0.1071},
        "y":{"CO2":0.0913, "H2O":0.0881, "N2":0.8206, "MEA":0.0},
        "nb":3, "rl":0.341, "cap":99.61, "ICT":{1:313.71, 2:316.47},
        "ICF":{1:2.06,2:1.56}},
    "K13":{"FL":81.1129157106, "TL":315.08, "P":107560, "FG":21.7268, "TG":315.24,
        "x":{"CO2":0.0183, "H2O":0.8701, "MEA":0.1116},
        "y":{"CO2":0.1024, "H2O":0.0807, "N2":0.8169, "MEA":0.0},
        "nb":3, "rl":0.36, "cap":97.96, "ICT":{1:None, 2:None},
        "ICF":{1:None,2:None}},
    "K14":{"FL":104.8247582804, "TL":313.56, "P":107860, "FG":28.4117, "TG":320.82,
        "x":{"CO2":0.0248, "H2O":0.8643, "MEA":0.1109},
        "y":{"CO2":0.0984, "H2O":0.1057, "N2":0.7959, "MEA":0.0},
        "nb":3, "rl":0.42, "cap":98.26, "ICT":{1:316.28, 2:316.49},
        "ICF":{1:2.69,2:2.27}},
    "K15":{"FL":103.834498248, "TL":313.75, "P":108080, "FG":28.2957, "TG":315.98,
        "x":{"CO2":0.0259, "H2O":0.8587, "MEA":0.1154},
        "y":{"CO2":0.0997, "H2O":0.0833, "N2":0.8169, "MEA":0.0},
        "nb":3, "rl":0.413, "cap":99.42, "ICT":{1:315.27, 2:316.49},
        "ICF":{1:2.69,2:2.31}},
    "K16":{"FL":48.2225061483, "TL":317.6, "P":108700, "FG":28.3396, "TG":318.57,
        "x":{"CO2":0.0154, "H2O":0.8601, "MEA":0.1244},
        "y":{"CO2":0.1009, "H2O":0.0945, "N2":0.8046, "MEA":0.0},
        "nb":3, "rl":0.476, "cap":93.46, "ICT":{1:316.49, 2:316.51},
        "ICF":{1:1.37,2:0.68}},
    "K17":{"FL":80.7777952835, "TL":315.18, "P":108120, "FG":21.7265, "TG":314.18,
        "x":{"CO2":0.0190, "H2O":0.8676, "MEA":0.1134},
        "y":{"CO2":0.1010, "H2O":0.0771, "N2":0.8219, "MEA":0.0},
        "nb":2, "rl":0.354, "cap":97.59, "ICT":{1:None}},
    #"K18":{"FL":81.3743685928, "TL":315.39, "P":109180, "FG":22.0889, "TG":319.22,
    #    "x":{"CO2":0.0157, "H2O":0.8729, "MEA":0.1114},
    #    "y":{"CO2":0.1099, "H2O":0.0985, "N2":0.7916, "MEA":0.0},
    #    "nb":1, "rl":0.349, "cap":92.85, "ICT":{},"ICF":{}},
    #"K19":{"FL":143.6114829772, "TL":314.05, "P":108020, "FG":13.9505, "TG":319.33,
    #    "x":{"CO2":0.0184, "H2O":0.8814, "MEA":0.1001},
    #    "y":{"CO2":0.1174, "H2O":0.0998, "N2":0.7828, "MEA":0.0},
    #    "nb":1, "rl":0.276, "cap":98.21, "ICT":{}, "ICF":{}},
    #"K20":{"FL":39.1348005536, "TL":318.45, "P":108120, "FG":12.8353, "TG":319.24,
    #    "x":{"CO2":0.0075, "H2O":0.8921, "MEA":0.1003},
    #    "y":{"CO2":0.1172, "H2O":0.0995, "N2":0.7832, "MEA":0.0},
    #    "nb":1, "rl":0.393, "cap":95.55, "ICT":{}, "ICF":{}},
    #"K21":{"FL":39.4187129867, "TL":318.33, "P":107760, "FG":13.3001, "TG":319.26,
    #    "x":{"CO2":0.0072, "H2O":0.8961, "MEA":0.0968},
    #    "y":{"CO2":0.1097, "H2O":0.1007, "N2":0.7897, "MEA":0.0},
    #    "nb":2, "rl":0.385, "cap":96.32, "ICT":{1:348.7983153}},
    #"K22":{"FL":83.8083127732, "TL":314.39, "P":107510, "FG":13.3683, "TG":319.21,
    #    "x":{"CO2":0.0127, "H2O":0.8897, "MEA":0.0976},
    #    "y":{"CO2":0.1166, "H2O":0.0994, "N2":0.7840, "MEA":0.0},
    #    "nb":2, "rl":0.291, "cap":99.49, "ICT":{1:310.9360983}},
    #"K23":{"FL":82.9873806047, "TL":314.38, "P":107370, "FG":13.8152, "TG":319.26,
    #    "x":{"CO2":0.0141, "H2O":0.8840, "MEA":0.1019},
    #    "y":{"CO2":0.1094, "H2O":0.1008, "N2":0.7898, "MEA":0.0},
    #    "nb":2, "rl":0.283, "cap":99.58, "ICT":{1:310.8061622}}
}

def summy_stats(s="W", v='flux'):
    col_set = [i for i in case_data.keys() if i.startswith(s)]
    n = len(col_set)
    mean = 0
    sst = 0
    if n > 0:
        mean = sum(case_data[i][v] for i in col_set)/n
        sst = sum((case_data[i][v] - mean)**2 for i in col_set)
        stdev = sqrt(sst/(n-1))
    return {"n":n, "sst":sst, "mean":mean, "stdev":stdev}

# I know putting this here like this is a bit odd, but it's a shortcut
wwc_stats = summy_stats("W", 'flux')
nccc_stats = summy_stats("K", 'cap')

"""
DLW, Sept 2018:
Create the lookup dictionary and Num_Scenarios for use by PySP callbacks
"""
keylist = list(case_data.keys()) # we are going to change case data
for key in keylist:
    if key.startswith(case_char): del case_data[key]
col_set = sorted(case_data.keys()) # Keys for column sub models

lookup = {}
for i,case in enumerate(col_set):
    lookup[i+1] = case


def col_init2(col, x, y, FG, FL, TL, TG, P):
    """
    This is a generic initalization routine for a single absorber bed, working
    for packed bed and WWC cases.
    """
    opt = SolverFactory('ipopt')
    strip_bounds = TransformationFactory('contrib.strip_var_bounds')
    # Set up the column conditions for initialization, a lot of fixed stuff
    # will get unfixed by initialization
    abs_xin = x
    abs_yin = y
    # Normalize the compositions (just in case they dont add up to 1)
    abs_xin_tot = sum(abs_xin[i] for i in abs_xin)
    abs_yin_tot = sum(abs_yin[i] for i in abs_yin)
    for i in abs_xin:
        abs_xin[i] = abs_xin[i] / abs_xin_tot
    for i in abs_yin:
        abs_yin[i] = abs_yin[i] / abs_yin_tot
    abs_P = P
    abs_Tv_in = TG
    abs_Tl_in = TL
    if "Absorber" == col.config.col_type:
        abs_uv = col.uv[0,1].value*FG/10.70 # rough estimate of velocity from flow
        abs_ul = col.ul[0,0].value*FL/28.21 # rough estimate of velocity from flow
        col.L.value = 60.7184/3.0*0.3048 # each bed for NCCC is 1/3 of about 60.7 ft
        col.D.value = 0.64135 # NCCC column diameter
    elif "WWC" == col.config.col_type:
        abs_uv=0.218150087
        abs_ul=0.13525
        col.r_wwc.fix(0.63e-2)
        col.R_wwc.fix(1.27e-2)
        col.L.value = 9.1e-2
        col.D.value = 2.205e-2
    for z in col.z:
        col.ul[0,z].fix(abs_ul)
        col.uv[0,z].fix(abs_uv)
        col.P[0,z].value = abs_P
        col.T_l[0,z].value = abs_Tl_in
        col.T_v[0,z].value = abs_Tv_in
        model_util.dict_set(col.y_l, abs_xin, pre_idx=(0,z))
        model_util.dict_set(col.y_v, abs_yin, pre_idx=(0,z))
    col.initialize('default', m_steps=[0.5, 1.0])
    # Switch from spcifiying velocities to flow rates for absorber
    col.ul[0,0].unfix()
    col.uv[0,1].unfix()
    print("Adjusting flue gas flow from {} mol/s to {} mol/s".format(
        col.F_vap_in[0].value, FG))
    print("Adjusting solvent flow from {} mol/s to {} mol/s".format(
        col.F_liq_in[0].value, FL))
    col.F_liq_in[0].fix(FL)
    col.F_vap_in[0].fix(FG)
    # Solve the model then bed is all ready to go
    strip_bounds.apply_to(col, reversible=True) # when not optimizing with ipopt works better w/o bounds
    results = opt.solve(col, tee=False)
    print(results.solver.message)
    strip_bounds.revert(col)

def col_init(col, case):
    """
    Initialize the whole absorber bed train.  This is excessive for most cases,
    but I wanted a general method that would work for everything.
    1) initiaize all beds
    2) make several passes up and down the train of beds solving them
       independently and moving results to inlet of next, to close recycle a bit
    3) connect beds
    4) solve all together
    """
    opt = SolverFactory('ipopt')
    strip_bounds = TransformationFactory('contrib.strip_var_bounds')
    # Initialize absorber.
    print("\n\n----------------------------")
    print("\nInitialize Case: {} Bed: {}".format(case, 1))

    #Initialize first, since don't have rest of beds, vapor inlet is not right
    col_init2(
        col=col.bed[1].col,
        x=case_data[case]["x"],
        y=case_data[case]["y"],
        FG=case_data[case]["FG"],
        FL=case_data[case]["FL"],
        TG = case_data[case]["TG"],
        TL = case_data[case]["TL"],
        P = case_data[case]["P"])

    #Initialize rest, using liquid from first, but still wrong vapor inlet
    for i in col.bed:
        if i == 1: continue
        print("\n\n----------------------------")
        print("\nInitialize Case: {} Bed: {}".format(case, i))
        col_init2(
            col=col.bed[i].col,
            x={"CO2":value(col.bed[i-1].col.y_l[0,1, "CO2"]),
               "H2O":value(col.bed[i-1].col.y_l[0,1, "H2O"]),
               "MEA":value(col.bed[i-1].col.y_l[0,1, "MEA"])},
            y=case_data[case]["y"],
            FG=case_data[case]["FG"],
            FL = value(col.bed[i-1].col.F_liq[0,1]),
            TG = case_data[case]["TG"],
            TL = value(col.bed[i-1].col.T_l[0,1]),
            P = case_data[case]["P"])

    def re_up(stee=False):
        """
        Function to go up the series of beds solving and moving the resulting
        vapor out up the train. (this is used later)
        """
        for i in reversed(list(col.con_set)):
            col.bed[i].col.y_v[0,1,"CO2"].value \
                = col.bed[i+1].col.y_v[0,0,"CO2"].value
            col.bed[i].col.y_v[0,1,"H2O"].value \
                = col.bed[i+1].col.y_v[0,0,"H2O"].value
            col.bed[i].col.y_v[0,1,"MEA"].value \
                = col.bed[i+1].col.y_v[0,0,"MEA"].value
            col.bed[i].col.y_v[0,1,"N2"].value \
                = col.bed[i+1].col.y_v[0,0,"N2"].value
            col.bed[i].col.T_v[0,1].value \
                = col.bed[i+1].col.T_v[0,0].value
            col.bed[i].col.F_vap_in[0].value \
                = col.bed[i+1].col.F_vap_out[0].value
            strip_bounds.apply_to(col, reversible=True)
            results = opt.solve(col.bed[i].col, tee=stee)
            print("Bed: {}, {}".format(i, results.solver.message))
            strip_bounds.revert(col)

    def re_down(stee=False):
        """
        Function to go down the series of beds solving and moving the resulting
        liquid out down the train. (this is used later)
        """
        for i in col.con_set:
            if case_data[case]["ICT"][i] is None:
                col.bed[i+1].col.T_l[0,0].value = col.bed[i].col.T_l[0,1].value
            else:
                col.bed[i+1].col.T_l[0,0].value = case_data[case]["ICT"][i]
            col.bed[i+1].col.F_liq_in[0].value \
                = col.bed[i].col.F_liq_out[0].value
            col.bed[i+1].col.y_l[0,0,"CO2"].value \
                = col.bed[i].col.y_l[0,1,"CO2"].value
            col.bed[i+1].col.y_l[0,0,"H2O"].value \
                = col.bed[i].col.y_l[0,1,"H2O"].value
            col.bed[i+1].col.y_l[0,0,"MEA"].value \
                = col.bed[i].col.y_l[0,1,"MEA"].value
            strip_bounds.apply_to(col, reversible=True)
            results = opt.solve(col.bed[i+1].col, tee=stee)
            print("Bed: {}, {}".format(i+1, results.solver.message))
            print(col.bed[i+1].col.y_l[0,0,"CO2"].value)
            strip_bounds.revert(col)

    if case_data[case]["nb"] > 1:
        print("\n\n---------------------------------")
        for i in range(12):
            # make a bunch of passes up and down the absorber train to get closer
            # to the right valor and liquid inlet in each bed.
            print("Up and Down Passes, Case: {} Pass: {}".format(case, i))
            re_up(stee=False)
            re_down(stee=False)

    for i in col.bed_set:
        # unfix the liquid and vapor variables that are connected between beds
        # these are set by equality constraints at the level of the whole
        # absorber train
        if i > 1:
            col.bed[i].col.F_liq_in[0].unfix()
            col.bed[i].col.T_l[0,0].unfix()
            col.bed[i].col.y_l[0,0,"CO2"].unfix()
            col.bed[i].col.y_l[0,0,"H2O"].unfix()
            col.bed[i].col.y_l[0,0,"MEA"].unfix()
        if i < col.bed_set[-1]:
            col.bed[i].col.F_vap_in[0].unfix()
            col.bed[i].col.T_v[0,1].unfix()
            col.bed[i].col.y_v[0,1,"CO2"].unfix()
            col.bed[i].col.y_v[0,1,"H2O"].unfix()
            col.bed[i].col.y_v[0,1,"MEA"].unfix()
            col.bed[i].col.y_v[0,1,"N2"].unfix()

    # Now solve whole train together
    print("\n\n---------------------------------")
    print("Initial Case: {}".format(case))
    strip_bounds.apply_to(col, reversible=True)
    results = opt.solve(col, tee=True)
    strip_bounds.revert(col)

    #print some stuff to make sure results look okay
    if case.startswith("K"):
        print("Rich Loading: {}".format(col.rich_loading.value))
        print("CO2 Capture Fraction: {}".format(col.cap.value))
    elif case.startswith("W"):
        col.bed[1].col.make_profile(t=0)
        print(col.bed[1].col.profile_1)

def generate_case(case):
    """
    Rule to create an absorber column case.  If case starts with K make NCCC case
    if case starts with W make a WWC case.
    """
    #the reason for the concrete=True flag here is because the submodel is
    #being built and before being attached to another model, so at least at
    #the time its constructed blk is the top level.
    blk = FlowsheetBlock(name=case, concrete=True)
    last = case_data[case]["nb"] # number of beds (last bed index)
    blk.bed_set = RangeSet(1, last)
    blk.con_set = RangeSet(1, last-1)
    pp = blk.pp = PropertyParams()
    if case.startswith("W"):  # Make a WWC model
        typ = 'WWC'
        packing = 'WWC001'
        isothermal = True
        tp = blk.tp = TransportParams(packing=packing)
        wwc_mode = 'absorption'
        nfe = 11
    elif case.startswith("K"): # Make a packed column model
        typ = 'Absorber'
        packing = 'mellapak_250Y'
        isothermal = False
        tp = blk.tp = TransportParams(packing=packing)
        wwc_mode = None
        nfe = 11
    else:
        raise Exception("Bad case name {}".format(case))

    def rule_col2(blk2, i2):
        """
        Rule to make 1 absorber bed model. (some consist of up to 3 beds)
        """
        blk2.build()
        blk2.col = Column(name="Absorber", dae_znfe=nfe, col_type=typ, tp=tp, pp=pp,
                     dynamic=False, packing=packing, wwc_mode=wwc_mode,
                     isothermal=isothermal, doc="CO2 absorber",
                     dae_ztransform="dae.finite_difference",
                     dae_zscheme="BACKWARD")

    # Create the beds (in W cases and some K cases there is only 1)
    blk.bed = UnitBlock(blk.bed_set, rule=rule_col2, name="Beds")

    # The next bunch of constraints connect the beds in a column
    def rule_yln(blk, i, j):
        return blk.bed[i].col.y_l[0,1, j] == blk.bed[i+1].col.y_l[0,0,j]
    blk.yln = Constraint(blk.con_set, ["CO2", "H2O", "MEA"], rule=rule_yln)

    def rule_ygn(blk, i, j):
        return blk.bed[i].col.y_v[0,1, j] == blk.bed[i+1].col.y_v[0,0,j]
    blk.ygn = Constraint(blk.con_set, ["CO2", "H2O", "MEA", "N2"], rule=rule_ygn)

    def rule_Tln(blk, i):
        if case_data[case]["ICT"][i] is None:
            return blk.bed[i].col.T_l[0,1] == blk.bed[i+1].col.T_l[0,0]
        else:
            return blk.bed[i+1].col.T_l[0,0] == \
                (case_data[case]["ICF"][i]/blk.bed[i].col.prop_liq[0,1].mw_ave/blk.bed[i].col.F_liq[0,1]) \
                *case_data[case]["ICT"][i] + \
                (1 - case_data[case]["ICF"][i]/blk.bed[i].col.prop_liq[0,1].mw_ave/blk.bed[i].col.F_liq[0,1]) \
                *blk.bed[i].col.T_l[0,1]
    blk.Tln = Constraint(blk.con_set, rule=rule_Tln)

    def rule_Tgn(blk, i):
        return blk.bed[i].col.T_v[0,1] == blk.bed[i+1].col.T_v[0,0]
    blk.Tgn = Constraint(blk.con_set, rule=rule_Tgn)

    def rule_fln(blk, i):
        return blk.bed[i].col.F_liq[0,1] == blk.bed[i+1].col.F_liq[0,0]
    blk.fln = Constraint(blk.con_set, rule=rule_fln)

    def rule_fgn(blk, i):
        return blk.bed[i].col.F_vap[0,1] == blk.bed[i+1].col.F_vap[0,0]
    blk.fgn = Constraint(blk.con_set, rule=rule_fgn)

    # the beds and connected now add constraints to calculate eithere rich
    # loading for NCCC of average CO2 flux for WWC cases.
    if case.startswith("K"):
        blk.rich_loading = Var(initialize=0.3)
        blk.cap = Var(initialize=95.0)
        blk.eq_rich_loading = Constraint(expr=blk.rich_loading ==
            blk.bed[last].col.y_l[0,1,"CO2"]/blk.bed[last].col.y_l[0,1,"MEA"])
        blk.eq_cap = Constraint(expr=blk.cap/100.0 == 1 -
            blk.bed[1].col.y_v[0,0,"CO2"]*blk.bed[1].col.F_vap[0,0]/
            blk.bed[last].col.y_v[0,1,"CO2"]/blk.bed[last].col.F_vap[0,1])
        blk.FirstStageCost = Expression(expr=0)
        blk.SE = Expression(expr=(blk.cap - case_data[case]["cap"])**2)
        blk.SecondStageCost = Expression(expr=blk.SE/nccc_stats["stdev"]**2)
    elif case.startswith("W"):
        blk.flux = Var(initialize=0.003, bounds=(-1, 1.0))
        blk.eq_flux = Constraint(expr=blk.flux == \
            sum(blk.bed[1].col.N_v[0,z,"CO2"]/blk.bed[1].col.trans[0,z].ae\
            for z in blk.bed[1].col.z if z != 0)/(len(blk.bed[1].col.z) - 1))
        blk.FirstStageCost = Expression(expr=0)
        blk.SE = Expression(expr=(blk.flux - case_data[case]["flux"])**2)
        blk.SecondStageCost = Expression(expr=blk.SE/wwc_stats["stdev"]**2)

    blk.objective = Objective(expr=blk.FirstStageCost + blk.SecondStageCost)
    #initialize
    # Check for init, that's already done
    fname = os.path.join("pysp_init", case+".json")
    if os.path.isfile(fname):
        from_json(blk, fname=fname, wts=StoreSpec(ignore_missing=True))
    else:
        if not os.path.exists("pysp_init"): # DLW, Sept 2018
            os.mkdir("pysp_init")
        col_init(blk, case)
        to_json(blk, fname=fname)

    # Solve at inital parameter values.
    opt = SolverFactory('ipopt')
    set_bounds(blk)
    #blk.strip_bounds()
    #results = opt.solve(blk, tee=True)
    #blk.restore_bounds()
    unfix(blk)
    #to_json(blk, fname=fname)

    return blk

def pysp_scenario_tree_model_callback():
    # uses data in Num_Scenarios defined in file namespace
    st_model = CreateConcreteTwoStageScenarioTreeModel(Num_Scenarios)

    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()

    st_model.StageCost[first_stage] = 'FirstStageCost'
    st_model.StageVariables[first_stage].add('pp.Keq_a[1]')
    st_model.StageVariables[first_stage].add('pp.Keq_a[2]')
    st_model.StageCost[second_stage] = 'SecondStageCost'
    return st_model

def pysp_instance_creation_callback(scen_tree_model,scenario_name, node_names):
    # uses lookup, which is defined in the file namespace
    i = int(re.compile(r'(\d+)$').search(scenario_name).group(1))
    print("{}, {}".format(i, lookup[i]))
    instance = generate_case(lookup[i])
    return instance

def print_res(model, free_case):
    params = (
    ("Reaction Equilibrium parameters:", None),
    ("a_1", model.cases[free_case].col.pp.Keq_a[1]),
    ("b_1", model.cases[free_case].col.pp.Keq_b[1]),
    ("c_1", model.cases[free_case].col.pp.Keq_c[1]),
    ("a_2", model.cases[free_case].col.pp.Keq_a[2]),
    ("b_2", model.cases[free_case].col.pp.Keq_b[2]),
    ("c_2", model.cases[free_case].col.pp.Keq_c[2]),
    ("Raction Kinetic Parameters", None),
    ("A_H2O", model.cases[free_case].col.tp.A_H2O),
    ("E_H2O", model.cases[free_case].col.tp.E_H2O),
    ("A_MEA", model.cases[free_case].col.tp.A_MEA),
    ("E_MEA", model.cases[free_case].col.tp.E_MEA),
    ("NRTL parameters:", None),
    ("A_co2_h2o", model.cases[free_case].col.pp.nrtl_parA["CO2","H2O"]),
    ("Mass transfer parameters:", None),
    ("Cv", model.cases[free_case].col.tp.Cv),
    ("av", model.cases[free_case].col.tp.kv_para),
    ("bv", model.cases[free_case].col.tp.kv_parb),
    ("Cl", model.cases[free_case].col.tp.Cl),
    ("Effective interfacial area parameters:", None),
    ("X", model.cases[free_case].col.tp.ae_parX),
    ("a", model.cases[free_case].col.tp.ae_para),
    ("b", model.cases[free_case].col.tp.ae_parb)
    )
    print("\n")
    print("             Value      Fixed    LB         UB       ")
    print("-----------------------------------------------------")
    for i in params:
        if i[1] is None:
            print(i[0])
        else:
            print("    {:8s} {:10.3e} {:8s} {:10.3e} {:10.3e}".format(
                i[0],
                i[1].value,
                str(i[1].fixed),
                i[1].lb,
                i[1].ub))
    print("\n")
    print("WWC R**2: {}".format(1 - value(model.sse_wwc)/wwc_stats["sst"]))
    print("NCCC R**2: {}".format(1 - value(model.sse_nccc)/nccc_stats["sst"]))
    print("WWC SSE: {}".format(value(model.sse_wwc)))
    print("NCCC SSE: {}".format(value(model.sse_nccc)))
    print(wwc_stats["stdev"])
    print(nccc_stats["stdev"])

def csv_T_profile(model, cols_set):
    for case in sorted(cols_set):
        p = None
        fname = "profile_{}.csv".format(case)
        for bed in sorted(model.cases[case].col.bed.keys()):
            if p is None:
                p = model.cases[case].col.bed[bed].col.make_profile()
            else:
                p = p.append(model.cases[case].col.bed[bed].col.make_profile(), ignore_index=True)
        p.to_csv(fname)

def print_table(model, col_set):
    """
    Summarize the NCCC and WWC results compared to data
    """
    print("\nNCCC Cases")
    print("       Model          Data           Model              Data")
    print("Case   Rich Loading   Rich Loading   Capture Fraction   Capture Fraction")
    print("------------------------------------------------------------------------")
    for i in sorted(col_set):
        if not i.startswith("K"): continue
        print("{:4s} {:14.4f} {:14.4f} {:18.6f} {:18.6f}".format(
            i,
            value(model.cases[i].col.rich_loading),
            case_data[i]["rl"],
            value(model.cases[i].col.cap),
            case_data[i]["cap"]))
    print("\nWWC Cases")
    print("       Model          Data")
    print("Case   Flux           Flux")
    print("-----------------------------------------------------")
    for i in sorted(col_set):
        if not i.startswith("W"): continue
        print("{:4s} {:14.4e} {:14.4e}".format(
            i,
            value(model.cases[i].col.flux),
            case_data[i]["flux"]))

def bound_check(model, tol=1e-4):
    """
    Check that all variables are within bounds.  Here we are solving the system
    of equations (plain old simulation) with no bounds.  But then using bounds
    when we do optimization, so is important to check that variables are still
    in bounds. This function prints variables that are out of bounds.
    """
    print("\n\nHere are the variables outside bounds:")
    i = 0
    for o in model.component_objects(Var, descend_into=True):
        for key, item in o.iteritems():
            if item.lb is not None and item.ub is not None and \
               (item.value < item.lb or item.value > item.ub) and \
               abs(item.value - item.lb) > 1e-20:
                print("\t{} = {}  [{}, {}]".format(
                    item, value(item), item.lb, item.ub))
                i += 1
    if not i:
        print("    None")
    if tol > 0:
        print("\n\nHere are the variables near nonzero bounds:")
        i = 0
        for o in model.component_objects(Var, descend_into=True):
            for key, item in o.iteritems():
                if (item.lb is not None and item.lb != 0.0 and\
                    abs((item.value - item.lb)/item.lb) < tol) or\
                    (item.ub is not None and item.ub != 1.0 and\
                    abs((item.value - item.ub)/item.ub) < tol):
                    print("\t{} = {}  [{}, {}]".format(
                        item, value(item), item.lb, item.ub))
                    i += 1
        if not i:
            print("    None")

def run_wwc_validation(model, csv_file):
    """
    Use the existing WWC001 model to run aditional validation cases to with data
    not used in the fitting.  Save the profile files and report the flux.
    """
    wwc = model.cases["W001"]
    wwc.col.pp.Keq_a.fix()
    wwc.col.pp.Keq_b.fix()
    wwc.col.tp.Cv.fix()
    wwc.col.tp.kv_para.fix()
    wwc.col.tp.kv_parb.fix()
    wwc.col.tp.Cl.fix()
    wwc.col.tp.ae_parX.fix()
    wwc.col.tp.ae_para.fix()
    wwc.col.tp.ae_parb.fix()
    c1f = os.path.join("pysp_init", "W001.json")
    with open(csv_file, 'rb') as f:
        csvr = csv.reader(f)
        csvr.next()
        K = {}
        print("\n\n---------\n WWC Kv Validation")
        print("case, Kv_co2, flux")
        for row in csvr:
            from_json(wwc.col, fname=c1f, wts=StoreSpec(ignore_missing=True))
            #if int(row[0]) in [37,49,75,97]: continue
            case = "W{:03d}".format(int(row[0]))
            T = float(row[6])
            P = float(row[8])
            FL = float(row[1])
            FG = float(row[13])
            y = {"CO2":float(row[10]), "N2":float(row[11]), "H2O":float(row[12])}
            x = {"CO2":float(row[3]), "H2O":float(row[4]), "MEA":float(row[2])}
            #print(case)
            #print("T = {}, P = {}, FL = {}, FG = {}".format(T, P, FL, FG))
            #print(y)
            #print(x)
            wwc.col.bed[1].col.P[0,0].value = P
            wwc.col.bed[1].col.T_l[0,0].value = T
            wwc.col.bed[1].col.T_v[0,1].value = T
            model_util.dict_set(wwc.col.bed[1].col.y_l, x, pre_idx=(0,0))
            model_util.dict_set(wwc.col.bed[1].col.y_v, y, pre_idx=(0,1))
            wwc.col.bed[1].col.F_liq_in[0].value = FL
            wwc.col.bed[1].col.F_vap_in[0].value = FG
            opt = SolverFactory('ipopt')
            strip_bounds = TransformationFactory('contrib.strip_var_bounds')
            strip_bounds.apply_to(wwc.col, reversible=True)
            results = opt.solve(wwc, tee=False)
            strip_bounds.revert(wwc.col)
            fname = "profile_{}.csv".format(case)
            p = wwc.col.bed[1].col.make_profile()
            p.to_csv(fname)
            K[case] = p["Kv_co2"].mean()
            print("{}, {}, {}".format(case, K[case], wwc.col.flux.value))

def plot_parity(model, col_set, orig=None):
    wwc_data_x = [case_data[i]["flux"] for i in col_set if i.startswith("W")]
    wwc_data_y = [model.cases[i].col.flux.value for i in col_set if i.startswith("W")]
    nccc_data_x = [case_data[i]["cap"] for i in col_set if i.startswith("K")]
    nccc_data_y = [model.cases[i].col.cap.value for i in col_set if i.startswith("K")]

    plt.figure(num=1,figsize=(12,6))
    plt.subplot(121)
    if orig is not None:
        plt.scatter(nccc_data_x,orig["NCCC"],marker="o",color='black')
    plt.scatter(nccc_data_x,nccc_data_y,marker="x",color='black')
    plt.plot([50, 100], [50,100],color='black')
    plt.axis([50,100,50,100])
    plt.xlabel('Measured CO$_2$ Capture (%)',fontsize=15)
    plt.ylabel('Predicted CO$_2$ Capture (%)',fontsize=15)
    plt.tight_layout()
    plt.tick_params(labelsize=14)
    plt.title('Packed',fontsize=14)
    plt.subplot(122)
    if orig is not None:
        plt.scatter(wwc_data_x,orig["WWC"],marker="o",color='black')
    plt.scatter(wwc_data_x,wwc_data_y,marker="x",color='black')
    plt.plot([0,0.015], [0,0.015],color='black')
    plt.xlabel('Measured CO$_2$ Flux (mol/s/m$^{2}$)',fontsize=15)
    plt.ylabel('Predicted CO$_2$ Flux, (mol/s/m$^{2}$)',fontsize=15)
    plt.axis([0,0.015,0,0.015])
    plt.tight_layout()
    plt.tick_params(labelsize=14)
    plt.title('WWC',fontsize=16)
    plt.tight_layout()
    plt.tick_params(labelsize=16)
    plt.show()


def main():
    """
    """
    opt_opt = {  # ipopt options
        "linear_solver":"ma57",
        "halt_on_ampl_error":"no",
        "linear_scaling_on_demand":"yes",
        "bound_push":1e-14,
        "warm_start_init_point":"no",
        "warm_start_bound_push":1e-14,
        "compute_red_hessian":"yes",
        "output_file":"covariance_matrix.txt"}
    
    p_opt = { # pyomo solve options
        "tee":True,
        "symbolic_solver_labels":False,
        "keepfiles":False}

    model = FlowsheetBlock(concrete=True)
    #for key in case_data.keys():
    #    if key.startswith("K"): del case_data[key]
    col_set = sorted(case_data.keys()) # Keys for column sub models
    free_case = col_set[0]
    def rule_case(blk, i):
        blk.col = generate_case(i)
        blk.col.objective.deactivate()
    model.cases = UnitBlock(col_set, rule=rule_case)
    def link_params(blk, case, param):
        if case == free_case:
            return Constraint.Skip
        elif param == "Keq_a1":
            return model.cases[case].col.pp.Keq_a[1] == model.cases[free_case].col.pp.Keq_a[1]
        elif param == "Keq_a2":
            return model.cases[case].col.pp.Keq_a[2] == model.cases[free_case].col.pp.Keq_a[2]
        elif param == "Keq_b1":
            return model.cases[case].col.pp.Keq_b[1] == model.cases[free_case].col.pp.Keq_b[1]
        elif param == "Keq_b2":
            return model.cases[case].col.pp.Keq_b[2] == model.cases[free_case].col.pp.Keq_b[2]
        elif param == "Keq_c1":
            return model.cases[case].col.pp.Keq_c[1] == model.cases[free_case].col.pp.Keq_c[1]
        elif param == "Keq_c2":
            return model.cases[case].col.pp.Keq_c[2] == model.cases[free_case].col.pp.Keq_c[2]
        elif param == "nrtl_A_co2_h2o":
            return model.cases[case].col.pp.nrtl_parA["CO2","H2O"] == model.cases[free_case].col.pp.nrtl_parA["CO2","H2O"]
        elif param == "Cv":
            return model.cases[case].col.tp.Cv == model.cases[free_case].col.tp.Cv
        elif param == "kv_para":
            return model.cases[case].col.tp.kv_para == model.cases[free_case].col.tp.kv_para
        elif param == "kv_parb":
            return model.cases[case].col.tp.kv_parb == model.cases[free_case].col.tp.kv_parb
        elif param == "Cl":
            return model.cases[case].col.tp.Cl == model.cases[free_case].col.tp.Cl
        elif param == "ae_X":
            return model.cases[case].col.tp.ae_parX == model.cases[free_case].col.tp.ae_parX
        elif param == "ae_a":
            return model.cases[case].col.tp.ae_para == model.cases[free_case].col.tp.ae_para
        elif param == "ae_b":
            return model.cases[case].col.tp.ae_parb == model.cases[free_case].col.tp.ae_parb
        elif param == "A_H2O":
            return model.cases[case].col.tp.A_H2O == model.cases[free_case].col.tp.A_H2O
        elif param == "E_H2O":
            return model.cases[case].col.tp.E_H2O == model.cases[free_case].col.tp.E_H2O
        elif param == "A_MEA":
            return model.cases[case].col.tp.A_MEA == model.cases[free_case].col.tp.A_MEA
        elif param == "E_MEA":
            return model.cases[case].col.tp.E_MEA == model.cases[free_case].col.tp.E_MEA
    model.plinks = Constraint(
        col_set,
        [#"Keq_a1",
         #"Keq_b1",
         #"Keq_a2",
         #"Keq_b2",
         #"Keq_c1",
         #"Keq_c2",
         #"nrtl_A_co2_h2o",
         "A_H2O", "E_H2O", "A_MEA", "E_MEA",
         "Cv",
         "Cl",
         "kv_para",
         "kv_parb",
         "ae_X",
         "ae_a",
         "ae_b"
        ],
        rule=link_params)
    model.objective = Objective(expr=sum(model.cases[case].col.SecondStageCost for case in col_set))
    model.sse_wwc = Expression(expr=sum(model.cases[i].col.SE for i in col_set if i.startswith("W")))
    model.sse_nccc = Expression(expr=sum(model.cases[i].col.SE for i in col_set if i.startswith("K")))
    
    #: sIpopt suffix FOR REDUCED HESSIAN
    model.red_hessian = Suffix(direction=Suffix.EXPORT)

    #:suffix ordering
    model.cases[free_case].col.tp.A_H2O.set_suffix_value(model.red_hessian, 1)
    model.cases[free_case].col.tp.E_H2O.set_suffix_value(model.red_hessian, 2)
    model.cases[free_case].col.tp.A_MEA.set_suffix_value(model.red_hessian, 3)
    model.cases[free_case].col.tp.E_MEA.set_suffix_value(model.red_hessian, 4)
    
    model.cases[free_case].col.tp.Cv.set_suffix_value(model.red_hessian, 5)
    model.cases[free_case].col.tp.Cl.set_suffix_value(model.red_hessian, 6)
    model.cases[free_case].col.tp.kv_para.set_suffix_value(model.red_hessian, 7)
    model.cases[free_case].col.tp.kv_parb.set_suffix_value(model.red_hessian, 8)
    
    model.cases[free_case].col.tp.ae_parX.set_suffix_value(model.red_hessian, 9)
    model.cases[free_case].col.tp.ae_para.set_suffix_value(model.red_hessian, 10)
    model.cases[free_case].col.tp.ae_parb.set_suffix_value(model.red_hessian, 11)

    ###exe = r"C:\cygwin64\home\PAKULA\CoinIpopt\build\bin\ipopt_sens"
    ###opt = SolverFactory('ipopt',executable =exe)
    opt = SolverFactory('ipopt_sens')
    print_table(model, col_set)
    print_res(model, free_case)
    #plot_parity(model, col_set)
    orig = {}
    orig["NCCC"] = [model.cases[i].col.cap.value for i in col_set if i.startswith("K")]
    orig["WWC"] = [model.cases[i].col.flux.value for i in col_set if i.startswith("W")]
    bound_check(model, tol=0)
    #csv_T_profile(model, col_set)
    results = opt.solve(model, options=opt_opt, **p_opt)
    print_table(model, col_set)
    print_res(model, free_case)
    bound_check(model, tol=0)
    csv_T_profile(model, col_set)
    plot_parity(model, col_set)
    #plot_parity(model, col_set, orig)

if __name__ == "__main__":
    main()
