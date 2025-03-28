#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

# --------------------------------------------------
# @load
model = pyo.AbstractModel()
model.A = pyo.Set()
data = pyo.DataPortal()
data.load(filename='A.tab', set=model.A)
instance = model.create_instance(data)
# @load
instance.pprint()
# --------------------------------------------------
# @set1
model = pyo.AbstractModel()
model.A = pyo.Set()
data = pyo.DataPortal()
data.load(filename='A.tab', set=model.A)
instance = model.create_instance(data)
# @set1
instance.pprint()
# --------------------------------------------------
# @set2
model = pyo.AbstractModel()
model.C = pyo.Set(dimen=2)
data = pyo.DataPortal()
data.load(filename='C.tab', set=model.C)
instance = model.create_instance(data)
# @set2
instance.pprint()
# --------------------------------------------------
# @set3
model = pyo.AbstractModel()
model.D = pyo.Set(dimen=2)
data = pyo.DataPortal()
data.load(filename='D.tab', set=model.D, format='set_array')
instance = model.create_instance(data)
# @set3
instance.pprint()

# --------------------------------------------------
# @param1
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.z = pyo.Param()
data.load(filename='Z.tab', param=model.z)
instance = model.create_instance(data)
# @param1
instance.pprint()
# --------------------------------------------------
# @param2
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(initialize=['A1', 'A2', 'A3'])
model.y = pyo.Param(model.A)
data.load(filename='Y.tab', param=model.y)
instance = model.create_instance(data)
# @param2
instance.pprint()
# --------------------------------------------------
# @param4
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(initialize=['A1', 'A2', 'A3'])
model.x = pyo.Param(model.A)
model.w = pyo.Param(model.A)
data.load(filename='XW.tab', param=(model.x, model.w))
instance = model.create_instance(data)
# @param4
instance.pprint()
# --------------------------------------------------
# @param3
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.y = pyo.Param(model.A)
data.load(filename='Y.tab', param=model.y, index=model.A)
instance = model.create_instance(data)
# @param3
instance.pprint()
# --------------------------------------------------
# @param5
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.w = pyo.Param(model.A)
data.load(filename='XW.tab', select=('A', 'W'), param=model.w, index=model.A)
instance = model.create_instance(data)
# @param5
instance.pprint()
# --------------------------------------------------
# @param6
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(initialize=['A1', 'A2', 'A3'])
model.I = pyo.Set(initialize=['I1', 'I2', 'I3', 'I4'])
model.u = pyo.Param(model.I, model.A)
data.load(filename='U.tab', param=model.u, format='array')
instance = model.create_instance(data)
# @param6
instance.pprint()
# --------------------------------------------------
# @param7
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(initialize=['A1', 'A2', 'A3'])
model.I = pyo.Set(initialize=['I1', 'I2', 'I3', 'I4'])
model.t = pyo.Param(model.A, model.I)
data.load(filename='U.tab', param=model.t, format='transposed_array')
instance = model.create_instance(data)
# @param7
instance.pprint()
# --------------------------------------------------
# @param8
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.s = pyo.Param(model.A)
data.load(filename='S.tab', param=model.s, index=model.A)
instance = model.create_instance(data)
# @param8
instance.pprint()
# --------------------------------------------------
# @param9
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(initialize=['A1', 'A2', 'A3', 'A4'])
model.y = pyo.Param(model.A)
data.load(filename='Y.tab', param=model.y)
instance = model.create_instance(data)
# @param9
instance.pprint()
# --------------------------------------------------
# @param10
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(dimen=2)
model.p = pyo.Param(model.A)
data.load(filename='PP.tab', param=model.p, index=model.A)
instance = model.create_instance(data)
# @param10
instance.pprint()
# --------------------------------------------------
# @param11
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.B = pyo.Set()
model.q = pyo.Param(model.A, model.B)
data.load(filename='PP.tab', param=model.q, index=(model.A, model.B))
# instance = model.create_instance(data)
# @param11
# --------------------------------------------------
# @concrete1
data = pyo.DataPortal()
data.load(filename='A.tab', set="A", format="set")

model = pyo.ConcreteModel()
model.A = pyo.Set(initialize=data['A'])
# @concrete1
model.pprint()
# --------------------------------------------------
# @concrete2
data = pyo.DataPortal()
data.load(filename='Z.tab', param="z", format="param")
data.load(filename='Y.tab', param="y", format="table")

model = pyo.ConcreteModel()
model.z = pyo.Param(initialize=data['z'])
model.y = pyo.Param(['A1', 'A2', 'A3'], initialize=data['y'])
# @concrete2
model.pprint()
# --------------------------------------------------
# @getitem
data = pyo.DataPortal()
data.load(filename='A.tab', set="A", format="set")
print(data['A'])  # ['A1', 'A2', 'A3']

data.load(filename='Z.tab', param="z", format="param")
print(data['z'])  # 1.1

data.load(filename='Y.tab', param="y", format="table")
for key in sorted(data['y']):
    print("%s %s" % (key, data['y'][key]))
# @getitem
# --------------------------------------------------
# @excel1
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(dimen=2)
model.p = pyo.Param(model.A)
data.load(filename='excel.xls', range='PPtable', param=model.p, index=model.A)
instance = model.create_instance(data)
# @excel1
instance.pprint()
# --------------------------------------------------
# @excel2
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(dimen=2)
model.p = pyo.Param(model.A)
# data.load(filename='excel.xls', range='AX2:AZ5',
#                    param=model.p, index=model.A)
instance = model.create_instance(data)
# @excel2
instance.pprint()
# --------------------------------------------------
# @db1
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set(dimen=2)
model.p = pyo.Param(model.A)
data.load(
    filename='PP.sqlite', using='sqlite3', table='PPtable', param=model.p, index=model.A
)
instance = model.create_instance(data)
# @db1
data = pyo.DataPortal()
data.load(
    filename='PP.sqlite',
    using='sqlite3',
    table='PPtable',
    param=model.p,
    index=model.A,
    text_factory=str,
)
instance = model.create_instance(data)
instance.pprint()
# --------------------------------------------------
# @db2
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.p = pyo.Param(model.A)
data.load(
    filename='PP.sqlite',
    using='sqlite3',
    query="SELECT A,PP FROM PPtable",
    param=model.p,
    index=model.A,
)
instance = model.create_instance(data)
# @db2
data = pyo.DataPortal()
data.load(
    filename='PP.sqlite',
    using='sqlite3',
    query="SELECT A,PP FROM PPtable",
    param=model.p,
    index=model.A,
    text_factory=str,
)
instance = model.create_instance(data)
instance.pprint()
# --------------------------------------------------
# @db3
if False:
    model = pyo.AbstractModel()
    data = pyo.DataPortal()
    model.A = pyo.Set()
    model.p = pyo.Param(model.A)
    data.load(
        filename="Driver={MySQL ODBC 5.2 UNICODE Driver}; Database=Pyomo; Server=localhost; User=pyomo;",
        using='pypyodbc',
        query="SELECT A,PP FROM PPtable",
        param=model.p,
        index=model.A,
    )
    instance = model.create_instance(data)
    # @db3
    data = pyo.DataPortal()
    data.load(
        filename="Driver={MySQL ODBC 5.2 UNICODE Driver}; Database=Pyomo; Server=localhost; User=pyomo;",
        using='pypyodbc',
        query="SELECT A,PP FROM PPtable",
        param=model.p,
        index=model.A,
        text_factory=str,
    )
    instance = model.create_instance(data)
    instance.pprint()
# --------------------------------------------------
# @json1
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.B = pyo.Set(dimen=2)
model.C = pyo.Set(model.A)
model.p = pyo.Param()
model.q = pyo.Param(model.A)
model.r = pyo.Param(model.B)
data.load(filename='T.json')
# @json1
data = pyo.DataPortal()
data.load(filename='T.json', convert_unicode=True)
instance = model.create_instance(data)
instance.pprint()
# --------------------------------------------------
# @yaml1
model = pyo.AbstractModel()
data = pyo.DataPortal()
model.A = pyo.Set()
model.B = pyo.Set(dimen=2)
model.C = pyo.Set(model.A)
model.p = pyo.Param()
model.q = pyo.Param(model.A)
model.r = pyo.Param(model.B)
data.load(filename='T.yaml')
# @yaml1
instance = model.create_instance(data)
instance.pprint()
# --------------------------------------------------

# @namespaces1
model = pyo.AbstractModel()
model.C = pyo.Set(dimen=2)
data = pyo.DataPortal()
data.load(filename='C.tab', set=model.C, namespace='ns1')
data.load(filename='D.tab', set=model.C, namespace='ns2', format='set_array')
instance1 = model.create_instance(data, namespaces=['ns1'])
instance2 = model.create_instance(data, namespaces=['ns2'])
# @namespaces1
instance1.pprint()
instance2.pprint()
