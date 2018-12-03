# @Import_Syntax
import pyomo.kernel as pmo
# @Import_Syntax

data = None

# @AbstractModels
def create(data):
    instance = pmo.block()
    # ... define instance ...
    return instance
instance = create(data)
# @AbstractModels
del data
del instance
# @ConcreteModels
m = pmo.block()
m.b = pmo.block()
# @ConcreteModels



# @Sets_1
m.s = [1,2]

# @Sets_1
# @Sets_2
# [0,1,2]
m.q = range(3)
# @Sets_2



# @Parameters_single
m.p = pmo.parameter(0)

# @Parameters_single
# @Parameters_dict
# pd[1] = 0, pd[2] = 1
m.pd = pmo.parameter_dict()
for k,i in enumerate(m.s):
    m.pd[i] = pmo.parameter(k)


# @Parameters_dict
# @Parameters_list
# uses 0-based indexing
# pl[0] = 0, pl[0] = 1, ...
m.pl = pmo.parameter_list()
for j in m.q:
    m.pl.append(
        pmo.parameter(j))
# @Parameters_list



# @Variables_single
m.v = pmo.variable(value=1,
                   lb=1,
                   ub=4)
# @Variables_single
# @Variables_dict
m.vd = pmo.variable_dict()
for i in m.s:
    m.vd[i] = pmo.variable(ub=9)
# @Variables_dict
# @Variables_list
# used 0-based indexing
m.vl = pmo.variable_list()
for j in m.q:
    m.vl.append(
        pmo.variable(lb=i))

# @Variables_list



# @Constraints_single
m.c = pmo.constraint(
    sum(m.vd.values()) <= 9)
# @Constraints_single
# @Constraints_dict
m.cd = pmo.constraint_dict()
for i in m.s:
    for j in m.q:
        m.cd[i,j] = \
            pmo.constraint(
                body=m.vd[i],
                rhs=j)
# @Constraints_dict
# @Constraints_list
# uses 0-based indexing
m.cl = pmo.constraint_list()
for j in m.q:
    m.cl.append(
        pmo.constraint(
            lb=-5,
            body=m.vl[j]-m.v,
            ub=5))
# @Constraints_list



# @Expressions_single
m.e = pmo.expression(-m.v)
# @Expressions_single
# @Expressions_dict
m.ed = pmo.expression_dict()
for i in m.s:
    m.ed[i] = \
        pmo.expression(-m.vd[i])
# @Expressions_dict
# @Expressions_list
# uses 0-based indexed
m.el = pmo.expression_list()
for j in m.q:
    m.el.append(
        pmo.expression(-m.vl[j]))
# @Expressions_list



# @Objectives_single
m.o = pmo.objective(-m.v)
# @Objectives_single
# @Objectives_dict
m.od = pmo.objective_dict()
for i in m.s:
    m.od[i] = \
        pmo.objective(-m.vd[i])
# @Objectives_dict
# @Objectives_list
# uses 0-based indexing
m.ol = pmo.objective_list()
for j in m.q:
    m.ol.append(
        pmo.objective(-m.vl[j]))
# @Objectives_list



# @SOS_single
m.sos1 = pmo.sos1(m.vd.values())


m.sos2 = pmo.sos2(m.vl)


# @SOS_single
# @SOS_dict
m.sd = pmo.sos_dict()
m.sd[1] = pmo.sos1(m.vd.values())
m.sd[2] = pmo.sos1(m.vl)







# @SOS_dict
# @SOS_list
# uses 0-based indexing
m.sl = pmo.sos_list()
for i in m.s:
    m.sl.append(pmo.sos1(
        [m.vl[i], m.vd[i]]))
# @SOS_list



# @Suffix_single
m.dual = pmo.suffix(
    direction=pmo.suffix.IMPORT)
# @Suffix_single
# @Suffix_dict
m.suffixes = pmo.suffix_dict()
m.suffixes['dual'] = pmo.suffix(
    direction=pmo.suffix.IMPORT)
# @Suffix_dict



# @Piecewise_1d
breakpoints = [1,2,3,4]
values = [1,2,1,2]
m.f = pmo.variable()
m.pw = pmo.piecewise(
    breakpoints,
    values,
    input=m.v,
    output=m.f,
    bound='eq')
# @Piecewise_1d



pmo.pprint(m)
