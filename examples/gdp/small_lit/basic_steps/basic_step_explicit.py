from pyomo.environ import *
from pyomo.gdp import *
from pyomo.gdp.basic_step import apply_basic_step

m = ConcreteModel()

m.x1 = Var(bounds=(-1,6), initialize=0)
m.x2 = Var(bounds=( 0,7), initialize=3.5)

m.objective = Objective(expr=0.2*m.x1 + m.x2)

m.disjuncts = Disjunct([1,2,3],[1,2])
m.disjuncts11 = Disjunct()
m.disjuncts11.c = Constraint(expr=m.x1**2 + (1./4)*(m.x2 - 5)**2 <= 1)

m.disjuncts21 = Disjunct()
m.disjuncts21.c = Constraint(expr=m.x1**2 + (1./4)*(m.x2 - 2)**2 <= 1)

m.disjuncts31 = Disjunct()
m.disjuncts31.c = Constraint(expr=m.x1**2 + (1./4)*(m.x2 - 3.5)**2 <= 1)


# NOTE: you MUST do 1./4 ... 1/4 is integer math (and equals 0!)
m.disjuncts12 = Disjunct()
m.disjuncts12.c = Constraint(expr=(m.x1 - 5)**2 + (1./4)*(m.x2 - 2)**2 <= 1)

m.disjuncts22 = Disjunct()
m.disjuncts22.c = Constraint(expr=(m.x1 - 5)**2 + (1./4)*(m.x2 - 5)**2 <= 1)

m.disjuncts32 = Disjunct()
m.disjuncts32.c = Constraint(expr=(m.x1 - 5)**2 + (1./4)*(m.x2 - 3.5)**2 <= 1)


# m.disjunctions = Disjunction([1,2,3], rule=disjunctions, xor=True)
m.disjunction1 = Disjunction(expr=[m.disjuncts11, m.disjuncts12], xor=True)
#m.disjunction1.xor = False
m.disjunction2 = Disjunction(expr=[m.disjuncts21, m.disjuncts22], xor=True)
#m.disjunction2.xor = False
m.disjunction3 = Disjunction(expr=[m.disjuncts31, m.disjuncts32], xor=True)
#m.disjunction3.xor = False
# Give a default BigM for testing with the Big-M relaxation
m.BigM = Suffix()
m.BigM[None] = 100
#m.pprint()
#
# Sanity check: solve the base model
#

#m1 = TransformationFactory('gdp.chull').create_using(m)
m1 = TransformationFactory('gdp.bigm').create_using(m)

solver = SolverFactory('gams')

solver.solve(m1, solver='baron')

print("Base model CHULL:")
m1.pprint()
m1.objective.display()
m1.x1.display()
m1.x2.display()

#
# A quick & dirty routine to apply a basic step to 2+ disjunctions.
# Note that it creates the full factorial set of disjuncts and does not
# prune the infeasible disjuncts from the disjunction.  That is lefy for
# "future work".
#

def basic_step(*disjunctions):
    # Get the disjuncts for each source disjunction
    disjuncts = []
    num_bs_disjuncts = 1
    for d in disjunctions:
        # HACK #1: until I get the Disjunction component updated,
        # getting the disjuncts from a Disjunction is a bit tricky
        disjuncts.append( d.parent_component()._disjuncts[d.index()] )
        # compute the number of disjuncts in the new disjunction
        num_bs_disjuncts *= len(disjuncts[-1])

    # Form the new disjuncts in the new resulting disjunction
    bs = Block(concrete=True)
    bs.DISJ = Set(initialize=range(len(disjuncts)))
    bs.disjuncts = Disjunct(range(num_bs_disjuncts))
    idx = [0]*len(disjuncts)
    idx[-1] -= 1
    for i in range(num_bs_disjuncts):
        # calculate the next index in the permutations of source disjuncts
        for j in reversed(range(len(disjuncts))):
            idx[j] += 1
            if idx[j] < len(disjuncts[j]):
                break
            else:
                idx[j] = 0

        # *COPY* the source disjuncts into this new disjunct, but store
        # them as blocks (not disjuncts).  this effectively prevents
        # the GDP transformations from seeing them as Disjuncts (so they
        # don't try and relax them)
        bs.disjuncts[i].src_disjunct = Block(
            bs.DISJ, 
            rule=lambda b,j: disjuncts[j][idx[j]].clone())

        # It might be nice to tie the original source disjuncts'
        # indicator variables with the current (new) indicator_var.  It
        # is not necessary for this problem, however it would become
        # necessary if there were logical constraints defined globally
        # using the original disjunct indicator_vars.
        bs.disjuncts[i].bind_indicators = Constraint(
            bs.DISJ, 
            rule=lambda b,j: b.indicator_var==b.src_disjunct[j].indicator_var)

        # HACK #2: chull/bigm can't handle subblocks at the moment.
        # Copy the contents of the block out
        for j in bs.DISJ:
            for c in bs.disjuncts[i].src_disjunct[j].component_objects():
                n = c.local_name
                bs.disjuncts[i].src_disjunct[j].del_component(c)
                bs.disjuncts[i].add_component("%s_%s__%s" % (i,j,n), c)
        bs.disjuncts[i].del_component(bs.disjuncts[i].src_disjunct)

    # Define the new basic step disjunction over the new disjunct blocks
    bs.disjunction = Disjunction(expr=list(bs.disjuncts.values()))

    # deactivate the original disjunctions / disjuncts
    for d in disjunctions:
        d.deactivate()
    for dlist in disjuncts:
        for d in dlist:
            d.deactivate()

    return bs


# Apply the basic step to 1+2 and evaluate the CHull.

m2 = m.clone()
# m2.bs = basic_step(m2.disjunctions[1], m2.disjunctions[2])
# m2.basic_step = apply_basic_step([m2.disjunctions[1], m2.disjunctions[2]])
m2.basic_step = apply_basic_step([m2.disjunction1, m2.disjunction2])
#m2.disjunction3.xor = False
#m2.basic_step.disjunction.xor = False
print("###################################")
m2.pprint()

TransformationFactory('gdp.bigm').apply_to(m2, bigM=100)
SolverFactory('gams').solve(m2, solver='baron', keepfiles=True, tee=True, mtype='minlp')
#SolverFactory('gdpopt').solve(m2, strategy='GLOA')

print("Post basic-step model CHULL:")
m2.objective.display()
m2.x1.display()
m2.x2.display()
