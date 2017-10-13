# Generation and Transmission Expansion Planning (MILP)
# IDAES project
# author: Cristiana L. Lara
# date: 10/09/2017

__author__ = "Cristiana L. Lara"

from pyomo.environ import *
from GTEPdata import *

m = ConcreteModel()

#####################Declaration of sets#########################################

m.r = Set(initialize=['Northeast', 'West', 'Coastal', 'South', 'Panhandle'],
 		ordered=True)
m.i = Set(initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old',
		'nuc-st-old', 'pv-old', 'wind-old', 'nuc-st-new','wind-new', 'pv-new',
		'csp-new', 'coal-igcc-new', 'coal-igcc-ccs-new', 'ng-cc-new',
		'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
m.th = Set(within=m.i,
		initialize=['nuc-st-old', 'nuc-st-new', 'coal-st-old1', 'coal-igcc-new',
	   	'coal-igcc-ccs-new', 'ng-ct-old', 'ng-cc-old', 'ng-st-old','ng-cc-new',
		'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
m.rn = Set(within=m.i,
	    initialize=['pv-old', 'pv-new', 'csp-new', 'wind-old', 'wind-new'],
	    ordered=True)
m.co = Set(within=m.th,
	    initialize=['coal-st-old1', 'coal-igcc-new', 'coal-igcc-ccs-new'],
	    ordered=True)
m.ng = Set(within=m.th,
	    initialize=['ng-ct-old', 'ng-cc-old', 'ng-st-old', 'ng-cc-new',
	    'ng-cc-ccs-new', 'ng-ct-new'], ordered=True)
m.nu = Set(within=m.th,
	    initialize=['nuc-st-old', 'nuc-st-new'], ordered=True)
m.pv = Set(within=m.rn,
	    initialize=['pv-old', 'pv-new'], ordered=True)
m.csp = Set(within=m.rn,
	    initialize=['csp-new'], ordered=True)
m.wi = Set(within=m.rn,
	    initialize=['wind-old', 'wind-new'], ordered=True)
m.old = Set(within=m.i,
	    initialize=['coal-st-old1', 'ng-ct-old', 'ng-cc-old', 'ng-st-old',
		'nuc-st-old', 'pv-old', 'wind-old'], ordered=True)
m.new = Set(within=m.i,
	    initialize=['nuc-st-new', 'wind-new', 'pv-new', 'csp-new',
		'coal-igcc-new','coal-igcc-ccs-new', 'ng-cc-new', 'ng-cc-ccs-new',
		'ng-ct-new'], ordered=True)
m.rold = Set(within=m.old,
	    initialize=['pv-old', 'wind-old'], ordered=True)
m.rnew = Set(within=m.new,
 	    initialize=['wind-new', 'pv-new', 'csp-new'], ordered=True)
m.told = Set(within=m.old,
 	    initialize=['nuc-st-old', 'coal-st-old1', 'ng-ct-old', 'ng-cc-old',
		'ng-st-old'], ordered=True)
m.tnew = Set(within=m.new,
 	    initialize=['nuc-st-new', 'coal-igcc-new','coal-igcc-ccs-new',
		'ng-cc-new','ng-cc-ccs-new', 'ng-ct-new'], ordered=True)

m.t = RangeSet(30)

m.ss = Set(initialize=['spring', 'summer', 'fall', 'winter'], ordered=True)
m.s = Set(initialize=['1:00','2:00','3:00','4:00','5:00','6:00','7:00','8:00',
                      '9:00','10:00','11:00','12:00','13:00','14:00','15:00',
                      '16:00','17:00','18:00','19:00','20:00','21:00','22:00',
                      '23:00','24:00'], ordered=True)

#Superset of iterations
m.iter = RangeSet(50)
#Set of iterations for which cuts are generated
m.k = Set(within=m.iter, dimen=1)

####################Import parameters###########################################

m.L = Param(m.r,m.t,m.ss,m.s, default=0, initialize=L)
m.n_ss = Param(m.ss, default=0, initialize=n_ss)
m.L_max = Param(m.t, default=0, initialize=L_max)
m.cf = Param(m.i, m.r, m.t, m.ss, m.s, default=0, initialize=cf)
m.Qg_np = Param(m.i, m.r, default=0, initialize=Qg_np)
m.Ng_old = Param(m.i, m.r, default=0, initialize=Ng_old)
m.Ng_max = Param(m.i, m.r, default=0, initialize=Ng_max)
m.Qinst_UB = Param(m.i, m.t, default=0, initialize=Qinst_UB)
m.LT = Param(m.i, initialize=LT, default=0)
m.Tremain = Param(m.t, default=0, initialize=Tremain)
m.Ng_r = Param(m.old, m.r, m.t, default=0, initialize=Ng_r)
m.q_v = Param(m.i, default=0, initialize=q_v)
m.Pg_min = Param(m.i, default=0, initialize=Pg_min)
m.Ru_max = Param(m.i, default=0, initialize=Ru_max)
m.Rd_max = Param(m.i, default=0, initialize=Rd_max)
m.f_start = Param(m.i, default=0, initialize=f_start)
m.C_start = Param(m.i, default=0, initialize=C_start)
m.frac_spin = Param(m.i, default=0, initialize=frac_spin)
m.frac_Qstart = Param(m.i, default=0, initialize=frac_Qstart)
m.t_loss = Param(m.r, m.r, default=0, initialize=t_loss)
m.t_up = Param(m.r, m.r, default=0, initialize=t_up)
m.dist = Param(m.r, m.r, default=0, initialize=dist)
m.if_ = Param(m.t, default=0, initialize=if_)
m.ED = Param(m.t, default=0, initialize=ED)
m.Rmin = Param(m.t, default=0, initialize=Rmin)
m.hr = Param(m.i, m.r, default=0, initialize=hr)
m.P_fuel = Param(m.i, m.t, default=0, initialize=P_fuel)
m.EF_CO2 = Param(m.i, default=0, initialize=EF_CO2)
m.FOC = Param(m.i, m.t, default=0, initialize=FOC)
m.VOC = Param(m.i, m.t, default=0, initialize=VOC)
m.CCm = Param(m.i, default=0, initialize=CCm)
m.DIC = Param(m.i, m.t, default=0, initialize=DIC)
m.LEC = Param(m.i, default=0, initialize=LEC)
m.PEN = Param(m.t, default=0, initialize=PEN)
m.PENc = Param(default=0, initialize=PENc)
m.tx_CO2 = Param(m.t, default=0, initialize=tx_CO2)
m.RES_min = Param(m.t, default=0, initialize=RES_min)
m.hs = Param(initialize=hs, default=0)
m.ir = Param(initialize=ir, default=0)

def i_r_filter(m,i,r):
    return i in m.new or (i in m.old and m.Ng_old[i,r] != 0)
m.i_r = Set(initialize=m.i*m.r, filter=i_r_filter)

########################### Decomposition Parameters ###########################
m.ngo_rn_par = Param(m.rn, m.r, m.t, default=0, initialize=0, mutable=True)
m.ngo_th_par = Param(m.th, m.r, m.t, default=0, initialize=0, mutable=True)
m.ngb_rn_par = Param(m.rn, m.r, m.t, default=0, initialize=0, mutable=True)
m.ngb_th_par = Param(m.th, m.r, m.t, default=0, initialize=0, mutable=True)
m.ngo_rn_par_k = Param(m.rn, m.r, m.t, m.iter, default=0, initialize=0, mutable=True)
m.ngo_th_par_k = Param(m.th, m.r, m.t, m.iter, default=0, initialize=0, mutable=True)
m.ngb_rn_par_k = Param(m.rn, m.r, m.t, m.iter, default=0, initialize=0, mutable=True)
m.ngb_th_par_k = Param(m.th, m.r, m.t, m.iter, default=0, initialize=0, mutable=True)
m.cost=Param(m.t,m.iter, default=0, initialize=0, mutable=True)
m.mltp_o_rn=Param(m.rn,m.r,m.t,m.iter, default=0, initialize=0, mutable=True)
m.mltp_o_th=Param(m.th,m.r,m.t,m.iter, default=0, initialize=0, mutable=True)
m.mltp_b_rn=Param(m.rn,m.r,m.t,m.iter, default=0, initialize=0, mutable=True)
m.mltp_b_th=Param(m.th,m.r,m.t,m.iter, default=0, initialize=0, mutable=True)
m.cost_t=Param(m.t,m.iter, default=0, initialize=0, mutable=True)

################ Block of Equations per time period ############################
def planning_block_rule(b,t):
	# Declaration of bounds
	def bound_P(_b,i,r,ss,s):
		if i in m.old:
			return (0, m.Qg_np[i,r]*m.Ng_old[i,r])
		else:
			return (0, m.Qg_np[i,r]*m.Ng_max[i,r])

	def bound_Pflow(_b,r,r_,ss,s):
	    if r_ != r:
	        return (0, t_up[r,r_])
	    else:
	        return (0,0)

	def bound_o_rn(_b,rn,r):
		if rn in m.rold:
			return (0, m.Ng_old[rn,r])
		else:
			return (0, m.Ng_max[rn,r])

	def bound_b_rn(_b,rn,r):
		if rn in m.rold:
			return (0, 0)
		else:
			return (0, floor(m.Qinst_UB[rn,t]/m.Qg_np[rn,r]))

	def bound_r_rn(_b,rn,r):
		if rn in m.rold:
			return (0, m.Ng_old[rn,r])
		else:
			return (0, m.Ng_max[rn,r])

	def bound_e_rn(_b,rn,r):
		if rn in m.rold:
			return (0, m.Ng_r[rn,r,t])
		else:
			return (0, m.Ng_max[rn,r])

	def bound_o_th(_b,th,r):
		if th in m.told:
			return (0, m.Ng_old[th,r])
		else:
			return (0, m.Ng_max[th,r])

	def bound_b_th(_b,th,r):
		if th in m.told:
			return (0, 0)
		else:
			return (0, floor(m.Qinst_UB[th,t]/m.Qg_np[th,r]))

	def bound_r_th(_b,th,r):
		if th in m.told:
			return (0, m.Ng_old[th,r])
		else:
			return (0, m.Ng_max[th,r])

	def bound_e_th(_b,th,r):
		if th in m.told:
			return (0, m.Ng_r[th,r,t])
		else:
			return (0, m.Ng_max[th,r])

	def bound_UC(_b,th,r,ss,s):
		if th in m.told:
			return (0, m.Ng_old[th,r])
		else:
			return (0, m.Ng_max[th,r])

	# Declaration of Local Variables
	b.P = Var(m.i, m.r, m.ss, m.s, within=NonNegativeReals, bounds=bound_P,
		doc="power output (MW)")
	b.cu = Var(m.r, m.ss, m.s, within=NonNegativeReals,
		doc="curtailment slack in reagion r (MW)")
	b.RES_def = Var(within=NonNegativeReals,
	    doc="deficit from renewable quota target (MW)")
	b.P_flow = Var(m.r, m.r, m.ss, m.s, within=NonNegativeReals,
		bounds=bound_Pflow, doc="power flow from one region to the other (MW)")
	b.Q_spin = Var(m.th, m.r, m.ss, m.s, within=NonNegativeReals,
		doc="spinning reserve (MW)")
	b.Q_Qstart = Var(m.th, m.r, m.ss, m.s, within=NonNegativeReals,
		doc="quick-start reserve (MW)")
	b.ngr_rn = Var(m.rn, m.r, bounds=bound_r_rn, domain= NonNegativeReals,
		doc="number of renewable generaters that are retired in cluster i, \
		region r, year t")
	for rnew, r  in  m.rnew * m.r :
	    if t == 1:
	        b.ngr_rn[rnew,r].fix(0.0)
	    else:
	        b.ngr_rn[rnew,r].unfix()

	b.nge_rn = Var(m.rn, m.r, bounds=bound_e_rn, domain= NonNegativeReals,
		doc="number of renewable generaters that had their life extended in \
		cluster i, region r, year t")
	for rnew, r in m.rnew * m.r:
	    if t == 1:
	        b.nge_rn[rnew,r].fix(0.0)
	    else:
	        b.nge_rn[rnew,r].unfix()

	b.ngr_th = Var(m.th, m.r, bounds=bound_r_th, domain=NonNegativeIntegers,
		doc="number of thermal generaters that are retired in cluster i, \
		region r, year t")
	for tnew, r in m.tnew * m.r:
	    if t == 1:
	        b.ngr_th[tnew,r].fix(0.0)
	    else:
	        b.ngr_th[tnew,r].unfix()

	b.nge_th = Var(m.th, m.r, bounds=bound_e_th, domain=NonNegativeIntegers,
		doc="number of thermal generaters that had their life extended in \
		cluster i, region r, year t")
	for tnew, r in m.tnew * m.r:
	    if t==1:
	        b.nge_th[tnew,r].fix(0.0)
	    else:
	        b.nge_th[tnew,r].unfix()

	b.u = Var(m.th, m.r, m.ss, m.s, bounds=bound_UC, domain=NonNegativeIntegers,
		doc="unit commitment status")
	b.su = Var(m.th, m.r, m.ss, m.s, bounds=bound_UC, domain=NonNegativeIntegers,
		doc="startup indicator")
	b.sd = Var(m.th, m.r, m.ss, m.s, bounds=bound_UC, domain=NonNegativeIntegers,
		doc="shutdown indicator")
	for th,r,ss,s in m.th*m.r*m.ss*m.s:
	    if m.s.ord(s) == 1:
	        b.sd[th,r,ss,s].fix(0.0)
	    else:
	        b.sd[th,r,ss,s].unfix()

	b.alphafut = Var(within=Reals) #cost-to-go function

	# Declaration of State Variables
	b.ngo_rn = Var(m.rn, m.r, bounds=bound_o_rn, domain= NonNegativeReals,
		doc="number of renewable generaters that are operational in cluster i, \
		region r, year t")

	b.ngb_rn = Var(m.rn, m.r, bounds=bound_b_rn, domain= NonNegativeReals,
		doc="number of renewable generaters that are built in cluster i,\
		region r, year t")
	for rold, r in m.rold * m.r:
	    b.ngb_rn[rold,r].fix(0.0)

	b.ngo_th = Var(m.th, m.r, bounds=bound_o_th, domain=NonNegativeIntegers,
		doc="number of thermal generaters that are operational in cluster i, \
		region r, year t")

	b.ngb_th = Var(m.th, m.r, bounds=bound_b_th, domain=NonNegativeIntegers,
		doc="number of thermal generaters that are built in cluster i, region r,\
		year t")
	for told, r in  m.told * m.r:
	    b.ngb_th[told,r].fix(0.0)

	# Declaration of the Copy of State Variables
	b.ngo_rn_prev = Var(m.rn, m.r, bounds=bound_o_rn, domain= NonNegativeReals,
		doc="number of renewable generaters that are operational in cluster i,\
		region r, year t")
	b.ngb_rn_LT = Var(m.rn, m.r, bounds=bound_b_rn, domain= NonNegativeReals,
		doc="number of renewable generaters that are built in cluster i, \
		region r, year t")
	for rold, r in m.rold * m.r:
	    b.ngb_rn_LT[rold,r].fix(0.0)

	b.ngo_th_prev = Var(m.th, m.r, bounds=bound_o_th, domain=NonNegativeReals,
		doc="number of thermal generaters that are operational in cluster i, \
		region r, year t")
	b.ngb_th_LT = Var(m.th, m.r, bounds=bound_b_th, domain=NonNegativeReals,
		doc="number of thermal generaters that are built in cluster i, region r,\
		year t")
	for told, r in  m.told * m.r:
	    b.ngb_th_LT[told,r].fix(0.0)

	#Objective function
	def obj_rule(_b):
		return m.if_[t]*\
			(sum(m.n_ss[ss]*m.hs* \
            	sum((m.VOC[i,t] + m.hr[i,r]*m.P_fuel[i,t] + m.EF_CO2[i]
					*m.tx_CO2[t]*m.hr[i,r])*_b.P[i,r,ss,s] for i,r in m.i_r) \
                for ss in m.ss for s in m.s) \
			+ sum(m.FOC[rn,t]*m.Qg_np[rn,r]*_b.ngo_rn[rn,r] \
			    for rn,r in m.i_r if rn in m.rn) \
			+ sum(m.FOC[th,t]*m.Qg_np[th,r]*_b.ngo_th[th,r] \
			    for th,r in m.i_r if th in m.th) \
			+ sum(m.n_ss[ss]*m.hs*_b.su[th,r,ss,s]*m.Qg_np[th,r] \
				*(m.f_start[th]*m.P_fuel[th,t] + m.f_start[th]*m.EF_CO2[th] \
                	*m.tx_CO2[t] + m.C_start[th]) \
                for th,r in m.i_r if th in m.th for ss in m.ss for s in m.s) \
			+ sum(m.DIC[rnew,t]*m.CCm[rnew]*m.Qg_np[rnew,r]*_b.ngb_rn[rnew,r] \
			    for rnew,r in m.i_r if rnew in m.rnew) \
			+ sum(m.DIC[tnew,t]*m.CCm[tnew]*m.Qg_np[tnew,r]*_b.ngb_th[tnew,r] \
			    for tnew,r in m.i_r if tnew in m.tnew) \
			+ sum(m.DIC[rn,t]*m.LEC[rn]*m.Qg_np[rn,r]*_b.nge_rn[rn,r] \
			    for rn,r in m.i_r if rn in m.rn) \
			+ sum(m.DIC[th,t]*m.LEC[th]*m.Qg_np[th,r]*_b.nge_th[th,r] \
			    for th,r in m.i_r if th in m.th) \
			+ m.PEN[t]*_b.RES_def \
			+ m.PENc*sum(_b.cu[r,ss,s] \
				for r in m.r for ss in m.ss for s in m.s))  \
			* 10**(-9)\
			+ _b.alphafut
	b.obj = Objective(rule=obj_rule, sense=minimize)

	# Investment Constraints
	def min_RN_req(_b):
		return sum(m.n_ss[ss]*m.hs*sum(_b.P[rn,r,ss,s]-_b.cu[r,ss,s] \
		                       			for rn,r in m.i_r if rn in m.rn) \
                   for ss in m.ss for s in m.s)\
				   + _b.RES_def >= m.RES_min[t]*m.ED[t]
	b.min_RN_req = Constraint(rule=min_RN_req)

	def min_reserve(_b):
		return sum(m.Qg_np[rn,r]*_b.ngo_rn[rn,r]*m.q_v[rn] \
					for rn,r in m.i_r if rn in m.rn) \
	     	+ sum(m.Qg_np[th,r]*_b.ngo_th[th,r] for th,r in m.i_r if th in m.th) \
			>= (1 + m.Rmin[t])*m.L_max[t]
	b.min_reserve = Constraint(rule=min_reserve)

	def inst_RN_UB(_b,rnew):
		return sum(_b.ngb_rn[rnew,r] for r in m.r) \
               <= m.Qinst_UB[rnew,t] / sum(m.Qg_np[rnew,r] /len(m.r) for r in m.r)
	b.inst_RN_UB = Constraint(m.rnew, rule=inst_RN_UB)

	def inst_TH_UB(_b,tnew):
		return sum(_b.ngb_th[tnew,r] for r in m.r) \
               <= m.Qinst_UB[tnew,t] / sum(m.Qg_np[tnew,r] /len(m.r) for r in m.r)
	b.inst_TH_UB = Constraint(m.tnew, rule=inst_TH_UB)

	#Operating Constraints
	def en_bal(_b,r,ss,s):
		return sum(_b.P[i,r,ss,s] for i in m.i if (i,r) in m.i_r) \
             + sum(_b.P_flow[r_,r,ss,s]*(1-m.t_loss[r,r_]*m.dist[r,r_])-\
			 _b.P_flow[r,r_,ss,s] for r_ in m.r if r_ != r)\
			 == m.L[r,t,ss,s] + _b.cu[r,ss,s]
	b.en_bal = Constraint(m.r, m.ss, m.s, rule=en_bal)

	def capfactor(_b,rn,r,ss,s):
		if (rn,r) in m.i_r and rn in m.rn:
			return _b.P[rn,r,ss,s] == m.Qg_np[rn,r]*m.cf[rn,r,t,ss,s] \
				*_b.ngo_rn[rn,r]
		return Constraint.Skip
	b.capfactor = Constraint(m.rn, m.r, m.ss, m.s, rule=capfactor)

	def min_output(_b,th,r,ss,s):
		if (th,r) in m.i_r and th in m.th:
			return _b.u[th,r,ss,s]*m.Pg_min[th]*m.Qg_np[th,r] <= _b.P[th,r,ss,s]
		return Constraint.Skip
	b.min_output = Constraint(m.th,m.r,m.ss,m.s, rule=min_output)

	def max_output(_b,th,r,ss,s):
		if (th,r) in m.i_r and th in m.th:
			return _b.u[th,r,ss,s]*m.Qg_np[th,r] >= _b.P[th,r,ss,s] \
												+ _b.Q_spin[th,r,ss,s]
		return Constraint.Skip
	b.max_output = Constraint(m.th,m.r,m.ss,m.s, rule=max_output)

	def unit_commit1(_b,th,r,ss,s,s_):
		if (th,r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s)-1):
			return _b.u[th,r,ss,s] == _b.u[th,r,ss,s_] + _b.su[th,r,ss,s] \
									- _b.sd[th,r,ss,s]
		return Constraint.Skip
	b.unit_commit1 = Constraint(m.th,m.r,m.ss,m.s,m.s, rule=unit_commit1)

	def ramp_up(_b,th,r,ss,s,s_):
		if (th,r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s)-1):
			return _b.P[th,r,ss,s]-_b.P[th,r,ss,s_] <= m.Ru_max[th]*m.hs* \
				m.Qg_np[th,r]*(_b.u[th,r,ss,s]-_b.su[th,r,ss,s]) + \
				max(m.Pg_min[th],m.Ru_max[th]*m.hs)*m.Qg_np[th,r]*_b.su[th,r,ss,s]
		return Constraint.Skip
	b.ramp_up = Constraint(m.th,m.r,m.ss,m.s,m.s, rule=ramp_up)

	def ramp_down(_b,th,r,ss,s,s_):
		if (th,r) in m.i_r and th in m.th and (m.s.ord(s_) == m.s.ord(s)-1):
			return _b.P[th,r,ss,s_]-_b.P[th,r,ss,s] <= m.Rd_max[th]*m.hs* \
				m.Qg_np[th,r]*(_b.u[th,r,ss,s]-_b.su[th,r,ss,s]) + \
				max(m.Pg_min[th],m.Rd_max[th]*m.hs)*m.Qg_np[th,r]*_b.sd[th,r,ss,s]
		return Constraint.Skip
	b.ramp_down = Constraint(m.th,m.r,m.ss,m.s,m.s, rule=ramp_down)

	def total_op_reserve(_b,r,ss,s):
		return sum(_b.Q_spin[th,r,ss,s]+_b.Q_Qstart[th,r,ss,s] for th in m.th) \
			>= 0.075*m.L[r,t,ss,s]
	b.total_op_reserve = Constraint(m.r,m.ss,m.s, rule=total_op_reserve)

	def total_spin_reserve(_b,r,ss,s):
		return sum(_b.Q_spin[th,r,ss,s] for th in m.th) >= 0.015*m.L[r,t,ss,s]
	b.total_spin_reserve = Constraint(m.r,m.ss,m.s, rule=total_spin_reserve)

	def reserve_cap_1(_b,th,r,ss,s):
		if (th,r) in m.i_r and th in m.th:
			return _b.Q_spin[th,r,ss,s] <= _b.u[th,r,ss,s]*m.Qg_np[th,r]*\
				m.frac_spin[th]
		return Constraint.Skip
	b.reserve_cap_1 = Constraint(m.th,m.r,m.ss,m.s, rule=reserve_cap_1)

	def reserve_cap_2(_b,th,r,ss,s):
		if (th,r) in m.i_r and th in m.th:
			return _b.Q_Qstart[th,r,ss,s] <= (_b.ngo_th[th,r] - _b.u[th,r,ss,s])\
				* m.Qg_np[th,r] * m.frac_Qstart[th]
		return Constraint.Skip
	b.reserve_cap_2 = Constraint(m.th, m.r, m.ss, m.s, rule=reserve_cap_2)

	def logic_RN_1(_b,rn,r):
		if (rn,r) in m.i_r and rn in m.rn:
			if t == 1:
				return _b.ngb_rn[rn,r] - _b.ngr_rn[rn,r] == _b.ngo_rn[rn,r] - \
					m.Ng_old[rn,r]
			else:
				return _b.ngb_rn[rn,r] - _b.ngr_rn[rn,r] == _b.ngo_rn[rn,r] - \
						_b.ngo_rn_prev[rn,r]
		return Constraint.Skip
	b.logic_RN_1 = Constraint(m.rn,m.r, rule=logic_RN_1)

	def logic_TH_1(_b,th,r):
		if (th,r) in m.i_r and th in m.th:
			if t == 1:
				return _b.ngb_th[th,r] - _b.ngr_th[th,r] == _b.ngo_th[th,r] - \
					m.Ng_old[th,r]
			else:
				return _b.ngb_th[th,r] - _b.ngr_th[th,r] == _b.ngo_th[th,r] - \
					_b.ngo_th_prev[th,r]
		return Constraint.Skip
	b.logic_TH_1 = Constraint(m.th,m.r, rule=logic_TH_1)

	def logic_RNew_2(_b,rnew,r):
		if (rnew,r) in m.i_r and rnew in m.rnew:
			return _b.ngb_rn_LT[rnew,r] == _b.ngr_rn[rnew,r]+_b.nge_rn[rnew,r]
		return Constraint.Skip
	b.logic_RNew_2 = Constraint(m.rnew,m.r, rule=logic_RNew_2)

	def logic_TNew_2(_b,tnew,r):
		if (tnew,r) in m.i_r and tnew in m.tnew:
			return _b.ngb_th_LT[tnew,r] == _b.ngr_th[tnew,r]+_b.nge_th[tnew,r]
		return Constraint.Skip
	b.logic_TNew_2 = Constraint(m.tnew,m.r, rule=logic_TNew_2)

	def logic_ROld_2(_b,rold,r):
		if (rold,r) in m.i_r and rold in m.rold:
			return _b.ngr_rn[rold,r]+_b.nge_rn[rold,r] == m.Ng_r[rold,r,t]
		return Constraint.Skip
	b.logic_ROld_2 = Constraint(m.rold, m.r, rule=logic_ROld_2)

	def logic_TOld_2(_b,told,r):
		if (told,r) in m.i_r and told in m.told:
			return _b.ngr_th[told,r]+_b.nge_th[told,r] == m.Ng_r[told,r,t]
		return Constraint.Skip
	b.logic_TOld_2 = Constraint(m.told, m.r, rule=logic_TOld_2)

	def logic_3(_b,th,r,ss,s):
		if (th,r) in m.i_r and th in m.th:
			return _b.u[th,r,ss,s] <= _b.ngo_th[th,r]
		return Constraint.Skip
	b.logic_3 = Constraint(m.th,m.r,m.ss,m.s, rule=logic_3)

	#linking equalities
	def link_equal1(_b,rn,r):
		if (rn,r) in m.i_r and rn in m.rn and t > 1:
			return _b.ngo_rn_prev[rn,r] == m.ngo_rn_par[rn,r,t-1]
		return Constraint.Skip
	b.link_equal1 = Constraint(m.rn, m.r, rule=link_equal1)

	def link_equal2(_b,th,r):
		if (th,r) in m.i_r and th in m.th and t > 1:
			return _b.ngo_th_prev[th,r] == m.ngo_th_par[th,r,t-1]
		return Constraint.Skip
	b.link_equal2 = Constraint(m.th, m.r, rule=link_equal2)

	def link_equal3(_b,rn,r):
		if (rn,r) in m.i_r and rn in m.rn and t > LT[rn]:
			return _b.ngb_rn_LT[rn,r] == m.ngb_rn_par[rn,r,t-m.LT[rn]]
		return Constraint.Skip
	b.link_equal3 = Constraint(m.rn, m.r, rule=link_equal3)

	def link_equal4(_b,th,r):
		if (th,r) in m.i_r and th in m.th and t > LT[th]:
			return _b.ngb_th_LT[th,r] == m.ngo_th_par[th,r,t-m.LT[th]]
		return Constraint.Skip
	b.link_equal4 = Constraint(m.th, m.r, rule=link_equal4)

	#Benders cut
	b.fut_cost = ConstraintList()

m.Bl = Block(m.t, rule=planning_block_rule)

################################################################################
####################### post_process function ##################################
################################################################################

#Parameters to store results
m.P_result = Param(m.i, m.r, m.t, m.ss, m.s, m.iter, default=0, initialize=0, \
    mutable=True)
m.cu_result = Param(m.r, m.t, m.ss, m.s, m.iter, default=0, initialize=0,\
    mutable=True)
m.RES_def_result = Param(m.t, m.iter, default=0, initialize=0, mutable=True)
m.P_flow_result = Param(m.r, m.r, m.t, m.ss, m.s, m.iter, default=0,   \
    initialize=0, mutable=True)
m.Q_spin_result = Param(m.th, m.r, m.t, m.ss, m.s, m.iter, default=0, \
    initialize=0, mutable=True)
m.Q_Qstart_result = Param(m.th, m.r, m.t, m.ss, m.s, m.iter, default=0, \
    initialize=0, mutable=True)
m.ngo_rn_result = Param(m.rn, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.ngb_rn_result = Param(m.rn, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.ngr_rn_result = Param(m.rn, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.nge_rn_result = Param(m.rn, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.ngo_th_result = Param(m.th, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.ngb_th_result = Param(m.th, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.ngr_th_result = Param(m.th, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.nge_th_result = Param(m.th, m.r, m.t, m.iter, default=0, initialize=0, \
    mutable=True)
m.u_result = Param(m.th, m.r, m.t, m.ss, m.s, m.iter, default=0, initialize=0,\
    mutable=True)
m.su_result = Param(m.th, m.r, m.t, m.ss, m.s, m.iter, default=0, initialize=0,\
    mutable=True)
m.sd_result = Param(m.th, m.r, m.t, m.ss, m.s, m.iter, default=0, initialize=0,\
    mutable=True)

m.gs = Set(initialize=['coal', 'NG', 'nuc', 'solar','wind'],
	   doc="energy sources", ordered=True)

def post_process():

	for iter_ in m.iter:
		def fixOp(m,iter_):
			return sum(m.if_[t]*(sum(m.FOC[rn,t]*m.Qg_np[rn,r]\
                    *m.ngo_rn_result[rn,r,t,iter_]for rn,r in m.i_r if rn in m.rn) \
                    + sum(m.FOC[th,t]*m.Qg_np[th,r]*m.ngo_th_result[th,r,t,iter_] \
                      for th,r in m.i_r if th in m.th)) for t in m.t)* 10**(-9)
		m.fixOp = Expression(m.iter, rule=fixOp)

		def varOp(m,iter_):
			return sum(m.if_[t]*sum(m.n_ss[ss]*m.hs*m.VOC[i,t]\
                    *m.P_result[i,r,t,ss,s,iter_] for i,r in m.i_r \
                    for ss in m.ss for s in m.s) for t in m.t)*10**(-9)
		m.varOp = Expression(m.iter, rule=varOp)

		def start(m,iter_):
			return sum(m.if_[t]*sum(m.n_ss[ss]*m.hs*m.su_result[th,r,t,ss,s,iter_]\
                    *m.Qg_np[th,r]*(m.f_start[th]*m.P_fuel[th,t]+m.f_start[th]\
                    *m.EF_CO2[th]*m.tx_CO2[t]+m.C_start[th]) \
                    for th,r in m.i_r if th in m.th for ss in m.ss for s in m.s) \
                    for t in m.t)*10**(-9)
		m.start = Expression(m.iter, rule=start)

		def inv(m,iter_):
			return sum(m.if_[t]*(sum(m.DIC[rnew,t]*m.CCm[rnew]*m.Qg_np[rnew,r]\
                *m.ngb_rn_result[rnew,r,t,iter_] for rnew,r in m.i_r if rnew in m.rnew) \
                +sum(m.DIC[tnew,t]*m.CCm[tnew]*m.Qg_np[tnew,r]*m.ngb_th_result[tnew,r,t,iter_]\
                for tnew,r in m.i_r if tnew in m.tnew)) for t in m.t)*10**(-9)
		m.inv = Expression(m.iter, rule=inv)

		def ext(m,iter_):
			return sum(m.if_[t]*(sum(m.DIC[rn,t]*m.LEC[rn]*m.Qg_np[rn,r]*\
                m.nge_rn_result[rn,r,t,iter_] for rn,r in m.i_r if rn in m.rn)\
                + sum(m.DIC[th,t]*m.LEC[th]*m.Qg_np[th,r]*m.nge_th_result[th,r,t,iter_] \
                for th,r in m.i_r if th in m.th)) for t in m.t)*10**(-9)
		m.ext = Expression(m.iter, rule=ext)

		def fuel(m,iter_):
			return sum(m.if_[t]*sum(m.n_ss[ss]*m.hs*m.hr[i,r]*m.P_fuel[i,t]*\
                m.P_result[i,r,t,ss,s,iter_] for i,r in m.i_r for ss in m.ss \
                for s in m.s)for t in m.t)*10**(-9)
		m.fuel = Expression(m.iter, rule=fuel)

		def tot(m,iter_):
			return m.fixOp[iter_] + m.varOp[iter_] + m.start[iter_] + m.inv[iter_] + \
                m.fuel[iter_] + m.ext[iter_]
		m.tot = Expression(m.iter, rule=tot)

		def generation(m, i, t,iter_): #Generation per technology
			if i in m.rn:
				return sum(m.ngo_rn_result[i,r,t,iter_]*m.Qg_np[i,r] for r in m.r if (i,r) in m.i_r)*10**(-3)
			else:
				return sum(m.ngo_th_result[i,r,t,iter_]*m.Qg_np[i,r] for r in m.r if (i,r) in m.i_r)*10**(-3)
		m.generation = Expression(m.i,m.t,m.iter, rule=generation)

		def generation_source(m,gs,t,iter_): #Generation per source
			if gs in m.gs and gs == 'coal':
				return sum(m.generation[i,t,iter_] for i in m.co)
			elif gs in m.gs and gs == 'NG':
				return sum(m.generation[i,t,iter_] for i in m.ng)
			elif gs in m.gs and gs == 'nuc':
				return sum(m.generation[i,t,iter_] for i in m.nu)
			elif gs in m.gs and gs == 'solar':
				return sum(m.generation[i,t,iter_] for i in m.pv)
			elif gs in m.gs and gs == 'wind':
				return sum(m.generation[i,t,iter_] for i in m.wi)
		m.generation_source = Expression(m.gs,m.t,m.iter, rule=generation_source)

		if iter_ == m.iter.last():
			print ("Fixed Operating Cost", value(m.fixOp[iter_]))
	        print ("Variable Operating Cost", value(m.varOp[iter_]))
	        print ("Startup Cost", value(m.start[iter_]))
	        print ("Investment Cost", value(m.inv[iter_]))
	        print ("Life Extension Cost", value(m.ext[iter_]))
	        print ("Fuel Cost", value(m.fuel[iter_]))
	        print ("obj", value(m.tot[iter_]))

		if iter_ == m.iter.last():
			print ("Capacity per source per year in GW")
			for (gs,t) in m.gs*m.t:
				print (gs, t, value(m.generation_source[gs,t,iter_]))
