import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import meshio
import pymeshlab
import pyomo.environ as pyo
import pyomo.opt as pyopt
import seuif97
###############################################################################################################################################
print("""pyomo_universal_pwl_format.py demonstrates concept of new universal human readable format for function approximation in PYOMO for:
1) stationary process modeling subblocks (for nonlinear system of equations)
2) state estimation (process data reconciliation) and gross error detection
""")
###############################################################################################################################################
# Data preparation for example
print(
"""
Nonlinear function x3(x1,x2) used for demonstration
x1 - absolute pressure in turbine condenser atm
x2 - wet steam entropy in turbine condenser, kJ/(kg·K)
x3 - wrt steam enthalpy in turbine condenser, kcal/kg
""")
x1 = np.linspace(0.03,0.3,32)
x2 = np.linspace(7.4,8.5,32)
x1_, x2_ = np.meshgrid(x1,x2)
df = pd.DataFrame(np.concatenate([x1_.reshape(-1,1), x2_.reshape(-1,1)],axis=1), columns=['x1','x2'])
df['x3'] = df[['x1','x2']].apply(lambda x: seuif97.ps2h(x[0]/10,x[1])/4.186,axis=1)
Scaler_1 = StandardScaler()
dfn = pd.DataFrame(data = Scaler_1.fit_transform(df),index=df.index, columns=df.columns)
meshio.write("function_grid.ply", mesh=meshio.Mesh(points=dfn.values, cells = []), binary=False)
ms = pymeshlab.MeshSet()
ms.load_new_mesh("function_grid.ply")
ms.generate_surface_reconstruction_ball_pivoting()
ms.meshing_remove_unreferenced_vertices()
ms.save_current_mesh("function_grid.off",save_vertex_normal=False)
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=7)
ms.save_current_mesh("function_grid_simple.off",save_vertex_normal=False,save_polygonal=True)
with open("function_grid_simple.off",'r') as f:
    lines = f.readlines()
vert_num, simp_num, _ = lines[1].split()
vert_num = int(vert_num)
simp_num = int(simp_num)
Simplex_list = [x[1:].split() for x in lines[vert_num+2:]]
dfs = pd.DataFrame([x.split() for x in lines[2:vert_num+2]], columns=['x','y','z']).astype(float)
dfs['sind'] = [[] for x in dfs.index]
for k,v in enumerate(Simplex_list):
    for vert in v:
        dfs.at[int(vert),'sind'] += [int(k)+1]  
dfs.columns =  list(df.columns) + ['simplex_numb']
dfs[df.columns] = Scaler_1.inverse_transform(dfs[df.columns])
df_zero_mode = pd.DataFrame([[0,0,0,[0]],[0,1,0,[0]]],columns=['x1','x2','x3','simplex_numb'])
df_PWL = pd.concat([df_zero_mode,dfs])
df_PWL.reset_index(inplace=True, drop=True)

df['flg'] = 0 
df2 = df_PWL.iloc[:,:-1]
df2['flg'] = 1

print("""
df_PWL - dataframe representing uneversal format for PWL approximation: 
1) vertexes are index by rows
2) columns till last - vertex coordinates
3) last column - list of simplex numbers, that includes rows vertex

Function "mesh block" builds PYOMO concrete model from df_PWL.
Returned by "mesh block" PYOMO model represents the union of convex simplexes.
Can be use to model even discontinuous functions and nonzero volume domains.

df_PWL dataframe:
""")
print(df_PWL)
px.scatter_3d(data_frame=pd.concat([df,df2],axis=0), x = 'x1', y = 'x2', z = 'x3', color='flg',opacity=0.2)
print('\n'*3)
print('-'*150)
###############################################################################################################################################
# Pyomo code for concept demonstartion
m = pyo.ConcreteModel()
def mesh_block(df_PWL):
    """
    Function "mesh block" builds PYOMO concrete model from df_PWL.
    Returned by "mesh block" PYOMO model represents the union of convex simplexes.
    Can be use to model even discontinuous functions and nonzero volume domains.
    """
    m = pyo.Block(concrete=True)
    x_ind = df_PWL.columns.to_list()[:-1]
    lambda_ind = list(range(len(df_PWL)))
    tmp1 = []
    for item in df_PWL['simplex_numb'].values:
        tmp1 += item
    simlex_ind = list(set(tmp1))
    m.x = pyo.Var(x_ind)
    m.lambdas = pyo.Var(lambda_ind, bounds=(0,1))
    m.simplex_indicator = pyo.Var(simlex_ind,within=pyo.Binary)
    m.constr_simplex_indicator = pyo.Constraint(expr=sum(m.simplex_indicator[j] for j in simlex_ind)==1)
    m.constr_lincomb = pyo.ConstraintList()
    for jn,jv in enumerate(x_ind):
        expr = - m.x[jv]
        for i in lambda_ind:
            expr += m.lambdas[i]*df_PWL.iloc[:,:-1].values[i,jn]
        m.constr_lincomb.add(expr==0)
    m.constr_convex_comb = pyo.Constraint(expr=sum(m.lambdas[j] for j in lambda_ind)==1)
    m.constr_triangulation = pyo.ConstraintList()
    for j in lambda_ind:
        expr = m.lambdas[j]
        for i in df_PWL.iloc[j,-1]:
            expr += -m.simplex_indicator[i]
        m.constr_triangulation.add(expr<=0)
    return m
m.PWL_subblock_1 =  mesh_block(df_PWL)

# Code extention for state estimation (process data reconciliation) and gross error detection demonstration
top_x_list = []
for block_item in m.component_data_objects(pyo.Block):
    top_x_list += list(block_item.x.keys())
m.x = pyo.Var(top_x_list, within =pyo.NonNegativeReals)
m.x_meas = pyo.Var(top_x_list, within =pyo.NonNegativeReals)
metter_presision1 = 0.01 
metter_presision2 = 0.01
metter_presision3 = 0.01
measurements_metadata = {}
measurements_metadata['x1'] = {'norm':(df_PWL['x1'].max()-df_PWL['x1'].min())*metter_presision1}
measurements_metadata['x2'] = {'norm':(df_PWL['x2'].max()-df_PWL['x2'].min())*metter_presision2}
measurements_metadata['x3'] = {'norm':(df_PWL['x3'].max()-df_PWL['x3'].min())*metter_presision3}
measurements_list = list(measurements_metadata.keys())
deviation_bounds = 999.99
m.residual = pyo.Var(measurements_list, bounds=(-deviation_bounds,deviation_bounds))
m.res2 = pyo.Var(measurements_list)
m.con1 = pyo.ConstraintList()
for block_item in m.component_data_objects(pyo.Block):
    for var_item in top_x_list:
        try:
            m.con1.add(expr = block_item.x[var_item] == m.x[var_item])
        except:
            pass
def res_rule(m,measurement):
    return m.residual[measurement] == (m.x[measurement] - m.x_meas[measurement])/measurements_metadata[measurement]['norm']
m.res_list = pyo.Constraint(measurements_list, rule=res_rule)
N = 16
sigma_threshold  = 2.99
g_range =  np.concatenate([np.array([-deviation_bounds]),np.linspace(-sigma_threshold,sigma_threshold,N),np.array([deviation_bounds])])
qdf = pd.DataFrame({'dW': g_range, 'dW2': (g_range**2).clip(0.0,sigma_threshold**2)})
def v2(b,measurement):
    b.PWL = pyo.Piecewise(m.res2[measurement], m.residual[measurement], pw_pts=list(qdf['dW'].values),
                      pw_constr_type='EQ', f_rule=list(qdf['dW2'].values), pw_repn='SOS2')
m.quad_v = pyo.Block(measurements_list, rule=v2)
m.Imbalance = pyo.Var()
m.Imbalance_contraint = pyo.Constraint(expr= m.Imbalance == sum(m.res2[measurement] for measurement in measurements_list))
m.obj = pyo.Objective(expr = m.Imbalance)
def update_measurements(m, measurements_values_list):
    measurements_names_list = list(measurements_values_list.keys())
    m.x_meas.unfix()
    m.res_list.deactivate()
    for measurement in measurements_names_list:
        m.x_meas[measurement].fix(measurements_values_list[measurement])
        m.res_list[measurement].activate() 
    return m
m.update_measurements = update_measurements

###############################################################################################################################################
# Example of usage
measurements = {}
measurements['x1'] =0.04
measurements['x2'] = 7.5
m = m.update_measurements(m, measurements)
opt = pyo.SolverFactory('cbc')
opt.options["limits/gap"] = 0.0
opt.options["limits/absgap"] = 0.0
opt.options["limits/time"] = 2
status = opt.solve(m, tee=False)
print("""
Model solution x3 for given x1, x2
""")
m.x.display()
print(f"""\nx3 exact value = {seuif97.ps2h(m.x['x1'].value/10,m.x['x2'].value)/4.186:3.3f}""")
print('\n'*3)
print('-'*150)
m.PWL_subblock_1.pprint()
m.write('PWL_model.mps',io_options={'symbolic_solver_labels': True})
print('\n'*3)
print('-'*150)
df2plot = pd.concat([df,df2],axis=0)
df2plot.flg.replace({0:'exact',1:'pwl'},inplace=True)
fig1 = px.scatter_3d(data_frame=df2plot, x = 'x1', y = 'x2', z = 'x3', color='flg',opacity=0.2)
fig1.write_html('PWL_figure.html', auto_open=True)
