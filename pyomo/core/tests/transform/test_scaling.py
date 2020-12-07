#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#


import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.opt.base.solvers import UnknownSolver

class TestScaleModelTransformation(unittest.TestCase):

    def test_linear_scaling(self):
        model = pe.ConcreteModel()
        model.x = pe.Var([1,2,3], bounds=(-10,10), initialize=5.0)
        model.z = pe.Var(bounds=(10,20))
        model.obj = pe.Objective(expr=model.z + model.x[1])

        # test scaling of duals as well
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
        model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
        
        def con_rule(m, i):
            if i == 1:
                return m.x[1] + 2*m.x[2] + 1*m.x[3] == 4.0
            if i == 2:
                return m.x[1] + 2*m.x[2] + 2*m.x[3] == 5.0
            if i == 3:
                return m.x[1] + 3.0*m.x[2] + 1*m.x[3] == 5.0
        model.con = pe.Constraint([1,2,3], rule=con_rule)
        model.zcon = pe.Constraint(expr=model.z >= model.x[2])

        x_scale = 0.5
        obj_scale = 2.0
        z_scale = -10.0
        con_scale1 = 0.5
        con_scale2 = 2.0
        con_scale3 = -5.0
        zcon_scale = -3.0
        
        unscaled_model = model.clone()
        unscaled_model.scaling_factor = pe.Suffix(direction=pe.Suffix.EXPORT)
        unscaled_model.scaling_factor[unscaled_model.obj] = obj_scale
        unscaled_model.scaling_factor[unscaled_model.x] = x_scale
        unscaled_model.scaling_factor[unscaled_model.z] = z_scale
        unscaled_model.scaling_factor[unscaled_model.con[1]] = con_scale1 
        unscaled_model.scaling_factor[unscaled_model.con[2]] = con_scale2
        unscaled_model.scaling_factor[unscaled_model.con[3]] = con_scale3
        unscaled_model.scaling_factor[unscaled_model.zcon] = zcon_scale
                
        scaled_model = pe.TransformationFactory('core.scale_model').create_using(unscaled_model)

        # print('*** unscaled ***')
        # unscaled_model.pprint()
        # print('*** scaled ***')
        # scaled_model.pprint()

        glpk_solver =  pe.SolverFactory('glpk')
        if isinstance(glpk_solver, UnknownSolver) or \
           (not glpk_solver.available()):
            raise unittest.SkipTest("glpk solver not available")

        glpk_solver.solve(unscaled_model)
        glpk_solver.solve(scaled_model)

        # check vars
        self.assertAlmostEqual(pe.value(unscaled_model.x[1]), pe.value(scaled_model.scaled_x[1])/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.x[2]), pe.value(scaled_model.scaled_x[2])/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.x[3]), pe.value(scaled_model.scaled_x[3])/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.z), pe.value(scaled_model.scaled_z)/z_scale, 4)
        # check var lb
        self.assertAlmostEqual(pe.value(unscaled_model.x[1].lb), pe.value(scaled_model.scaled_x[1].lb)/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.x[2].lb), pe.value(scaled_model.scaled_x[2].lb)/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.x[3].lb), pe.value(scaled_model.scaled_x[3].lb)/x_scale, 4)
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(pe.value(unscaled_model.z.lb), pe.value(scaled_model.scaled_z.ub)/z_scale, 4)
        # check var ub
        self.assertAlmostEqual(pe.value(unscaled_model.x[1].ub), pe.value(scaled_model.scaled_x[1].ub)/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.x[2].ub), pe.value(scaled_model.scaled_x[2].ub)/x_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.x[3].ub), pe.value(scaled_model.scaled_x[3].ub)/x_scale, 4)
        # note: z_scale is negative, therefore, the inequality directions swap
        self.assertAlmostEqual(pe.value(unscaled_model.z.ub), pe.value(scaled_model.scaled_z.lb)/z_scale, 4)
        # check var multipliers (rc)
        self.assertAlmostEqual(pe.value(unscaled_model.rc[unscaled_model.x[1]]), pe.value(scaled_model.rc[scaled_model.scaled_x[1]])*x_scale/obj_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.rc[unscaled_model.x[2]]), pe.value(scaled_model.rc[scaled_model.scaled_x[2]])*x_scale/obj_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.rc[unscaled_model.x[3]]), pe.value(scaled_model.rc[scaled_model.scaled_x[3]])*x_scale/obj_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.rc[unscaled_model.z]), pe.value(scaled_model.rc[scaled_model.scaled_z])*z_scale/obj_scale, 4)
        # check constraint multipliers
        self.assertAlmostEqual(pe.value(unscaled_model.dual[unscaled_model.con[1]]),pe.value(scaled_model.dual[scaled_model.scaled_con[1]])*con_scale1/obj_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.dual[unscaled_model.con[2]]),pe.value(scaled_model.dual[scaled_model.scaled_con[2]])*con_scale2/obj_scale, 4)
        self.assertAlmostEqual(pe.value(unscaled_model.dual[unscaled_model.con[3]]),pe.value(scaled_model.dual[scaled_model.scaled_con[3]])*con_scale3/obj_scale, 4)

        # put the solution from the scaled back into the original
        pe.TransformationFactory('core.scale_model').propagate_solution(scaled_model, model)

        # compare var values and rc with the unscaled soln
        for vm in model.component_objects(ctype=pe.Var, descend_into=True):
            cuid = pe.ComponentUID(vm)
            vum = cuid.find_component_on(unscaled_model)
            self.assertEqual((vm in model.rc), (vum in unscaled_model.rc)) 
            if vm in model.rc:
                self.assertAlmostEqual(pe.value(model.rc[vm]), pe.value(unscaled_model.rc[vum]), 4)
            for k in vm:
                vmk = vm[k]
                vumk = vum[k]
                self.assertAlmostEqual(pe.value(vmk), pe.value(vumk), 4)
                self.assertEqual((vmk in model.rc), (vumk in unscaled_model.rc)) 
                if vmk in model.rc:
                    self.assertAlmostEqual(pe.value(model.rc[vmk]), pe.value(unscaled_model.rc[vumk]), 4)

        # compare constraint duals and value
        for model_con in model.component_objects(ctype=pe.Constraint, descend_into=True):
            cuid = pe.ComponentUID(model_con)
            unscaled_model_con = cuid.find_component_on(unscaled_model)
            self.assertEqual((model_con in model.rc), (unscaled_model_con in unscaled_model.rc)) 
            if model_con in model.dual:
                self.assertAlmostEqual(pe.value(model.dual[model_con]), pe.value(unscaled_model.dual[unscaled_model_con]), 4)
            for k in model_con:
                mk = model_con[k]
                umk = unscaled_model_con[k]
                self.assertEqual((mk in model.dual), (umk in unscaled_model.dual)) 
                if mk in model.dual:
                    self.assertAlmostEqual(pe.value(model.dual[mk]), pe.value(unscaled_model.dual[umk]), 4)
        
if __name__ == "__main__":
    unittest.main()
