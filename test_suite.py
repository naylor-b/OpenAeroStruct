from __future__ import division, print_function
import sys
from time import time
import numpy as np
import unittest

from OpenAeroStruct import OASProblem

class TestAero(unittest.TestCase):

    currentResult = []

    def run(self, result=None):
        self.currentResult.append(result) # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method

    def test_aero_analysis_flat(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : False})
        OAS_prob.add_surface({'span_cos_spacing' : 0})
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob

        self.assertAlmostEqual(prob['wing_perf.CL'], .46173591841167, places=5)
        self.assertAlmostEqual(prob['wing_perf.CD'], .005524603647, places=5)

    def test_aero_analysis_flat_multiple(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : False})
        OAS_prob.add_surface({'span_cos_spacing' : 0.})
        OAS_prob.add_surface({'name' : 'tail',
                              'span_cos_spacing' : 0.,
                              'offset' : np.array([0., 0., 1000000.])})
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], .46173591841167, places=5)
        self.assertAlmostEqual(prob['tail_perf.CL'], .46173591841167, places=5)

    def test_aero_analysis_flat_side_by_side(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : False})
        OAS_prob.add_surface({'name' : 'wing',
                              'span' : 5.,
                              'num_y' : 3,
                              'span_cos_spacing' : 0.,
                              'symmetry' : False,
                              'offset' : np.array([0., -2.5, 0.])})
        OAS_prob.add_surface({'name' : 'tail',
                              'span' : 5.,
                              'num_y' : 3,
                              'span_cos_spacing' : 0.,
                              'symmetry' : False,
                              'offset' : np.array([0., 2.5, 0.])})
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], 0.46173591841167183, places=5)
        self.assertAlmostEqual(prob['tail_perf.CL'], 0.46173591841167183, places=5)
        self.assertAlmostEqual(prob['wing_perf.CD'], .005524603647, places=5)
        self.assertAlmostEqual(prob['tail_perf.CD'], .005524603647, places=5)

    def test_aero_analysis_flat_full(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : False})
        surf_dict = {'symmetry' : False}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], .45655138, places=5)
        self.assertAlmostEqual(prob['wing_perf.CD'], 0.0055402121081108589, places=5)

    def test_aero_analysis_flat_viscous_full(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : False,
                               'with_viscous' : True})
        surf_dict = {'symmetry' : False}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], .45655138, places=5)
        self.assertAlmostEqual(prob['wing_perf.CD'], 0.018942466133780547, places=5)

    def test_aero_optimization(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : True})
        OAS_prob.add_surface()
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)
        OAS_prob.add_desvar('wing.sweep', lower=10., upper=30.)
        OAS_prob.add_desvar('wing.dihedral', lower=-10., upper=20.)
        OAS_prob.add_desvar('wing.taper', lower=.5, upper=2.)
        OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
        OAS_prob.add_objective('wing_perf.CD', scaler=1e4)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CD'], 0.0040503761683420006, places=5)

    def test_aero_optimization_chord_monotonic(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : True,
                               'with_viscous' : False})
        OAS_prob.add_surface({
        'chord_dist' : 'random',
        'num_y' : 21,
        'monotonic_taper' : True,
        'span_cos_spacing' : 0.,
        })
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.chord_dist_cp', lower=0.1, upper=5.)
        OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
        OAS_prob.add_constraint('wing_perf.CL', equals=0.1)
        OAS_prob.add_constraint('wing.S_ref', equals=10)
        OAS_prob.add_constraint('wing.monotonic', upper=0.)
        OAS_prob.add_objective('wing_perf.CD', scaler=1e4)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CD'], 0.00060238294097975553, places=5)
        self.assertAlmostEqual(prob['wing.monotonic'][0], -0.88962717780055134, places=4)

    def test_aero_viscous_optimization(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : True,
                               'with_viscous' : True})
        OAS_prob.add_surface()
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.twist_cp', lower=-10., upper=15.)
        OAS_prob.add_desvar('wing.sweep', lower=10., upper=30.)
        OAS_prob.add_desvar('wing.dihedral', lower=-10., upper=20.)
        OAS_prob.add_desvar('wing.taper', lower=.5, upper=2.)
        OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
        OAS_prob.add_objective('wing_perf.CD', scaler=1e4)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CD'], 0.016916491728620451, places=5)

    def test_aero_viscous_chord_optimization(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : True,
                               'with_viscous' : True})
        OAS_prob.add_surface()
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.chord_dist_cp', lower=0.1, upper=3.)
        OAS_prob.add_constraint('wing_perf.CL', equals=0.5)
        OAS_prob.add_constraint('wing.S_ref', equals=10)
        OAS_prob.add_objective('wing_perf.CD', scaler=1e4)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CD'], 0.019326203641337643, places=5)

    def test_aero_multiple_opt(self):
        OAS_prob = OASProblem({'type' : 'aero',
                               'optimize' : True})
        surf_dict = { 'name' : 'wing',
                      'span' : 5.,
                      'num_y' : 3,
                      'span_cos_spacing' : 0.}
        OAS_prob.add_surface(surf_dict)
        surf_dict.update({'name' : 'tail',
                       'offset' : np.array([0., 0., 10.])})
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()

        OAS_prob.add_desvar('tail.twist_cp', lower=-10., upper=15.)
        OAS_prob.add_desvar('tail.sweep', lower=10., upper=30.)
        OAS_prob.add_desvar('tail.dihedral', lower=-10., upper=20.)
        OAS_prob.add_desvar('tail.taper', lower=.5, upper=2.)
        OAS_prob.add_constraint('tail_perf.CL', equals=0.5)
        OAS_prob.add_objective('tail_perf.CD', scaler=1e4)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], .41543435621928004, places=4)
        self.assertAlmostEqual(prob['tail_perf.CL'], .5, places=5)
        self.assertAlmostEqual(prob['wing_perf.CD'], .0075400306289957033, places=5)
        self.assertAlmostEqual(prob['tail_perf.CD'], 0.0079322194100007078, places=5)


class TestStruct(unittest.TestCase):

    currentResult = []

    def run(self, result=None):
        self.currentResult.append(result) # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method

    def test_struct_analysis(self):
        OAS_prob = OASProblem({'type' : 'struct',
                               'optimize' : False})
        surf_dict = {'symmetry' : False}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing.weight'], 2080.284115390823, places=3)

    def test_struct_analysis_symmetry(self):
        OAS_prob = OASProblem({'type' : 'struct',
                               'optimize' : False})
        surf_dict = {'symmetry' : True}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing.weight'], 2080.284115390823, places=3)

    def test_struct_optimization(self):
        OAS_prob = OASProblem({'type' : 'struct',
                               'optimize' : True})
        OAS_prob.add_surface({'symmetry' : False})
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing.failure', upper=0.)
        OAS_prob.add_objective('wing.weight', scaler=1e-3)

        OAS_prob.run()
        prob = OAS_prob.prob

        self.assertAlmostEqual(prob['wing.weight'], 535.93857888840353, places=2)

    def test_struct_optimization_symmetry(self):
        OAS_prob = OASProblem({'type' : 'struct',
                               'optimize' : True})
        OAS_prob.add_surface()
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.thickness_cp', lower=0.001, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing.failure', upper=0.)
        OAS_prob.add_objective('wing.weight', scaler=1e-3)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing.weight'], 532.76522856377937, places=2)


class TestAeroStruct(unittest.TestCase):

    currentResult = []

    def run(self, result=None):
        self.currentResult.append(result) # remember result for use in tearDown
        unittest.TestCase.run(self, result) # call superclass run method

    def test_aerostruct_analysis(self):
        OAS_prob = OASProblem({'type' : 'aerostruct',
                               'optimize' : False})
        surf_dict = {'num_y' : 13,
                  'num_x' : 2,
                  'wing_type' : 'CRM',
                  'CL0' : 0.2,
                  'CD0' : 0.015,
                  'symmetry' : False}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], 0.73065749603248542)
        self.assertAlmostEqual(prob['wing_perf.failure'], -0.58779315534201515, places=5)
        self.assertAlmostEqual(prob['fuelburn'], 126201.90559473804, places=2)

    def test_aerostruct_analysis_symmetry(self):
        OAS_prob = OASProblem({'type' : 'aerostruct',
                               'optimize' : False})
        surf_dict = {'symmetry' : True,
                  'num_y' : 13,
                  'num_x' : 2,
                  'wing_type' : 'CRM',
                  'CL0' : 0.2,
                  'CD0' : 0.015}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()
        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['wing_perf.CL'], 0.76727243345999141)
        self.assertAlmostEqual(prob['wing_perf.failure'], -0.6066768432330637, places=5)
        self.assertAlmostEqual(prob['fuelburn'], 136257.17861685093, places=2)

    def test_aerostruct_optimization(self):
        OAS_prob = OASProblem({'type' : 'aerostruct',
                               'optimize' : True})
        surf_dict = {'num_y' : 7,
                  'num_x' : 2,
                  'wing_type' : 'CRM',
                  'CL0' : 0.2,
                  'CD0' : 0.015,
                  'symmetry' : False}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.thickness_cp', lower=0.01, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing_perf.failure', upper=0.)
        OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
        OAS_prob.add_constraint('eq_con', equals=0.)
        OAS_prob.add_objective('fuelburn', scaler=1e-5)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['fuelburn'], 76043.052616603629, places=0)
        self.assertAlmostEqual(prob['wing_perf.failure'], 0., places=4)

    def test_aerostruct_optimization_symmetry(self):
        OAS_prob = OASProblem({'type' : 'aerostruct',
                               'optimize' : True})
        surf_dict = {'symmetry' : True,
                  'num_y' : 7,
                  'num_x' : 3,
                  'wing_type' : 'CRM',
                  'CL0' : 0.2,
                  'CD0' : 0.015}
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()

        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.thickness_cp', lower=0.01, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing_perf.failure', upper=0.)
        OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
        OAS_prob.add_constraint('eq_con', equals=0.)
        OAS_prob.add_objective('fuelburn', scaler=1e-4)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['fuelburn'], 80456.958369551561, places=0)
        self.assertAlmostEqual(prob['wing_perf.failure'], 0, places=5)

    def test_aerostruct_optimization_symmetry_multiple(self):
        OAS_prob = OASProblem({'type' : 'aerostruct',
                               'optimize' : True})
        surf_dict = {'name' : 'wing',
                     'symmetry' : True,
                     'num_y' : 5,
                     'num_x' : 2,
                     'wing_type' : 'CRM',
                     'CL0' : 0.2,
                     'CD0' : 0.015,
                     'num_twist' : 2,
                     'num_thickness' : 2}
        OAS_prob.add_surface(surf_dict)
        surf_dict.update({'name' : 'tail',
                          'offset':np.array([0., 0., 1.e7])})
        OAS_prob.add_surface(surf_dict)
        OAS_prob.setup()

        # Add design variables and constraints for both the wing and tail
        OAS_prob.add_desvar('wing.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('wing.thickness_cp', lower=0.01, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('wing_perf.failure', upper=0.)
        OAS_prob.add_desvar('tail.twist_cp', lower=-15., upper=15.)
        OAS_prob.add_desvar('tail.thickness_cp', lower=0.01, upper=0.25, scaler=1e2)
        OAS_prob.add_constraint('tail_perf.failure', upper=0.)

        OAS_prob.add_desvar('alpha', lower=-10., upper=10.)
        OAS_prob.add_constraint('eq_con', equals=0.)
        OAS_prob.add_objective('fuelburn', scaler=1e-5)

        OAS_prob.run()
        prob = OAS_prob.prob
        self.assertAlmostEqual(prob['fuelburn'], 181838.09546516923, places=1)
        self.assertAlmostEqual(prob['wing_perf.failure'], 0, places=5)
        self.assertAlmostEqual(np.linalg.norm(prob['wing.twist_cp']), np.linalg.norm(prob['tail.twist_cp']), places=1)


if __name__ == "__main__":

    # Get user-inputted argument if provided
    try:
        arg = sys.argv[1]
        arg_provided = True
    except:
        arg_provided = False

    # Based on user input, run one subgroup of tests
    if arg_provided:
        if 'aero' == arg:
            test_classes = [TestAero]
        elif 'struct' == arg:
            test_classes = [TestStruct]
        elif 'aerostruct' == arg:
            test_classes = [TestAeroStruct]
    else:
        arg = 'full'
        test_classes = [TestAero, TestStruct, TestAeroStruct]

    print()
    print('+==================================================+')
    print('             Running ' + arg + ' test suite')
    print('+==================================================+')
    print()

    failures = []
    errors = []

    for test_class in test_classes:

        # Set up the test suite and run the tests corresponding to this subgroup
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unittest.TextTestRunner().run(suite)

        failures.extend(test_class.currentResult[-1].failures)
        errors.extend(test_class.currentResult[-1].errors)

    if len(failures) or len(errors):
        sys.exit(1)
