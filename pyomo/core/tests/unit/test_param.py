#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# Unit Tests for Param() Objects
#
# PyomoModel                Base test class
# SimpleParam                Test scalar parameter
# ArrayParam1                Test arrays of parameters
# ArrayParam2                Test arrays of parameter with explicit zero default
# ArrayParam3                Test arrays of parameter with nonzero default
# TestIO                Test initialization from an AMPL *.dat file
#

import math
import os
import sys

import pyutilib.services
import pyutilib.th as unittest

from pyomo.environ import *

from six import iteritems, itervalues, StringIO

class ParamTester(object):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, no default
        #
        self.model.Z = Set(initialize=[1,3])
        self.model.A = Param(self.model.Z, **kwds)
        self.instance = self.model.create_instance()

        self.expectTextDomainError = False
        self.expectNegativeDomainError = False

    def tearDown(self):
        self.model = None
        self.instance = None

    def validateDict(self, ref, test):
        test = dict(test)
        ref = dict(ref)
        self.assertEqual( len(test), len(ref) )
        for key in test.keys():
            self.assertTrue( key in ref )
            if ref[key] is None:
                self.assertTrue( test[key] is None or test[key].value is None )
            else:
                self.assertEqual( ref[key], value( test[key] ) )

    def test_value(self):
        if self.instance.A.is_indexed():
            self.assertRaises(TypeError, value, self.instance.A)
            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)

        if self.instance.A._default_val is None:
            val_list = self.sparse_data.items()
        else:
            val_list = self.data.items()

        for key, val in val_list:
            if key is None:
                continue
            tmp = value(self.instance.A[key])
            self.assertEqual(type(tmp), type(val))
            self.assertEqual(tmp, val)

            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)

    def test_call(self):
        #"""Check the use of the __call__ method"""
        self.assertRaises(TypeError, self.instance.A)

    def test_get_valueattr(self):
        try:
            tmp = self.instance.A.value
            self.fail("Array Parameters should not contain a value")
        except AttributeError:
            pass

    # JDS: I would like this test to work, but there is no way to
    # prevent a user from adding new attributes to the (indexed) Param
    # instance.
    #
    #def test_set_valueattr(self):
    #    try:
    #        self.instance.A.value = 4.3
    #        self.fail("Array Parameters should not contain a value")
    #    except AttributeError:
    #        pass

    def test_set_value(self):
        try:
            self.instance.A = 4.3
            self.fail("Array Parameters should not be settable")
        except ValueError:
            pass

    def test_getitem(self):
        for key, val in iteritems(self.data):
            try:
                test = self.instance.A[key]
                self.assertEqual( value(test), val )
            except ValueError:
                if val is not None:
                    raise

    def test_setitem_index_error(self):
        try:
            self.instance.A[2] = 4.3
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            self.fail("Expected KeyError because 2 is not a valid key")
        except KeyError:
            pass
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

    def test_setitem_preexisting(self):
        keys = self.instance.A.sparse_keys()
        if not keys or None in keys:
            return

        idx = sorted(keys)[0]
        self.assertEqual(self.instance.A[idx], self.data[idx])
        if self.instance.A._mutable:
            self.assertTrue( isinstance( self.instance.A[idx],
                                         pyomo.core.base.param._ParamData ) )
        else:
            self.assertEqual(type(self.instance.A[idx]), float)

        try:
            self.instance.A[idx] = 4.3
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            self.assertEqual( self.instance.A[idx], 4.3)
            self.assertTrue( isinstance(self.instance.A[idx],
                                        pyomo.core.base.param._ParamData ) )
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

        try:
            self.instance.A[idx] = -4.3
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            if self.expectNegativeDomainError:
                self.fail("Expected setitem[%s] to fail with negative data"
                          % (idx,))
            self.assertEqual( self.instance.A[idx], -4.3 )
        except ValueError:
            if not self.expectNegativeDomainError:
                self.fail(
                    "Unexpected exception (%s) for setitem[%s] = negative data"
                    % ( str(sys.exc_info()[1]), idx ) )
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

        try:
            self.instance.A[idx] = 'x'
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            if self.expectTextDomainError:
                self.fail("Expected setitem[%s] to fail with text data",
                          (idx,))
            self.assertEqual( value(self.instance.A[idx]), 'x' )
        except ValueError:
            if not self.expectTextDomainError:
                self.fail(
                    "Unexpected exception (%s) for setitem[%s] with text data"
                    % ( str(sys.exc_info()[1]), idx ) )
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

    def test_setitem_default_override(self):
        sparse_keys = set(self.instance.A.sparse_keys())
        keys = sorted(self.instance.A.keys())
        if len(keys) == len(sparse_keys):
            # No default value possible
            return
        if self.instance.A._default_val is None:
            # No default value defined
            return

        while True:
            idx = keys.pop(0)
            if not idx in sparse_keys:
                break

        self.assertEqual( value(self.instance.A[idx]),
                          self.instance.A._default_val )
        if self.instance.A._mutable:
            self.assertEqual( type(self.instance.A[idx]),
                              pyomo.core.base.param._ParamData )
        else:
            self.assertEqual(type(self.instance.A[idx]),
                             type(value(self.instance.A._default_val)))

        try:
            self.instance.A[idx] = 4.3
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            self.assertEqual( self.instance.A[idx], 4.3)
            self.assertEqual( type(self.instance.A[idx]),
                              pyomo.core.base.param._ParamData )
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

        try:
            self.instance.A[idx] = -4.3
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            if self.expectNegativeDomainError:
                self.fail("Expected setitem[%s] to fail with negative data"
                          % (idx,))
            self.assertEqual( self.instance.A[idx], -4.3 )
        except ValueError:
            if not self.expectNegativeDomainError:
                self.fail(
                    "Unexpected exception (%s) for setitem[%s] = negative data"
                    % ( str(sys.exc_info()[1]), idx ) )
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

        try:
            self.instance.A[idx] = 'x'
            if not self.instance.A._mutable:
                self.fail("Expected setitem[%s] to fail for immutable Params"
                          % (idx,))
            if self.expectTextDomainError:
                self.fail("Expected setitem[%s] to fail with text data"
                          % (idx,))
            self.assertEqual( value(self.instance.A[idx]), 'x' )
        except ValueError:
            if not self.expectTextDomainError:
                self.fail(
                    "Unexpected exception (%s) for setitem[%s] with text data"
                    % ( str(sys.exc_info()[1]), idx) )
        except TypeError:
            # immutable Params should raise a TypeError exception
            if self.instance.A._mutable:
                raise

    def test_dim(self):
        key = list(self.data.keys())[0]
        try:
            key = tuple(key)
        except TypeError:
            key = (key,)
        self.assertEqual( self.instance.A.dim(), len(key))

    def test_is_indexed(self):
        self.assertTrue(self.instance.A.is_indexed())

    def test_keys(self):
        test = self.instance.A.keys()
        #self.assertEqual( type(test), list )
        if self.instance.A._default_val is None:
            self.assertEqual( sorted(test), sorted(self.sparse_data.keys()) )
        else:
            self.assertEqual( sorted(test), sorted(self.data.keys()) )

    def test_values(self):
        expectException = False
        #    len(self.sparse_data) < len(self.data) and \
        #    not self.instance.A._mutable
        try:
            test = self.instance.A.values()
            #self.assertEqual( type(test), list )
            test = zip(self.instance.A.keys(), test)
            if self.instance.A._default_val is None:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_items(self):
        expectException = False
        #                  len(self.sparse_data) < len(self.data) and \
        #                  not self.instance.A._default_val is None and \
        #                  not self.instance.A._mutable
        try:
            test = self.instance.A.items()
            #self.assertEqual( type(test), list )
            if self.instance.A._default_val is None:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_iterkeys(self):
        test = self.instance.A.iterkeys()
        self.assertEqual( sorted(test), sorted(self.instance.A.keys()) )

    def test_itervalues(self):
        expectException = False
        #                  len(self.sparse_data) < len(self.data) and \
        #                  not self.instance.A._default_val is None and \
        #                  not self.instance.A._mutable
        try:
            test = itervalues(self.instance.A)
            test = zip(self.instance.A.keys(), test)
            if self.instance.A._default_val is None:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_iteritems(self):
        expectException = False
        #                  len(self.sparse_data) < len(self.data) and \
        #                  not self.instance.A._default_val is None and \
        #                  not self.instance.A._mutable
        try:
            test = iteritems(self.instance.A)
            if self.instance.A._default_val is None:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise


    def test_sparse_keys(self):
        test = self.instance.A.sparse_keys()
        self.assertEqual( type(test), list )
        self.assertEqual( sorted(test), sorted(self.sparse_data.keys()) )

    def test_sparse_values(self):
        #self.instance.pprint()
        test = self.instance.A.sparse_values()
        self.assertEqual( type(test), list )
        #print test
        #print self.sparse_data.items()
        test = zip(self.instance.A.keys(), test)
        self.validateDict(self.sparse_data.items(), test)

    def test_sparse_items(self):
        test = self.instance.A.sparse_items()
        self.assertEqual( type(test), list )
        self.validateDict(self.sparse_data.items(), test)


    def test_sparse_iterkeys(self):
        test = self.instance.A.sparse_iterkeys()
        self.assertEqual( sorted(test), sorted(self.sparse_data.keys()) )

    def test_sparse_itervalues(self):
        test = self.instance.A.sparse_itervalues()
        test = zip(self.instance.A.keys(), test)
        self.validateDict(self.sparse_data.items(), test)

    def test_sparse_iteritems(self):
        test = self.instance.A.sparse_iteritems()
        self.validateDict(self.sparse_data.items(), test)


    def test_len(self):
        #"""Check the use of len"""
        if self.instance.A._default_val is None:
            self.assertEqual( len(self.instance.A), len(self.sparse_data) )
            self.assertEqual( len(list(self.instance.A.keys())), len(self.sparse_data) )
        else:
            self.assertEqual( len(self.instance.A), len(self.data) )
            self.assertEqual( len(list(self.instance.A.keys())), len(self.data) )
        self.assertEqual( len(list(self.instance.A.sparse_keys())), len(self.sparse_data) )

    def test_index(self):
        #"""Check the use of index"""
        self.assertEqual( len(self.instance.A.index_set()), len(list(self.data.keys())) )

    def test_get_default(self):
        if len(self.sparse_data) == len(self.data):
            # nothing to test
            return
        idx = list(set(self.data) - set(self.sparse_data))[0]
        expectException = self.instance.A._default_val is None and \
                          not self.instance.A._mutable
        try:
            test = self.instance.A[idx]
            if expectException:
                self.fail("Expected the test to raise an exception")
            self.assertFalse(expectException)
            expectException = self.instance.A._default_val is None
            try:
                ans = value(test)
                self.assertEquals(ans, value(self.instance.A._default_val))
                self.assertFalse(expectException)
            except:
                if not expectException:
                    raise
        except ValueError:
            if not expectException:
                raise


class ArrayParam_mutable_sparse_noDefault\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, no default
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize={1:1.3}, **kwds)

        self.sparse_data = {1:1.3}
        self.data = {1:1.3, 3:None}

class ArrayParam_mutable_sparse_intDefault\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, int default
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize={1:1.3}, default=0, **kwds)

        self.sparse_data = {1:1.3}
        self.data = {1:1.3, 3:0}


class ArrayParam_mutable_sparse_floatDefault\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, float default
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize={1:1.3}, default=99.5, **kwds)

        self.sparse_data = {1:1.3}
        self.data = {1:1.3, 3:99.5}


class ArrayParam_mutable_dense_intDefault_scalarInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Dense single-index Param, float default, init with scalar
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize=1.3, default=99.5, **kwds)

        self.sparse_data = {1:1.3, 3:1.3}
        self.data = self.sparse_data

class ArrayParam_mutable_dense_intDefault_scalarParamInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Dense single-index Param, float default, init with scalar
        #
        self.model = AbstractModel()
        self.model.p = Param(initialize=1.3)
        ParamTester.setUp(self, mutable=True, initialize=self.model.p, default=99.5, **kwds)

        self.sparse_data = {1:1.3, 3:1.3}
        self.data = self.sparse_data

class ArrayParam_mutable_dense_intDefault_sparseParamInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Dense single-index Param, float default, init with scalar
        #
        self.model = AbstractModel()
        self.model.p = Param([1,3], initialize={1:1.3}, default=9.5)
        ParamTester.setUp(self, mutable=True, initialize=self.model.p, default=99.5, **kwds)

        self.sparse_data = {1:1.3, 3:9.5}
        self.data = self.sparse_data

class ArrayParam_mutable_dense_intDefault_denseParamInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Dense single-index Param, float default, init with scalar
        #
        self.model = AbstractModel()
        self.model.p = Param([1,3], initialize={1:1.3, 3:2.3})
        ParamTester.setUp(self, mutable=True, initialize=self.model.p, default=99.5, **kwds)

        self.sparse_data = {1:1.3, 3:2.3}
        self.data = self.sparse_data


class ArrayParam_mutable_dense_intDefault_dictInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        def A_init(model, i):
            return 1.5+i
        #
        # Dense single-index Param, no default, init with rule
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize=A_init, **kwds)

        self.sparse_data = {1:2.5, 3:4.5}
        self.data = self.sparse_data


class ArrayParam_mutable_dense_intDefault_ruleInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        def A_init(model):
            return {1:2.5, 3:4.5}
        #
        # Dense single-index Param, no default, init with rule
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=True, initialize=A_init, **kwds)

        self.sparse_data = {1:2.5, 3:4.5}
        self.data = self.sparse_data


class ArrayParam_immutable_sparse_noDefault\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, no default
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=False, initialize={1:1.3}, **kwds)

        self.sparse_data = {1:1.3}
        self.data = {1:1.3, 3:None}


class ArrayParam_immutable_sparse_intDefault\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, int default
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=False, initialize={1:1.3}, default=0, **kwds)

        self.sparse_data = {1:1.3}
        self.data = {1:1.3, 3:0}


class ArrayParam_immutable_sparse_floatDefault\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, float default
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=False, initialize={1:1.3}, default=99.5, **kwds)

        self.sparse_data = {1:1.3}
        self.data = {1:1.3, 3:99.5}


class ArrayParam_immutable_dense_intDefault_scalarInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Dense single-index Param, float default, init with scalar
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=False, initialize=1.3, default=99.5, **kwds)

        self.sparse_data = {1:1.3, 3:1.3}
        self.data = self.sparse_data



class ArrayParam_immutable_dense_intDefault_scalarParamInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Dense single-index Param, float default, init with scalar
        #
        self.model = AbstractModel()
        self.model.p = Param(initialize=1.3)
        ParamTester.setUp(self, mutable=False, initialize=self.model.p, default=99.5, **kwds)

        self.sparse_data = {1:1.3, 3:1.3}
        self.data = self.sparse_data


class ArrayParam_immutable_dense_intDefault_dictInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        def A_init(model, i):
            return 1.5+i
        #
        # Dense single-index Param, no default, init with rule
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=False, initialize=A_init, **kwds)

        self.sparse_data = {1:2.5, 3:4.5}
        self.data = self.sparse_data


class ArrayParam_immutable_dense_intDefault_ruleInit\
          (ParamTester, unittest.TestCase):

    def setUp(self, **kwds):
        def A_init(model):
            return {1:2.5, 3:4.5}
        #
        # Dense single-index Param, no default, init with rule
        #
        self.model = AbstractModel()
        ParamTester.setUp(self, mutable=False, initialize=A_init, **kwds)

        self.sparse_data = {1:2.5, 3:4.5}
        self.data = self.sparse_data


class ArrayParam6(unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Create Model
        #
        self.model = AbstractModel()
        self.repn = '_bogus_'
        self.instance = None

    def tearDown(self):
        self.model = None
        self.instance = None

    def test_index1(self):
        self.model.A = Set(initialize=range(0,4))
        def B_index(model):
            for i in model.A:
                if i%2 == 0:
                    yield i
        def B_init(model, i, j):
            if j:
                return 2+i
            return -(2+i)
        self.model.B = Param( B_index, [True,False],
                              initialize=B_init)
        self.instance = self.model.create_instance()
        #self.instance.pprint()
        self.assertEqual(set(self.instance.B.keys()),
                         set([(0,True),(2,True),(0,   False),(2,False)]))
        self.assertEqual(self.instance.B[0,True],2)
        self.assertEqual(self.instance.B[0,False],-2)
        self.assertEqual(self.instance.B[2,True],4)
        self.assertEqual(self.instance.B[2,False],-4)

    def test_index2(self):
        self.model.A = Set(initialize=range(0,4))
        @set_options(dimen=3)
        def B_index(model):
            return [(i,2*i,i*i) for i in model.A if i%2 == 0]
        def B_init(model, i, ii, iii, j):
            if j:
                return 2+i
            return -(2+i)
        self.model.B = Param(B_index, [True,False], initialize=B_init)
        self.instance = self.model.create_instance()
        #self.instance.pprint()
        self.assertEqual(set(self.instance.B.keys()),set([(0,0,0,True),(2,4,4,True),(0,0,0,False),(2,4,4,False)]))
        self.assertEqual(self.instance.B[0,0,0,True],2)
        self.assertEqual(self.instance.B[0,0,0,False],-2)
        self.assertEqual(self.instance.B[2,4,4,True],4)
        self.assertEqual(self.instance.B[2,4,4,False],-4)

    def test_index3(self):
        self.model.A = Set(initialize=range(0,4))
        def B_index(model):
            return [(i,2*i,i*i) for i in model.A if i%2 == 0]
        def B_init(model, i, ii, iii, j):
            if j:
                return 2+i
            return -(2+i)
        self.model.B = Param(B_index, [True,False], initialize=B_init)
        try:
            self.instance = self.model.create_instance()
            self.fail("Expected ValueError because B_index returns a tuple")
        except ValueError:
            pass

    def test_index4(self):
        self.model.A = Set(initialize=range(0,4))
        @set_options(within=Integers)
        def B_index(model):
            return [i/2.0 for i in model.A]
        def B_init(model, i, j):
            if j:
                return 2+i
            return -(2+i)
        self.model.B = Param(B_index, [True,False], initialize=B_init)
        try:
            self.instance = self.model.create_instance()
            self.fail("Expected ValueError because B_index returns invalid index values")
        except ValueError:
            pass

    def test_dimen1(self):
        model=AbstractModel()
        model.A = Set(dimen=2, initialize=[(1,2),(3,4)])
        model.B = Set(dimen=3, initialize=[(1,1,1),(2,2,2),(3,3,3)])
        model.C = Set(dimen=1, initialize=[9,8,7,6,5])
        model.x = Param(model.A, model.B, model.C, initialize=-1)
        #model.y = Param(model.B, initialize=(1,1))
        model.y = Param(model.B, initialize=((1,1,7),2))
        instance=model.create_instance()
        self.assertEqual( instance.x.dim(), 6)
        self.assertEqual( instance.y.dim(), 3)

    def test_setitem(self):
        model = ConcreteModel()
        model.a = Set(initialize=[1,2,3])
        model.b = Set(initialize=['a','b','c'])
        model.c = model.b * model.b
        model.p = Param(model.a, model.c, within=NonNegativeIntegers, default=0, mutable=True)
        #print(model.p._index.keys())
        model.p[1,'a','b'] = 1
        model.p[1,('a','b')] = 1
        model.p[(1,'b'),'b'] = 1
        try:
            model.p[1,5,7] = 1
            self.fail("Expected KeyError")
        except KeyError:
            pass


class ScalarTester(ParamTester):

    def setUp(self, **kwds):
        #
        # "Sparse" scalar Param, no default
        #
        self.model.Z = Set(initialize=[1,3])
        self.model.A = Param(**kwds)
        self.instance = self.model.create_instance()
        #self.instance.pprint()

        self.expectTextDomainError = False
        self.expectNegativeDomainError = False


    def test_value_scalar(self):
        #"""Check the value of the parameter"""
        if self.sparse_data.get(None,None) is None:
            self.assertRaises(ValueError, value, self.instance.A)
            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)
        else:

            val = self.data[None]
            tmp = value(self.instance.A)
            self.assertEqual( type(tmp), type(val))
            self.assertEqual( tmp, val )

            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)

    def test_call(self):
        #"""Check the use of the __call__ method"""
        if self.sparse_data.get(None,None) is None: #not self.sparse_data:
            self.assertRaisesRegexp(
                ValueError, ".*undefined and no default value",
                self.instance.A.__call__ )
        else:
            self.assertEqual(self.instance.A(), self.data[None])

    def test_get_valueattr(self):
        self.assertEqual(self.instance.A.value, self.data[None])

    def test_set_valueattr(self):
        self.instance.A.value = 4.3
        self.assertEqual(self.instance.A.value, 4.3)
        if not self.sparse_data:
            self.assertRaises(ValueError, self.instance.A)
        else:
            self.assertEqual(self.instance.A(), 4.3)

    def test_get_value(self):
        if not self.sparse_data or (None in self.sparse_data and self.sparse_data[None] is None):
            try:
                value(self.instance.A)
                self.fail("Expected value error")
            except ValueError:
                pass
        else:
            self.assertEqual( self.instance.A, self.data[None])

    def test_set_value(self):
        self.instance.A = 4.3
        self.assertEqual(self.instance.A.value, 4.3)
        self.assertEqual(self.instance.A(), 4.3)


    def test_is_indexed(self):
        self.assertFalse(self.instance.A.is_indexed())

    def test_dim(self):
        #"""Check the use of dim"""
        self.assertEqual( self.instance.A.dim(), 0)


class ScalarParam_mutable_noDefault(ScalarTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, no default
        #
        self.model = AbstractModel()
        ScalarTester.setUp(self, mutable=True, **kwds)

        self.sparse_data = {None:None}
        self.data = {None:None}


class ScalarParam_mutable_init(ScalarTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, no default
        #
        self.model = AbstractModel()
        ScalarTester.setUp(self, mutable=True, initialize=1.3, **kwds)

        self.sparse_data = {None:1.3}
        self.data = {None:1.3}


class ScalarParam_mutable_floatDefault(ScalarTester, unittest.TestCase):

    def setUp(self, **kwds):
        #
        # Sparse single-index Param, no default
        #
        self.model = AbstractModel()
        ScalarTester.setUp(self, mutable=True, default=1.3, **kwds)

        self.sparse_data = {None:1.3}
        self.data = {None:1.3}


class TestIO(unittest.TestCase):

    def setUp(self):
        #
        # Create Model
        #
        self.model = AbstractModel()
        self.instance = None

    def tearDown(self):
        if os.path.exists("param.dat"):
            os.remove("param.dat")
        self.model = None
        self.instance = None

    def test_io1(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "param A := 3.3;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.A=Param()
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( value(self.instance.A), 3.3 )

    def test_io2(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "set Z := 1 3 5;\n" )
        OUTPUT.write( "param A :=\n" )
        OUTPUT.write( "1 2.2\n" )
        OUTPUT.write( "3 2.3\n" )
        OUTPUT.write( "5 2.5;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.Z=Set()
        self.model.A=Param(self.model.Z)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( len(self.instance.A), 3 )

    def test_io3(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "set Z := 1 3 5;\n" )
        OUTPUT.write( "param : A B :=\n" )
        OUTPUT.write( "1 2.2 3.3\n" )
        OUTPUT.write( "3 2.3 3.4\n" )
        OUTPUT.write( "5 2.5 3.5;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.Z=Set()
        self.model.A=Param(self.model.Z)
        self.model.B=Param(self.model.Z)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( len(self.instance.A), 3 )
        self.assertEqual( len(self.instance.B), 3 )
        self.assertEqual( self.instance.B[5], 3.5 )

    def test_io4(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "set Z := A1 A2 A3;\n" )
        OUTPUT.write( "set Y := 1 2 3;\n" )
        OUTPUT.write( "param A: A1 A2 A3 :=\n" )
        OUTPUT.write( "1 1.3 2.3 3.3\n" )
        OUTPUT.write( "2 1.4 2.4 3.4\n" )
        OUTPUT.write( "3 1.5 2.5 3.5\n" )
        OUTPUT.write( ";\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.Z=Set()
        self.model.Y=Set()
        self.model.A=Param(self.model.Y,self.model.Z)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( len(self.instance.Y), 3 )
        self.assertEqual( len(self.instance.Z), 3 )
        self.assertEqual( len(self.instance.A), 9 )
        self.assertEqual( self.instance.A[1, 'A2'], 2.3 )

    def test_io5(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "set Z := A1 A2 A3;\n" )
        OUTPUT.write( "set Y := 1 2 3;\n" )
        OUTPUT.write( "param A (tr): A1 A2 A3 :=\n" )
        OUTPUT.write( "1 1.3 2.3 3.3\n" )
        OUTPUT.write( "2 1.4 2.4 3.4\n" )
        OUTPUT.write( "3 1.5 2.5 3.5\n" )
        OUTPUT.write( ";\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.Z=Set()
        self.model.Y=Set()
        self.model.A=Param(self.model.Z,self.model.Y)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( len(self.instance.Y), 3 )
        self.assertEqual( len(self.instance.Z), 3 )
        self.assertEqual( len(self.instance.A), 9 )
        self.assertEqual( self.instance.A['A2',1], 2.3 )

    def test_io6(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "set Z := 1 3 5;\n" )
        OUTPUT.write( "param A default 0.0 :=\n" )
        OUTPUT.write( "1 2.2\n" )
        OUTPUT.write( "3 .\n" )
        OUTPUT.write( "5 2.5;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.Z=Set()
        self.model.A=Param(self.model.Z)
        self.instance = self.model.create_instance("param.dat")
        #self.instance.pprint()
        self.assertEqual( len(self.instance.A), 3 )
        self.assertEqual( self.instance.A[3], 0.0 )

    def test_io7(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "param A := True;\n" )
        OUTPUT.write( "param B := False;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.A=Param(within=Boolean)
        self.model.B=Param(within=Boolean)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( value(self.instance.A), True )
        self.assertEqual( value(self.instance.B), False )

    def test_io8(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "param : A : B :=\n" )
        OUTPUT.write( "\"A\" 3.3\n" )
        OUTPUT.write( "\"B\" 3.4\n" )
        OUTPUT.write( "\"C\" 3.5;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.A=Set()
        self.model.B=Param(self.model.A)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( self.instance.A.data(), set(['A','B','C']) )

    def test_io9(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "param : A : B :=\n" )
        OUTPUT.write( "\"A\" 0.1\n" )
        OUTPUT.write( "\"B\" 1e-1\n" )
        OUTPUT.write( "\"b\" 1.4e-1\n" )
        OUTPUT.write( "\"C\" 1E-1\n" )
        OUTPUT.write( "\"c\" 1.4E-1\n" )
        OUTPUT.write( "\"D\" 1E+1\n" )
        OUTPUT.write( "\"d\" 1.4E+1\n" )
        OUTPUT.write( "\"AA\" -0.1\n" )
        OUTPUT.write( "\"BB\" -1e-1\n" )
        OUTPUT.write( "\"bb\" -1.4e-1\n" )
        OUTPUT.write( "\"CC\" -1E-1\n" )
        OUTPUT.write( "\"cc\" -1.4E-1\n" )
        OUTPUT.write( "\"DD\" -1E+1\n" )
        OUTPUT.write( "\"dd\" -1.4E+1;\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.A=Set()
        self.model.B=Param(self.model.A)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( self.instance.B['A'], 0.1)
        self.assertEqual( self.instance.B['B'], 0.1)
        self.assertEqual( self.instance.B['b'], 0.14)
        self.assertEqual( self.instance.B['C'], 0.1)
        self.assertEqual( self.instance.B['c'], 0.14)
        self.assertEqual( self.instance.B['D'], 10)
        self.assertEqual( self.instance.B['d'], 14)
        self.assertEqual( self.instance.B['AA'], -0.1)
        self.assertEqual( self.instance.B['BB'], -0.1)
        self.assertEqual( self.instance.B['bb'], -0.14)
        self.assertEqual( self.instance.B['CC'], -0.1)
        self.assertEqual( self.instance.B['cc'], -0.14)
        self.assertEqual( self.instance.B['DD'], -10)
        self.assertEqual( self.instance.B['dd'], -14)

    def test_io10(self):
        OUTPUT=open("param.dat","w")
        OUTPUT.write( "data;\n" )
        OUTPUT.write( "set A1 := a b c d e f g h i j k l ;\n" )
        OUTPUT.write( "set A2 := 2 4 6 ;\n" )
        OUTPUT.write( "param B :=\n" )
        OUTPUT.write( " [*,2,*] a b 1 c d 2 e f 3\n" )
        OUTPUT.write( " [*,4,*] g h 4 i j 5\n" )
        OUTPUT.write( " [*,6,*] k l 6\n" )
        OUTPUT.write( ";\n" )
        OUTPUT.write( "end;\n" )
        OUTPUT.close()
        self.model.A1=Set()
        self.model.A2=Set()
        self.model.B=Param(self.model.A1,self.model.A2,self.model.A1)
        self.instance = self.model.create_instance("param.dat")
        self.assertEqual( set(self.instance.B.sparse_keys()), set([('e', 2, 'f'), ('c', 2, 'd'), ('a', 2, 'b'), ('i', 4, 'j'), ('g', 4, 'h'), ('k', 6, 'l')]))



class TestParamConditional(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()

    def tearDown(self):
        self.model = None

    def test1(self):
        self.model.p = Param(initialize=1.0)
        try:
            if self.model.p:
                pass
            self.fail("Expected ValueError because parameter was undefined")
        except ValueError:
            pass
        instance = self.model.create_instance()
        if instance.p:
            pass
        else:
            self.fail("Wrong condition value")

    def test2(self):
        self.model.p = Param(initialize=0.0)
        try:
            if self.model.p:
                pass
            self.fail("Expected ValueError because parameter was undefined")
        except ValueError:
            pass
        instance = self.model.create_instance()
        if instance.p:
            self.fail("Wrong condition value")
        else:
            pass


class MiscParamTests(unittest.TestCase):

    def test_constructor(self):
        a = Param(name="a")
        try:
            b = Param(foo="bar")
            self.fail("Cannot pass in 'foo' as an option to Param")
        except ValueError:
            pass
        model=AbstractModel()
        model.b = Param(initialize=[1,2,3])
        try:
            model.c = Param(model.b)
            self.fail("Can't index a parameter with a parameter")
        except TypeError:
            pass
        #
        model = AbstractModel()
        model.a = Param(initialize={None:3.3})
        instance = model.create_instance()

    def test_empty_index(self):
        # Verify that we can initialize a parameter with an empty set.
        model = ConcreteModel()
        model.A = Set()
        def rule(model, i):
            return 0.0
        model.p = Param(model.A, initialize=rule)

    def test_get_uninitialized(self):
        model=AbstractModel()
        model.a = Param()
        model.b = Set(initialize=[1,2,3])
        model.c = Param(model.b, initialize=2, within=Reals)

        instance=model.create_instance()
        # Test that value(instance.a) throws ValueError
        self.assertRaises(ValueError, value, instance.a)
        #
        # GAH: commenting out this check, other components
        #      (like Var) do not raise a ValueError
        #
        # Test that instance.a() throws ValueError
        #self.assertRaises(ValueError, instance.a)

    def test_indexOverRange_abstract(self):
        model = AbstractModel()
        model.p = Param(range(1,3), range(2), initialize=1.0)
        inst = model.create_instance()
        self.assertEqual( sorted(inst.p.keys()),
                          [(1,0), (1,1), (2,0), (2,1)] )
        self.assertEqual( inst.p[1,0], 1.0 )
        self.assertRaises( KeyError, inst.p.__getitem__, (0, 0) )

    def test_indexOverRange_concrete(self):
        inst = ConcreteModel()
        inst.p = Param(range(1,3), range(2), initialize=1.0)
        self.assertEqual( sorted(inst.p.keys()),
                          [(1,0), (1,1), (2,0), (2,1)] )
        self.assertEqual( inst.p[1,0], 1.0 )
        self.assertRaises( KeyError, inst.p.__getitem__, (0, 0) )


    def test_get_set(self):
        model=AbstractModel()
        model.a = Param(initialize=2, mutable=True)
        model.b = Set(initialize=[1,2,3])
        model.c = Param(model.b, initialize=2, within=Reals, mutable=True)
        #try:
            #model.a.value = 3
            #self.fail("can't set the value of an unitialized parameter")
        #except AttributeError:
            #pass
        instance=model.create_instance()
        instance.a.value=3
        #try:
            #instance.a.default='2'
            #self.fail("can't set a bad default value")
        #except ValueError:
            #pass
        self.assertEqual(2 in instance.c, True)

        try:
            instance.a[1] = 3
            self.fail("can't index a scalar parameter")
        except KeyError:
            pass
        try:
            instance.c[4] = 3
            self.fail("can't index a parameter with a bad index")
        except KeyError:
            pass
        try:
            instance.c[3] = 'a'
            self.fail("can't set a parameter with a bad value")
        except ValueError:
            pass

    def test_iter(self):
        model=AbstractModel()
        model.b = Set(initialize=[1,2,3])
        model.c = Param(model.b,initialize=2)
        instance = model.create_instance()
        for i in instance.c:
            self.assertEqual(i in instance.c, True)

    def test_valid(self):
        def d_valid(model, a):
            return True
        def e_valid(model, a, i, j):
            return True
        model=AbstractModel()
        model.b = Set(initialize=[1,3,5])
        model.c = Param(initialize=2, within=None)
        model.d = Param(initialize=(2,3), validate=d_valid)
        model.e = Param(model.b,model.b,initialize={(1,1):(2,3)}, validate=e_valid)
        instance = model.create_instance()
        #instance.e.check_values()
        #try:
            #instance.c.value = 'b'
            #self.fail("can't have a non-numerical parameter")
        #except ValueError:
            #pass


def createNonIndexedParamMethod(func, init_xy, new_xy, tol=1e-10):

    def testMethod(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=init_xy[0], mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=func(model.Q1)<=model.x)

        self.assertAlmostEqual(init_xy[1], value(model.CON[None].lower), delta=1e-10)

        model.Q1 = new_xy[0]

        self.assertAlmostEqual(new_xy[1], value(model.CON[None].lower), delta=tol)

    return testMethod

def createIndexedParamMethod(func, init_xy, new_xy, tol=1e-10):

    def testMethod(self):
        model = ConcreteModel()
        model.P = Param([1,2],initialize=init_xy[0], mutable=True)
        model.Q = Param([1,2],default=init_xy[0], mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = init_xy[0]
        model.R[2] = init_xy[0]
        model.x = Var()
        model.CON1 = Constraint(expr=func(model.P[1])<=model.x)
        model.CON2 = Constraint(expr=func(model.Q[1])<=model.x)
        model.CON3 = Constraint(expr=func(model.R[1])<=model.x)

        self.assertAlmostEqual(init_xy[1], value(model.CON1[None].lower), delta=tol)
        self.assertAlmostEqual(init_xy[1], value(model.CON2[None].lower), delta=tol)
        self.assertAlmostEqual(init_xy[1], value(model.CON3[None].lower), delta=tol)

        model.P[1] = new_xy[0]
        model.Q[1] = new_xy[0]
        model.R[1] = new_xy[0]

        self.assertAlmostEqual(new_xy[1], value(model.CON1[None].lower), delta=tol)
        self.assertAlmostEqual(new_xy[1], value(model.CON2[None].lower), delta=tol)
        self.assertAlmostEqual(new_xy[1], value(model.CON3[None].lower), delta=tol)

    return testMethod

def assignTestsNonIndexedParamTests(cls, problem_list):
    for val in problem_list:
        attrName = 'test_mutable_'+val[0]+'_expr'
        setattr(cls,attrName,createNonIndexedParamMethod(eval(val[0]),val[1],val[2]))

def assignTestsIndexedParamTests(cls, problem_list):
    for val in problem_list:
        attrName = 'test_mutable_'+val[0]+'_expr'
        setattr(cls,attrName,createIndexedParamMethod(eval(val[0]),val[1],val[2]))

instrinsic_test_list = [('sin', (0.0,0.0), (math.pi/2.0,1.0)), \
                        ('cos', (0.0,1.0), (math.pi/2.0,0.0)), \
                        ('log', (1.0,0.0), (math.e,1.0)), \
                        ('log10', (1.0,0.0), (10.0,1.0)),\
                        ('tan', (0.0,0.0), (math.pi/4.0,1.0)),\
                        ('cosh', (0.0,1.0), (math.acosh(1.5),1.5)),\
                        ('sinh', (0.0,0.0), (math.asinh(0.5),0.5)),\
                        ('tanh', (0.0,0.0), (math.atanh(0.8),0.8)),\
                        ('asin', (0.0,0.0), (math.sin(1.0),1.0)),\
                        ('acos', (1.0,0.0), (math.cos(1.0),1.0)),\
                        ('atan', (0.0,0.0), (math.tan(1.0),1.0)),\
                        ('exp', (0.0,1.0), (math.log(2),2.0)),\
                        ('sqrt', (1.0,1.0), (4.0,2.0)),\
                        ('asinh', (0.0,0.0), (math.sinh(2.0),2.0)),\
                        ('acosh', (1.0,0.0), (math.cosh(2.0),2.0)),\
                        ('atanh', (0.0,0.0), (math.tanh(2.0),2.0)),\
                        ('ceil', (0.5,1.0), (1.5,2.0)),\
                        ('floor', (0.5,0.0), (1.5, 1.0))\
                       ]


class MiscNonIndexedParamBehaviorTests(unittest.TestCase):

    # Test that non-indexed params are mutable
    def test_mutable_self(self):
        model = ConcreteModel()
        model.Q = Param(initialize=0.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q<=model.x)

        self.assertEqual(0.0, value(model.CON[None].lower))

        model.Q = 1.0

        self.assertEqual(1.0, value(model.CON[None].lower))

    # Test that display actually displays the correct param value
    def test_mutable_display(self):
        tmp_stream = pyutilib.services.TempfileManager.create_tempfile(suffix = '.param_display.test')
        model = ConcreteModel()
        model.Q = Param(initialize=0.0, mutable=True)
        self.assertEqual(model.Q, 0.0)
        #print model.Q._data
        #print value(model.Q)
        f = StringIO()
        display(model.Q, f)
        tmp = f.getvalue().splitlines()
        val = float(tmp[-1].split(':')[-1].strip())
        self.assertEqual(model.Q, val)

        model.Q = 1.0
        self.assertEqual(model.Q,1.0)
        f = StringIO()
        display(model.Q,f)
        tmp = f.getvalue().splitlines()
        val = float(tmp[-1].split(':')[-1].strip())
        self.assertEqual(model.Q, val)

    # Test that pprint actually displays the correct param value
    def test_mutable_pprint(self):
        model = ConcreteModel()
        model.Q = Param(initialize=0.0, mutable=True)
        self.assertEqual(model.Q, 0.0)
        buf = StringIO()
        model.Q.pprint(ostream=buf)
        val = float(buf.getvalue().splitlines()[-1].split(':')[-1].strip())
        self.assertEqual(model.Q, val)

        buf.buf = ''
        model.Q = 1.0
        self.assertEqual(model.Q,1.0)
        model.Q.pprint(ostream=buf)
        val = float(buf.getvalue().splitlines()[-1].split(':')[-1].strip())
        self.assertEqual(model.Q, val)

    # Test mutability of non-indexed
    # params involved in sum expression
    def test_mutable_sum_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=0.0, mutable=True)
        model.Q2 = Param(initialize=0.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q1+model.Q2<=model.x)

        self.assertEqual(0.0, value(model.CON[None].lower))

        model.Q1 = 3.0
        model.Q2 = 2.0

        self.assertEqual(5.0, value(model.CON[None].lower))

    # Test mutability of non-indexed
    # params involved in prod expression
    def test_mutable_prod_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=0.0, mutable=True)
        model.Q2 = Param(initialize=0.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q1*model.Q2<=model.x)

        self.assertEqual(0.0, value(model.CON[None].lower))

        model.Q1 = 3.0
        model.Q2 = 2.0

        self.assertEqual(6.0, value(model.CON[None].lower))

    # Test mutability of non-indexed
    # params involved in pow expression
    def test_mutable_pow_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=1.0, mutable=True)
        model.Q2 = Param(initialize=1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q1**model.Q2<=model.x)

        self.assertEqual(1.0, value(model.CON[None].lower))

        model.Q1 = 3.0
        model.Q2 = 2.0

        self.assertEqual(9.0, value(model.CON[None].lower))

    # Test mutability of non-indexed
    # params involved in abs expression
    def test_mutable_abs_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=-1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=abs(model.Q1)<=model.x)

        self.assertEqual(1.0, value(model.CON[None].lower))

        model.Q1 = -3.0

        self.assertEqual(3.0, value(model.CON[None].lower))


# Add test methods for all intrinsic functions
assignTestsNonIndexedParamTests(MiscNonIndexedParamBehaviorTests,instrinsic_test_list)


class MiscIndexedParamBehaviorTests(unittest.TestCase):

    # Test that indexed params are mutable
    def test_mutable_self1(self):
        model = ConcreteModel()
        model.P = Param([1], mutable=True)
        model.P[1] = 1.0
        model.x = Var()
        model.CON = Constraint(expr=model.P[1]<=model.x)

        self.assertEqual(1.0, value(model.CON[None].lower))

        model.P[1] = 2.0

        self.assertEqual(2.0, value(model.CON[None].lower))

    # Test that indexed params are mutable
    # when initialized with 'initialize'
    def test_mutable_self2(self):
        model = ConcreteModel()
        model.P = Param([1],initialize=1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.P[1]<=model.x)

        self.assertEqual(1.0, value(model.CON[None].lower))

        model.P[1] = 2.0

        self.assertEqual(2.0, value(model.CON[None].lower))

    # Test that indexed params are mutable
    # when initialized with 'default'
    def test_mutable_self3(self):
        model = ConcreteModel()
        model.P = Param([1],default=1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.P[1]<=model.x)

        self.assertEqual(1.0, value(model.CON[None].lower))

        model.P[1] = 2.0

        self.assertEqual(2.0, value(model.CON[None].lower))

    # Test the behavior when using the 'default' keyword
    # in param initialization
    def test_mutable_self4(self):
        model = ConcreteModel()
        model.P = Param([1,2],default=1.0, mutable=True)

        self.assertEqual(model.P[1],1.0)
        self.assertEqual(model.P[2],1.0)
        model.P[1].value = 0.0
        self.assertEqual(model.P[1],0.0)
        self.assertEqual(model.P[2],1.0)

        model.Q = Param([1,2],default=1.0, mutable=True)
        self.assertEqual(model.Q[1],1.0)
        self.assertEqual(model.Q[2],1.0)
        model.Q[1] = 0.0
        self.assertEqual(model.Q[1],0.0)
        self.assertEqual(model.Q[2],1.0)

    # Test that display actually displays the correct param value
    def test_mutable_display(self):
        tmp_stream = pyutilib.services.TempfileManager.create_tempfile(suffix = '.param_display.test')
        model = ConcreteModel()
        model.P = Param([1,2],default=0.0, mutable=True)
        model.Q = Param([1,2],initialize=0.0, mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        # check initial values are correct

        # check that the correct value is printed
        # Treat the param using default a little differently
        for Item in [model.P]:
            f = StringIO()
            display(Item,f)
            tmp = f.getvalue().splitlines()
            self.assertEqual(len(tmp), 2)
        for Item in [model.Q, model.R]:
            f = StringIO()
            display(Item,f)
            tmp = f.getvalue().splitlines()
            for tmp_ in tmp[2:]:
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(0, val)

        #**** NOTE: Accessing the
        #     value of indexed params which utilize
        #     the default keyword actually causes the internal
        #     rep to become dense for that index, which
        #     changes display output
        for Item in [model.P, model.Q, model.R]:
            for i in [1,2]:
                self.assertEqual(Item[i],0.0)

        # check that the correct value is printed
        # Treat the param using default a little differently
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            display(Item,f)
            tmp = f.getvalue().splitlines()
            for tmp_ in tmp[2:]:
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(0, val)

        model.P[1] = 1.0
        model.P[2] = 2.0
        model.Q[1] = 1.0
        model.Q[2] = 2.0
        model.R[1] = 1.0
        model.R[2] = 2.0

        # check that the correct value is printed
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            display(Item,f)
            tmp = f.getvalue().splitlines()
            i = 0
            for tmp_ in tmp[2:]:
                i += 1
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(i, val)

    # Test that pprint actually displays the correct param value
    def test_mutable_pprint(self):
        tmp_stream = pyutilib.services.TempfileManager.create_tempfile(suffix = '.param_display.test')
        model = ConcreteModel()
        model.P = Param([1,2],default=0.0, mutable=True)
        model.Q = Param([1,2],initialize=0.0, mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        # check initial values are correct

        # check that the correct value is printed
        # Treat the param using default a little differently
        for Item in [model.P]:
            f = StringIO()
            display(Item,f)
            tmp = f.getvalue().splitlines()
            self.assertEqual(len(tmp), 2)
        for Item in [model.Q, model.R]:
            f = StringIO()
            display(Item,f)
            tmp = f.getvalue().splitlines()
            for tmp_ in tmp[2:]:
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(0, val)

        #**** NOTE: Accessing the
        #     value of indexed params which utilize
        #     the default keyword actually causes the internal
        #     rep to become dense for that index, which
        #     changes pprint output
        for Item in [model.P, model.Q, model.R]:
            for i in [1,2]:
                self.assertEqual(Item[i],0.0)

        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            Item.pprint(ostream=f)
            tmp = f.getvalue().splitlines()
            for i in [1,2]:
                val = float(tmp[i+1].split(':')[-1].strip())
                self.assertEqual(0, val)

        model.P[1] = 1.0
        model.P[2] = 2.0
        model.Q[1] = 1.0
        model.Q[2] = 2.0
        model.R[1] = 1.0
        model.R[2] = 2.0

        # check that the correct value is printed
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            Item.pprint(ostream=f)
            tmp = f.getvalue().splitlines()
            for i in [1,2]:
                val = float(tmp[i+1].split(':')[-1].strip())
                self.assertEqual(i, val)

    # Test mutability of indexed
    # params involved in sum expression
    # and that params behave the same when initialized in
    # different ways
    def test_mutable_sum_expr(self):
        model = ConcreteModel()
        model.P = Param([1,2],default=0.0, mutable=True)
        model.Q = Param([1,2],initialize=0.0, mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        model.x = Var()
        model.CON1 = Constraint(expr=model.P[1]+model.P[2]<=model.x)
        model.CON2 = Constraint(expr=model.Q[1]+model.Q[2]<=model.x)
        model.CON3 = Constraint(expr=model.R[1]+model.R[2]<=model.x)

        self.assertEqual(0.0, value(model.CON1[None].lower))
        self.assertEqual(0.0, value(model.CON2[None].lower))
        self.assertEqual(0.0, value(model.CON3[None].lower))

        model.P[1] = 3.0
        model.P[2] = 2.0
        model.Q[1] = 3.0
        model.Q[2] = 2.0
        model.R[1] = 3.0
        model.R[2] = 2.0

        self.assertEqual(5.0, value(model.CON1[None].lower))
        self.assertEqual(5.0, value(model.CON2[None].lower))
        self.assertEqual(5.0, value(model.CON3[None].lower))

    # Test mutability of indexed
    # params involved in prod expression
    # and that params behave the same when initialized in
    # different ways
    def test_mutable_prod_expr(self):
        model = ConcreteModel()
        model.P = Param([1,2],initialize=0.0, mutable=True)
        model.Q = Param([1,2],default=0.0, mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        model.x = Var()
        model.CON1 = Constraint(expr=model.P[1]*model.P[2]<=model.x)
        model.CON2 = Constraint(expr=model.Q[1]*model.Q[2]<=model.x)
        model.CON3 = Constraint(expr=model.R[1]*model.R[2]<=model.x)

        self.assertEqual(0.0, value(model.CON1[None].lower))
        self.assertEqual(0.0, value(model.CON2[None].lower))
        self.assertEqual(0.0, value(model.CON3[None].lower))

        model.P[1] = 3.0
        model.P[2] = 2.0
        model.Q[1] = 3.0
        model.Q[2] = 2.0
        model.R[1] = 3.0
        model.R[2] = 2.0

        self.assertEqual(6.0, value(model.CON1[None].lower))
        self.assertEqual(6.0, value(model.CON2[None].lower))
        self.assertEqual(6.0, value(model.CON3[None].lower))

    # Test mutability of indexed
    # params involved in pow expression
    # and that params behave the same when initialized in
    # different ways
    def test_mutable_pow_expr(self):
        model = ConcreteModel()
        model.P = Param([1,2],initialize=0.0, mutable=True)
        model.Q = Param([1,2],default=0.0, mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        model.x = Var()
        model.CON1 = Constraint(expr=model.P[1]**model.P[2]<=model.x)
        model.CON2 = Constraint(expr=model.Q[1]**model.Q[2]<=model.x)
        model.CON3 = Constraint(expr=model.R[1]**model.R[2]<=model.x)

        self.assertEqual(1.0, value(model.CON1[None].lower))
        self.assertEqual(1.0, value(model.CON2[None].lower))
        self.assertEqual(1.0, value(model.CON3[None].lower))

        model.P[1] = 3.0
        model.P[2] = 2.0
        model.Q[1] = 3.0
        model.Q[2] = 2.0
        model.R[1] = 3.0
        model.R[2] = 2.0

        self.assertEqual(9.0, value(model.CON1[None].lower))
        self.assertEqual(9.0, value(model.CON2[None].lower))
        self.assertEqual(9.0, value(model.CON3[None].lower))

    # Test mutability of indexed
    # params involved in abs expression
    # and that params behave the same when initialized in
    # different ways
    def test_mutable_abs_expr(self):
        model = ConcreteModel()
        model.P = Param([1,2],initialize=-1.0, mutable=True)
        model.Q = Param([1,2],default=-1.0, mutable=True)
        model.R = Param([1,2], mutable=True)
        model.R[1] = -1.0
        model.R[2] = -1.0
        model.x = Var()
        model.CON1 = Constraint(expr=abs(model.P[1])<=model.x)
        model.CON2 = Constraint(expr=abs(model.Q[1])<=model.x)
        model.CON3 = Constraint(expr=abs(model.R[1])<=model.x)

        self.assertEqual(1.0, value(model.CON1[None].lower))
        self.assertEqual(1.0, value(model.CON2[None].lower))
        self.assertEqual(1.0, value(model.CON3[None].lower))

        model.P[1] = -3.0
        model.Q[1] = -3.0
        model.R[1] = -3.0

        self.assertEqual(3.0, value(model.CON1[None].lower))
        self.assertEqual(3.0, value(model.CON2[None].lower))
        self.assertEqual(3.0, value(model.CON3[None].lower))

# Add test methods for all intrinsic functions
assignTestsIndexedParamTests(MiscIndexedParamBehaviorTests,instrinsic_test_list)


if __name__ == "__main__":
    unittest.main()
