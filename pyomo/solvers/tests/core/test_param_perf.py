import os
thisdir = os.path.dirname(os.path.abspath(__file__))

import time
from pyomo.core import *
import pyutilib.th as unittest

_plot_filename = os.path.join(thisdir, "param_performance.pdf")
_pdf_out = None

def setUpModule():
    global _plot_filename
    global _pdf_out
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_pdf import PdfPages
        _pdf_out = PdfPages(_plot_filename)
    except:
        _pdf_out = None

def tearDownModule():
    global _pdf_out
    if _pdf_out:
        _pdf_out.close()

def plot_results(page_title, results):

    import matplotlib.pyplot as plt

    pyomo_set_iter_time = results.pop('pyomo set iter')
    python_set_iter_time = results.pop('python set iter')
    pyomo_set_contains_time = results.pop('pyomo set contains')
    python_set_contains_time = results.pop('python set contains')
    results = results[None]

    results = sorted(results, key=lambda x: x['ord'], reverse=True)
    ind = [x for x,res in enumerate(results,4)]
    labels = [res['label'] for x,res in enumerate(results,4)]
    construct_times = [res['construct'] for x,res in enumerate(results,4)]
    access_times = [res['access'] for x,res in enumerate(results,4)]
    fig = plt.figure()
    plt.title(page_title, fontsize=22)
    fig.set_size_inches(25,8)
    #plt.figure(figsize=(10, 3))
    p1 = plt.barh(ind, construct_times, color='r')
    p2 = plt.barh(ind, access_times, color='y',
                  left=construct_times)
    p3 = plt.barh([3], [python_set_iter_time], color='k')
    p4 = plt.barh([2], [pyomo_set_iter_time], color='k')
    p3 = plt.barh([1], [python_set_contains_time], color='k')
    p5 = plt.barh([0], [pyomo_set_contains_time], color='k')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #set plot limits
    max_y = len(ind)+4
    plt.ylim(0, max_y)
    max_x = max(c+a for c,a in zip(construct_times, access_times))
    plt.xlim(0, max_x)

    #add the numbers to the side of each bar
    for i, c, a in zip(ind, construct_times, access_times):
        plt.annotate("%.6f"%(c+a),
                     xy=(float(c+a)+max_x*0.01, i+.5),
                     va='center',
                     fontsize=12)
    plt.annotate("%.6f"%(python_set_iter_time),
                 xy=(float(python_set_iter_time)+max_x*0.01, 3+.5),
                 va='center',
                 fontsize=12)
    plt.annotate("%.6f"%(pyomo_set_iter_time),
                 xy=(float(pyomo_set_iter_time)+max_x*0.01, 2+.5),
                 va='center',
                 fontsize=12)
    plt.annotate("%.6f"%(python_set_contains_time),
                 xy=(float(python_set_contains_time)+max_x*0.01, 1+.5),
                 va='center',
                 fontsize=12)
    plt.annotate("%.6f"%(pyomo_set_contains_time),
                 xy=(float(pyomo_set_contains_time)+max_x*0.01, 0+.5),
                 va='center',
                 fontsize=12)

    all_labels = ['Pyomo Set() contains','Python set() contains',
                  'Pyomo Set() iteration','Python set() iteration']+labels
    ticks = plt.yticks([i+.5 for i in range(len(all_labels))],
                       all_labels,
                       fontsize=10)
    xt = plt.xticks()[0]
    plt.xticks(xt, [' '] * len(xt))

    plt.legend((p1[0],p2[0]),
               ('Consruction','Access All'),loc=4)

    plt.savefig(_pdf_out,format='pdf')

def _setup_cls(self):
    self.results = {}
    self.results['set'] = 0.0
    self.results[None] = []
    self.model = ConcreteModel()
    
    self.model.s = self._create_index()
    
    m_s = self.model.s
    start = time.time()
    s_raw = set(i for i in m_s)
    self.results['pyomo set iter'] = time.time()-start
    start = time.time()
    set(i for i in s_raw)
    self.results['python set iter'] = time.time()-start
    start = time.time()
    [(i in m_s) for i in s_raw]
    self.results['pyomo set contains'] = time.time()-start
    s_raw_copy = [i for i in s_raw]
    start = time.time()
    [(i in s_raw) for i in s_raw_copy]
    self.results['python set contains'] = time.time()-start


class TestParamPerformanceBase(object):

    def _test_usage(self, order, tag, cls, *args, **kwds):
        res = {}
        res['ord'] = order
        res['label'] = tag
        start = time.time()
        self.model.test_component = cls(*args, **kwds)
        res['construct'] = time.time()-start
        test_component = self.model.test_component
        raw_index = [i for i in test_component]
        start = time.time()
        x = [test_component[i] for i in raw_index]
        res['access'] = time.time()-start
        res['total'] = res['access']+res['construct']
        self.results[None].append(res)

    def _test_param_mutable_default(self):
        self._test_usage(0,
                         "Param(mutable, default)",
                         Param,
                         self.model.s,
                         mutable=True,
                         default=1.0)

    def _test_param_mutable_default_denseinit(self):
        self._test_usage(1,
                         "Param(mutable, default, dense_init)",
                         Param,
                         self.model.s,
                         mutable=True,
                         default=1.0,
                         initialize_as_dense=True)

    def _test_param_mutable(self):
        self._test_usage(2,
                         "Param(mutable)",
                         Param,
                         self.model.s,
                         mutable=True,
                         initialize=1.0)

    def _test_param_mutable_denseinit(self):
        self._test_usage(3,
                         "Param(mutable, dense_init)",
                         Param,
                         self.model.s,
                         mutable=True,
                         initialize=1.0,
                         initialize_as_dense=True)

    def _test_param_default(self):
        self._test_usage(4,
                         "Param(default)",
                         Param,
                         self.model.s,
                         mutable=False,
                         default=1.0)

    def _test_param_default_denseinit(self):
        self._test_usage(5,
                         "Param(default, dense_init)",
                         Param,
                         self.model.s,
                         mutable=False,
                         default=1.0,
                         initialize_as_dense=True)

    def _test_param(self):
        self._test_usage(6,
                         "Param()",
                         Param,
                         self.model.s,
                         mutable=False,
                         initialize=1.0)

    def _test_param_denseinit(self):
        self._test_usage(7,
                         "Param(dense_init)",
                         Param,
                         self.model.s,
                         mutable=False,
                         initialize=1.0,
                         initialize_as_dense=True)

    def _test_dict(self):
        self._test_usage(8,
                         "dict()",
                         dict,
                         ((i, 1.0) for i in self.model.s))


@unittest.category('performance')
class TestParamPerformanceRangeSet(unittest.TestCase,
                                   TestParamPerformanceBase):


    @classmethod
    def setUpClass(self):
        _setup_cls(self)

    def tearDown(self):
        self.model.del_component('test_component')
        # the above won't work for non component types (e.g., dict)
        try:
            del self.model.test_component
        except:
            pass

    @classmethod
    def _create_index(self):
        
        N = 4000000
        return RangeSet(N)

    # These could live on the base class, except nosetests would try to
    # execute them there.
    def test_param_mutable_default(self):
        TestParamPerformanceBase._test_param_mutable_default(self)
    def test_param_mutable_default_denseinit(self):
        TestParamPerformanceBase._test_param_mutable_default_denseinit(self)
    def test_param_mutable(self):
        TestParamPerformanceBase._test_param_mutable(self)
    def test_param_mutable_denseinit(self):
        TestParamPerformanceBase._test_param_mutable_denseinit(self)
    def test_param_default(self):
        TestParamPerformanceBase._test_param_default(self)
    def test_param_default_denseinit(self):
        TestParamPerformanceBase._test_param_default_denseinit(self)
    def test_param(self):
        TestParamPerformanceBase._test_param(self)
    def test_param_denseinit(self):
        TestParamPerformanceBase._test_param_denseinit(self)
    def test_dict(self):
        TestParamPerformanceBase._test_dict(self)

    @classmethod
    def tearDownClass(self):
        try:
            plot_results("Param Usage - Large RangeSet Index", self.results)
        except:
            print("Results plotting failed")


@unittest.category('performance')
class TestParamPerformanceSetProduct(unittest.TestCase,
                                     TestParamPerformanceBase):


    @classmethod
    def setUpClass(self):
        _setup_cls(self)

    def tearDown(self):
        self.model.del_component('test_component')
        # the above won't work for non component types (e.g., dict)
        try:
            del self.model.test_component
        except:
            pass

    @classmethod
    def _create_index(self):

        m = self.model
        N = 17
        m.s1 = Set(initialize=range(N))
        m.s2 = Set(dimen=2,initialize=[(i,j) for i in range(N) \
                                           for j in range(N)])
        return m.s2*m.s2*m.s1

    # These could live on the base class, except nosetests would try to
    # execute them there.
    def test_param_mutable_default(self):
        TestParamPerformanceBase._test_param_mutable_default(self)
    def test_param_mutable_default_denseinit(self):
        TestParamPerformanceBase._test_param_mutable_default_denseinit(self)
    def test_param_mutable(self):
        TestParamPerformanceBase._test_param_mutable(self)
    def test_param_mutable_denseinit(self):
        TestParamPerformanceBase._test_param_mutable_denseinit(self)
    def test_param_default(self):
        TestParamPerformanceBase._test_param_default(self)
    def test_param_default_denseinit(self):
        TestParamPerformanceBase._test_param_default_denseinit(self)
    def test_param(self):
        TestParamPerformanceBase._test_param(self)
    def test_param_denseinit(self):
        TestParamPerformanceBase._test_param_denseinit(self)
    def test_dict(self):
        TestParamPerformanceBase._test_dict(self)

    @classmethod
    def tearDownClass(self):
        try:
            plot_results("Param Usage - High Dimensional Set Product Index", self.results)
        except:
            print("Results plotting failed")


if __name__ == "__main__":
    unittest.main()
