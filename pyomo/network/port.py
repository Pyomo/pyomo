#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'Port' ]

import logging, sys
from six import iteritems, itervalues
from weakref import ref as weakref_ref

from pyomo.common.timing import ConstructionTimer
from pyomo.common.plugin import Plugin, implements
from pyomo.common.modeling import unique_component_name

from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.component import ComponentData
from pyomo.core.base.indexed_component import \
    IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.numvalue import as_numeric, value
from pyomo.core.expr.current import identify_variables
from pyomo.core.base.label import alphanum_label_from_name
from pyomo.core.base.plugin import register_component, \
    IPyomoScriptModifyInstance, TransformationFactory
from pyomo.core.kernel.component_map import ComponentMap

from pyomo.network.util import replicate_var

logger = logging.getLogger('pyomo.network')


class _PortData(ComponentData):
    """
    This class defines the data for a single Port

    Attributes
    ----------
        vars:`dict`
            A dictionary mapping added names to variables
    """

    __slots__ = ('vars', '_arcs', '_sources', '_dests', '_rules', '_splitfracs')

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

        self.vars = {}
        self._arcs = []
        self._sources = []
        self._dests = []
        self._rules = {}
        self._splitfracs = ComponentMap()

    def __getstate__(self):
        state = super(_PortData, self).__getstate__()
        for i in _PortData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    def __getattr__(self, name):
        """Returns `self.vars[name]` if it exists"""
        if name in self.vars:
            return self.vars[name]
        # Since the base classes don't support getattr, we can just
        # throw the "normal" AttributeError
        raise AttributeError("'%s' object has no attribute '%s'"
                             % (self.__class__.__name__, name))

    def arcs(self, active=None):
        """A list of Arcs in which this Port is a member"""
        return self._collect_ports(active, self._arcs)

    def sources(self, active=None):
        """A list of Arcs in which this Port is a destination"""
        return self._collect_ports(active, self._sources)

    def dests(self, active=None):
        """A list of Arcs in which this Port is a source"""
        return self._collect_ports(active, self._dests)

    def _collect_ports(self, active, port_list):
        # need to call the weakrefs
        if active is None:
            return [_a() for _a in port_list]
        tmp = []
        for _a in port_list:
            a = _a()
            if a.active == active:
                tmp.append(a)
        return tmp

    def set_value(self, value):
        """Cannot specify the value of a port"""
        raise ValueError("Cannot specify the value of a port: '%s'" % self.name)

    def polynomial_degree(self):
        """Returns the maximum polynomial degree of all port members"""
        ans = 0
        for v in self.iter_vars():
            tmp = v.polynomial_degree()
            if tmp is None:
                return None
            ans = max(ans, tmp)
        return ans

    def is_fixed(self):
        """Return True if all vars/expressions in the Port are fixed"""
        return all(v.is_fixed() for v in self.iter_vars())

    def is_potentially_variable(self):
        """Return True as ports may (should!) contain variables"""
        return True

    def is_binary(self):
        """Return True if all variables in the Port are binary"""
        return len(self) and all(
            v.is_binary() for v in self.iter_vars(expr_vars=True))

    def is_integer(self):
        """Return True if all variables in the Port are integer"""
        return len(self) and all(
            v.is_integer() for v in self.iter_vars(expr_vars=True))

    def is_continuous(self):
        """Return True if all variables in the Port are continuous"""
        return len(self) and all(
            v.is_continuous() for v in self.iter_vars(expr_vars=True))

    def add(self, var, name=None, rule=None, **kwds):
        """
        Add `var` to this Port, casting it to a Pyomo numeric if necessary

        Arguments
        ---------
            var
                A variable or some `NumericValue` like an expression
            name: `str`
                Name to associate with this member of the Port
            rule: `function`
                Function implementing the desired expansion procedure
                for this member. `Port.Equality` by default, other
                options include `Port.Extensive`. Customs are allowed.
            kwds
                Keyword arguments that will be passed to rule
        """
        if var is not None:
            try:
                # indexed components are ok, but as_numeric will error on them
                # make sure they have this attribute
                var.is_indexed()
            except AttributeError:
                var = as_numeric(var)
        if name is None:
            name = var.local_name
        if name in self.vars and self.vars[name] is not None:
            # don't throw warning if replacing an implicit (None) var
            logger.warning("Implicitly replacing variable '%s' in Port '%s'.\n"
                           "To avoid this warning, use Port.remove() first."
                           % (name, self.name))
        self.vars[name] = var
        if rule is None:
            rule = Port.Equality
        if rule is Port.Extensive:
            # avoid name collisions
            if (name.endswith("_split") or name.endswith("_equality") or
                    name == "splitfrac"):
                raise ValueError(
                    "Extensive variable '%s' on Port '%s' may not end "
                    "with '_split' or '_equality'" % (name, self.name))
        self._rules[name] = (rule, kwds)

    def remove(self, name):
        """Remove this member from the port"""
        if name not in self.vars:
            raise ValueError("Cannot remove member '%s' not in Port '%s'"
                             % (name, self.name))
        self.vars.pop(name)
        self._rules.pop(name)

    def rule_for(self, name):
        """Return the rule associated with the given port member"""
        return self._rules[name][0]

    def is_equality(self, name):
        """Return True if the rule for this port member is Port.Equality"""
        return self.rule_for(name) is Port.Equality

    def is_extensive(self, name):
        """Return True if the rule for this port member is Port.Extensive"""
        return self.rule_for(name) is Port.Extensive

    def fix(self):
        """
        Fix all variables in the port at their current values.
        For expressions, fix every variable in the expression.
        """
        for v in self.iter_vars(expr_vars=True, fixed=False):
            v.fix()

    def unfix(self):
        """
        Unfix all variables in the port.
        For expressions, unfix every variable in the expression.
        """
        for v in self.iter_vars(expr_vars=True, fixed=True):
            v.unfix()

    free = unfix

    def iter_vars(self, expr_vars=False, fixed=None, names=False):
        """
        Iterate through every member of the port, going through
        the indices of indexed members.

        Arguments
        ---------
            expr_vars: `bool`
                If True, call `identify_variables` on expression type members
            fixed: `bool`
                Only include variables/expressions with this type of fixed
            names: `bool`
                If True, yield (name, var/expr) pairs
        """
        for name, mem in iteritems(self.vars):
            if not mem.is_indexed():
                itr = (mem,)
            else:
                itr = itervalues(mem)
            for v in itr:
                if fixed is not None and v.is_fixed() != fixed:
                    continue
                if expr_vars and v.is_expression_type():
                    for var in identify_variables(v):
                        if fixed is not None and var.is_fixed() != fixed:
                            continue
                        if names:
                            yield name, var
                        else:
                            yield var
                else:
                    if names:
                        yield name, v
                    else:
                        yield v

    def set_split_fraction(self, arc, val, fix=True):
        """
        Set the split fraction value to be used for an arc during
        arc expansion when using `Port.Extensive`.
        """
        if arc not in self.dests():
            raise ValueError("Port '%s' is not a source of Arc '%s', cannot "
                             "set split fraction" % (self.name, arc.name))
        self._splitfracs[arc] = (val, fix)

    def get_split_fraction(self, arc):
        """
        Returns a tuple (val, fix) for the split fraction of this arc that
        was set via `set_split_fraction` if it exists, and otherwise None.
        """
        res = self._splitfracs.get(arc, None)
        if res is None:
            return None
        else:
            return res


class Port(IndexedComponent):
    """
    A collection of variables, which may be connected to other ports

    The idea behind Ports is to create a bundle of variables that can
    be manipulated together by connecting them to other ports via Arcs.
    A preprocess transformation will look for Arcs and expand them into
    a series of constraints that involve the original variables contained
    within the Port. The way these constraints are built can be specified
    for each Port member when adding members to the port, but by default
    the Port members will be equated to each other. Additionally, other
    objects such as expressions can be added to Ports as long as they, or
    their indexed members, can be manipulated within constraint expressions.

    Parameters
    ----------
        rule: `function`
            A function that returns a dict of (name: var) pairs to be
            initially added to the Port. Instead of var it could also be a
            tuples of (var, rule). Or it could return an iterable of either
            vars or tuples of (var, rule) for implied names.
        initialize
            Follows same specifications as rule's return value, gets
            initially added to the Port
        implicit
            An iterable of names to be initially added to the Port as
            implicit vars
        extends: `Port`
            A Port whose vars will be added to this Port upon construction
    """

    def __new__(cls, *args, **kwds):
        if cls != Port:
            return super(Port, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return SimplePort.__new__(SimplePort)
        else:
            return IndexedPort.__new__(IndexedPort)

    def __init__(self, *args, **kwd):
        self._rule = kwd.pop('rule', None)
        self._initialize = kwd.pop('initialize', {})
        self._implicit = kwd.pop('implicit', {})
        self._extends = kwd.pop('extends', None)
        kwd.setdefault('ctype', Port)
        IndexedComponent.__init__(self, *args, **kwd)

    # This method must be defined on subclasses of
    # IndexedComponent that support implicit definition
    def _getitem_when_not_present(self, idx):
        """Returns the default component data value."""
        tmp = self._data[idx] = _PortData(component=self)
        return tmp

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):  #pragma:nocover
            logger.debug( "Constructing Port, name=%s, from data=%s"
                          % (self.name, data) )

        if self._constructed:
            return

        timer = ConstructionTimer(self)
        self._constructed = True

        # Construct _PortData objects for all index values
        if self.is_indexed():
            self._initialize_members(self._index)
        else:
            self._data[None] = self
            self._initialize_members([None])

        # get rid of these references
        self._rule = None
        self._initialize = None
        self._implicit = None
        self._extends = None # especially important as this is another port

        timer.report()

    def _initialize_members(self, initSet):
        for idx in initSet:
            tmp = self[idx]
            for key in self._implicit:
                tmp.add(None, key)
            if self._extends:
                for key, val in iteritems(self._extends.vars):
                    tmp.add(val, key, self._extends.rule_for(key))
            if self._initialize:
                self._add_from_container(tmp, self._initialize)
            if self._rule:
                items = apply_indexed_rule(
                    self, self._rule, self._parent(), idx)
                self._add_from_container(tmp, items)

    def _add_from_container(self, port, items):
        if type(items) is dict:
            for key, val in iteritems(items):
                if type(val) is tuple:
                    port.add(val[0], key, val[1])
                else:
                    port.add(val, key)
        else:
            for val in self._initialize:
                if type(val) is tuple:
                    port.add(val[0], rule=val[1])
                else:
                    port.add(val)

    def _pprint(self, ostream=None, verbose=False):
        """Print component information."""
        def _line_generator(k, v):
            for _k, _v in sorted(iteritems(v.vars)):
                if _v is None:
                    _len = '-'
                elif _v.is_indexed():
                    _len = len(_v)
                else:
                    _len = 1
                yield _k, _len, str(_v)
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None)],
             iteritems(self._data),
             ( "Name", "Size", "Variable"),
             _line_generator)

    def display(self, prefix="", ostream=None):
        """
        Print component state information

        This duplicates logic in Component.pprint()
        """
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab = "    "
        ostream.write(prefix + self.local_name + " : ")
        ostream.write("Size=" + str(len(self)))

        ostream.write("\n")
        def _line_generator(k,v):
            for _k, _v in sorted(iteritems(v.vars)):
                if _v is None:
                    _val = '-'
                elif not _v.is_indexed():
                    _val = str(value(_v))
                else:
                    _val = "{%s}" % (
                        ', '.join('%r: %r' % (
                            x, value(_v[x])) for x in sorted(_v._data)))
                yield _k, _val
        tabular_writer(ostream, prefix+tab,
                       ((k, v) for k, v in iteritems(self._data)),
                       ("Name", "Value"), _line_generator)

    @staticmethod
    def Equality(port, name, index_set):
        """Arc Expansion procedure to generate simple equality constraints"""
        # Iterate over every arc off this port. Since this function will
        # be called for every port, we need to check if it already exists.
        for arc in port.arcs(active=True):
            Port._add_equality_constraint(arc, name, index_set)

    @staticmethod
    def Extensive(port, name, index_set, include_splitfrac=False,
            write_var_sum=True):
        """
        Arc Expansion procedure for extensive variable properties

        This procedure is the rule to use when variable quantities should
        be split for outlets and combined for inlets.

        This will first go through every destination of the port and create
        a new variable on the arc's expanded block of the same index as the
        current variable being processed. It will also create a splitfrac
        variable on the expanded block as well. Then it will generate
        constraints for the new variable that relates it to the port member
        variable by the split fraction. Following this, an indexed constraint
        is written that states that the sum of all the new variables equals
        the parent. However, if `write_var_sum=False` is passed, instead of
        this last indexed constraint, a single constraint will be written
        that states the sum of the split fractions equals 1.

        Then, this procedure will go through every source of the port and
        create a new variable (unless it already exists), and then write
        a constraint that states the sum of all the incoming new variables
        must equal the parent variable.

        Model simplifications:

            If the port has a 1-to-1 connection on either side, it will not
            create the new variables and instead write a simple equality
            constraint for that side.

            If the outlet side is not 1-to-1 but there is only one outlet,
            it will not create a splitfrac variable or write the split
            constraint, but it will still write the outsum constraint
            which will be a simple equality.

            If the port only contains a single Extensive variable, the
            splitfrac variables and the splitting constraints will
            be skipped since they will be unnecessary. However, they
            can be still be included by passing include_splitfrac=True.

        .. note::
            If split fractions are skipped, the `write_var_sum=False`
            option is not allowed.
        """
        port_parent = port.parent_block()
        out_vars = Port._Split(port, name, index_set,
            include_splitfrac=include_splitfrac, write_var_sum=write_var_sum)
        in_vars = Port._Combine(port, name, index_set)

    @staticmethod
    def _Combine(port, name, index_set):
        port_parent = port.parent_block()
        var = port.vars[name]
        in_vars = []
        sources = port.sources(active=True)

        if not len(sources):
            return in_vars

        if len(sources) == 1 and len(sources[0].source.dests(active=True)) == 1:
            # This is a 1-to-1 connection, no need for evar, just equality.
            arc = sources[0]
            Port._add_equality_constraint(arc, name, index_set)
            return in_vars

        for arc in sources:
            eblock = arc.expanded_block

            # Make and record new variables for every arc with this member.
            evar = Port._create_evar(port.vars[name], name, eblock, index_set)
            in_vars.append(evar)

        # Create constraint: var == sum of evars
        # Same logic as Port._Split
        cname = unique_component_name(port_parent, "%s_%s_insum" %
            (alphanum_label_from_name(port.local_name), name))
        def rule(m, *args):
            if len(args):
                return sum(evar[args] for evar in in_vars) == var[args]
            else:
                return sum(evar for evar in in_vars) == var
        con = Constraint(index_set, rule=rule)
        port_parent.add_component(cname, con)

        return in_vars

    @staticmethod
    def _Split(port, name, index_set, include_splitfrac=False,
            write_var_sum=True):
        port_parent = port.parent_block()
        var = port.vars[name]
        out_vars = []
        no_splitfrac = False
        dests = port.dests(active=True)

        if not len(dests):
            return out_vars

        if len(dests) == 1:
            # No need for splitting on one outlet.
            # Make sure they do not try to fix splitfrac not at 1.
            splitfracspec = port.get_split_fraction(dests[0])
            if splitfracspec is not None:
                if splitfracspec[0] != 1 and splitfracspec[1] == True:
                    raise ValueError(
                        "Cannot fix splitfrac not at 1 for port '%s' with a "
                        "single dest '%s'" % (port.name, dests[0].name))

            no_splitfrac = True

            if len(dests[0].destination.sources(active=True)) == 1:
                # This is a 1-to-1 connection, no need for evar, just equality.
                arc = dests[0]
                Port._add_equality_constraint(arc, name, index_set)
                return out_vars

        for arc in dests:
            eblock = arc.expanded_block

            # Make and record new variables for every arc with this member.
            evar = Port._create_evar(port.vars[name], name, eblock, index_set)
            out_vars.append(evar)

            if no_splitfrac:
                continue

            # Create and potentially initialize split fraction variables.
            # This function will be called for every Extensive member of this
            # port, but we only need one splitfrac variable per arc, so check
            # if it already exists before making a new one. However, we do not
            # need a splitfrac if there is only one Extensive data object,
            # so first check whether or not we need it.

            if eblock.component("splitfrac") is None:
                if not include_splitfrac:
                    num_data_objs = 0
                    for k, v in iteritems(port.vars):
                        if port.is_extensive(k):
                            if v.is_indexed():
                                num_data_objs += len(v)
                            else:
                                num_data_objs += 1
                            if num_data_objs > 1:
                                break

                    if num_data_objs <= 1:
                        # Do not make splitfrac, do not make split constraints.
                        # Make sure they didn't specify splitfracs.
                        # This inner loop will only run once.
                        for arc in dests:
                            if port.get_split_fraction(arc) is not None:
                                raise ValueError(
                                    "Cannot specify splitfracs for port '%s' "
                                    "(found arc '%s') because this port only "
                                    "has one variable. To have control over "
                                    "splitfracs, please pass the "
                                    " include_splitfrac=True argument." %
                                    (port.name, arc.name))
                        no_splitfrac = True
                        continue

                eblock.splitfrac = Var()
                splitfracspec = port.get_split_fraction(arc)
                if splitfracspec is not None:
                    eblock.splitfrac = splitfracspec[0]
                    if splitfracspec[1]:
                        eblock.splitfrac.fix()

            # Create constraint for this member using splitfrac.
            cname = "%s_split" % name
            def rule(m, *args):
                if len(args):
                    return evar[args] == eblock.splitfrac * var[args]
                else:
                    return evar == eblock.splitfrac * var
            con = Constraint(index_set, rule=rule)
            eblock.add_component(cname, con)

        if write_var_sum:
            # Create var total sum constraint: var == sum of evars
            # Need to alphanum port name in case it is indexed.
            cname = unique_component_name(port_parent, "%s_%s_outsum" %
                (alphanum_label_from_name(port.local_name), name))
            def rule(m, *args):
                if len(args):
                    return sum(evar[args] for evar in out_vars) == var[args]
                else:
                    return sum(evar for evar in out_vars) == var
            con = Constraint(index_set, rule=rule)
            port_parent.add_component(cname, con)
        else:
            # OR create constraint on splitfrac vars: sum == 1
            if no_splitfrac:
                raise ValueError(
                    "Cannot choose to write split fraction sum constraint for "
                    "ports with a single destination or a single Extensive "
                    "variable.\nSplit fractions are skipped in this case to "
                    "simplify the model.\nPlease use write_var_sum=True on "
                    "this port (the default).")
            cname = unique_component_name(port_parent,
                "%s_frac_sum" % alphanum_label_from_name(port.local_name))
            con = Constraint(expr=
                sum(a.expanded_block.splitfrac for a in dests) == 1)
            port_parent.add_component(cname, con)

        return out_vars

    @staticmethod
    def _add_equality_constraint(arc, name, index_set):
        # This function will add the equality constraint if it doesn't exist.
        eblock = arc.expanded_block
        cname = name + "_equality"
        if eblock.component(cname) is not None:
            # already exists, skip
            return
        port1, port2 = arc.ports
        def rule(m, *args):
            if len(args):
                return port1.vars[name][args] == port2.vars[name][args]
            else:
                return port1.vars[name] == port2.vars[name]
        con = Constraint(index_set, rule=rule)
        eblock.add_component(cname, con)

    @staticmethod
    def _create_evar(member, name, eblock, index_set):
        # Name is same, conflicts are prevented by a check in Port.add.
        # The new var will mirror the original var and have same index set.
        # We only need one evar per arc, so check if it already exists
        # before making a new one.
        evar = eblock.component(name)
        if evar is None:
            evar = replicate_var(member, name, eblock, index_set)
        return evar


class SimplePort(Port, _PortData):

    def __init__(self, *args, **kwd):
        _PortData.__init__(self, component=self)
        Port.__init__(self, *args, **kwd)


class IndexedPort(Port):
    pass


register_component(
    Port, "A bundle of variables that can be connected to other ports.")
