#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import Factory
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import MouseTrap
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TransformationTimer


class TransformationInfo(object):
    pass


class TransformationData(object):
    """
    This is a container class that supports named data objects.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, name):
        if not name in self._data:
            self._data[name] = TransformationInfo()
        return self._data[name]


class Transformation(object):
    """
    Base class for all model transformations.
    """

    def __init__(self, **kwds):
        kwds["name"] = kwds.get("name", "transformation")
        # super(Transformation, self).__init__(**kwds)

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    @deprecated(
        "Transformation.apply() has been deprecated.  Please use either "
        "Transformation.apply_to() for in-place transformations or "
        "Transformation.create_using() for transformations that create a "
        "new, independent transformed model instance.",
        version='4.3.11323',
    )
    def apply(self, model, **kwds):
        inplace = kwds.pop('inplace', True)
        if inplace:
            self.apply_to(model, **kwds)
        else:
            return self.create_using(model, **kwds)

    def apply_to(self, model, **kwds):
        """
        Apply the transformation to the given model.
        """
        timer = TransformationTimer(self, 'in-place')
        if not hasattr(model, '_transformation_data'):
            model._transformation_data = TransformationData()
        reverse_token = self._apply_to(model, **kwds)
        timer.report()

        return reverse_token

    def create_using(self, model, **kwds):
        """
        Create a new model with this transformation
        """
        timer = TransformationTimer(self, 'out-of-place')
        if not hasattr(model, '_transformation_data'):
            model._transformation_data = TransformationData()
        new_model = self._create_using(model, **kwds)
        timer.report()
        return new_model

    def _apply_to(self, model, **kwds):
        raise RuntimeError("The Transformation.apply_to method is not implemented.")

    def _create_using(self, model, **kwds):
        # Put all the kwds onto the model so that when we clone the
        # model any references to things on the model are correctly
        # updated to point to the new instance.  Note that users &
        # transformation developers cannot rely on things happening by
        # argument side effect.
        name = unique_component_name(model, '_kwds')
        setattr(model, name, kwds)
        instance = model.clone()
        kwds = getattr(instance, name)
        delattr(model, name)
        delattr(instance, name)
        self._apply_to(instance, **kwds)
        return instance


class ReverseTransformationToken(object):
    """
    Class returned by reversible transformations' apply_to methods that
    can be passed back to the transformation in order to revert its changes
    to the model.

    We store the transformation that created it, so that we have some basic
    error checking when the user attempts to revert, and we store a dictionary
    that can be whatever the transformation wants/needs in order to revert
    itself.

    args:
        transformation: The class of the transformation that created this token
        model: The model being transformed when this token was created
        targets: The targets on 'model' being transformed when this token
                 was created.
        reverse_dict: Dictionary with everything the transformation needs to
                      undo itself.
    """

    def __init__(self, transformation, model, targets, reverse_dict):
        self._transformation = transformation
        self._model = model
        self._targets = ComponentSet(targets)
        self._reverse_dict = reverse_dict

    @property
    def transformation(self):
        return self._transformation

    @property
    def reverse_dict(self):
        return self._reverse_dict

    def check_token_valid(self, cls, model, targets):
        if cls is not self._transformation:
            raise ValueError(
                "Attempting to reverse transformation of class '%s' "
                "using a token created by a transformation of class "
                "'%s'. Cannot revert transformation with a token from "
                "another transformation." % (cls, self._transformation)
            )
        if model is not self._model:
            raise MouseTrap(
                "A reverse transformation was called on model '%s', but the "
                "transformation that created this token was created from "
                "model '%s'. We do not currently support reversing "
                "transformations on clones of the transformed model."
                % (model.name, self._model.name)
            )
        # TODO: Do we need to pass targets into this? I'm thinking no because
        # people can untransform selectively. We just need to enforce that the
        # targets given here were indeed transformed, but they don't have to
        # correspond exactly to what happened before. I think we leave that for
        # the transformation to sort out


TransformationFactory = Factory('transformation type')


@deprecated(version='4.3.11323')
def apply_transformation(*args, **kwds):
    if len(args) == 0:
        return list(TransformationFactory)
    xfrm = TransformationFactory(args[0])
    if len(args) == 1 or xfrm is None:
        return xfrm
    tmp = (args[1],)
    return xfrm.apply(*tmp, **kwds)
