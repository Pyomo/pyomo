"""
distribution_factory.py

This module will export a distribution_factory function which should essentially
accept a name for a distribution and return the class associated with that distribution.
This will work by performing a scan through all the modules in this directory and finding
all the classes that are "registered" as distributions.

Registering a  entails using the class decorator register which is also
exported from this module.
"""

import os
import importlib

distribution_registry = {}


# Right now, this function simply scans through all modules in the current working directory
# It is perhaps more desirable (and safer) to have a list of modules to scan through
# This would be even easier to implement; dlw dec 2017: this needs to be fixed
#   so it will work as part of pyomo if needed.

def import_all_classes():
    """
    Imports all classes in the current directory and stores
    all registered distribution classes in the distribution_registry object
    """
    for file in os.listdir(os.getcwd()):
        if file.endswith(".py"):
            module_name = file.split('.')[0]

            try:
                module = importlib.__import__(module_name)
            except:
                print("Failed to import module {}".format(module_name))
                continue

            for var in module.__dict__:
                obj = getattr(module, var)
                if hasattr(obj, "is_registered_distribution") and getattr(obj, "is_registered_distribution"):
                    distribution_registry[obj.registered_name] = obj


def register_distribution(name, ndim=None):
    """
    Class decorator with arguments to register a class for use in distribution_factory
    One must specify the name of the distribution to register under and the number of
    dimensions the input argument must be (if there is a requirement).

    This will add the attributes is_registered_distribution, registered_name, and registered_ndim to the class

    Names will be case insensitive.

    Examples:
        To register a function, simply use this as a decorator before any class to be registered

        @register(name='clayton', ndim=2)
        class ClaytonCopula(CopulaBase):
            ...

    Args:
        name (str): The name the distribution will be registered under
        ndim (:obj: `int`, optional): The required dimensionality of input. Defaults to None signaling no requirement

    Returns:
        A class decorator to register a class as a distribution
    """

    def class_decorator(cls):
        cls.is_registered_distribution = True
        cls.registered_name = name.lower()
        cls.registered_ndim = ndim
        return cls

    return class_decorator


def distribution_factory(name):
    """
    This function will accept a name and return the associated
    distribution class raising an error if the name is unrecognized.
    Names should be case insensitive

    Args:
        name (str): The name of the distribution wanted
    Returns:
        The distribution class associated with name
    """
    import_all_classes()

    try:
        distribution = distribution_registry[name.lower()]
    except KeyError:
        possible_names = '\n'.join(sorted(distribution_registry.keys(), key=str.lower))
        raise NameError("The specified distribution {} is unrecognized. "
                        "Possible distributions are:\n{}".format(name, possible_names))

    return distribution


if __name__ == '__main__':
    import numpy as np
    from distributions import UnivariateEmpiricalDistribution
    keys = ['wind', 'solar']
    data = {'wind': np.random.randn(100).tolist(), 'solar': np.random.randn(100).tolist()}
    marginals = {'wind': UnivariateEmpiricalDistribution(data['wind']),
                 'solar': UnivariateEmpiricalDistribution(data['solar'])}

    gaussian_copula = distribution_factory('gaussian-copula')
    fitted_distribution = gaussian_copula(keys, data, marginals)
    print(fitted_distribution.C([0, 2]))
