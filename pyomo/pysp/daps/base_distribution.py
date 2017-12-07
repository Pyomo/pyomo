"""
This abstract base class is the parental class of all distribution classes.
It is so far only useable for univariate distributions.
"""
from abc import *
from scipy import integrate
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import collections
import random
#import gosm_options ...  daps cannot depend on this
import os
### from timer import * # do not import * says DLW
import scipy.stats as sps

class GOSM_options(object):
    """
    distributions from gosm are being used here outside of gosm.
    TBD: create shared distributions and figure out what to do with options.
    dlw December 2017
    """
    def __init__(self):
        self.cdf_inverse_tolerance = 1e-4
        self.cdf_inverse_max_refinements = 10
gosm_options = GOSM_options()
        
class BaseDistribution(object):
    __metaclass__ = ABCMeta

    # --------------------------------------------------------------------
    # Abstract methods (have to be implemented within the subclass)
    # --------------------------------------------------------------------

    @abstractmethod
    def __init__(self, name = None, dimension = 0, input_data = None):
        """
        Initializes the distribution.

        Args:
            name (str): the name of the distribution
            dimension (int): the dimension of the distribution
            input_data: dict, OrderedDict or list of input data
        """
        self.name = name
        self.dimension = dimension

        # Lower and upper bound of the considered data
        self.alpha = None
        self.beta = None

        # Input data
        self.input_data = input_data

    @abstractmethod
    def pdf(self, x):
        """
        Evaluates the probability density function at a given point x.

        Args:
            x (float): the point at which the pdf is to be evaluated

        Returns:
            float: the value of the pdf
        """
        pass

    # --------------------------------------------------------------------
    # Non-abstract methods (already implemented within the base class)
    # --------------------------------------------------------------------

    def plot(self, plot_pdf = 1, plot_cdf = 1, plot_input_data = 0, output_file = None, title = None, xlabel = None, ylabel = None):
        """
        Plots the pdf/cdf within the interval [alpha, beta].
        If required, the input data is added as a scatter plot, where the size of the data points is correlated
        to their respective number of occurrences.
        If no output file is specified, the plots are shown at runtime.

        Args:
            plot_pdf (binary): 1, if the plot should include the pdf, 0 otherwise
            plot_cdf (binary): 1, if the plot should include the cdf, 0 otherwise
            plot_input_data (binary): 1, if the plot should include the input data, 0 otherwise
            output_file (string): name of an output file to save the plot
            title (string): the title of the plot
            xlabel (string): the name of the x-axis
            ylabel (string): the name of the y-axis
        """
        raise RuntimeError("plot in daps.base_distributions.py not supported due to dependence on gosm options") 
        """
        if plot_pdf == 0 and plot_cdf == 0 and plot_input_data == 0:
            print('Error: The print method was called, but no functions were supposed to be plotted.')
            return

        # If the dimension is 1, plot the pdf.
        if self.dimension == 1:

            directory = gosm_options.output_directory + os.sep + 'plots' + os.sep + 'error_distributions'
            if not os.path.isdir(directory):
                os.makedirs(directory)

            x_range = np.arange(self.alpha, self.beta + gosm_options.plot_variable_gap, gosm_options.plot_variable_gap)
            plt.figure(1)

            # Plot the pdf if required.
            if plot_pdf == 1:
                y_range = []
                for x in x_range:
                    y_range.append(self.pdf(x))
                plt.plot(x_range, y_range, label = 'PDF', color = 'blue')

            # Plot the cdf if required.
            if plot_cdf == 1:
                y_range = []
                for x in x_range:
                    y_range.append(self.cdf(x))
                plt.plot(x_range, y_range, label = 'CDF', color = 'red')

            # Plot the input data if required.
            if plot_input_data == 1:

                # Size the input data points according to the number of occurrences.
                counter = collections.Counter(self.input_data)

                # The scaling factor 20 was chosen arbitrarily.
                size = [counter[i] * 20 for i in self.input_data]
                plt.scatter(self.input_data, len(self.input_data) * [0], size, label = 'Input Data', color = 'grey')

            # Display a legend.
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, shadow=True)

            # Display a grid and the axes.
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.axvline(x=0, color='k')

            # Name the axes.
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.title(title, y=1.08)

            if output_file is None:
                # Display the plot.
                plt.show()
            else:
                # Save the plot.
                plt.savefig(directory + os.sep + output_file)

            plt.close(1)

        return
        """
        
    def cdf(self, x):
        """
        Evaluates the cumulative distribution function at a given point x.

        Args:
            x (float): the point at which the cdf is to be evaluated

        Returns:
            float: the value of the cdf
        """

        if x <= self.alpha:
            return 0
        elif x >= self.beta:
            return 1
        else:
            return integrate.quad(self.pdf, self.alpha, x)[0]

    def cdf_inverse(self, x):
        """
        Evaluates the inverse cumulative distribution function at a given point x.

        Args:
            x (float): the point at which the inverse cdf is to be evaluated

        Returns:
            float: the value of the inverse cdf
        """

        # This method calculates the cdf of start and then increases (if the cdf value is less than or equal x) or
        # decreases (if the cdf value is greater than x) start iteratively by one stepsize until x is passed.
        # It returns the increased (or decreased) start value and its cdf value.
        def approximate_inverse_value(start):
            cdf_val = self.cdf(start)
            if x >= cdf_val:
                while x >= cdf_val:
                    start += stepsize
                    cdf_val = self.cdf(start)
            else:
                while x <= cdf_val:
                    start -= stepsize
                    cdf_val = self.cdf(start)
            return cdf_val, start

        # Handle some special cases.
        if x < 0 or x > 1:
            return None
        elif abs(x) <= gosm_options.cdf_inverse_tolerance:
            return self.alpha
        elif abs(x-1) <= gosm_options.cdf_inverse_tolerance:
            return self.beta
        else:

            # Initialize variables.
            approx_x = 0
            result = None
            number_of_refinement = 0

            # The starting stepsize was chosen arbitrarily.
            stepsize = (self.beta - self.alpha)/10

            while abs(approx_x - x) > gosm_options.cdf_inverse_tolerance and number_of_refinement <= gosm_options.cdf_inverse_max_refinements:

                # If this is the first iteration, start at one of the bounds of the domain.
                if number_of_refinement == 0:

                    # If x is greater than or equal 0.5, start the approximation at the upper bound of the domain.
                    if x >= 0.5:
                        approx_x, result = approximate_inverse_value(self.beta)

                    # If x is less than 0.5, start the approximation at the lower bound of the domain.
                    else:
                        approx_x, result = approximate_inverse_value(self.alpha)
                else:

                    # If this is not the first iteration, halve the stepsize and call the approximation method.
                    stepsize /= 2
                    approx_x, result = approximate_inverse_value(result)

                number_of_refinement += 1

            return result

    def mean(self):
        """
        Computes the mean value (expectation) of the distribution.

        Returns:
            float: the mean value
        """

        # Use region_expectation to compute the mean value.
        return self.region_expectation((self.alpha, self.beta))

    def region_expectation(self, region):
        """
        Computes the mean value (expectation) of a specified region.

        Args:
            region: the region (tuple of dimension 2) of which the expectation is to be computed

        Returns:
            float: the expectation

        """

        # Check whether region is a tuple of dimension 2.
        if isinstance(region, tuple) and len(region) == 2:
            a, b = region
            if a > b:
                raise RuntimeError('Error: The upper bound of \'region\' can\'t be less than the lower bound.')
        else:
            raise RuntimeError('Error: Parameter \'region\' must be a tuple of dimension 2.')

        integral = 0

        # Define a modified pdf.
        modified_pdf = lambda x: x * self.pdf(x)

        # Compute the expected value by integration, if the dimension equals 1.
        if self.dimension == 1:
            integral = integrate.quad(modified_pdf, a, b)[0]

        return integral

    def region_probability(self, region):
        """
        Computes the probability of a specified region.

        Args:
            region: the region of which the probability is to be computed

        Returns:
            float: the probability
        """

        # Compute the region's probability by integration, if the dimension equals 1.
        if self.dimension == 1:

            # Check whether region is a tuple of dimension 2.
            if isinstance(region, tuple) and len(region) == 2:
                a, b = region
                integral = integrate.quad(self.pdf, a, b)[0]
            else:
                raise RuntimeError('Error: Parameter \'region\' must be a tuple of dimension 2.')

            return integral

        return

    def sample_one(self):
        """
        Returns a single sample of the distribution

        Returns:
            float: the sample

        """

        return self.cdf_inverse(random.random())

    @staticmethod
    def seed_reset(seed=None):
        """
        Resets the random seed for sampling.
        If no argument is passed, the current time is used.

        Args:
            seed: the random seed

        """

        random.seed(seed)

        return


class MultiDistr(BaseDistribution):
    """
    This class will be the basis of copulas building.
    """

    def __init__(self, dimkeys=None, input_data=None):
        """
        Args :
            dimkeys (List) : keys for each dimension in dictin (e.g. a list of ints)
            input_data (any) : the raw data; given as lists for each dimension
        """

        self.dimkeys = dimkeys
        # first put the data into structures that scipy likes
        dataarray = []
        length = len(input_data[self.dimkeys[0]])
        for k in self.dimkeys:
            if len(input_data[k]) != length:
                raise RuntimeError('The number of elements in the data for dps {} is not equal the number of elements '
                                   'in the data for dps {}.'.format(self.dimkeys[0],k))
            dataarray.append(input_data[k])

        self.mean = np.mean(dataarray, axis=1)
        # Call the parent's constructor.
        super(MultiDistr, self).__init__(dimension=len(dimkeys), input_data=dataarray)


    def pdf(self):
        raise RuntimeError('pdf not implemented.')

    def rect_prob(self, lowerdict, upperdict):

        tempdict = dict.fromkeys(self.dimkeys)
        def f(n):

            # recursive function that will calculate the cdf
            if n == 0:
                return self.cdf(tempdict)
            else:
                tempdict[self.dimkeys[n - 1]] = upperdict[self.dimkeys[n - 1]]
                leftresult = f(n - 1)
                tempdict[self.dimkeys[n - 1]] = lowerdict[self.dimkeys[n - 1]]
                rightresult = f(n - 1)
                return leftresult - rightresult

        return f(self.dimension)

