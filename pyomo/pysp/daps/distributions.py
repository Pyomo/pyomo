import numpy
import math
from pyomo.environ import *
from pyomo.pysp.daps.base_distribution import BaseDistribution
from pyomo.pysp.daps.base_distribution import MultiDistr
from collections import OrderedDict
### from statistics import DomainFinder dlw nov 2017
import scipy.stats as sps
import numpy as np
from scipy.stats import mvn
import scipy.special as sps
### TBD: dlw dec 2017: deal with gosm (it has a similar file)
from pyomo.pysp.daps.distribution_factory import register_distribution

inf = 1e9


@register_distribution(name="univariate-normal", ndim=1)
class UnivariateNormalDistribution(BaseDistribution):
    """
        This class creates a univariate normal distribution (i.e. with a gaussian density).
        
        To be simpler this class use only floats and not dictionaries.
        To use dictionaries, one must make a MultiNormalDistribution with dimension 1.
        
        One can call it either :
            with a mean and/or a variance
            by default the mean and the variance will be calculated thanks to the datas
        
        mean (float): the mean of the distribution it is a number, not a dictionary
        variance (float): the variance of the distribution
    """

    def __init__(self, input_data=None, var=None, mean=None):
        if not (var is None):
            self.var = var
        else:
            self.var = np.var(input_data)

        if not (mean is None):
            self.mean = mean
        else:
            self.mean = np.mean(input_data)


        super(UnivariateNormalDistribution, self).__init__(self, dimension=1, input_data=input_data)

    def pdf(self, x):
        """
    
        Args:
            x (float): The values where you want to compute the pdf

        Returns:
            (float) The value of the probability density function of this distribution on x.
        """
        return ((1 / (2 * self.var * np.pi) ** 0.5)) * np.exp(-1 / 2 * (x - self.mean) ** 2 / self.var)

    def cdf(self, x):
        """

            Args:
                x (float): The values where you want to compute the pdf
    
            Returns:
                (float) The value of the cumulative density function of this distribution on x.
        """

        return sps.norm.cdf(x, loc=self.mean, scale=self.var ** 0.5)

    def rect_prob(self,down,up):
        """
        
        Args:
            up (float): the upper values where you want to compute the probability
            down (float): the upper values where you want to compute the probability

        Returns: the probability of being between up and down

        """
        return (self.cdf(up)-self.cdf(down))

@register_distribution(name="univariate-student",ndim=1)
class UnivariateStudentDistribution(BaseDistribution):
    def __init__(self, input_data=None, df=None, mean=None):
        if not (df is None):
            self.df = df
        else:
            self.var =  np.var(input_data)
            if self.var <= 1:
                print('input_data gives Var < 1 : Impossible to define a student distribution')
                print('Degree of freedom is by default set to 1')
                self.df=1
            else:
                self.df = 2*self.var/(self.var-1)

        if not (mean is None):
            self.mean = mean
        else:
            self.mean = np.mean(input_data)

        super(UnivariateStudentDistribution, self).__init__(self, dimension=1, input_data=input_data)

    def pdf(self, x):
        """

        Args:
            x (float): The values where you want to compute the pdf

        Returns:
            (float) The value of the probability density function of this distribution on x.
        """
        return  sps.gamma((self.df+1)/2)/(np.sqrt(self.df*np.pi)*sps.gamma(self.df/2))*(1+(x-self.mean)**2/self.df)**(-(self.df+1)/2)


    def cdf(self, x):
        """

            Args:
                x (float): The values where you want to compute the pdf

            Returns:
                (float) The value of the cumulative density function of this distribution on x.
        """
        if (x-self.mean) >=0:
            return 1-1/2*sps.betainc(self.df/2,1/2,self.df/((x-self.mean)**2+self.df))
        else:
            return 1/2*sps.betainc(self.df/2,1/2,self.df/((x-self.mean)**2+self.df))


    def rect_prob(self, down, up):
        """

        Args:
            up (float): the upper values where you want to compute the probability
            down (float): the upper values where you want to compute the probability

        Returns: the probability of being between up and down

        """
        return (self.cdf(up) - self.cdf(down))

@register_distribution(name='multivariate-normal')
class MultiNormalDistribution(MultiDistr):
    """
        This class creates a multi variate normal distribution.
        One can call it either :
            with a mean and/or a covariance matrix
            by default ie without a mean/covariance input,
            the mean and/or the covariance will be calculated thanks to the datas

    """
    def __init__(self, dimkeys=None, input_data=None, cov =None, mean =None):
        """
        Args :
            dimkeys (List) : keys or each dimension in dictin (e.g. a list of ints)
            input_data (dict) : the raw data; given as lists for each dimension
            mean (dict) : the mean of the distribution
            cov (matrix) : the covariance matrix of this distribution. Do not confuse with the correlation matrix which is the covariance but reduced
        """

        super(MultiNormalDistribution, self).__init__(dimkeys, input_data)
        if not (mean is None):
            self.mean=mean
        else:
            self.mean = dict.fromkeys(self.dimkeys)
            for i in self.dimkeys:
                self.mean[i] = np.mean(input_data[i])

        if not (cov is None):
            self.cov = cov
        else:
            self.cov = np.cov(self.input_data)


    def pdf(self,valuedict):
        return sps.multivariate_normal.pdf(list(valuedict.values()),mean=list(self.mean.values()),cov=self.cov)

    def cdf(self,valuedict):
        infinity = 10**7
        low_list_infinity = []
        for i in range(self.dimension):
            low_list_infinity =[-infinity]+low_list_infinity
        result,i = sps.mvn.mvnun(low_list_infinity, list(valuedict.values()), list(self.mean.values()), self.cov)
        return result

    def rect_prob(self, lowerdict, upperdict):
        """
        
        Args:
            lowerdict (dict): The lower bound where you want to compute the probability.
            upperdict (dict): The upper bound where you want to compute the probability.

        Returns:
            The probability of being in the rectangle defines by the upper and lower bounds.
        
        We here just use mvnun which compute the same rectangular probability but with different types.
        """
        result,i = sps.mvn.mvnun(list(lowerdict.values()),list(upperdict.values()),list(self.mean.values()),self.cov)
        return result

@register_distribution(name="univariate-empirical", ndim=1)
class UnivariateEmpiricalDistribution(BaseDistribution):
    """
    This class will fit an empirical distribution to a set of given self.input_data.
    """

    def __init__(self, input_data):
        """
        Initializes the distribution.

        Args:
            input_data: list of data points
        """

        # Check the type of the input data and sort it.
        if isinstance(input_data, list):
            input_data = sorted(input_data)
        else:
            raise RuntimeError('Error: Unknown type of input data.')

        # Calls the constructor of the parental class.
        super(UnivariateEmpiricalDistribution, self).__init__('UnivariateEmpiricalDistribution', 1, input_data)

        self.alpha = input_data[0]
        self.beta = input_data[len(input_data)-1]

        return

    def pdf(self, x):
        """
        Evaluates the probability of a given point x.

        Args:
            x (float): the point at which the probability is to be evaluated

        Returns:
            float: the probability
        """

        def condition(y):
            return y == x

        # Count all self.input_data that are equal to x.
        number = sum(1 for y in self.input_data if condition(y))

        return number/len(self.input_data)

    def cdf(self, x, lower_bound=None, upper_bound=None):
        """
        This method calculates a empirical cdf, which is fitted to the data by interpolation.
        If a lower bound is provided, any point smaller will have cdf value 0.
        If an upper bound is provided, any point larger will have cdf value 1.
        If either is not provided the value is estimated using the line between the nearest two self.input_data.

        Args:
            x (float): the point at which the cdf is to be evaluated
            lower_bound (float): the lower bound
            upper_bound (float): the upper bound

        Returns:
            float: the value of the cdf

        Notes:
            This method was copied from PINT's distributions class on 03/24/2017.
        """

        n = len(self.input_data)
        if n == 0:
            raise RuntimeError("Your list of self.input_data to calculate the inverse cdf is empty. "
                               "One possible reason could be, that your day-ahead forecast file and "
                               "your historic forecast file do not match.")
        lower_neighbor = None
        lower_neighbor_index = None
        upper_neighbor = None
        upper_neighbor_index = None
        for index in range(n):
            if self.input_data[index] <= x:
                lower_neighbor = self.input_data[index]
                lower_neighbor_index = index
            if self.input_data[index] > x:
                upper_neighbor = self.input_data[index]
                upper_neighbor_index = index
                break

        if lower_neighbor == x:
            cdf_x = (lower_neighbor_index + 1) / (n + 1)

        elif lower_neighbor is None:  # x is smaller than all of the values in self.input_data
            if lower_bound is None:
                x1 = self.input_data[0]
                index1 = self._count_less_than_or_equal(self.input_data, x1)

                x2 = self.input_data[index1]
                index2 = self._count_less_than_or_equal(self.input_data, x2)

                y1 = index1 / (n + 1)
                y2 = index2 / (n + 1)
                interpolating_line = interpolate_line(x1, y1, x2, y2)
                cdf_x = max(0, interpolating_line(x))
            else:
                if lower_bound > x:
                    cdf_x = 0
                else:
                    x1 = lower_bound
                    x2 = upper_neighbor
                    y1 = 0
                    y2 = 1 / (n + 1)
                    interpolating_line = interpolate_line(x1, y1, x2, y2)
                    cdf_x = interpolating_line(x)

        elif upper_neighbor is None:  # x is greater than all of the values in self.input_data
            if upper_bound is None:
                j = n - 1
                while self.input_data[j] == self.input_data[n - 1]:
                    j -= 1
                x1 = self.input_data[j]
                x2 = self.input_data[n - 1]
                y1 = (j+1) / (n + 1)
                y2 = n / (n + 1)
                interpolating_line = interpolate_line(x1, y1, x2, y2)
                cdf_x = min(1, interpolating_line(x))
            else:
                if upper_bound < x:
                    cdf_x = 1
                else:
                    x1 = lower_neighbor
                    x2 = upper_bound
                    y1 = n / (n + 1)
                    y2 = 1
                    interpolating_line = interpolate_line(x1, y1, x2, y2)
                    cdf_x = interpolating_line(x)
        else:
            x1 = lower_neighbor
            x2 = upper_neighbor
            y1 = (lower_neighbor_index + 1) / (n + 1)
            y2 = (upper_neighbor_index + 1) / (n + 1)
            interpolating_line = interpolate_line(x1, y1, x2, y2)
            cdf_x = interpolating_line(x)

        return cdf_x

    def cdf_inverse(self, x, lower_bound=None, upper_bound=None):
        """
        This method calculates a empirical inverse cdf, which is fitted to the data by interpolation.

        Args:
            x (float): the point at which the inverse cdf is to be evaluated
            lower_bound (float): the lower bound
            upper_bound (float): the upper bound

        Returns:
            float: the value of the inverse cdf

        Notes:
            This method was copied from PINT's distributions class on 03/24/2017.
        """

        n = len(self.input_data)
        if x < 0 or x > 1:
            raise RuntimeError('A x has to be between 0 and 1!')
        # compute 'index' of this x
        index = x * (n + 1) - 1
        first_index = self._count_less_than_or_equal(self.input_data, self.input_data[0]) - 1

        if index < first_index:
            if lower_bound is None:
                # take linear function through (0, self.input_data[0]) and (1, self.input_data[1])
                # NOTE: self.input_data[0]) could occur several times, so find the highest index j with self.input_data[j] = self.input_data[0]
                if n == 0:
                    raise RuntimeError("Your list of self.input_data to calculate the inverse cdf is empty. "
                                       "One possible reason could be, that your day-ahead forecast file and "
                                       "your historic forecast file do not match.")

                first_index += 1
                second_index = self._count_less_than_or_equal(self.input_data, self.input_data[first_index])
                interpolating_line = interpolate_line(first_index / (n + 1), self.input_data[0],
                                                      second_index / (n + 1), self.input_data[first_index])

                return interpolating_line(x)
            else:
                return lower_bound * (1 / (n + 1) - x) / (1 / (n + 1)) + \
                       self.input_data[0] * x / (1 / (n + 1))
        elif index > n - 1:
            if upper_bound is None:
                # take linear function through (n-2, self.input_data[n-2]) and (n-1, self.input_data[n-1])
                # NOTE: self.input_data[n-1] could occur several times,
                # so find the lowest index j with self.input_data[j] = self.input_data[n-1]
                j = n - 1
                while self.input_data[j] == self.input_data[j - 1]:
                    j -= 1
                    if j - 1 == -len(self.input_data):
                        print("Warning: all values for segmentation are the same (", self.input_data[j], ")")
                        return self.input_data[j]
                # g(x) = a*x + b
                a = self.input_data[j] - self.input_data[j - 1]
                b = self.input_data[j - 1] - (self.input_data[j] - self.input_data[j - 1]) * (j - 1)
                return a * index + b
            else:
                return self.input_data[n - 1] * \
                       (1 - x) / (1 - n / (n + 1)) + \
                       upper_bound * (x - n / (n + 1)) / (1 - n / (n + 1))
        else:
            if math.floor(index) == index:
                return self.input_data[math.floor(index)]
            else:
                interpolating_line = interpolate_line(x1=math.floor(index), y1=self.input_data[math.floor(index)],
                                                      x2=math.ceil(index), y2=self.input_data[math.ceil(index)])
                return interpolating_line(index)

    def _count_less_than_or_equal(self, xs, x):
        """
        Counts the number of elements less than or equal to x in
        a sorted list xs

        Args:
            xs: A sorted list of elements
            x: An element that you wish to find the number of elements less than it

        Returns:
            int: The number of elements in xs less than or equal to x
        """
        count = 0
        for elem in xs:
            if elem <= x:
                count += 1
            else:
                break
        return count

def interpolate_line(x1, y1, x2, y2):
    """
    This functions accepts two points (passed in as four arguments)
    and returns the function of the line which passes through the points.

    Args:
        x1 (float): x-value of point 1
        y1 (float): y-value of point 1
        x2 (float): x-value of point 2
        y2 (float): y-value of point 2

    Returns:
        function: the function of the line
    """

    if x1 == x2:
        raise ValueError("x1 and x2 must be different values")

    def f(x):
        slope = (y2 - y1) / (x2 - x1)
        return slope * (x - x1) + y1

    return f

#=========
@register_distribution(name="univariate-discrete", ndim=1)
class UnivariateDiscrete(BaseDistribution):
    """
        This class creates a discrete univariate distribution.
        The constructor takes an ordered dict of breakpoints.
    """

    def __init__(self, breakpoints):
        """
        Univariate Discrete distribution constructor.
        args:
            breakpoints (OrderedDict): [value] := probability,
            which need to be in increasing value and with prob that sums to 1.
            Written for 3.x+
        """
        if not isinstance(breakpoints, OrderedDict):
            raise RuntimeError("DiscreteDistribution expecting breakpoints to be a dict")
            
        self.breakpoints = breakpoints
        # check the breakpoints
        tol = 1e-6
        sumprob = 0
        self.mean = 0
        Esqsum = 0
        lastval, prob = list(self.breakpoints.items())[0]
        for val, prob in self.breakpoints.items():
            sumprob += prob
            self.mean += prob * val
            Esqsum += prob * val * val
            if val < lastval:
                raise RuntimeError("DiscreteDistribution dict must be ordered by val:"+str(val)+" < "+str(lastval))
            lastval = val
        self.var = self.mean*self.mean - Esqsum
        if sumprob - 1 > tol: # could use gosm_options.cdf_tolerance
            raise ValueError("Discrete distribution with total prob="
                             +str(sumprob)+" tolerance="+str(tol))
        super(UnivariateDiscrete, self).__init__(self, dimension=1)

    def pdf(self, x):
        raise RuntimeError("pdf called for a discrete distribution.")
        
    def cdf(self, x):
        """
            Cummulative Distribution Function: prob(X < x), which is weird
            Args:
                x (float): The value where you want to compute the cdf
    
            Returns:
                (float) The value of the cumulative density function of this distribution on x.
        """
        lastval, prob = list(self.breakpoints.items())[0]
        if x < lastval:
            return 0
        elif x == lastval:
            return prob
        sumprob = 0
        for val, prob in self.breakpoints.items():
            sumprob += prob
            if x == val:
                return sumprob
            if x > lastval and x < val:
                return sumprob - prob
            lastval = val
        return sumprob  # should be one if we got this far

    def cdf_inverse(self, x):
        """
        Evaluates the inverse of the cdf at probability value x, but
        that does not really fly for discrete distrs...
        """
        raise RuntimeError("cdf called for a discrete distribution.")

    def sample_one(self):
        """
        Returns a single sample from the distribution

        Returns:
            float or int: the sample
        """
        p = np.random.uniform()
        sumprob = 0
        for val, prob in self.breakpoints.items():
            sumprob += prob
            if sumprob >= p:
                return val
        # if the probs dont' quite sum to one...
        val, prob = list(self.breakpoints.items())[-1]
        return val
    
    def rect_prob(self,down,up):
        """
        
        Args:
            up (float): the upper values where you want to compute the probability
            down (float): the upper values where you want to compute the probability

        Returns: the probability of being between up and down

        """
        return (self.cdf(up)-self.cdf(down))
