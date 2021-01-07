#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
from pyomo.common.dependencies import (
    matplotlib, matplotlib_available,
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy, scipy_available,
    check_min_version, attempt_import
)

plt = matplotlib.pyplot
stats = scipy.stats

# occasionally dependent conda packages for older distributions
# (e.g. python 3.5) get released that are either broken not
# compatible, resulting in a SyntaxError
sns, seaborn_available = attempt_import(
    'seaborn', alt_names=['sns'],
    catch_exceptions=(ImportError, SyntaxError)
)

imports_available = numpy_available & scipy_available & pandas_available \
                    & matplotlib_available & seaborn_available

def _get_variables(ax,columns):
    sps = ax.get_subplotspec()
    nx = sps.get_geometry()[1]
    ny = sps.get_geometry()[0]
    cell = sps.get_geometry()[2]
    xloc = int(np.mod(cell,nx))
    yloc = int(np.mod((cell-xloc)/nx, ny))

    xvar = columns[xloc]
    yvar = columns[yloc]
    #print(sps.get_geometry(), cell, xloc, yloc, xvar, yvar)
    
    return xvar, yvar, (xloc, yloc)


def _get_XYgrid(x,y,ncells):
    xlin = np.linspace(min(x)-abs(max(x)-min(x))/2, max(x)+abs(max(x)-min(x))/2, ncells)
    ylin = np.linspace(min(y)-abs(max(y)-min(y))/2, max(y)+abs(max(y)-min(y))/2, ncells)
    X, Y = np.meshgrid(xlin, ylin)
    
    return X,Y


def _get_data_slice(xvar,yvar,columns,data,theta_star):

    search_ranges = {} 
    for var in columns:
        if var in [xvar,yvar]:
            search_ranges[var] = data[var].unique()
        else:
            search_ranges[var] = [theta_star[var]]

    data_slice = pd.DataFrame(list(itertools.product(*search_ranges.values())),
                            columns=search_ranges.keys())
    
    # griddata will not work with linear interpolation if the data 
    # values are constant in any dimension
    for col in data[columns].columns:
        cv = data[col].std()/data[col].mean() # Coefficient of variation
        if cv < 1e-8: 
            temp = data.copy()
            # Add variation (the interpolation is later scaled)
            if cv == 0:
                temp[col] = temp[col] + data[col].mean()/10
            else:
                temp[col] = temp[col] + data[col].std()
            data = data.append(temp, ignore_index=True)
    
    data_slice['obj'] = scipy.interpolate.griddata(
        np.array(data[columns]),
        np.array(data[['obj']]),
        np.array(data_slice[columns]),
        method='linear',
        rescale=True,
    )
        
    X = data_slice[xvar]
    Y = data_slice[yvar]
    Z = data_slice['obj']
    
    return X,Y,Z
    
# Note: seaborn 0.11 no longer expects color and label to be passed to the 
#       plotting functions. label is kept here for backward compatibility
def _add_scatter(x,y,color,columns,theta_star,label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax, columns)
    
    ax.scatter(theta_star[xvar], theta_star[yvar], c=color, s=35)
    
    
def _add_rectangle_CI(x,y,color,columns,lower_bound,upper_bound,label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)

    xmin = lower_bound[xvar]
    ymin = lower_bound[yvar]
    xmax = upper_bound[xvar]
    ymax = upper_bound[yvar]
    
    ax.plot([xmin, xmax], [ymin, ymin], color=color)
    ax.plot([xmax, xmax], [ymin, ymax], color=color)
    ax.plot([xmax, xmin], [ymax, ymax], color=color)
    ax.plot([xmin, xmin], [ymax, ymin], color=color)


def _add_scipy_dist_CI(x,y,color,columns,ncells,alpha,dist,theta_star,label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    X,Y = _get_XYgrid(x,y,ncells)
    
    data_slice = []
    
    if isinstance(dist, stats._multivariate.multivariate_normal_frozen):
        for var in theta_star.index:
            if var == xvar:
                data_slice.append(X)
            elif var == yvar:
                data_slice.append(Y)
            elif var not in [xvar,yvar]:
                data_slice.append(np.array([[theta_star[var]]*ncells]*ncells))
        data_slice = np.dstack(tuple(data_slice))
        
    elif isinstance(dist, stats.kde.gaussian_kde):
        for var in theta_star.index:
            if var == xvar:
                data_slice.append(X.ravel())
            elif var == yvar:
                data_slice.append(Y.ravel())
            elif var not in [xvar,yvar]:
                data_slice.append(np.array([theta_star[var]]*ncells*ncells))
        data_slice = np.array(data_slice)
    else:
        return
        
    Z = dist.pdf(data_slice)
    Z = Z.reshape((ncells, ncells))
    
    ax.contour(X,Y,Z, levels=[alpha], colors=color) 
    
    
def _add_obj_contour(x,y,color,columns,data,theta_star,label=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)

    try:
        X, Y, Z = _get_data_slice(xvar,yvar,columns,data,theta_star)
        
        triang = matplotlib.tri.Triangulation(X, Y)
        cmap = plt.cm.get_cmap('Greys')
        
        plt.tricontourf(triang,Z,cmap=cmap)
    except:
        print('Objective contour plot for', xvar, yvar,'slice failed')

def _set_axis_limits(g, axis_limits, theta_vals, theta_star):
    
    if theta_star is not None:
        theta_vals = theta_vals.append(theta_star, ignore_index=True)
        
    if axis_limits is None:
        axis_limits = {}
        for col in theta_vals.columns:
            theta_range = np.abs(theta_vals[col].max() - theta_vals[col].min())
            if theta_range < 1e-10:
                theta_range  = theta_vals[col].max()/10
            axis_limits[col] = [theta_vals[col].min() - theta_range/4, 
                                theta_vals[col].max() + theta_range/4]
    for ax in g.fig.get_axes():
        xvar, yvar, (xloc, yloc) = _get_variables(ax,theta_vals.columns)
        if xloc != yloc: # not on diagonal
            ax.set_ylim(axis_limits[yvar])
            ax.set_xlim(axis_limits[xvar])
        else: # on diagonal
            ax.set_xlim(axis_limits[xvar])

            
def pairwise_plot(theta_values, theta_star=None, alpha=None, distributions=[], 
                  axis_limits=None, title=None, add_obj_contour=True, 
                  add_legend=True, filename=None):
    """
    Plot pairwise relationship for theta values, and optionally alpha-level 
    confidence intervals and objective value contours
    
    Parameters
    ----------
    theta_values: DataFrame or tuple
    
        * If theta_values is a DataFrame, then it contains one column for each theta variable 
          and (optionally) an objective value column ('obj') and columns that contains 
          Boolean results from confidence interval tests (labeled using the alpha value). 
          Each row is a sample.
          
          * Theta variables can be computed from ``theta_est_bootstrap``, 
            ``theta_est_leaveNout``, and  ``leaveNout_bootstrap_test``.
          * The objective value can be computed using the ``likelihood_ratio_test``.
          * Results from confidence interval tests can be computed using the  
           ``leaveNout_bootstrap_test``, ``likelihood_ratio_test``, and 
           ``confidence_region_test``.

        * If theta_values is a tuple, then it contains a mean, covariance, and number 
          of samples (mean, cov, n) where mean is a dictionary or Series 
          (indexed by variable name), covariance is a DataFrame (indexed by 
          variable name, one column per variable name), and n is an integer.
          The mean and covariance are used to create a multivariate normal 
          sample of n theta values. The covariance can be computed using 
          ``theta_est(calc_cov=True)``.
        
    theta_star: dict or Series, optional
        Estimated value of theta.  The dictionary or Series is indexed by variable name.  
        Theta_star is used to slice higher dimensional contour intervals in 2D
    alpha: float, optional
        Confidence interval value, if an alpha value is given and the 
        distributions list is empty, the data will be filtered by True/False 
        values using the column name whose value equals alpha (see results from
        ``leaveNout_bootstrap_test``, ``likelihood_ratio_test``, and 
        ``confidence_region_test``)
    distributions: list of strings, optional
        Statistical distribution used to define a confidence region, 
        options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde, and 
        'Rect' for rectangular.
        Confidence interval is a 2D slice, using linear interpolation at theta_star.
    axis_limits: dict, optional
        Axis limits in the format {variable: [min, max]}
    title: string, optional
        Plot title
    add_obj_contour: bool, optional
        Add a contour plot using the column 'obj' in theta_values.
        Contour plot is a 2D slice, using linear interpolation at theta_star.
    add_legend: bool, optional
        Add a legend to the plot
    filename: string, optional
        Filename used to save the figure
    """
    assert isinstance(theta_values, (pd.DataFrame, tuple))
    assert isinstance(theta_star, (type(None), dict, pd.Series, pd.DataFrame))
    assert isinstance(alpha, (type(None), int, float))
    assert isinstance(distributions, list)
    assert set(distributions).issubset(set(['MVN', 'KDE', 'Rect']))
    assert isinstance(axis_limits, (type(None), dict))
    assert isinstance(title, (type(None), str))
    assert isinstance(add_obj_contour, bool)
    assert isinstance(filename, (type(None), str))
    
    # If theta_values is a tuple containing (mean, cov, n), create a DataFrame of values
    if isinstance(theta_values, tuple):
        assert(len(theta_values) == 3)
        mean = theta_values[0]
        cov = theta_values[1]
        n = theta_values[2]
        if isinstance(mean, dict):
            mean = pd.Series(mean)
        theta_names = mean.index
        mvn_dist = stats.multivariate_normal(mean, cov)
        theta_values = pd.DataFrame(mvn_dist.rvs(n, random_state=1), columns=theta_names)
            
    assert(theta_values.shape[0] > 0)
    
    if isinstance(theta_star, dict):
        theta_star = pd.Series(theta_star)
    if isinstance(theta_star, pd.DataFrame):
        theta_star = theta_star.loc[0,:]
    
    theta_names = [col for col in theta_values.columns if (col not in ['obj']) 
                        and (not isinstance(col, float)) and (not isinstance(col, int))]
    
    # Filter data by alpha
    if (alpha in theta_values.columns) and (len(distributions) == 0):
        thetas = theta_values.loc[theta_values[alpha] == True, theta_names]
    else:
        thetas = theta_values[theta_names]
    
    if theta_star is not None:
        theta_star = theta_star[theta_names]
    
    legend_elements = []
    
    g = sns.PairGrid(thetas)
    
    # Plot histogram on the diagonal
    # Note: distplot is deprecated and will be removed in a future
    #       version of seaborn, use histplot.  distplot is kept for older
    #       versions of python.
    if check_min_version(sns, "0.11"):
        g.map_diag(sns.histplot)
    else:
        g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False) 
    
    # Plot filled contours using all theta values based on obj
    if 'obj' in theta_values.columns and add_obj_contour:
        g.map_offdiag(_add_obj_contour, columns=theta_names, data=theta_values, 
                      theta_star=theta_star)
        
    # Plot thetas
    g.map_offdiag(plt.scatter, s=10)
    legend_elements.append(matplotlib.lines.Line2D(
        [0], [0], marker='o', color='w', label='thetas',
        markerfacecolor='cadetblue', markersize=5))
    
    # Plot theta*
    if theta_star is not None:
        g.map_offdiag(_add_scatter, color='k', columns=theta_names, theta_star=theta_star)
        
        legend_elements.append(matplotlib.lines.Line2D(
            [0], [0], marker='o', color='w', label='theta*',
            markerfacecolor='k', markersize=6))
    
    # Plot confidence regions
    colors = ['r', 'mediumblue', 'darkgray']
    if (alpha is not None) and (len(distributions) > 0):
        
        if theta_star is None:
            print("""theta_star is not defined, confidence region slice will be 
                  plotted at the mean value of theta""")
            theta_star = thetas.mean()
        
        mvn_dist = None
        kde_dist = None
        for i, dist in enumerate(distributions):
            if dist == 'Rect':
                lb, ub = fit_rect_dist(thetas, alpha)
                g.map_offdiag(_add_rectangle_CI, color=colors[i], columns=theta_names, 
                            lower_bound=lb, upper_bound=ub)
                legend_elements.append(matplotlib.lines.Line2D(
                    [0], [0], color=colors[i], lw=1, label=dist))
                
            elif dist == 'MVN':
                mvn_dist = fit_mvn_dist(thetas)
                Z = mvn_dist.pdf(thetas)
                score = stats.scoreatpercentile(Z, (1-alpha)*100) 
                g.map_offdiag(_add_scipy_dist_CI, color=colors[i], columns=theta_names, 
                            ncells=100, alpha=score, dist=mvn_dist, 
                            theta_star=theta_star)
                legend_elements.append(matplotlib.lines.Line2D(
                    [0], [0], color=colors[i], lw=1, label=dist))
                
            elif dist == 'KDE':
                kde_dist = fit_kde_dist(thetas)
                Z = kde_dist.pdf(thetas.transpose())
                score = stats.scoreatpercentile(Z, (1-alpha)*100) 
                g.map_offdiag(_add_scipy_dist_CI, color=colors[i], columns=theta_names, 
                            ncells=100, alpha=score, dist=kde_dist, 
                            theta_star=theta_star)
                legend_elements.append(matplotlib.lines.Line2D(
                    [0], [0], color=colors[i], lw=1, label=dist))
            
    _set_axis_limits(g, axis_limits, thetas, theta_star)
    
    for ax in g.axes.flatten():
        ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='both')
        
        if add_legend:
            xvar, yvar, loc = _get_variables(ax, theta_names)
            if loc == (len(theta_names)-1,0):
                ax.legend(handles=legend_elements, loc='best', prop={'size': 8})
    if title:
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title) 
        
    # Work in progress
    # Plot lower triangle graphics in separate figures, useful for presentations
    lower_triangle_only = False
    if lower_triangle_only:
        for ax in g.axes.flatten():
            xvar, yvar, (xloc, yloc) = _get_variables(ax, theta_names)
            if xloc < yloc: # lower triangle
                ax.remove()
                
                ax.set_xlabel(xvar)
                ax.set_ylabel(yvar)
                
                fig = plt.figure()
                ax.figure=fig
                fig.axes.append(ax)
                fig.add_axes(ax)
                
                f, dummy = plt.subplots()
                bbox = dummy.get_position()
                ax.set_position(bbox) 
                dummy.remove()
                plt.close(f)

                ax.tick_params(reset=True)
                
                if add_legend:
                    ax.legend(handles=legend_elements, loc='best', prop={'size': 8})
                
        plt.close(g.fig)
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
        
def fit_rect_dist(theta_values, alpha):
    """
    Fit an alpha-level rectangular distribution to theta values
    
    Parameters
    ----------
    theta_values: DataFrame
        Theta values, columns = variable names
    alpha: float, optional
        Confidence interval value
    
    Returns
    ---------
    tuple containing lower bound and upper bound for each variable
    """
    assert isinstance(theta_values, pd.DataFrame)
    assert isinstance(alpha, (int, float))
    
    tval = stats.t.ppf(1-(1-alpha)/2, len(theta_values)-1) # Two-tail
    m = theta_values.mean()
    s = theta_values.std()
    lower_bound = m-tval*s
    upper_bound = m+tval*s
    
    return lower_bound, upper_bound
    
def fit_mvn_dist(theta_values):
    """
    Fit a multivariate normal distribution to theta values
    
    Parameters
    ----------
    theta_values: DataFrame
        Theta values, columns = variable names
    
    Returns
    ---------
    scipy.stats.multivariate_normal distribution
    """
    assert isinstance(theta_values, pd.DataFrame)
    
    dist = stats.multivariate_normal(
        theta_values.mean(), theta_values.cov(), allow_singular=True)
    return dist

def fit_kde_dist(theta_values):
    """
    Fit a Gaussian kernel-density distribution to theta values
    
    Parameters
    ----------
    theta_values: DataFrame
        Theta values, columns = variable names
    
    Returns
    ---------
    scipy.stats.gaussian_kde distribution
    """
    assert isinstance(theta_values, pd.DataFrame)

    dist = stats.gaussian_kde(theta_values.transpose().values)
    
    return dist

def _get_grouped_data(data1, data2, normalize, group_names):
    if normalize:
        data_median = data1.median()
        data_std = data1.std()
        data1 = (data1 - data_median)/data_std
        data2 = (data2 - data_median)/data_std
        
    # Combine data1 and data2 to create a grouped histogram
    data = pd.concat({group_names[0]: data1, 
                    group_names[1]: data2})
    data.reset_index(level=0, inplace=True)
    data.rename(columns={'level_0': 'set'}, inplace=True)
    
    data = data.melt(id_vars='set', value_vars=data1.columns, var_name='columns')
    
    return data

def grouped_boxplot(data1, data2, normalize=False, group_names=['data1', 'data2'],
                    filename=None):
    """
    Plot a grouped boxplot to compare two datasets
    
    The datasets can be normalized by the median and standard deviation of data1.
    
    Parameters
    ----------
    data1: DataFrame
        Data set, columns = variable names
    data2: DataFrame
        Data set, columns = variable names
    normalize : bool, optional
        Normalize both datasets by the median and standard deviation of data1
    group_names : list, optional
        Names used in the legend
    filename: string, optional
        Filename used to save the figure
    """
    assert isinstance(data1, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)
    assert isinstance(normalize, bool)
    assert isinstance(group_names, list)
    assert isinstance(filename, (type(None), str))
        
    data = _get_grouped_data(data1, data2, normalize, group_names)
    
    plt.figure()
    sns.boxplot(data=data, hue='set', y='value', x='columns', 
                order=data1.columns)

    plt.gca().legend().set_title('')
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

def grouped_violinplot(data1, data2, normalize=False, group_names=['data1', 'data2'],
                       filename=None):
    """
    Plot a grouped violinplot to compare two datasets
    
    The datasets can be normalized by the median and standard deviation of data1.
    
    Parameters
    ----------
    data1: DataFrame
        Data set, columns = variable names
    data2: DataFrame
        Data set, columns = variable names
    normalize : bool, optional
        Normalize both datasets by the median and standard deviation of data1
    group_names : list, optional
        Names used in the legend
    filename: string, optional
        Filename used to save the figure
    """
    assert isinstance(data1, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)
    assert isinstance(normalize, bool)
    assert isinstance(group_names, list)
    assert isinstance(filename, (type(None), str))
    
    data = _get_grouped_data(data1, data2, normalize, group_names)
    
    plt.figure()
    sns.violinplot(data=data, hue='set', y='value', x='columns',
                   order=data1.columns, split=True)
    
    plt.gca().legend().set_title('')
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
