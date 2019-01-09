import numpy as np
import pandas as pd
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
except:
    pass
try:
    from scipy import stats
except:
    pass


def _get_variables(ax,columns):
    sps = ax.get_subplotspec()
    nx = sps.get_geometry()[0]
    ny = sps.get_geometry()[1]
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

def _add_scatter(x,y,color,label,columns,theta_star=None):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax, columns)
    
    ax.scatter(x, y, s=10)
    if theta_star is not None:
        ax.scatter(theta_star[xvar], theta_star[yvar], c='k', s=35)
    
def _add_rectangle_CI(x,y,color,label,columns,alpha):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    tval = stats.t.ppf(1-(1-alpha)/2, len(x)-1) # Two-tail
    xm = x.mean()
    ym = y.mean()
    xs = x.std()
    ys = y.std()
    
    ax.plot([xm-tval*xs, xm+tval*xs], [ym-tval*ys, ym-tval*ys], color='r')
    ax.plot([xm+tval*xs, xm+tval*xs], [ym-tval*ys, ym+tval*ys], color='r')
    ax.plot([xm+tval*xs, xm-tval*xs], [ym+tval*ys, ym+tval*ys], color='r')
    ax.plot([xm-tval*xs, xm-tval*xs], [ym+tval*ys, ym-tval*ys], color='r')
"""
def _add_multivariate_normal_CI(x,y,color,label,columns,ncells,alpha,mvn_dist,theta_star):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    X,Y = _get_XYgrid(x,y,ncells)
    
    data_slice = []
    for var in theta_star.index:
        if var == xvar:
            data_slice.append(X)
        elif var == yvar:
            data_slice.append(Y)
        elif var not in [xvar,yvar]:
            data_slice.append(np.array([[theta_star[var]]*ncells]*ncells))
    data_slice = np.dstack(tuple(data_slice))
        
    Z = mvn_dist.pdf(data_slice)
    Z = Z.reshape((ncells, ncells))
    
    ax.contour(X,Y,Z, levels=[alpha], colors='b') 

def _add_gaussian_kde_CI(x,y,color,label,columns,ncells,alpha,kde_dist,theta_star):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    X,Y = _get_XYgrid(x,y,ncells)
    
    data_slice = []
    for var in theta_star.index:
        if var == xvar:
            data_slice.append(X.ravel())
        elif var == yvar:
            data_slice.append(Y.ravel())
        elif var not in [xvar,yvar]:
            data_slice.append(np.array([theta_star[var]]*ncells*ncells))
    data_slice = np.array(data_slice)
        
    Z = kde_dist.pdf(data_slice)
    Z = Z.reshape((ncells, ncells))
    
    ax.contour(X,Y,Z, levels=[alpha], colors='r') 
"""
def _add_scipy_dist_CI(x,y,color,label,columns,ncells,alpha,dist,theta_star):
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
        print('Invalid distribution')
        
    Z = dist.pdf(data_slice)
    Z = Z.reshape((ncells, ncells))
    
    ax.contour(X,Y,Z, levels=[alpha], colors='r') 
"""
def _add_LR_contour(x,y,color,label,columns,LR,compare,theta_star):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    import itertools
    from scipy.interpolate import griddata
    
    search_ranges = {} 
    for var in columns:
        if var in [xvar,yvar]:
            search_ranges[var] = LR[var].unique()
        else:
            search_ranges[var] = [theta_star[var]]

    data_slice = pd.DataFrame(list(itertools.product(*search_ranges.values())),
                            columns=search_ranges.keys())
    data_slice['obj'] = griddata(np.array(LR[columns]),
                         np.array(LR[['obj']]),
                         np.array(data_slice[columns]),
                         method='linear',
                         rescale=True)
    
    X = data_slice[xvar]
    Y = data_slice[yvar]
    Z = data_slice['obj']
    
    triang = tri.Triangulation(X, Y)
    cmap = plt.cm.get_cmap('Greys')
    
    plt.tricontour(triang,Z,[compare], colors='r')
    plt.tricontourf(triang,Z,cmap=cmap)
"""
def _set_axis_limits(g, axis_limits, theta_vals):
    
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
            
            
def pairwise_plot(theta_values, theta_star=None, CI_dist=None, alpha=0.8, 
                  axis_limits=None, filename=None):
    """
    Plot pairwise relationship for theta values.
    
    Parameters
    ----------
    theta_values: `pandas DataFrame` (columns = variable names)
        Theta estimates.  
    theta_star: `dict` or `pandas Series` (index or key = variable names) (optional)
        Theta*
    CI_dist: `string` (optional)
        Statistical distribution used for confidence intervals,  
        options = 'rectangular', 'multivariate_normal', and 'gaussian_kde'
    alpha: `float` (optional)
        Confidence interval value, default = 0.8
    axis_limits: `dict` or `pandas Series` (optional)
        Axis limits in the format {variable: [min, max]}
    filename: `string` (optional)
        Filename used to save the figure
        
    Returns
    ----------
    if CI_dist = 'multivariate_normal' or 'gaussian_kde', the scipy 
    distribution is returned
    """

    if len(theta_values) == 0:
        return('Empty data')    
    if isinstance(theta_star, dict):
        theta_star = pd.Series(theta_star)
    
    columns = theta_values.columns

    g = sns.PairGrid(theta_values)
    g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)

    g.map_upper(_add_scatter, columns=columns, theta_star=theta_star)
    g.map_lower(_add_scatter, columns=columns, theta_star=theta_star)
    
    if CI_dist is not None:
        if CI_dist == 'rectangular':
            scipy_dist = None
            g.map_lower(_add_rectangle_CI, columns=columns, alpha=alpha)
            g.map_upper(_add_rectangle_CI, columns=columns, alpha=alpha)
        elif CI_dist in ['multivariate_normal', 'gaussian_kde']:
            if CI_dist == 'multivariate_normal':
                scipy_dist = stats.multivariate_normal(theta_values.mean(), 
                                    theta_values.cov(), allow_singular=True)
                Z = scipy_dist.pdf(theta_values)
            elif CI_dist == 'gaussian_kde':
                scipy_dist = stats.gaussian_kde(theta_values.transpose().values)
                Z = scipy_dist.pdf(theta_values.transpose())
                
            score = stats.scoreatpercentile(Z.transpose(), (1-alpha)*100) 
            g.map_lower(_add_scipy_dist_CI, columns=columns, ncells=100, 
                            alpha=score, dist=scipy_dist, theta_star=theta_star)
            g.map_upper(_add_scipy_dist_CI, columns=columns, ncells=100, 
                            alpha=score, dist=scipy_dist, theta_star=theta_star)

    _set_axis_limits(g, axis_limits, theta_values)
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    
    if CI_dist is not None:
        return scipy_dist
#
#def pairwise_likelihood_ratio_plot(theta_LR, objval, alpha, S, theta_star,
#                                   axis_limits=None, filename=None):
#    """
#    Plot pairwise relationship for each variable, filtered by chi-squared
#    alpha test using the objective for each entry.
#    
#    Parameters
#    ----------
#    theta_LR: `pandas DataFrame` (columns = variable names plus 'obj') 
#        Objective for each estimate of theta (returned by 
#        parmest.likelihood_ratio).
#    objval: `float`
#        Objective value for theta star
#    alpha: `float`
#        Alpha value for the chi-squared test
#    S: `float`
#        Number of samples
#    theta_star: `dict` or `pandas Series` (index = variable names)
#        Theta values used in 2D distribution slice, generally set to theta*
#    axis_limits: `dict` or `pandas Series` (optional)
#        Axis limits in the format {variable: [min, max]}        
#    filename: `string` (optional)
#        Filename used to save the figure
#        
#    Returns
#    -----------
#    Theta values filtered by chi-squared alpha test 
#    """  
#    if isinstance(theta_star, dict):
#        theta_star = pd.Series(theta_star)
#    
#    chi2_val = stats.chi2.ppf(alpha, 2)
#    compare = objval * ((chi2_val / (S - 2)) + 1)
#    alpha_region = theta_LR[theta_LR['obj'] < compare]
#    theta_est = alpha_region.drop('obj', axis=1)
#    
#    columns = theta_est.columns
#    
#    g = sns.PairGrid(theta_est)
#    g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)
#    
#    g.map_upper(_add_LR_contour, columns=columns, LR=theta_LR, compare=compare, 
#                theta_star=theta_star)
#    g.map_upper(_add_scatter, columns=columns, theta_star=theta_star)
#    
#    g.map_lower(_add_LR_contour, columns=columns, LR=theta_LR, compare=compare, 
#                theta_star=theta_star)
#    g.map_lower(_add_scatter, columns=columns, theta_star=theta_star)
#
#    _set_axis_limits(g, axis_limits, theta_est)
#
#    if filename is None:
#        plt.show()
#    else:
#        plt.savefig(filename)
#        plt.close()
#    
#    return alpha_region
#
#
#def pairwise_bootstrap_plot(theta_est, alpha, theta_star, axis_limits=None, 
#                             filename=None):
#    """
#    Plot pairwise relationship for theta estimates along with confidence 
#    intevals using multivariate normal and kernel density estimate distributions
#    
#    Parameters
#    ----------
#    theta_est: `pandas DataFrame` (columns = variable names)
#        Theta estimate (returned by parmest.bootstrap). 
#    alpha: `float`
#        Confidence interval
#    theta_star: `dict` or `pandas Series` (index = variable names)
#        Theta values used in 2D distribution slice, generally set to theta*
#    axis_limits: `dict` or `pandas Series` (optional)
#        Axis limits in the format {variable: [min, max]}
#    filename: `string` (optional)
#        Filename used to save the figure
#    
#    Returns
#    --------
#    Mutlivariate normal distribution (scipy.stats.multivariate_normal), 
#    gaussian kde distribution (scipy.stats.gaussian_kde)
#    """
#    if 'samples' in theta_est.columns:
#        theta_est = theta_est.drop('samples', axis=1)      
#    if isinstance(theta_star, dict):
#        theta_star = pd.Series(theta_star)
#        
#    mvn_dist = stats.multivariate_normal(theta_est.mean(), theta_est.cov(), allow_singular=True)
#    mvnZ = mvn_dist.pdf(theta_est)
#    mvn_score = stats.scoreatpercentile(mvnZ.transpose(), (1-alpha)*100) 
#    
#    kde_dist = stats.gaussian_kde(theta_est.transpose().values) # data.shape = (#dim, #data)    
#    kdeZ = kde_dist.pdf(theta_est.transpose())
#    kde_score = stats.scoreatpercentile(kdeZ.transpose(), (1-alpha)*100) 
#    
#    columns = theta_est.columns
#    ncells = 100
#                    
#    g = sns.PairGrid(theta_est)
#    g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)
#    #g.map_diag(sns.distplot, fit=stats.norm, hist=False,  fit_kws={'color': 'b'}) #, kde=False, norm_hist=False) # histogram and kde estimate
#    #g.map_diag(sns.kdeplot) #, color='r')
#
#    g.map_lower(_add_scatter, columns=columns, theta_star=theta_star)
#    g.map_lower(_add_rectangle_CI, columns=columns, alpha=alpha)
#    g.map_lower(_add_multivariate_normal_CI, columns=columns, ncells=ncells, 
#                alpha=mvn_score, mvn_dist=mvn_dist, theta_star=theta_star)
#    g.map_lower(_add_gaussian_kde_CI, columns=columns, ncells=ncells, 
#                alpha=kde_score, kde_dist=kde_dist, theta_star=theta_star)
#    
#    g.map_upper(_add_scatter, columns=columns, theta_star=theta_star)
#    g.map_upper(_add_rectangle_CI, columns=columns, alpha=alpha)
#    g.map_upper(_add_multivariate_normal_CI, columns=columns, ncells=ncells, 
#                alpha=mvn_score, mvn_dist=mvn_dist, theta_star=theta_star)
#    g.map_upper(_add_gaussian_kde_CI, columns=columns, ncells=ncells, 
#                alpha=kde_score, kde_dist=kde_dist, theta_star=theta_star)
#
#    _set_axis_limits(g, axis_limits, theta_est)
#    
#    if filename is None:
#        plt.show()
#    else:
#        plt.savefig(filename)
#        plt.close()
#    
#    """
#    import itertools
#    for xcol, ycol in itertools.combinations(theta_est.columns,2):
#        g = sns.JointGrid(x=xcol, y=ycol, data=theta_est);
#        fig = g.fig
#        ax = fig.gca()
#        
#        g.plot_marginals(sns.distplot, kde=False, hist=True, norm_hist=False)
#        g.plot_joint(_add_scatter,color=None, label=None, columns=columns, 
#            theta_star=theta_star)
#        g.plot_joint(_add_rectangle_CI, color=None,label=None,columns=columns, alpha=alpha)
#        
#        #_set_axis_limits(g, axis_limits, theta_est)
#        #ax.set_xlim(axis_limits[xcol])
#        #ax.set_ylim(axis_limits[ycol])
#    """
#    return mvn_dist, kde_dist
