import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
except:
    pass
try:
    from scipy import stats
except:
    pass
import seaborn as sns


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
    
    ax.scatter(x, y, s=15)
    if theta_star is not None:
        ax.scatter(theta_star[xvar], theta_star[yvar], c='k', s=30)
    
def _add_rectangle_CI(x,y,color,label,columns,alpha):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    tval = stats.t.ppf(1-(1-alpha)/2, len(x)-1) # Two-tail
    xm = x.mean()
    ym = y.mean()
    xs = x.std()
    ys = y.std()
    
    ax.plot([xm-tval*xs, xm+tval*xs], [ym-tval*ys, ym-tval*ys], color='darkgrey')
    ax.plot([xm+tval*xs, xm+tval*xs], [ym-tval*ys, ym+tval*ys], color='darkgrey')
    ax.plot([xm+tval*xs, xm-tval*xs], [ym+tval*ys, ym+tval*ys], color='darkgrey')
    ax.plot([xm-tval*xs, xm-tval*xs], [ym+tval*ys, ym-tval*ys], color='darkgrey')

def _add_multivariate_normal_CI(x,y,color,label,columns,ncells,alpha,mvn_dist,theta_star):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    X,Y = _get_XYgrid(x,y,ncells)
    
    data = []
    for var in theta_star.index:
        if var == xvar:
            data.append(X)
        elif var == yvar:
            data.append(Y)
        elif var not in [xvar,yvar]:
            data.append(np.array([[theta_star[var]]*ncells]*ncells))
    pos = np.dstack(tuple(data))
        
    Z = mvn_dist.pdf(pos)
    Z = Z.reshape((ncells, ncells))
    ax.contour(X,Y,Z, levels=[alpha], colors='b') 

def _add_gaussian_kde_CI(x,y,color,label,columns,ncells,alpha,kde_dist,theta_star):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    X,Y = _get_XYgrid(x,y,ncells)
    
    data = []
    for var in theta_star.index:
        if var == xvar:
            data.append(X.ravel())
        elif var == yvar:
            data.append(Y.ravel())
        elif var not in [xvar,yvar]:
            data.append(np.array([theta_star[var]]*ncells*ncells))
    pos = np.array(data)
        
    Z = kde_dist.pdf(pos)
    Z = Z.reshape((ncells, ncells))
    ax.contour(X,Y,Z, levels=[alpha], colors='r') 

def _add_SSE_contour(x,y,color,label,columns,SSE,compare):
    ax = plt.gca()
    xvar, yvar, loc = _get_variables(ax,columns)
    
    x = SSE[xvar]
    y = SSE[yvar]
    z = SSE['SSE']
    triang = tri.Triangulation(x, y)
    plt.tricontour(triang,z,[compare], colors='r') 
      
def pairwise_plot(theta_est, theta_star=None, axis_limits=None, filename=None):
    """
    Plot pairwise relationship for theta estimates.
    
    Parameters
    ----------
    theta_est: `pandas DataFrame` (columns = variable names)
        Theta estimates.  If the DataFrame contains column names 
        'samples' or 'SSE', these columns will not be included in the plot.
    theta_star: `dict` or `pandas Series` (index = variable names)
        Theta*
    axis_limits: `dict` or `pandas Series` (optional)
        Axis limits in the format {variable: [min, max]}
    filename: `string` (optional)
        Filename used to save the figure
    """
    if 'samples' in theta_est.columns:
        theta_est = theta_est.drop('samples', axis=1)
    if 'SSE' in theta_est.columns:
        theta_est = theta_est.drop('SSE', axis=1)
    if isinstance(theta_star, dict):
        theta_star = pd.Series(theta_star)

    columns = theta_est.columns

    g = sns.PairGrid(theta_est)
    g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)

    g.map_upper(_add_scatter, columns=columns, theta_star=theta_star)
    g.map_lower(_add_scatter, columns=columns, theta_star=theta_star)

    if axis_limits is not None:
        for ax in g.fig.get_axes():
            xvar, yvar, (xloc, yloc) = _get_variables(ax,columns)
            if xloc != yloc: # not on diagonal
                ax.set_ylim(axis_limits[yvar])
                ax.set_xlim(axis_limits[xvar])

    if filename is not None:
        plt.savefig(filename)

def pairwise_likelihood_ratio_plot(theta_SSE, objval, alpha, S, 
                                   axis_limits=None, filename=None):
    """
    Plot pairwise relationship for each variable, filtered by chi-squared
    alpha test using the sum of the squared error for each entry.
    
    Parameters
    ----------
    theta_SSE: `pandas DataFrame` (columns = variable names plus SSE) 
        Sum of squared errors for each estimate of theta (returned by 
        parmest.likelihood_ratio).
    objval: `float`
        Objective value for theta star
    alpha: `float`
        Alpha value for the chi-squared test
    S: `float`
        Number of samples
    axis_limits: `dict` or `pandas Series` (optional)
        Axis limits in the format {variable: [min, max]}        
    filename: `string` (optional)
        Filename used to save the figure
    """  
    chi2_val = stats.chi2.ppf(alpha, 2)
    compare = objval * ((chi2_val / (S - 2)) + 1)
    alpha_region = theta_SSE[theta_SSE['SSE'] < compare]
    theta_est = alpha_region.drop("SSE", axis=1)
    
    if axis_limits is None:
        axis_limits = {}
        for col in theta_est:
            temp = np.abs(theta_est[col].max()-theta_est[col].min())
            axis_limits[col] = [theta_est[col].min()-temp/10, 
                                theta_est[col].max()+temp/10]
    columns = theta_est.columns
    
    g = sns.PairGrid(theta_est)
    g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)

    g.map_upper(_add_scatter, columns=columns)

    g.map_lower(_add_SSE_contour, columns=columns, SSE=theta_SSE, compare=compare)
    g.map_lower(_add_scatter, columns=columns)

    if axis_limits is not None:
        for ax in g.fig.get_axes():
            xvar, yvar, (xloc, yloc) = _get_variables(ax,columns)
            if xloc != yloc: # not on diagonal
                ax.set_ylim(axis_limits[yvar])
                ax.set_xlim(axis_limits[xvar])

    if filename is not None:
        plt.savefig(filename)
    
    return alpha_region

def pairwise_bootstrap_plot(theta_est, theta_star, alpha, axis_limits=None, 
                            filename=None):
    """
    Plot pairwise relationship for theta estimates along with confidence 
    intevals using multivariate normal and kernel density estimate distributions
    
    Parameters
    ----------
    theta_est: `pandas DataFrame` (columns = variable names)
        Theta estimate (returned by parmest.bootstrap). If the DataFrame 
        contains column names 'samples', these will not be included in the plot.
    theta_star: `dict` or `pandas Series` (index = variable names)
        Theta*
    alpha: `float`
        Confidence interval
    axis_limits: `dict` or `pandas Series` (optional)
        Axis limits in the format {variable: [min, max]}
    filename: `string` (optional)
        Filename used to save the figure
    
    Returns
    --------
    Mutlivariate normal distribution (scipy.stats.multivariate_normal), 
    gaussian kde distribution (scipy.stats.gaussian_kde)
    """
    if 'samples' in theta_est.columns:
        theta_est = theta_est.drop('samples', axis=1)      
    if isinstance(theta_star, dict):
        theta_star = pd.Series(theta_star)
          
    m = theta_est.mean()
    c = theta_est.cov()
    mvn_dist = stats.multivariate_normal(m, c, allow_singular=True)
    mvnZ = mvn_dist.pdf(theta_est)
    mvn_score = stats.scoreatpercentile(mvnZ.transpose(), (1-alpha)*100) 
    
    kde_dist = stats.gaussian_kde(theta_est.transpose().values) # data.shape = (#dim, #data)
    kdeZ = kde_dist.pdf(theta_est.transpose())
    kde_score = stats.scoreatpercentile(kdeZ.transpose(), (1-alpha)*100) 
    
    columns = theta_est.columns
    ncells = 100
    
    g = sns.PairGrid(theta_est)
    g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)
    #g.map_diag(sns.distplot, fit=stats.norm, hist=False,  fit_kws={'color': 'b'}) #, kde=False, norm_hist=False) # histogram and kde estimate
    #g.map_diag(sns.kdeplot) #, color='r')

    g.map_upper(_add_scatter, columns=columns, theta_star=theta_star)
    g.map_lower(_add_scatter, columns=columns, theta_star=theta_star)

    g.map_lower(_add_rectangle_CI, columns=columns, alpha=alpha)
    g.map_lower(_add_multivariate_normal_CI, columns=columns, ncells=ncells, 
                alpha=mvn_score, mvn_dist=mvn_dist, theta_star=theta_star)
    g.map_lower(_add_gaussian_kde_CI, columns=columns, ncells=ncells, 
                alpha=kde_score, kde_dist=kde_dist, theta_star=theta_star)

    if axis_limits is not None:
        for ax in g.fig.get_axes():
            xvar, yvar, (xloc, yloc) = _get_variables(ax,columns)
            if xloc != yloc: # not on diagonal
                ax.set_ylim(axis_limits[yvar])
                ax.set_xlim(axis_limits[xvar])
    if filename is not None:
        plt.savefig(filename)
        
    return mvn_dist, kde_dist
