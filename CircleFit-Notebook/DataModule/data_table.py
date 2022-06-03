# -*- coding: utf-8 -*-
"""Data_table class which is the base for the other data types like data_line,
data_complex and data_IQ. The idea is adapted from holoviews (holoviews.org)
and builds on pandas as excellent data processing tool.

Author: Christian Schneider <c.schneider@uibk.ac.at>
Date: 16.03.2018
"""
from .base import data_module_base
import pandas as pd
from .plot_style import color_scheme, cc, check_color, color_labels
import holoviews as hv
import holoviews.operation.datashader as hd
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting as bp
from bokeh.models import HoverTool
import scipy.optimize as sp_opt
import scipy.signal as sp_sig
import scipy.interpolate as sp_intp
from collections import OrderedDict


hv.extension('bokeh', logo=False)


class data_table(data_module_base):
    """Class for table like data with one or more independent variables and
    one or more dependent variables

    Parameters
    -----------
    data_arrays : list, array, np.array
        Data to save. Format should be
        [array_x, array_y, array_z, ...]
    data_names : list, array, np.array
        Labels for the data. Format should be
        ['Label first axis', 'Label second axis', ...]
    """

    def __init__(self, data_arrays=None, data_names=None):
        super().__init__()
        # Create default data and names
        if data_arrays is None:
            data_arrays = [[np.nan], [np.nan]]
        df_names = ['x', 'y', 'z'][:len(data_arrays)]
        df_names += ['x{}'.format(i) for i in range(3, len(data_arrays))]

        # Replace default names by the names given
        if data_names:
            for idx in range(len(data_names)):
                if data_names[idx]:
                    df_names[idx] = data_names[idx]

        # Create dictionary for pandas dataframe
        tmp_dict = {}
        for idx, d in enumerate(data_arrays):
            tmp_dict[df_names[idx]] = data_arrays[idx]

        # Create dataframe
        self.df = pd.DataFrame(data=tmp_dict)

        # Create easy access variables
        self.name_x = df_names[0]
        self.name_y = df_names[1]

        # Create further variables
        self._fit_executed = False
        self._fit_labeles = None

        # Order dataframe
        self.df = self.df[df_names]

    def _repr_html_(self):
        """Show pandas dataframe as default representation"""
        return '<h3>data_table</h3>' + self.df.head().to_html()

    def return_x(self):
        return np.array(self.df[self.name_x][self.idx_min:self.idx_max])

    def return_y(self):
        if '{}_smoothed'.format(self.name_y) in self.df.keys():
            # Return smoothed data
            name = '{}_smoothed'.format(self.name_y)
            return np.array(self.df[name][self.idx_min:self.idx_max])
        else:
            return np.array(self.df[self.name_y][self.idx_min:self.idx_max])

    @property
    def x(self):
        return self.return_x()

    @x.setter
    def x(self, value):
        if len(value) != len(self.x):
            self.df = self.df.reindex(range(len(value)))
        self.df[self.name_x][self.idx_min:self.idx_max] = value

    @property
    def y(self):
        return self.return_y()

    @y.setter
    def y(self, value):
        self.df[self.name_y][self.idx_min:self.idx_max] = value

    def rename_x(self, new_name):
        """Rename x variable"""
        self.df = self.df.rename(columns={self.name_x: new_name})
        self.name_x = new_name

    def rename_y(self, new_name):
        """Rename y variable"""
        self.df = self.df.rename(columns={self.name_y: new_name})
        self.name_y = new_name

    def rename(self, column_name, new_column_name):
        self.df = self.df.rename(column={column_name: new_column_name})

    def add_column(self, column, name):
        """Add a new column to the dataframe. New Column has to be equal or
        shorter then first column (independent variable)
        
        Parameters
        -----------
        column : list, array
            New column to add
        name : str
            Name for column
        """
        self.df[name] = pd.Series(column)

    def import_data(self, data_arrays, data_names=None):
        """Import data from new array. Naming highly recommended!
        Assumes first given array is the array of the independent variable.
        Parameters
        -----------
        data_arrays : list, array, np.array
            List of data arrays. Structure
            [[x, x, x, x, ....], [y, y, y,....], [z, z, z, z, ...]
        data_names : list, array, np.array, None
            List of names for arrays:
            ['x', 'y', 'Resistances', ... ]
        """
        if data_names:
            order_names = data_names
        else:
            order_names = self.df.columns

        # Check if column/dependent variable already exists, if not --> create
        for idx, d_name in zip(range(1, len(order_names) + 1),
                               order_names[1:]):
            if d_name not in self.df.keys():
                if (self.df[order_names[0]] == data_arrays[0]).all():
                    # Quick check: If independent variable is the same as
                    # already given, just add new column
                    self.df[d_name] = data_arrays[idx]
                else:
                    # Fill existing array with np.nan
                    self.df[d_name] = [np.nan for i in
                                       range(len(self.df[order_names[0]]))]

        # Add values if not already added
        for idx, d in zip(range(len(data_arrays[0])), data_arrays[0]):
            # Check if value for independent variable already exists
            df_idxs = self.df.index[self.df[order_names[0]] == d]
            if not df_idxs.empty:
                df_idx = df_idxs[0]  # take first element
                for i in range(1, len(data_arrays)):
                    if np.isfinite(self.df[order_names[i]][df_idx]):
                        # TODO Is running double if added above --> change
                        # Check if entry is np.nan, if not use mean
                        self.df[order_names[i]][df_idx] += data_arrays[i][idx]
                        self.df[order_names[i]][df_idx] /= 2
                    else:
                        # Else just take new value
                        self.df[order_names[i]][df_idx] = data_arrays[i][idx]
            else:
                self.df = self.df.append({key: value[idx] for key, value
                                          in zip(order_names, data_arrays)},
                                         ignore_index=True)
                pass

    def plot_hv(self, x=None, y=None, height=400, width=800,
                title='', color=None):
        """Plot table with Holoviews

        Parameters
        -----------
        x : None, str, list
            Column name(s) used for x axis
        y : None, str, list
            Column name(s) used for y axis
        height : int
            Height of plot in pixels
        width : int
            Width of plot in pixels
        title : str
            Title of plot
        color : str
            Color for plot
        """
        if x is None:
            x_val = self.name_x
        else:
            x_val = x

        if y is None:
            y_val = self.name_y
        else:
            y_val = y

        if color is None:
            color = color_scheme[0]
        elif color in cc.keys():
            color = cc[color]

        # Plot
        s = hv.Scatter(self.df[self.idx_min:self.idx_max],
                       x_val, y_val, label=title)
        scatter_plots = hd.dynspread(hd.datashade(hv.Curve(s), cmap=[color]))

        return scatter_plots.opts(plot=dict(height=height, width=width,
                                            show_grid=True))

    # Plotting #################################################################
    def plot(self, style='-o', x_col=None, y_col=None, color=None, xscale=1,
             yscale=1,
             plot_fit=True, linewidth=1.5, markersize=3,
             fit_linewidth=1, plot_errors=True, legend=None,
             legend_pos=0, engine=None, title='', show=True, fig=None,
             fitcolor='r', fit_on_top=False, **kwargs):
        """Plot data and optionally the fit.

        Choose between two plot-engines: 'bokeh' and 'pyplot'

        Parameters
        ----------
        style : str
            Specify plot style in matplotlib language. E.g. 'b-o' for
            blue dots connected with a line.
        x_col : str, None
            Name of column for plot on x axis. If none is given, first column is
            taken
        y_col : str, None
            Name of column for plot on y axis. If none is given, second column
            is taken
        color : str
            Color shortcut (eg. 'b') or specific color like '#123456'.
            Type cc to see available colors.
        xscale : float
            X scaling
        yscale : float
            Y scaling
        plot_fit : bool
            Plot the fit if available
        linewidth : int
            Thickness of lines
        markersize : int
            Size of markers
        fit_linewidth : int
            Thickness of fit line
        plot_errors : bool
            Plot error bars
        legend : list
            Custom legend for plot ['Label 1', 'Label 2']]
        legend_pos : int, str
            Location of legend
        engine : str
            Plot engine. Choose between 'bokeh' and 'pyplot'
        title : str
            Title of the plot
        show : bool
            Directly show plot. Useful if one wants to add labels etc.
            and show plot afterwards.
        fig : object
            Figure to plot into (bokeh/matplotlib figure object)
        fitcolor : str
            Color shortcut (like 'b') or specific color (like #123456) for
            fit
        fit_on_top : bool
            Data over fit or fit over data

        Returns
        --------
        object
            Returns a fit object for bokeh or matplotlib. This is useful, if
            you want to add for example another points, lines, labels, etc
            afterwards.
        """
        # Get default data or specified columns
        if x_col is None:
            x_col = self.name_x
        if y_col is None:
            if '{}_smoothed'.format(self.name_y) in self.df.keys():
                y_col = '{}_smoothed'.format(self.name_y)
            else:
                y_col = self.name_y

        x = np.array(self.df[x_col])[self.idx_min:self.idx_max]
        y = np.array(self.df[y_col])[self.idx_min:self.idx_max]

        # Check for default plot engine
        if engine is None:
            if len(x) > 5000:
                engine = 'h'  # Use holoviews for large data
            else:
                engine = 'b'  # Use bokeh for small data

        # Don't show plot if figure is given (normally one does not need this)
        if fig:
            show = False
        # Get fit function
        if plot_fit and self._fit_executed:
            exec(self._fit_function_code)
            possibles = globals().copy()
            possibles.update(locals())
            fitfunc = possibles.get(self._fit_function)
            if not fitfunc:
                raise Exception('Method %s not implemented' %
                                self._fit_function)

        # Check for color specified in style
        if color is None:
            color = check_color(style)

        if color in color_labels:
            c = cc[color]
        else:
            c = color

        # Bokeh
        if engine in ['bokeh', 'b']:
            # Create Figure if no figure is given
            if fig is None:
                tools = ['box_zoom', 'pan', 'wheel_zoom', 'reset',
                         'save', 'hover']
                fig = bp.figure(plot_width=800, plot_height=400, tools=tools,
                                toolbar_location='above', title=title)

            # Empty legend if no legend is given
            if legend is None:
                legend = ""

            # Plot Data
            if ("{} Errors".format(self.name_y) in self.df.keys() and
                    plot_errors):
                # Just plot dots if error bars given
                fig.circle(x, y, fill_color=c, line_color=c,
                           size=markersize, legend=legend, **kwargs)

                err_xs = []
                err_ys = []
                errs = self.df[self.name_y + ' Errors'][
                       self.idx_min:self.idx_max]
                for xs, ys, yerr in zip(x, y, errs):
                    err_xs.append((xs, xs))
                    err_ys.append((ys - yerr, ys + yerr))
                fig.multi_line(err_xs, err_ys, color=c)
            else:
                for kw in style:
                    if kw == '-':
                        fig.line(x * xscale, y * yscale, line_color=c,
                                 line_width=linewidth, legend=legend, **kwargs)
                    elif kw == 'o':
                        fig.circle(x, y, fill_color=c, line_color=c,
                                   size=markersize, legend=legend, **kwargs)

            # Plot Fit
            if plot_fit and self._fit_executed:
                # Get color
                if fitcolor in color_labels:
                    fc = cc[fitcolor]
                else:
                    fc = fitcolor
                # Plot fit
                fig.line(x * xscale, fitfunc(x, *self._fit_parameters) * yscale,
                         line_color=fc, line_width=fit_linewidth)

            # Format nicer HoverTool
            tooltips = [(x_col, "$x"), (y_col, "$y")]
            fig.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)

            # Add labels
            fig.xaxis.axis_label = x_col
            fig.yaxis.axis_label = y_col

            if show:
                bp.show(fig)
            return fig

        # Matplotlib
        elif engine in ['pyplot', 'p']:
            if not fig:
                try:
                    fig = plt.gcf()
                except:
                    fig = plt.figure()

            plt.title(title)
            # Plot errors
            # Data
            if (plot_errors and "{} Errors".format(self.name_y) in
                    self.df.keys()):
                yerrs = np.array(self.df[self.name_y + ' Errors'][
                                 self.idx_min:self.idx_max]) * yscale
                plt.errorbar(x * xscale, y * yscale, fmt=style, color=c,
                             yerr=yerrs, markersize=markersize, **kwargs)

            else:
                plt.plot(x * xscale, y * yscale, style, color=c,
                         linewidth=linewidth, markersize=markersize,
                         **kwargs)

            # Fit
            if plot_fit and self._fit_executed:
                if not fit_on_top:
                    fit_zorder = 0
                else:
                    fit_zorder = 99
                plt.plot(x * xscale, fitfunc(x, *self._fit_parameters) * yscale,
                         '-', color=fitcolor, linewidth=fit_linewidth,
                         zorder=fit_zorder)

            # Labels
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            # Styling
            plt.grid()
            plt.xlim([x[0] * xscale, x[-1] * xscale])

        elif engine in ['holoviews', 'h', 'hv']:
            return self.plot_hv(color=color, **kwargs)

    def plot_all(self, colormap=None):
        """Fast plot of the whole data in holoviews"""

        # Get default colormap
        if colormap is None:
            colormap = color_scheme

        # Get columns
        cols = self.df.columns[1:]

        # Create dots of the first point for legend (bug in holoviews)
        color_points = hv.NdOverlay({cols[i]:
                                         hv.Points([[self.x[0],
                                                     self.df[cols[i]][0]]],
                                                   label=str(cols[i]))
                                    .opts(style=dict(color=colormap[i]))
                                     for i in range(len(cols))})
        plot = color_points
        for col, c in zip(cols, colormap):
            plot *= self.plot_hv(y=col, color=c)
        return plot

    def fit(self, fitfunc, fit_p_init, boundaries = None,plot=True, plot_init=True,
            plot_params=True, labels=None, **kwargs):
        """Fit `fitfunc` to data with initial fit parameters `fit_p_init`

        Note
        -----
            After the fit, the data module will contain the fit, however it
            should be saved again. Don't forget this.

            The fit will be performed on the stored data, not the plotted
            one. If you use xscale or yscale different than the one you
            plot, you should take it into account

        Parameters
        -----------
        fitfunc : func
            Function to fit to. There exist already some functions in the
            DataModule library (DataModule/fit_functions)
        fit_p_init : list
            Initial (guessed) parameters
        plot : bool
            Plot fit after result
        plot_init : bool
            Plot initial guess.
        plot_params : bool
            Print paramaters for fit
        labels : list
            Label for fitparameters. E.g. ['offset', 'amplitude', 'fr']
        **kwargs : keywords
            Keywords for fitfunction like maxfev=10000, epsfcn=1e-10,
            factor=0.1, xtol=1e-9, ftol=1e-9...

        At the end of the fit the data module will contain the function
        used to fit and some fit-related functions will be enabled.
        The fit parameters are stored in
        self._fit_parameters, self._fit_par_errors
        and the average error (sigma) in self._fit_data_error.

        Returns
        list, list, float

        """
        xsel, ysel = self.return_x(), self.return_y()
        try:
            fit_p_fit, err = sp_opt.curve_fit(fitfunc, xsel, ysel, fit_p_init,
                                              **kwargs)
        except RuntimeError:
            print('At least one fit did not converge', end=' ', flush=True)
            fit_p_fit = np.array([np.nan for i in fit_p_init])
            err = np.array([[np.nan for i in fit_p_init] for j in fit_p_init])
            raise Exception(RuntimeError)

        if plot:
            # Just use bokeh since we don't want to publish this plot
            fig = bp.figure(title='Fit', plot_width=800, plot_height=400)
            if plot_init:
                # Plot initial parameter guess, fit and data
                fig.line(xsel, fitfunc(xsel, *fit_p_init), color=cc['g'],
                         legend='Init guess')

            fig.circle(xsel, ysel, color=cc['b'], legend='Data')
            fig.line(xsel, fitfunc(xsel, *fit_p_fit), color=cc['r'],
                     legend='Fit')
            bp.show(fig)

        # Save fitfunction as string
        import inspect
        self._fit_executed = True
        code = inspect.getsourcelines(fitfunc)[0]
        self._fit_function = fitfunc.__name__
        self._fit_function_code = ''.join(code)
        self._fit_parameters = fit_p_fit
        self._fit_par_errors = np.sqrt(np.diag(err))
        # Chi squared
        self._fit_data_error = (
                np.sum((fitfunc(xsel, *fit_p_fit) - ysel) ** 2) /
                (len(xsel) - 2))
        self._fit_labels = labels

        if plot_params:
            print(self.fit_pars())

        return fit_p_fit, self._fit_par_errors, self._fit_data_error

    def fit_func(self, x=None):
        """Calculates values of fit function for an x-array/values.

        Parameters
        -----------
        x : None, float, list, np.array
            X values. None means same x data as datamodule.

        ToDo
        -----
        Would like to rename the function to calc_fitfunc(self, x=None)
        """
        if not self._fit_executed:
            print('No fit was executed on this data')
            return

        exec(self._fit_function_code)
        possibles = globals().copy()
        possibles.update(locals())
        fitfunc = possibles.get(self._fit_function)
        if not fitfunc:
            raise Exception('Method %s not implemented' % self._fit_function)

        if x is None:
            return fitfunc(self.x, *self._fit_parameters)
        else:
            return fitfunc(x, *self._fit_parameters)

    def localmin(self, min_threshold=None, npoints=1, mode='clip'):
        """Obtain all the local minimas

        Parameters
        -----------
        min_threshold : float
            Only consider minima below this value
        npoints : int
            How many points on each side to use for the comparison.
        mode : str
            'clip' (def) or 'wrap'. If wrap is used, the data will considered
            periodic-like

        Returns
        --------
        np.array
            x and y values of local minima
        """
        xsel = self.x
        ysel = self.y

        if min_threshold is not None:
            msk = ysel <= min_threshold
            xsel = xsel[msk]
            ysel = ysel[msk]

        min_idx = sp_sig.argrelextrema(ysel, np.less, order=npoints, mode=mode)
        return np.vstack((xsel[min_idx], ysel[min_idx]))

    def localmax(self, max_threshold=None, npoints=1, mode='clip'):
        """Obtain all the local maxima

        Parameters
        -----------
        min_threshold : float
            Only consider maxima above this value
        npoints : int
            How many points on each side to use for the comparison.
        mode : str
            'clip' (def) or 'wrap'. If wrap is used, the data will considered
            periodic-like

        Returns
        --------
        np.array
            x and y values of local maxima
        """
        xsel = self.x
        ysel = self.y

        if max_threshold is not None:
            msk = ysel >= max_threshold
            xsel = xsel[msk]
            ysel = ysel[msk]

        min_idx = sp_sig.argrelextrema(ysel, np.greater, order=npoints,
                                       mode=mode)
        return np.vstack((xsel[min_idx], ysel[min_idx]))

    def smooth(self, nnb=21, polyorder=2):
        """Smooth data using the Savitzky-Golay filter.

        Information
            https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

        This has the advantage over moving_average, that the bias of smaller
        local minima/maxima is removed.

        Note
        -----
        Saves the smoothed data as dm.y values. To unsmooth the data run
        `unsmooth()`

        Parameters
        -----------
        nnb : int
            Window length of filter
        polyorder : int
            Polynomial order for the fit
        """
        y_filtered = sp_sig.savgol_filter(self.df[self.name_y], nnb, polyorder)
        self.import_data([self.x, y_filtered],
                         [self.name_x, '{}_smoothed'.format(self.df.keys()[1])])

    def unsmooth(self):
        """Recover unsmoothened y data."""
        try:
            del self.df['{}_smoothed'.format(self.name_y)]
        except KeyError:
            pass

    def interp(self, xnew, kind='cubic'):
        """Interpolates data to new x values.

        Parameters
        -----------
        xnew : list, np.array
            New x array
        kind : str
            Kind of interpolation.
            Choose between 'linear', 'nearest', 'zero', 'slinear', 'quadratic',
            'cubic'

        Returns
        --------
        DataModule
            Returns a new datamodule.

        Example
        --------
            >>> d1 = dm.load_datamodule('foo.dm')
            >>> xnew = np.linspace(d1.x.min(), d1.x.max(), 100)
            >>> d2 = d1.interp(xnew)
        """
        f = sp_intp.interp1d(self.x, self.y, kind=kind)
        return data_table([xnew, f(xnew)], [self.name_x, self.name_y])

    def fit_pars(self):
        """Returns the fit parameters as pandas DataFrame."""

        if not self._fit_executed:
            print('No Fit found. Please run a fit first.')
            return

        df = pd.DataFrame([list(self._fit_parameters),
                           list(self._fit_par_errors)])
        df.index = ['Value', 'Error']
        if self._fit_labels is not None:
            df.columns = self._fit_labels

        return df.T

    def y_value(self, x):
        """ Returns y value for given x array or value using linear
        interpolation.

        Parameters
        ----------
        x : list, np.array, float
            x value(s)

        Returns
        --------
        list, np.array, float
            Interpolated y value for given x value(s)
        """
        return np.interp(x, self.x, self.y)
