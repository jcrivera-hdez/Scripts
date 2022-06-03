# -*- coding: utf-8 -*-
"""
Author: Christian Schneider <c.schneider@uibk.ac.at>
Date: 17.03.2018

data_grid structure for our DataModule.
Powered by xarray (xarray.pydata.org)
"""
from .base import data_module_base
import holoviews as hv
import holoviews.operation.datashader as hd
import scipy.signal as sp_sig
import scipy.interpolate as sp_intp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .data_table import data_table
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .fit_functions import mode_fit, lorentzian_fit

hv.extension('bokeh', logo=False)


class data_grid(data_module_base):
    """Class for grid like data in multiple dimensions. Powered by the
    excellent library xarray for python.

    Initialization should be [x1_coords, x2_coords, ...., xN_coords,
    N_dim_data_tensor]
    Which means that the matrix for data is the last element of the given array.

    Supports also custom names for each dimension which can be given as an
    array. The last element is the name for the values (not the dimesion).
    Example: ['Probe Frequency (GHz)', 'Readout Frequency (GHz)', 'Mag (dB)']

    """

    def __init__(self, data_array, data_names=None):
        super().__init__()
        # Create default names
        df_names = ['x', 'y']
        df_names += ['x{}'.format(i) for i in range(2, len(data_array))]
        df_names[-1] = 'Value'  # Last element is matrix/tensor
        # Replace default names by the names given
        if data_names:
            for idx in range(len(data_names)):
                if data_names[idx]:
                    df_names[idx] = data_names[idx]
        # Create xarray
        self.df = xr.DataArray(data_array[-1],
                               dims=tuple(d_name for d_name in df_names[:-1]),
                               coords={d_name: d_vals for (d_name, d_vals) in
                                       zip(df_names[:-1], data_array[:-1])},
                               name=df_names[-1])

        # Idx variables for limiting data
        self.x_min = 0
        self.x_max = None
        self.y_min = 0
        self.y_max = None
        # Easy access variables
        self.name_x = df_names[0]
        self.name_y = df_names[1]
        self.name_v = df_names[-1]

    # Helpful data functions ###################################################
    def return_coord(self, coord_name):
        return np.array(self.df.coords[coord_name])

    @property
    def x(self):
        """Default for two dim grid: Return first dimension"""
        return self.return_coord(self.name_x)

    @x.setter
    def x(self, value):
        self.df.coords[self.name_x] = value

    @property
    def y(self):
        """Default for two dim grid: Return second dimension"""
        return self.return_coord(self.name_y)

    @y.setter
    def y(self, value):
        self.df.coords[self.name_y] = value

    @property
    def z(self):
        """Default for two dim grid: Return values"""
        return np.array(self.df.values)[self.x_min:self.x_max,
                        self.y_min:self.y_max]

    @z.setter
    def z(self, values):
        self.df.values[self.x_min:self.x_max, self.y_min:self.y_max] = values

    @property
    def values(self):
        return np.array(self.df.values)[self.x_min:self.x_max,
                        self.y_min:self.y_max]

    @values.setter
    def values(self, values):
        self.df.values[self.x_min:self.x_max, self.y_min:self.y_max] = values

    def rename_x(self, new_name):
        self.df = self.df.rename({self.name_x: new_name})
        self.name_x = new_name

    def rename_y(self, new_name):
        self.df = self.df.rename({self.name_y: new_name})
        self.name_y = new_name

    def rename_values(self, new_name):
        self.df = self.df.rename(new_name)
        self.name_v = new_name

    def rename_z(self, new_name):
        return self.rename_values(new_name)  # alias

    def select(self, xrng=None, yrng=None):
        """Limit data between specified ranges.

        This function will select the data in the specified range of the
        x-axis and y-axis. If nothing is specified all the data will be
        selected.

        Note
        -----
        To undo a selection just run `select()` without an argument.

        Parameters
        -----------
        xrng : list
            Start and stop x value [x_start, x_stop]
        yrng : list
            Start and stop y value [y_start, y_stop]
        """
        if xrng is None:
            x_idx = [0, None]
        else:
            x = self.df.coords[self.name_x]
            x_idx = np.where((x >= xrng[0]) & (x <= xrng[1]))[0]

        if yrng is None:
            y_idx = [0, None]
        else:
            y = self.df.coords[self.name_y]
            y_idx = np.where((y >= yrng[0]) & (y <= yrng[1]))[0]

        self.y_min = y_idx[0]
        self.y_max = y_idx[-1]
        self.x_min = x_idx[0]
        self.x_max = x_idx[-1]

    def extract_x(self, x0, plot=True):
        """Extract z values along  axis for specified x value x0.

        This function will return the data at the line corresponding at
        the specified value of the x-axis.
        If the value is not exact, it will take the closest one above the
        value.

        Parameters
        -----------
        x0 : float
            y value for which the data should be extracted
        plot : bool
            Plot the extracted datamodule directly

        Returns
        --------
        DataModule
            data_table module
        """
        x, y = self.df.dims[:2]
        kws = {x: x0}
        ex = self.df.sel(method='nearest', **kws)
        data = data_table([self.y, np.array(ex)[self.y_min:self.y_max]],
                          [y, self.name_v])
        data.par = self.par
        data.comments = self.comments
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop

        if plot:
            display(data.plot_hv(title='{} = {:.6e}'.format(x, x0)))

        return data

    def extract_y(self, y0, plot=True):
        """Extract z values along  axis for specified y value y0.

        This function will return the data at the line corresponding at
        the specified value of the x-axis.
        If the value is not exact, it will take the closest one above the
        value.

        Parameters
        -----------
        y0 : float
            y value for which the data should be extracted
        plot : bool
            Plot the extracted datamodule directly

        Returns
        --------
        DataModule
            data_table module
        """
        x, y = self.df.dims[:2]
        kws = {y: y0}
        ex = self.df.sel(method='nearest', **kws)
        data = data_table([self.x, np.array(ex)[self.x_min:self.x_max]],
                          [x, self.name_v])
        data.par = self.par
        data.comments = self.comments
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop

        if plot:
            display(data.plot_hv(title='{} = {:.6e}'.format(y, y0)))

        return data

    # Plotting #################################################################
    def plot(self, name_x=None, name_y=None, cmap='magma', height=400,
             width=800, z_min=None, z_max=None, mode='Image'):
        """Plot table with Holoviews

        Parameters
        -----------
        name_x : None, str
            Column name used for x axis
        name_y : None, str
            Column name used for y axis
        cmap : str
            Name of colormap
        height : int
            Height of plot in pixels
        width : int
            Width of plot in pixels
        z_min : None, float(Optional)
            Minimum value for z values. If you set this, the scale will not
            automatically updated to full contrast.
        z_max : None, float (Optional)
            Maximum value for z values
        mode : 'QuadMesh', 'Image'
            Choose mode for holoviews plotting
        """
        if name_x is None:
            x_vals = self.df.dims[0]
        else:
            x_vals = name_x

        if name_y is None:
            y_vals = self.df.dims[1]
        else:
            y_vals = name_y
        hv.opts({mode: {'plot': {'width': width, 'height': height},
                              'style': {'cmap': cmap}}})
        # Rename z values (to prevent bug in holoviews)
        df = self.df[self.x_min:self.x_max, self.y_min:self.y_max].rename('z')
        # Create dataset
        ds = hv.Dataset(df)
        # Create HoloObject
        if mode == 'QuadMesh':
            holo_object = hd.regrid(ds.to(hv.QuadMesh, [x_vals, y_vals]))
        else:
            holo_object = hd.regrid(ds.to(hv.Image, [x_vals, y_vals]))
        # Rescale
        holo_object = holo_object.redim.range(z=(z_min, z_max))
        return holo_object

    def pcolormesh(self):
        """Simple color plot without options. Quick and light"""
        df = self.df[self.x_min:self.x_max, self.y_min:self.y_max]
        df.plot.pcolormesh(self.name_x, self.name_y)

    def imshow(self, colormap='magma', levels=256, zlabel_pos="right",
               labelpad=0, cbar_y=0.5, **args):
        """Color plot using matplotlib

        Note
        -----
        Plots just the .select() data.

        Parameters
        -----------
        colormap : str
            Choose colormap from 'Magma' (Def), 'Inferno', 'Plasma', 'Viridis'
        levels : int
            Color levels. Default is 256
        data_type : str
            Choose if data is linear 'Lin', logarithmic 'Log' or 'Amplitude'
        xlabel : str
            Label for x axis
        ylabel : str
            Label for y axis
        zlabel : str
            Label for colorbar
        zlabel_pos : "top", "right", "right_invert"
            Position and orientation of zlabel next to colorbar. If location is
            wrong, play with labelpad and cbar_y
        """
        # Get colormap from matplotlib
        cmap = plt.cm.get_cmap(colormap, levels)

        # Just get select range
        df = self.df[self.x_min:self.x_max, self.y_min:self.y_max]

        # Plot
        ax = plt.subplot(111)
        im = df.plot.imshow(self.name_x, self.name_y, cmap=cmap,
                            add_colorbar=False, ax=ax)

        # Customizing colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        if zlabel_pos == "top":
            cbar.set_label(self.name_v, labelpad=labelpad - 32, y=cbar_y + 0.58,
                           rotation=0)
        elif zlabel_pos == "right":
            cbar.set_label(self.name_v, labelpad=labelpad + 5, y=cbar_y,
                           rotation=90)
        else:
            cbar.set_label(self.name_v, labelpad=labelpad + 15, y=cbar_y,
                           rotation=-90)
        return ax

    def contourf(self):
        df = self.df[self.x_min:self.x_max, self.y_min:self.y_max]
        df.plot.contourf(self.name_x, self.name_y)

    # Data processing functions ################################################
    def smoothx(self, nnb=21, polyorder=2):
        """Smooth data along x axis using the Savitzky-Golay filter

        Currently just for xyz data.

        Parameters
        -----------
        nnb : int
            Window length of filter. Must be an odd integer
        polyorder : int
            Polynomial order for the fit

        Returns
        --------
        np.array
            Filtered z matrix.
        """
        zsel = self.values
        for idx, row in enumerate(zsel):
            zsel[idx, :] = sp_sig.savgol_filter(row, nnb, polyorder)
        return zsel

    def smoothy(self, nnb=11, polyorder=2):
        """"Smooth data along y axis using the Savitzky-Golay filter

        Parameters
        -----------
        nnb : int
            Window length of filter. Must be an odd integer
        polyorder : int
            Polynomial order for the fit

        Returns
        --------
        np.array
            Filtered z matrix.
        """
        zsel = self.values
        for idx, row in enumerate(zsel.T):
            zsel[:, idx] = sp_sig.savgol_filter(row, nnb, polyorder)
        return zsel

    def interp(self, xnew, ynew, kind='cubic'):
        """Interpolates data to new x, y values `xnew` and `ynew`

        Parameters
        -----------
        xnew : np.array
            New x array
        ynew : np.array
            New y array
        kind : str
            Chose interpolation kind out of ['linear', 'cubic', 'quintic']

        Returns
        --------
        DataModule
            A new data_surface DataModule
        """
        # interp2d has a weird structure, therefore we have to transpose
        f = sp_intp.interp2d(self.x, self.y, self.values.T, kind=kind)
        return data_grid([xnew, ynew, f(xnew, ynew).T], [self.name_x,
                                                         self.name_y,
                                                         self.name_v])

    def extract_min_x(self, argument=True, nnB=21, polyorder=2):
        """Extract minimum z-values sweeping through x.

        Smooth data and extract minimum z values sweeping through x.

        Note
        -----
            Return depends on argument.
            - If set to True, it will return a new datamodule with (y, x)
            - If set to False, it will return a new datamodule with (y, min(z))

        Parameters
        -----------
        argument : bool
            Return corresponding x value (True) or corresponding minimum value
        nnB : int
            Window length for smoothing. Set to 1 to disable
                        smoothing.
        polyorder : int
            Polynomial order for smoothing

        Returns
        --------
        DataModule
            A `data_table` DataModule. Values depend on argument keyword (see
            above)
        """
        if nnB == 1:
            zsel = self.values
        else:
            zsel = self.smoothx(nnB, polyorder)  # Z data is smoothened along y

        tmp = np.zeros_like(self.x, dtype=np.float)

        # Get minima
        if argument:
            # Return y values
            for i in range(len(tmp)):
                tmp[i] = self.y[np.argmin(zsel[i, :])]
            name_tmp = self.name_y

        else:
            # Return z values
            for i in range(len(tmp)):
                tmp[i] = np.min(zsel[i, :])
            name_tmp = self.name_v

        data = data_table([self.x, tmp], [self.name_x, name_tmp])
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop
        return data

    def extract_min_y(self, argument=True, nnB=21, polyorder=2):
        """Extract minimum z-values sweeping through y.

        Smooth data and extract minimum z values sweeping through y.

        Note
        -----
            Return depends on argument.
            - If set to True, it will return a new datamodule with (y, x)
            - If set to False, it will return a new datamodule with (y, min(z))

        Parameters
        -----------
        argument : bool
            Return corresponding x value (True) or corresponding minimum value
        nnB : int
            Window length for smoothing. Set to 1 to disable
                        smoothing.
        polyorder : int
            Polynomial order for smoothing

        Returns
        --------
        DataModule
            A `data_table` DataModule. Values depend on argument keyword (see
            above)
        """
        if nnB == 1:
            zsel = self.values
        else:
            zsel = self.smoothx(nnB, polyorder)  # Z data is smoothened along y

        tmp = np.zeros_like(self.y, dtype=np.float)

        # Get minima
        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp[i] = self.x[np.argmin(zsel[:, i])]
            name_tmp = self.name_x

        else:
            # Return z values
            for i in range(len(tmp)):
                tmp[i] = np.min(zsel[:, i])
            name_tmp = self.name_v

        data = data_table([self.y, tmp], [self.name_y, name_tmp])
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop
        return data

    def extract_min_lorentzianfit_y(self, p0=None, argument=True, plot=False,
                                    adapt_p0=False):
        """Use Lorentzian with initial parameters p0 to fit the maximum z-value
        while sweeping through y.

        Note
        -----
            Return depends on argument.
            - If set to True, it will return a new datamodule with (y, x)
            - If set to False, it will return a new datamodule with (y, max(z))

        Parameters
        -----------
        p0 : list
            Initial guess for lorentzianfit pars ['center', 'width', 'offset',
            'amplitude']
        argument : bool
            Return x or z values
        plot : bool
            Plot each iteration
        adapt_p0 : bool
            Use p0 of last iteration

        Returns
        --------
        DataModule
            A `data_table` DataModule. Values depend on argument keyword (see
            above)
        """
        tmp = np.zeros_like(self.y)
        err_bars = []

        # Guess initial parameters, if none is given
        if p0 is None:
            tmp_data = data_table([self.x, self.values[:, 0]])
            idx_min = np.argmin(tmp_data.y)
            offset = np.mean(tmp_data.y)
            center = tmp_data.x[idx_min]
            width = center / 1e6  # kHz as guess
            amplitude = tmp_data.y[idx_min] - offset
            p0 = [center, width, offset, amplitude]

        # Go through y values and find x coordinate for maximum z value
        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp_dm = data_table([self.x, self.values[:, i]])
                fpars, fpars_err, _ = tmp_dm.fit(lorentzian_fit, p0, plot=plot,
                                                 plot_init=False,
                                                 plot_params=False,
                                                 maxfev=10000)
                tmp[i] = fpars[0]
                err_bars.append(fpars_err[0])
                tmp_name = self.name_x
                if adapt_p0:
                    p0 = fpars
        else:
            # Return z values
            for i in range(len(tmp)):
                tmp_dm = data_table([self.x, self.values[:, i]])
                fpars, fpars_err, _ = tmp_dm.fit(lorentzian_fit, p0, plot=plot,
                                                 plot_init=False,
                                                 plot_params=False,
                                                 maxfev=10000)
                tmp[i] = lorentzian_fit(fpars[0], *fpars)
                tmp_err_bar = np.sqrt((2 * fpars[3] / (np.pi * fpars[1] ** 2) *
                                       fpars_err[1]) ** 2 + fpars_err[2] ** 2 +
                                      (2 / (np.pi * fpars[1]) * fpars_err[
                                          3]) ** 2)
                err_bars.append(tmp_err_bar)
                tmp_name = self.name_v
                if adapt_p0:
                    p0 = fpars

        data = data_table([self.y, tmp, err_bars],
                          [self.name_y, tmp_name, '{} Errors'.format(tmp_name)])
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop
        return data

    def extract_min_modefit_y(self, p0=None, argument=True, plot=False):
        """Use mode fit with initial parameters p0 to fit the minimum z-value
        while sweeping through y.

        Note
        -----
            Return depends on argument.
            - If set to True, it will return a new datamodule with (y, x)
            - If set to False, it will return a new datamodule with (y, min(z))

        Parameters
        -----------
        p0 : list
            Initial guess for modefit pars ['offset', 'Qc', 'df', 'f0', 'Qi']
        argument : bool
            Return x or z values

        Returns
        --------
        DataModule
            A `data_table` DataModule. Values depend on argument keyword (see
            above)
        """
        tmp = np.zeros_like(self.y)
        err_bars = []

        # Guess initial parameters, if none is given
        if p0 is None:
            tmp_data = data_table([self.x, self.values[:, 0]])
            idx_min = np.argmin(tmp_data.y)

            offset = np.mean(tmp_data.y)
            Qc = 1e4  # Medium value
            Qi = 1e4  # Medium value
            f0 = tmp_data.x[idx_min]  # Frequency of minimum
            df = f0 / 1e6  # MHz?
            p0 = [offset, Qc, df, f0, Qi]

        # Go through y values and find x coordinate for minimum z value
        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp_dm = data_table([self.x, self.values[:, i]])
                fpars, fpars_err, _ = tmp_dm.fit(mode_fit, p0, plot=plot,
                                                 plot_init=False,
                                                 plot_params=False,
                                                 maxfev=10000)
                tmp[i] = fpars[3]
                err_bars.append(fpars_err[3])
                tmp_name = self.name_x
        else:
            # Return z values
            for i in range(len(tmp)):
                tmp_dm = data_table([self.x, self.values[:, i]])
                fpars, _, _ = tmp_dm.fit(mode_fit, p0, plot=plot,
                                         plot_init=False, plot_params=False,
                                         maxfev=10000)
                tmp[i] = mode_fit(fpars[3], *fpars)
                tmp_name = self.name_v

        data = data_table([self.y, tmp, err_bars],
                          [self.name_y, tmp_name, '{} Errors'.format(tmp_name)])
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop
        return data
