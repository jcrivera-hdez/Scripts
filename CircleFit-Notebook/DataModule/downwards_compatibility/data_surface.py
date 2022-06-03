# -*- coding: utf-8 -*-
"""
    Class for multi data z=f(x,y)

    Author: Iman, Oscar Gargiulo, Christian Schneider
"""
from DataModule.base import data_module_base
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp_sig
import scipy.interpolate as sp_intp
from DataModule.fit_functions import mode_fit, lorentzian_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from DataModule.data_table import data_table

# Bokeh
import bokeh.plotting as bp
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar


class data_surface(data_module_base):
    """Class for z=f(x,y) data."""

    def __init__(self, x=None, y=None, z=None):
        super().__init__()

        if x is None:
            self.x = np.array([])
            self.y = np.array([])
            self.z = np.array([])
            return

        if (y is None) or (z is None):
            print('ERROR: y or z not specified')
            raise Exception('EMPTYARRAY')

        print('data_surface depreciated. Please use data_grid')
        self.load_var(x, y, z)

    def load_var(self, x, y, z):
        """Import data from three tuples/lists/array.

        Parameters
        -----------
        x : np.array, list
            x values
        y : np.array, list
            y values
        z : [np.array, np.array]
            z values
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.select()

    def select(self, xrng=[0, 0], yrng=[0, 0]):
        """Limit data between x range and y range

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
        if xrng == [0, 0]:
            self.idx_min = 0
            self.idx_max = len(self.x)
        else:
            idx = np.where((self.x >= xrng[0]) & (self.x <= xrng[1]))[0]
            self.idx_min = idx[0]
            self.idx_max = idx[-1]

        if yrng == [0, 0]:
            self.ymin = 0
            self.ymax = len(self.y)
        else:
            idy = np.where((self.y >= yrng[0]) & (self.y <= yrng[1]))[0]
            self.ymin = idy[0]
            self.ymax = idy[-1]

        self.xsellen = self.idx_max-self.idx_min
        self.ysellen = self.ymax-self.ymin

    def return_ysel(self):
        """Return selected y range"""
        return self.y[self.ymin:self.ymax]

    def return_zsel(self):
        """Return selected z matrix"""
        return self.z[self.ymin:self.ymax, self.idx_min:self.idx_max]

    def pcolormesh(self):
        """Simple color plot without options. Quick and light."""
        plt.pcolormesh(self.return_xsel(),
                       self.return_ysel(),
                       self.return_zsel())

    def plot(self, colormap='Magma', show=True, z_min=None, z_max=None,
             **kwargs):
        """2D Plot of data with color encoded z values in bokeh.

        The plot uses just the .select() data.

        Parameters
        -----------
        colormap : str
            Choose colormap from 'Magma' (Def), 'Inferno', 'Plasma', 'Viridis'
        show : bool
            Show plot directly after calling the function.
        z_min : float
            Minimum z value for color scale
        z_max : float
            Maximum z value for color scale
        """
        # Just plot selection
        #xsel = self.return_xsel()
        #ysel = self.return_ysel()
        #zsel = self.return_zsel()
        xsel = self.x
        ysel = self.y
        zsel = self.z
        # Create color mapper
        if z_min is None:
            z_min = np.min(zsel)
        if z_max is None:
            z_max = np.max(zsel)

        color_mapper = LinearColorMapper(palette=colormap + '256',
                                         low=z_min, high=z_max)
        # Create figure with correct x and y ranges
        dx = np.abs(xsel[-1]-xsel[0])
        dy = np.abs(ysel[-1]-ysel[0])
        fig = bp.figure(x_range=(xsel[0], xsel[-1]),
                        y_range=(ysel[0], ysel[-1]),
                        width=800,
                        height=600,
                        toolbar_location='above')
        fig.image(image=[zsel], x=xsel[0], dw=dx,
                  y=ysel[0], dh=dy, color_mapper=color_mapper)
        # Create colorbar
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                             label_standoff=5, location=(0, 0),
                             title='  dB', title_standoff=10)
        fig.add_layout(color_bar, 'right')

        if show:
            bp.show(fig)
        return fig

    def imshow(self, colormap='magma', levels=256, data_type='Log',
               xlabel="", ylabel="", zlabel="dB", zlabel_pos="right",
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
        # Just plot the selection
        xsel = self.return_xsel()
        ysel = self.return_ysel()
        zsel = self.return_zsel()

        cmap = plt.cm.get_cmap(colormap, levels)

        ax = plt.subplot(111)
        if data_type == 'Log':
            im = plt.imshow(zsel, origin='lower', cmap=cmap, aspect='auto',
                       vmin=np.min(zsel), vmax=np.max(zsel),
                       extent=[xsel.min(), xsel.max(), ysel.min(), ysel.max()],
                       **args)

        elif data_type == 'Lin':
            zdata = 20*np.log10(zsel)
            im = plt.imshow(zdata, origin='lower', cmap=cmap,
                       aspect='auto', vmin=np.min(zdata),
                       vmax=np.max(zdata),
                       extent=[xsel.min(), xsel.max(), ysel.min(), ysel.max()],
                       **args)
        elif data_type == 'Amplitude':
            zdata = zsel
            im = plt.imshow(zdata, origin='lower', cmap=cmap,
                       aspect='auto', vmin=np.min(zdata),
                       vmax=np.max(zdata),
                       extent=[xsel.min(), xsel.max(), ysel.min(), ysel.max()],
                       **args)
        else:
            print('Wrong data_type inserted')
            raise Exception('DATATYPE')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        if zlabel_pos == "top":
            cbar.set_label(zlabel, labelpad=labelpad-32, y=cbar_y+0.58, rotation=0)
        elif zlabel_pos == "right":
            cbar.set_label(zlabel, labelpad=labelpad, y=cbar_y, rotation=90)
        else:
            cbar.set_label(zlabel, labelpad=labelpad+15, y=cbar_y, rotation=-90)
        return ax

    def smoothx(self, nnb=21, polyorder=2):
        """Smooth data along x axis using the Savitzky-Golay filter

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
        zsel = self.return_zsel().copy()
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
        zsel = self.return_zsel().copy()
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
        f = sp_intp.interp2d(self.return_xsel(), self.return_ysel(), kind=kind)
        return data_surface(xnew, f(xnew, ynew))

    def extract_y(self, y0, plot=True):
        """Extract z values along x axis for specified y value y0.

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
            data_line module
        """
        xsel = self.return_xsel()
        zsel = self.return_zsel()

        idx = np.argmax(self.y >= y0)
        data = data_table()

        data.par = self.par
        data.comments = self.comments
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop

        tmp = zsel[:][idx]
        data.load_var(xsel, tmp)

        if plot:
            data.plot(title='Y = {:.2e}'.format(y0))

        return data

    def extract_x(self, x0, plot=True):
        """Extract z values along y axis for specified x value x0.

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
            data_line module
        """
        ysel = self.return_ysel()
        zsel = self.return_zsel()

        idx = np.argmax(self.x >= x0)

        data = data_table()
        data.par = self.par
        data.comments = self.comments
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop

        tmp = zsel[:, idx]
        data.load_var(ysel, tmp)

        if plot:
            data.plot(title='X = {:.2e}'.format(x0))

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
            A `data_line` DataModule. Values depend on argument keyword (see
            above)
        """
        xsel = self.return_xsel()
        ysel = self.return_ysel()

        if nnB == 1:
            zsel = self.return_zsel()
        else:
            zsel = self.smoothx(nnB, polyorder)  # Z data is smoothened along y

        tmp = np.zeros_like(ysel, dtype=np.float)

        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp[i] = xsel[np.argmin(zsel[i, :])]

        else:
            # Return z values
            for i in range(len(tmp)):
                tmp[i] = np.min(zsel[i, :])

        data = data_table()
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop

        data.load_var(ysel, tmp)
        return data

    def extract_max_lorentzianfit_y(self, p0, argument=True, plot=False, 
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
            A `data_line` DataModule. Values depend on argument keyword (see
            above)
        """
        xsel = self.return_xsel()
        ysel = self.return_ysel()
        zsel = self.return_zsel()  # Z data is smoothened along y

        tmp = np.zeros_like(ysel)
        err_bars = []
        # Go through y values and find x coordinate for maximum z value
        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp_dm = data_table([xsel, zsel[i, :]])
                fpars, fpars_err, _ = tmp_dm.fit(lorentzian_fit, p0, plot=plot,
                                                 plot_init=False,
                                                 plot_params=False,
                                                 maxfev=10000)
                tmp[i] = fpars[0]
                err_bars.append(fpars_err[0])
                if adapt_p0:
                    p0 = fpars
        else:
            # Return z values
            for i in range(len(tmp)):
                tmp_dm = data_table([xsel, zsel[i, :]])
                fpars, fpars_err, _ = tmp_dm.fit(lorentzian_fit, p0, plot=plot,
                                                 plot_init=False,
                                                 plot_params=False,
                                                 maxfev=10000)
                tmp[i] = lorentzian_fit(fpars[0], *fpars)
                tmp_err_bar = np.sqrt((2*fpars[3]/(np.pi*fpars[1]**2)*fpars_err[1])**2 + fpars_err[2]**2 +
                                      (2/(np.pi*fpars[1])*fpars_err[3])**2)
                err_bars.append(tmp_err_bar)
                if adapt_p0:
                    p0 = fpars

        data = data_table()
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop
        data.load_var(ysel, tmp)

        return data, err_bars 

    def extract_min_modefit_y(self, p0, argument=True, plot=False):
        """Use Lorentzian with initial parameters p0 to fit the minimum z-value
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
            A `data_line` DataModule. Values depend on argument keyword (see
            above)
        """
        xsel = self.return_xsel()
        ysel = self.return_ysel()
        zsel = self.return_zsel()  # Z data is smoothened along y

        tmp = np.zeros_like(ysel)
        err_bars = []
        # Go through y values and find x coordinate for minimum z value
        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp_dm = data_table([xsel, zsel[i, :]])
                fpars, fpars_err, _ = tmp_dm.fit(mode_fit, p0, plot=plot,
                                                 plot_init=False,
                                                 plot_params=False,
                                                 maxfev=10000)
                tmp[i] = fpars[3]
                err_bars.append(fpars_err[3])
        else:
            # Return z values
            for i in range(len(tmp)):
                tmp_dm = data_table([xsel, zsel[i, :]])
                fpars, _, _ = tmp_dm.fit(mode_fit, p0, plot=plot,
                                         plot_init=False, plot_params=False,
                                         maxfev=10000)
                tmp[i] = mode_fit(fpars[3], *fpars)

        data = data_table()
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop
        data.load_var(ysel, tmp)

        return data, err_bars

    def extract_min_x(self, argument=True, nnB=21, polyorder=2):
        """Extract minimum z-values sweeping through x.

        Smooth data and extract minimum z values sweeping through x.

        Note
        -----
            Return depends on argument.
            - If set to True, it will return a new datamodule with (x, y)
            - If set to False, it will return a new datamodule with (x, min(z))

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
            A `data_line` DataModule. Values depend on argument keyword (see
            above)
        """
        xsel = self.return_xsel()
        ysel = self.return_ysel()
        if nnB == 1:
            zsel = self.return_zsel()
        else:
            zsel = self.smoothy()  # Z data is smoothened along y

        tmp = np.zeros_like(xsel, dtype=np.float)
        if argument:
            # Return x values
            for i in range(len(tmp)):
                tmp[i] = ysel[np.argmin(zsel[:, i])]

        else:
            # Return z values
            for i in range(len(tmp)):
                tmp[i] = np.min(zsel[:, i])

        data = data_table()
        data.par = self.par
        data.temp_start = self.temp_start
        data.temp_stop = self.temp_stop
        data.time_start = self.time_start
        data.time_stop = self.time_stop

        data.load_var(xsel, tmp)
        return data

    def extract_max_y(self, argument=True, nnB=21, polyorder=2):
        """Extract maximum z-values sweeping through y.

        Runs -1*data and the `extract_min_y` function.
        """
        self.z = -1*self.z
        new_dm = self.extract_min_y(argument, nnB, polyorder)
        self.z = -1*self.z
        return new_dm

    def extract_max_x(self, argument=True, nnB=21, polyorder=2):
        """Extract maximum z-values sweeping through x.

        Runs -1*data and the `extract_min_x` function
        """
        self.z = -1*self.z
        new_dm = self.extract_min_x(argument, nnB, polyorder)
        self.z = -1*self.z
        return new_dm

###############################################################################
# Needs rethinking ############################################################

    # Do we need this function?
    def contourf(self, colormap=plt.cm.hsv, lev=10, norm=0, **args):
        '''contourf is a 2D plot with the color for the depth,

        ToDo
        -----
            Do we need this function?

        some of the main contourf options have been simplified, this function 
        automatically builds the levels and there are 3 normalization options.
        Other contourf options can be passed as arguments (see pyplot doc).
        
        default lev is 10, a linear one will be used as default
        
        norm can be 0,1,2:
            0 (def): will evaluate a linear normalization for the colors
            1: a log color normalization will be used
            2: if data is in dB, it will be converted in linear scale before 
            using the log color normalization
        
        NOTE: default interpolation is on
        '''
        zsel = self.return_zsel()
        if norm is 0:
            normal = plt.cm.colors.Normalize(vmax=np.abs(zsel).max(),
                                             vmin=-np.abs(zsel).max())
        elif norm is 1:
            from matplotlib.colors import LogNorm
            normal = LogNorm()
        elif norm is 2:
            from matplotlib.colors import LogNorm
            normal = LogNorm()
            zsel = 10**(zsel/20)
        else:
            normal = norm
        levels = np.linspace(zsel.min(), zsel.max(), 10)
        cmap = colormap
        plt.contourf(self.return_xsel(), self.return_ysel(), zsel, levels,
                     cmap=plt.cm.get_cmap(cmap, len(levels)-1),
                     norm=normal, **args)
        plt.colorbar()


# Aliases #####################################################################
data_3d = data_surface
