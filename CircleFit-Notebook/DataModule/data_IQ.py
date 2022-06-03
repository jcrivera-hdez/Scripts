# -*- coding: utf-8 -*-
"""

"""
from .base import data_module_base
import numpy as np
import bokeh.plotting as bp
from bokeh.layouts import row, column
from bokeh.models import Spacer
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
from bokeh.models import LinearColorMapper, LogColorMapper, BasicTicker
from bokeh.models import ColorBar
from .fit_functions import gaussian_2d_mult
from .fit_functions import exp_boltzmann
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pandas as pd
from IPython.display import display



class data_IQ(data_module_base):
    """Class for IQ scatter data."""


    #Init on creation of new instant of data_IQ
    def __init__(self, I=None, Q=None):
        super().__init__()

        if I is None:
            self.I = np.array([])
            self.Q = np.array([])

        elif Q is None:
            print('Error: no Q values inserted')
            raise Exception('EMPTYARRAY')
        else:
            self.load_var(I, Q)

        self._fit_executed = False
        self._fit_labels = None
        self.Temperature = None


    def load_var(self, I, Q):
        """Import data from two tuples/lists/array.

        Parameters
        -----------
        I : list
            I-Array. Real values of Voltage
        Q : list
            Q-Array. Imaginary values of Voltage
        """

        I = np.array(I)
        Q = np.array(Q)

        if np.isscalar(I[0]) is False:
            print('Error: bad I-axis, maybe it is a list of list')
            raise Exception('NOTANARRAY')

        if np.isscalar(Q[0]) is False:
            print('Error: bad x-axis, maybe it is a list of list')
            raise Exception('NOTANARRAY')

        if len(I) != len(Q):
            print('WARNING: I and Q length mismatch')
        self.I = I
        self.Q = Q
        self.select()
            


    def select(self, rng=[0, -1]):
        """Select range of data.

        Plots, fits, etc will then only be applied on this range.
        If nothing is specified all the data will be select

        Parameters
        ----------
        xrng : list of int
            Start and Stop values of the range by idx
        """
        try:
            self.I[rng[0]]
            self.I[rng[1]]
            self.idx_min = rng[0]
            self.idx_max = rng[1]
        except IndexError:
            raise IndexError('Index out of range')

    def return_I_sel(self):
        """Returns the current selected I range

        Returns
        --------
        list
            Numpy array of x values in selected range
        """
        return self.I[self.idx_min:self.idx_max]

    def return_Q_sel(self):
        """Returns the current selected Q range

        Returns
        --------
        list
            Numpy array of x values in selected range
        """
        return self.Q[self.idx_min:self.idx_max]

    def bin(self, bins=100, IQ_range=None):
        """Bin data"""
        if not IQ_range:
            # Default: 2 times the maximum value
            iqrange = 1*np.amax(np.maximum(np.abs(self.return_I_sel()),
                                           np.abs(self.return_Q_sel())))
        else:
            iqrange = IQ_range

        IQ_hist, IQ_edges_x, IQ_edges_y = np.histogram2d(x=self.return_I_sel(),
                                                         y=self.return_Q_sel(),
                                                         bins=bins,
                                                         range=[[-iqrange,
                                                                 iqrange],
                                                                [-iqrange,
                                                                iqrange]])
        # X axis are rows --> transpose
        return IQ_hist.transpose(), IQ_edges_y, IQ_edges_x



    def fit_gaussian_2d(self, n, p_guess=[], bounds = (-np.inf,np.inf), bins=100, phi0=45, damp=0.3,
                        widths=None, plot=True, sigmas_to_plot = 2, plot_guess=False, maxfev=1e6,
                        print_res=True):
        """Fit multiple gaussians to 2d data. Requires number of gaussians to
        fit.

        Parameters
        -----------
        n : int
            Number of gaussians to fit
        p_guess : list, np.array, optional
            Initial guess for parameters.
            Structure: [x0, y0, amp0, width0, x1, ...]
        bounds: list of tuples
            borders for each free parameter
        bins : int
            Number of bins for fitting
        phi0 : float
            Angle between gaussians in degree
        damp : float
            Damping factor between next gaussians
        widths : None, list(n)
            Widths of gaussians
        plot : bool
            Plot fit
        plot_guess : bool
            Plot guess
        maxfev : float, int
            Maximum number of iterations for fit
        print_res : bool
            Print results
        """
        self.number_of_disks = n
        # Bin data
        IQ_hist, IQ_edges_x, IQ_edges_y = self.bin(bins)

        # Create complete mesh to display the guess and fitresults
        x = np.linspace(IQ_edges_x[0], IQ_edges_x[-1], bins)
        xy = np.meshgrid(x, x)

        # make an estimate for the initial parameters p0
        if len(p_guess) == 0:
            # Convert Phi0 in rad
            phi0 = phi0/180.*np.pi
            # Create guess values if none are given
            # First center is maximum. The rest goes clockwise further
            id_y0, id_x0 = np.unravel_index(np.argmax(IQ_hist), IQ_hist.shape)
            x0s = [x[id_x0]]
            y0s = [x[id_y0]]
            amplitude_guess = [np.max(IQ_hist)]
            for m in range(1, n):
                re_im = x0s[0] + 1j*y0s[0]
                x0s.append(np.real(re_im*np.exp(-1j*phi0*m)))
                y0s.append(np.imag(re_im*np.exp(-1j*phi0*m)))
                amplitude_guess.append(amplitude_guess[0]/m*damp)
            if not widths:
                widths = np.abs(x[-1]/10)
            widths_guess = [widths for i in range(n)]  # Guess same width
            p0 = [[x0s[i], y0s[i], amplitude_guess[i], widths_guess[i]] for i
                  in range(len(x0s))]
            p0 = np.array(p0).flatten()
        else:
            p0 = p_guess



        # Ravel function for fit_curve
        def fitfunc(x, *p):
            return gaussian_2d_mult(x, *p).ravel()

        # Fit
        pars, covars = curve_fit(fitfunc, xy, IQ_hist.ravel(), p0=p0,
                                 sigma = np.sqrt(IQ_hist.ravel() + 1), 
                                 bounds = bounds, maxfev=int(maxfev))
        
        #fix sign of standart deviations
        for i in range(len(pars)):
            if(i%4)==0:
                pars[i+3]=np.abs(pars[i+3])
            
        
        
        # Save Parameters
        self._fit_parameters = pars
        self._fit_parameters_stddev = np.sqrt(np.diag(covars))
        fit_pars = pd.DataFrame(data=pars.reshape(n, 4),
                                columns=['I', 'Q', 'Amplitude',
                                         'Width'])
        self.fitresults = fit_pars

        self.R_squared_gaussian_fit = self.__calculate_R_sqared(IQ_hist.ravel(), fitfunc(xy,
                                                           *pars))

        #Calculate overlap of the FITTED gaussians
        def calculate_overlap_of_fitted_gaussians(pars):
            if len(pars) > 7:
                x1 = pars[0]
                y1 = pars[1]
                w1 = pars[3]**2
                x2 = pars[4]
                y2 = pars[5]
                w2 = pars[7]**2
                if self.number_of_disks == 2:
                    return np.exp(-0.5 * ( (x1-x2)**2 + (y1-y2)**2 ) / (w1+w2) )
                if self.number_of_disks == 3:
                    x3 = pars[8]
                    y3 = pars[9]
                    w3 = pars[10]**2
                    nom1 =  ( (x1-x2)**2 + (y1-y2)**2 ) * w3 
                    nom2 =  ( (x1-x3)**2 + (y1-y3)**2 ) * w2
                    nom3 =  ( (x2-x3)**2 + (y2-y3)**2 ) * w1
                    den = 2 * (w1*w2 + w1*w3 + w2*w3)
                    
                    return np.exp(-(nom1 + nom2 + nom3)/den)
            else:
                raise Exception('Fucntion expects 4 parameters for each gaussian and is only definde for the case of 2 or 3 disks')
                return None
                
            
        self.overlap_of_fitted_gaussians = calculate_overlap_of_fitted_gaussians(pars)

            
        
        # Print results
        if print_res:
            pd.set_option('precision', 3)
            display(self.fitresults)

        if plot:
            # Plotting

            fig = self.plot_bins(bins, show=False, log_scale=True)
            
            for i in range(self.number_of_disks):
                x = self._fit_parameters[0+4*i]
                y = self._fit_parameters[1+4*i]
                w = self._fit_parameters[3+4*i]
                fig.ellipse([x],[y], width=2*sigmas_to_plot*w, height=2*sigmas_to_plot*w, fill_color=None, line_color = 'red')
                
            if plot_guess:
                for i in range(self.number_of_disks):
                    x = p0[0+4*i]
                    y = p0[1+4*i]
                    w = p0[3+4*i]
                    fig.ellipse([x],[y], width=2*sigmas_to_plot*w, height=2*sigmas_to_plot*w, fill_color=None, line_color = 'green')
            bp.show(fig)
        
        self._fit_executed = True
        return pars, covars, self.R_squared_gaussian_fit
    
    
    def __calculate_R_sqared(self, measured_data, modeled_data):
        """Calculate R squared (Coefficient of determination) as a measure of the 
        goodness of the fit. Comparison of how much better the model is, in 
        comparison to randomly distributet measurements.

        Parameters
        -----------
        measured_data : 1D array
            Measured data y_i
        modeled_data : 1D array
            Data f_i calculated with the model (fit)
        """
        avg = np.sum(measured_data)/len(measured_data)
        SS_tot = np.sum((measured_data-avg)**2)
        SS_res = np.sum((measured_data - modeled_data)**2)
        return 1 - SS_res/SS_tot

    def calculate_Temperature(self, f_ge, anharmonicity, p0=[500,0.1], plot=True,
                              testmode=True):
        if self._fit_executed == False:
            print('Please first perform a Fit using the function data_IQ.fit_gaussian_2d' )
            raise Exception('FITMISSING')
        elif self.number_of_disks < 2:
            print('Not enough information about the occupation of the states to calculate temperature' )
            raise Exception('NOTENOUGHDISKS')
        else:
            #Prepare x and y axes (with uncertainties)            
            def calculate_energy_axes(E0, alpha, n):
                x=[]
                for i in range(n):
                    if i == 0:
                        x.append(0)
                    else:
                        x.append((E0-alpha)*i+alpha)
                return x
            
            x_energy = calculate_energy_axes( f_ge, anharmonicity, self.number_of_disks)
            y_occupation, y_occupation_err = [[],[]]
            for i in range(self.number_of_disks):
                y_occupation.append(self._fit_parameters[2+4*i])
                y_occupation_err.append(self._fit_parameters_stddev[2+4*i])
            #y_occupation, y_occupation_err = zip(sorted(zip(y_occupation, y_occupation_err), key=lambda x: x[0]))
            y_occupation, y_occupation_err = list(zip(*list(reversed(sorted(zip(y_occupation, y_occupation_err))))))
            bose_fit_pars, bose_covars = curve_fit(exp_boltzmann, x_energy, y_occupation, p0, sigma=y_occupation_err)
            
        
            x_energy_least_squares = calculate_energy_axes( f_ge, anharmonicity,8)
            def func_to_minimize(pars):
               A=pars[0]
               T=pars[1]
               tmp1 = 0
               tmp2 = 0
               for i in range(len(x_energy_least_squares)):
                   if i < (self.number_of_disks - 1):
                       tmp1 += (y_occupation[i] - exp_boltzmann(x_energy_least_squares[i], A, T))**2
                   else:
                       tmp2 += exp_boltzmann(x_energy_least_squares[i], A, T)
               return tmp1 + (y_occupation[self.number_of_disks-1]-tmp2)**2
            
            result = minimize(func_to_minimize,[12000,0.12])
            lstsqr_results = [result['x'][0],result['x'][1]]
            self.Temperature = result['x'][1]
            

            
            self.R_squared_temperature_fit = self.__calculate_R_sqared(y_occupation, exp_boltzmann(np.array(x_energy[0:self.number_of_disks]), bose_fit_pars[0], bose_fit_pars[1]))
            
            if plot:
                fig = bp.figure()
                x_fit = np.linspace(0, x_energy[self.number_of_disks-1], 50*self.number_of_disks+1)
                #y_fit = exp_boltzmann(x_fit, bose_fit_pars[0], bose_fit_pars[1])
                y_fit = exp_boltzmann(x_fit, lstsqr_results[0], lstsqr_results[1])
                fig.scatter(x_fit, y_fit)
                fig.circle(x_energy[0:self.number_of_disks], y_occupation, 
                           color='red', size=10, line_alpha=0)

                
                
                fig.xaxis.axis_label = 'Energy / GHz'
                fig.yaxis.axis_label = 'Occupation'
                bp.show(fig)
                
            
            
            return x_energy, y_occupation, y_occupation_err, result, self.number_of_disks, bose_fit_pars, lstsqr_results, self.R_squared_temperature_fit
    

    
    # Plotting ################################################################
    def plot(self, IQ_range=None, alpha=0.1, fig=None, bins=30, hist=True,
             width=500, **kwargs):
        """Plot I/Q Plane with optional histograms"""
        def update(attr, old, new):
            inds = np.array(new['1d']['indices'])
            if len(inds) == 0 or len(inds) == len(self.return_I_sel()):
                hhist1, hhist2 = hzeros, hzeros
                vhist1, vhist2 = vzeros, vzeros
            else:
                neg_inds = np.ones_like(self.return_I_sel(), dtype=np.bool)
                neg_inds[inds] = False
                hhist1, _ = np.histogram(self.return_I_sel()[inds],
                                         bins=hedges)
                vhist1, _ = np.histogram(self.return_Q_sel()[inds],
                                         bins=vedges)
                hhist2, _ = np.histogram(self.return_I_sel()[neg_inds],
                                         bins=hedges)
                vhist2, _ = np.histogram(self.return_Q_sel()[neg_inds],
                                         bins=vedges)

            hh1.data_source.data["top"] = hhist1
            hh2.data_source.data["top"] = -hhist2
            vh1.data_source.data["right"] = vhist1
            vh2.data_source.data["right"] = -vhist2

        if not IQ_range:
            # Default: 2 times the maximum value
            IQ_range = 1*np.amax(np.maximum(np.abs(self.return_I_sel()),
                                            np.abs(self.return_Q_sel())))

        if not fig:
            # Create figure
            fig = bp.figure(x_range=(-IQ_range, IQ_range),
                            y_range=(-IQ_range, IQ_range),
                            plot_width=width,
                            plot_height=width,
                            output_backend="webgl",
                            min_border=10, min_border_left=50,
                            toolbar_location="above")

        fig.background_fill_color = "#fafafa"
        r = fig.scatter(self.return_I_sel(), self.return_Q_sel(), size=3, color="#3A5785", alpha=alpha)

        if hist:
            LINE_ARGS = dict(color="#3A5785", line_color=None)
            # create the horizontal histogram
            hhist, hedges = np.histogram(self.return_I_sel(), bins=bins)
            hzeros = np.zeros(len(hedges)-1)
            hmax = max(hhist)*1.1

            ph = bp.figure(toolbar_location=None, plot_width=fig.plot_width,
                           plot_height=150, x_range=fig.x_range,
                           y_range=(0, hmax), min_border=10,
                           min_border_left=50,
                           y_axis_location="right")
            ph.xgrid.grid_line_color = None
            ph.yaxis.major_label_orientation = np.pi/4
            ph.background_fill_color = "#fafafa"
            ph.xaxis.axis_label = 'I (V)'
            ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist,
                    color="white", line_color="#3A5785")
            hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:],
                          top=hzeros, alpha=0.5, **LINE_ARGS)
            hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:],
                          top=hzeros, alpha=0.1, **LINE_ARGS)

            # create the vertical histogram
            vhist, vedges = np.histogram(self.return_Q_sel(), bins=bins)
            vzeros = np.zeros(len(vedges)-1)
            vmax = max(vhist)*1.1

            pv = bp.figure(toolbar_location=None, plot_width=150,
                           plot_height=fig.plot_height, x_range=(0, vmax),
                           y_range=fig.y_range, min_border=10,
                           y_axis_location="right")
            pv.ygrid.grid_line_color = None
            pv.yaxis.axis_label = 'Q (V)'
            pv.xaxis.major_label_orientation = np.pi/4
            pv.background_fill_color = "#fafafa"

            pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist,
                    color="white", line_color="#3A5785")
            vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:],
                          right=vzeros, alpha=0.5, **LINE_ARGS)
            vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:],
                          right=vzeros, alpha=0.1, **LINE_ARGS)

            layout = column(row(fig, pv), row(ph, Spacer(width=100,
                                                         height=100)))

            bp.curdoc().add_root(layout)
            r.data_source.on_change('selected', update)
            bp.show(layout)
        else:
            # Just show scatter plot
            bp.show(fig)

    def plot_bins(self, bins=100, IQ_range=None, log_scale=False, cmap='Blues',
                  z_min=None, z_max=None, width=500, show=True):
        """Plot binned IQ plane

        Parameters
        -----------
        bins : int
            Number of bins of each axis
        IQ_range : float
            Maximum voltage to plot. The plot is a square with Lengths
            2*IQ_range around 0
        log_scale : bool
            Logarithmic color scale
        cmap : str
            Colormap for plotting. Choose from matlabs colorscales. E.g. Reds,
            Blues, Viridis, Magma, etc...
        z_min : float
            Lower limit for color scale
        z_max : float
            Upper limit for color scale
        width : float
            Width of bokeh figure
        """

        # Bin data if nothing is given
        if isinstance(bins, int):
            IQ_hist, IQ_edges_x, IQ_edges_y = self.bin(bins, IQ_range)
        else:
            # Else expect np.histogram2d data
            IQ_hist, IQ_edges_x, IQ_edges_y = bins
        # Create Colorscale
        if z_min is None:
            z_min = np.min(IQ_hist)
        if z_max is None:
            z_max = np.max(IQ_hist)
        __cmap = get_cmap(cmap)
        colors = [rgb2hex(i) for i in __cmap(range(256))]

        if log_scale:
            color_mapper = LogColorMapper(palette=colors,
                                          low=np.min(IQ_hist),
                                          high=np.max(IQ_hist))
        else:
            color_mapper = LinearColorMapper(palette=colors,
                                             low=np.min(IQ_hist),
                                             high=np.max(IQ_hist))

        # Create figure with correct x and  y ranges
        dx = np.abs(IQ_edges_x[-1] - IQ_edges_x[0])
        dy = np.abs(IQ_edges_y[-1] - IQ_edges_y[0])
        fig = bp.figure(x_range=(IQ_edges_x[0], IQ_edges_x[-1]),
                        y_range=(IQ_edges_y[0], IQ_edges_y[-1]),
                        width=width,
                        height=width,
                        toolbar_location='above')
        fig.image(image=[IQ_hist], x=IQ_edges_x[0], dw=dx,
                  y=IQ_edges_x[0], dh=dy, color_mapper=color_mapper)
        # Create colorbar
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                             label_standoff=5, location=(0, 0),
                             title='counts', title_standoff=10)
        fig.add_layout(color_bar, 'right')
        if show:
            bp.show(fig)
        else:
            return fig


