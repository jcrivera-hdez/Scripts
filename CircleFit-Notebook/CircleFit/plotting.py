# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:33:29 2016
plotting_utilities_module_version = '1.0.1'
@author: David Zoepfl, Christian Schneider


Last Updates:
v.1.1.4 (CHR) 16.01.2017
- Formatting of hovertool for fitted data

v1.1.3 - DAV, 22.12.16:
- improved plot_fitted_data()
- improved HoverTool for fitted data

v1.1.2 (CHR, 4.11.2016)
- improved plot_rawdata
- plot phase in degrees and mag in db

version 1.1.1 - DAV:
    - corrected plot_phase_fit to plot data as intended
    (wrong parameters before)

version 1.1.0 - DAZ:
    - using bokeh as standard opiton for plotting
        pyplot still available with plotting_engine = 'pyplot'

version 1.0.1 - DAZ:
    -minor modifications and improvemetns for reflection configuration
        - allow offres point to be set to +1 or -1, standard option remains +1
    - aspect ratio of all plots plotting a (fianl) circle is fixed, however no
    limits are set, which is preferable to go all the way from over to
    undercoupled
"""
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, gridplot, ColumnDataSource
from bokeh.models import HoverTool
from collections import OrderedDict
from DataModule.plot_style import cc
from .fit_toolbox import lorentzian_abs, tan_phase, notch_model
from .fit_toolbox import reflection_model, reflection_model_mag
from .fit_toolbox import notch_model_mag


def _get_tools():
    tools = ('box_zoom', 'pan', 'wheel_zoom', 'reset', 'save',
             'crosshair')
    return tools


def plot_rawdata(freq, data, title='-', engine='bokeh'):
    """
        Plot complex data in three subplots:
        | Im(data) over Re(data) | Mag(data) over freq | Ang(data) over freq |
    """
    # Calculate plot values
    mag = 20 * np.log10(np.abs(data))  # Power Mag in dB
    phase = np.unwrap(np.angle(data)) * 180 / np.pi  # Phase in degree

    # Holoviews
    if engine[0].lower() == 'h':
        pass

    if engine == 'bokeh':
        # Calculate data for HoverTool
        source_data = ColumnDataSource(
            data=dict(
                freq=freq,
                re=data.real,
                im=data.imag,
                phase=phase,
                mag=mag,
            ))
        tooltips = [('Frequency', '@freq GHz'), ("Re", "@re"), ("Im", "@im"),
                    ('Mag', '@mag dB'), ('Phase', '@phase°')]
        print(title)
        # Re over Im
        fig1 = figure(title='rawdata', tools=_get_tools())
        fig1.xaxis.axis_label = 'Re'
        fig1.yaxis.axis_label = 'Im'
        fig1.diamond(data.real, data.imag, size=4, fill_color='white')
        c1 = fig1.circle('re', 'im', source=source_data, size=3, legend='Data')
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c1])
        fig1.add_tools(hover)
        fig1.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)
        # Mag over freq
        fig2 = figure(tools=_get_tools())
        fig2.xaxis.axis_label = 'Frequency (GHz)'
        fig2.yaxis.axis_label = 'Magnitude (dB)'
        c2 = fig2.line('freq', 'mag', source=source_data, line_width=2)
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c2])
        fig2.add_tools(hover)
        fig2.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)

        # Phase over freq
        fig3 = figure(tools=_get_tools())
        fig3.xaxis.axis_label = 'Frequency (GHz)'
        fig3.yaxis.axis_label = 'Phase (deg)'
        c3 = fig3.line('freq', 'phase', source=source_data, line_width=2)
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c3])
        fig3.add_tools(hover)
        fig3.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)

        fig = gridplot([fig1, fig2, fig3], ncols=3, plot_width=300,
                       plot_height=300)

        show(fig)

    elif engine == 'pyplot':
        fig = plt.figure(figsize=(12, 4))
        fig.subplots_adjust(wspace=0.2)
        # Re over Im
        plt.subplot(131)
        plt.plot(data.real, data.imag, '.')
        plt.title('Im and Re')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.grid()
        plt.locator_params(axis='x', nbins=4)  # Reduce x ticks
        # Mag over Freq
        plt.subplot(132)
        plt.plot(freq, mag)
        plt.title('Magnitude (dB)')
        plt.xlabel('Frequency (GHz)')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])
        # Phase over Freq
        plt.subplot(133)
        plt.plot(freq, phase)
        plt.xlabel('Frequency (GHz)')
        plt.title('Phase (deg)')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])


def plot_cfit(circuit, engine='bokeh', title=''):
    """
        Plot the data and fit.
    """
    # Format data
    freq = circuit.freq
    data = circuit.value_raw
    fit_data = circuit.value_calc
    z_data_norm = circuit.circle_norm
    xc1 = circuit.fitresults_full_model.Value.xc
    yc1 = circuit.fitresults_full_model.Value.yc
    r = circuit.fitresults_full_model.Value.r
    if circuit.type == 'Notch':
        offres = 1
    else:
        offres = -1
    # Calculate plot values
    # # Data
    mag_data = 20 * np.log10(np.abs(circuit.value_raw))  # Power Mag in dB
    phase_data = np.unwrap(np.angle(circuit.value), 0.5) * 180 / np.pi  # Phase
    # # Fit
    # # # Correct electric delay
    fit_tmp = fit_data * np.exp(2j * np.pi * freq * circuit.delay)
    mag_fit = 20 * np.log10(np.abs(fit_tmp))  # Power Mag in dB
    phase_fit = np.unwrap(np.angle(fit_tmp), 0.5) * 180 / np.pi  # Phase

    if engine == 'bokeh':
        # Calculate data for HoverTool
        source_data = ColumnDataSource(
            data=dict(
                freq=freq,
                re=data.real,
                im=data.imag,
                phase=phase_data,
                mag=mag_data,
            ))
        source_fit = ColumnDataSource(
            data=dict(
                freq=freq,
                re=fit_data.real,
                im=fit_data.imag,
                phase=phase_fit,
                mag=mag_fit,
            ))
        source_norm = ColumnDataSource(
            data=dict(
                freq=freq,
                re=z_data_norm.real,
                im=z_data_norm.imag,
                phase=phase_data,
                mag=mag_data,
            ))
        tooltips = [('Frequency', '@freq GHz'), ("Re", "@re"), ("Im", "@im"),
                    ('Mag', '@mag dB'), ('Phase', '@phase°')]
        # Re/Im Data
        fig1 = figure(title=title, tools=_get_tools(), lod_threshold=100)
        fig1.xaxis.axis_label = 'Re (V)'
        fig1.yaxis.axis_label = 'Im (V)'
        c1 = fig1.circle('re', 'im', source=source_data, size=3, legend='Data')
        fig1.line('re', 'im', source=source_fit, line_width=2,
                  line_dash=[5, 5], color='firebrick', legend='Fit')
        fig1.legend.background_fill_alpha = 0.3
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c1])
        fig1.add_tools(hover)
        fig1.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)

        # Power Mag over frequency ############################################
        fig2 = figure(tools=_get_tools())
        fig2.xaxis.axis_label = 'Freq (GHz)'
        fig2.yaxis.axis_label = 'Magnitude (dB)'
        c2 = fig2.line('freq', 'mag', source=source_data, line_width=2)
        fig2.line('freq', 'mag', source=source_fit, line_width=3,
                  line_dash=[5, 5], color='firebrick')
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c2])
        fig2.add_tools(hover)
        fig2.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)

        # Phase over frequency ################################################
        fig3 = figure(tools=_get_tools())
        fig3.xaxis.axis_label = 'Freq (GHz)'
        fig3.yaxis.axis_label = 'Phase (deg)'
        c3 = fig3.line('freq', 'phase', source=source_data, line_width=2)
        fig3.line('freq', 'phase', source=source_fit, line_width=3,
                  line_dash=[5, 5], color='firebrick')
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c3])
        fig3.add_tools(hover)
        fig3.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)

        # Fitted circle #######################################################
        fig4 = figure(title='Normalized Circle',
                      lod_threshold=10, lod_factor=100)
        fig4.xaxis.axis_label = 'Re (V)'
        fig4.yaxis.axis_label = 'Im (V)'
        c4 = fig4.diamond('re', 'im', source=source_norm, size=10,
                          fill_color='white',
                          legend='Data (background subtracted)')
        # Format nicer HoverTool
        hover = HoverTool(renderers=[c4])
        fig4.add_tools(hover)
        fig4.select(dict(type=HoverTool)).tooltips = OrderedDict(tooltips)
        # Mark 1
        if circuit.type == 'Notch':
            if circuit.finalfit == 'full':
                d = notch_model(circuit.freq, *circuit._cir_fit_pars[0])
            else:
                Ql, absQc, fr = circuit._cir_fit_pars[0]
                phi0 = circuit.phi0
                d = notch_model(circuit.freq, Ql, absQc, fr, phi0)
            fig4.line(d.real, d.imag,
                      legend='Cirlce Fit', color='firebrick',
                      line_width=3)
        else:
            if circuit.finalfit == 'full':
                d = reflection_model(circuit.freq, *circuit._cir_fit_pars[0])
            else:
                Ql, absQc, fr, phi0, _ = circuit._cir_fit_pars[0]
                d = reflection_model(circuit.freq, Ql, absQc, fr, phi0)
            fig4.line(d.real, d.imag,
                      legend='Circle Fit', color='firebrick',
                      line_width=3)
        fig4.legend.background_fill_alpha = 0.3
        # Create grid
        fig = gridplot([fig2, fig3, fig1, fig4], ncols=2, plot_height=350,
                       plot_width=350)
        show(fig)

    elif engine == 'pyplot':
        fig = plt.figure(figsize=(20 / 2.54, 20 / 2.54))
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.35, hspace=0.35)  # Ensure space for label
        # Magnitude
        plt.subplot(221)
        plt.title('Magnitude')
        plt.plot(freq, mag_data)
        plt.plot(freq, mag_fit, '--', color=cc['r'], linewidth=2.0)
        plt.legend(['Data', 'Fit'], loc=3)
        plt.xlabel('Frequency (GHz')
        plt.ylabel('Magnitude (dB)')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])
        plt.locator_params(axis='x', nbins=5)
        # Phase
        plt.subplot(222)
        plt.title('Phase')
        plt.plot(freq, phase_data)
        plt.plot(freq, phase_fit, '--', color=cc['r'], linewidth=2.0)
        plt.legend(['Data', 'Fit'], loc=3)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Phase (deg)')
        plt.grid()
        plt.xlim([freq[0], freq[-1]])
        plt.locator_params(axis='x', nbins=5)
        # Real/Imag
        plt.subplot(223)
        plt.title('Real and Imaginary')
        plt.plot(data.real, data.imag, '.', markersize=4)
        plt.plot(fit_data.real, fit_data.imag, color=cc['r'], linewidth=1.3)
        plt.xlabel('Re (V)')
        plt.ylabel('Im (V)')
        plt.grid()
        plt.legend(['Data', 'Fit'])
        plt.locator_params(axis='x', nbins=5)
        plt.subplot(224)
        plt.title('Normalized Circle')
        plt.xlabel('Re (V)')
        plt.ylabel('Im (V)')
        plt.axis('equal')
        plt.grid()
        plt.plot(offres, 0, "o", color=cc['r'])
        plt.plot(z_data_norm.real, z_data_norm.imag, 'D', zorder=0,
                 color=cc['b'], label='Data')
        plt.gca().add_artist(plt.Circle((xc1, yc1), r, color=cc['r'],
                                        fill=False, linewidth=2, label='Fit'))
        plt.legend(loc=2)


def plot_delay(circuit):
    display(Markdown('## 1. Electric Delay'))
    if circuit._delayOffset is None:
        print('Delay set manually. Offset value unknown. Therefore the line '
              'could be running just parallel')
        fig = figure(width=800, height=400, title='Electric Delay fit')
        phase = np.unwrap(np.angle(circuit.value_raw)) * 180 / np.pi
        fig.circle(circuit.freq, phase, legend='Data')
        fig.line(circuit.freq,
                 circuit.delay * circuit.freq * 360,
                 color='firebrick', line_dash=[5, 5], legend='Linear Fit',
                 line_width=2)
        show(fig)
    else:
        fig = figure(width=800, height=400, title='Electric Delay fit')
        phase = np.unwrap(np.angle(circuit.value_raw)) * 180 / np.pi
        fig.circle(circuit.freq, phase, legend='Data')
        fig.line(circuit.freq, circuit._delayOffset -
                 circuit.delay * circuit.freq * 360,
                 color='firebrick', line_dash=[5, 5], legend='Linear Fit',
                 line_width=2)
        show(fig)


def plot_linear_slope(circuit):
    display(Markdown('## 2. Linear slope in magnitude signal\n'))
    try:
        fig = figure(width=800, height=400, title='Linear fit in dB ' +
                                                  'magnitude')
        fig.circle(circuit.freq, 20 * np.log10(np.abs(circuit.value_raw)),
                   legend='Data')
        fig.line(circuit.freq, circuit._bg_pars[-1] +
                 circuit._bg_pars[-2] * circuit.freq, color='firebrick',
                 line_dash=[5, 5], legend='Lorentzian Fit', line_width=2)
        show(fig)
    except AttributeError:
        print('No background subtraction\n')


def plot_lorentzian(circuit):
    display(Markdown('## 3. Lorentzian fit\n'))
    fig = figure(width=800, height=400, title='Lorentzian fit')
    fig.circle(circuit.freq, np.abs(circuit.value),
               legend='Data')
    fig.line(circuit.freq, lorentzian_abs(circuit.freq,
                                          *circuit._lor_pars[0]),
             legend='Lorentzian Fit', color='firebrick', line_dash=[5, 5],
             line_width=2)
    show(fig)
    errs = np.sqrt(np.diag(circuit._lor_pars[1]))
    df = pd.DataFrame({'Value': circuit._lor_pars[0], 'Error': errs},
                      index=['Offset', 'Amplitude', 'Ql', 'fr'],
                      columns=['Value', 'Error'])
    display(df)


def plot_circle_fit_I(circuit):
    display(Markdown('## 4. Circle fit to obtain offrespoint, a and ' +
                     'alpha\n'))
    fig = figure(width=400, height=400, title='Circle Fit I')
    fig.circle(circuit.value.real, circuit.value.imag, legend='Data')
    x_o = circuit.fitresults_full_model.Value.x_offres
    y_o = circuit.fitresults_full_model.Value.y_offres
    offrespoint = np.complex(x_o, y_o)
    xc, yc, r = circuit._circle_pars1
    p = np.linspace(0, 2 * np.pi, 100)
    circle = np.complex(xc, yc) + r * np.exp(1j * p)
    fig.circle(offrespoint.real, offrespoint.imag, color='firebrick',
               size=10, legend='Offrespoint')
    fig.line(circle.real, circle.imag, color='firebrick', line_width=2,
             legend='Circle Fit')
    show(fig)


def plot_phase_fit(circuit):
    xc, yc, r = circuit._circle_pars1
    p = np.linspace(0, 2 * np.pi, 100)
    display(Markdown('## 5. Phase fit for theta0'))
    data = (circuit.value - np.complex(xc, yc))
    fig = figure(width=800, height=400, title='Lorentzian fit')
    fig.circle(circuit.freq, np.unwrap(np.angle(data)),
               legend='Data')
    fig.line(circuit.freq, tan_phase(circuit.freq,
                                     *circuit._theta_pars[0]),
             legend='Tan Fit', color='firebrick', line_dash=[5, 5],
             line_width=2)
    show(fig)
    errs = np.sqrt(np.diag(circuit._theta_pars[1]))
    df = pd.DataFrame({'Value': circuit._theta_pars[0], 'Error': errs},
                      index=['Theta0', 'Ql', 'fr', 'compression'],
                      columns=['Value', 'Error'])
    display(df)


def plot_final_circle(circuit):
    display(Markdown('## 6. Final Circle fit'))
    fig = figure(width=400, height=400, title='Circle fit (final)' +
                                              'for Ql, absQc and fr')
    fig.circle(circuit.circle_norm.real, circuit.circle_norm.imag,
               legend='Data')
    if circuit.type == 'Notch':
        d = notch_model(circuit.freq, *circuit._cir_fit_pars[0])
        fig.line(d.real, d.imag,
                 legend='Cirlce Fit', color='firebrick',
                 line_dash=[5, 5], line_width=2)
    else:
        d = reflection_model(circuit.freq, *circuit._cir_fit_pars[0])
        fig.line(d.real, d.imag,
                 legend='Circle Fit', color='firebrick',
                 line_dash=[5, 5], line_width=2)
    show(fig)
    errs = np.sqrt(np.diag(circuit._cir_fit_pars[1]))
    df = pd.DataFrame({'Value': circuit._cir_fit_pars[0], 'Error': errs},
                      index=['Ql', 'absQc', 'fr', 'Phi0'],
                      columns=['Value', 'Error'])
    display(df)


def plot_steps(circuit):
    """
        Plots step by step what is done during the whole fitting
        procedure.
    """
    # 1. Delay
    plot_delay(circuit)

    # 2. Linear slope in magnitude
    plot_linear_slope(circuit)

    # 3. Lorentzian
    plot_lorentzian(circuit)

    # 4. Circle Fit I
    plot_circle_fit_I(circuit)

    # 5. Phase fit
    plot_phase_fit(circuit)

    # 6. Final circle fit
    plot_final_circle(circuit)


def plot_residuals(circuit):
    fig = figure(width=800, height=400, title='Residuals')
    fig.circle(circuit.freq, np.abs(circuit.value_raw - circuit.value_calc))
    show(fig)


def plot_weights(circuit):
    fig = figure(width=800, height=400, title='Weights')
    fig.circle(circuit.freq, circuit._weights)
    show(fig)
