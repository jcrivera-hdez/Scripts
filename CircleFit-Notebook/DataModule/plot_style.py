#!/usr/bin/python
"""This file contains a normalized plot style for our group."""

import matplotlib.pyplot as plt
import builtins
import re

# Colorscheme taken from http://colorbrewer2.org/
#color_scheme = ['#377EB8', '#E41A1C', '#4DAF4A', '#984EA3', '#FF7F00',
#                '#FFFF33', '#A65628', '#F781BF', '#000000']
color_scheme = ['#0072bd','#d95319','#edb120','#7e2f8e','#77ac30','#4dbeee',
                '#a2142f', '#000000']
#color_labels = ['b', 'r', 'g', 'p', 'mand', 'y', 'br', 'pink', 'k']
color_labels = ['b', 'mand', 'y', 'p', 'g', 'lb', 'r', 'k']
#color_scheme = Colorblind8
cc = dict(zip(color_labels, color_scheme))
builtins.cc = cc  # Make this a global variable for easy access

def set(font='Serif', fontsize=11, figsize=(8.6, 8.6),
        linewidth=1.5, color_scheme=color_scheme,
        color_labels=color_labels):
    """Function to set global plot style.

    Parameters
    -----------
    font : str
        Default font. Chose from ['Sans-Serif, 'Serif', 'Times New Roman', ...]
    fontsize : int
        Fontsize. Defaults is 11
    figsize : list
        Figure size in cm. [(x_dim (cm), y_dim (cm))]. For example (8.6, 8.6)]
        for the default PRL single column figure.
    linewidth : float
        Default linewidth
    color_scheme : list
        Colors for plot. Default is:
        [['#377EB8', '#E41A1C', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33',
        '#A65628', '#F781BF']
    color_labels : list
        Abbreviations array for colors. Default is:
        ['b', 'r', 'g', 'p', 'mand', 'y', 'br', 'pink']
    """
    params = {
               'font.size': fontsize,
               'backend': 'PDF',
               'font.family': font,
               'figure.figsize': (figsize[0]/2.54, figsize[1]/2.54),
               'axes.prop_cycle': plt.cycler('color', color_scheme),
               'axes.formatter.useoffset': False,
               'lines.linewidth': linewidth,
               'axes.axisbelow': True,  # Grid axis below data
               'grid.color': '#BFBFBF',
               'grid.linestyle': '-',
               'legend.fontsize': 10
             }

    plt.rcParams.update(params)
    builtins.cc = dict(zip(color_labels, color_scheme))


def check_color(style):
    """Help function, to check if color is part of the default colors"""
    for kw in list(cc.keys()):
        m = re.search(kw, style)
        if m:
            return m.group()

    # Return 'b' if nothing has found
    return 'b'
