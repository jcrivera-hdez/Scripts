# n -*- coding: utf-8 -*-
"""
    Class for simple data y=f(x)


    !!!DEPRECIATED!!!
    ONLY HERE FOR COMPATIBLITY TO OLD DATAMODULES. DO NOT USE OR MODIFY IT


    Author: Iman, Oscar Gargiulo, Christian Schneider
"""
from DataModule.base import data_module_base
import numpy as np


class data_line(data_module_base):
    """Class for real y=f(x) data."""

    def __init__(self, x=None, y=None):
        super().__init__()

        self._fit_executed = False
        self._fit_labels = None

        if x is None:
            self.x = np.array([])
            self.y = np.array([])
        else:
            if y is None:
                print('Error: no y-axis inserted')
                raise Exception('EMPTYARRAY')
            else:
                self.load_var(x, y)

        self.__errtype_list = ['ABS', 'REL', 'RELPC']

        print('Data_line depreciated. Please use data_table')
        # TODO Redesign
        self.xerr = (0, 0, True)
        self.yerr = (0, 0, True)

    def load_var(self, x, y):
        """Import data from two tuples/lists/array.

        Parameters
        -----------
        x : list
            X-Array. Typically frequencies
        y : list
            Y-Array. Typically magnitude or phase values
        """

        x = np.array(x)
        y = np.array(y)

        if x.size / len(x) != 1.:
            print('Error in the x-axis, check it!')
            raise Exception('NOTANARRAY')

        if y.size / len(y) != 1.:
            print('Error in the y-axis, check it!')
            raise Exception('NOTANARRAY')

        if np.isscalar(x[0]) is False:
            print('Error: bad x-axis, maybe it is a list of list')
            raise Exception('NOTANARRAY')

        if np.isscalar(y[0]) is False:
            print('Error: bad x-axis, maybe it is a list of list')
            raise Exception('NOTANARRAY')

        if len(x) != len(y):
            print('WARNING: x and y length mismatch')

        self.x = x
        self.y = y
        self.select()

    def return_ysel(self):
        """Return currently selected y values"""
        return self.y[self.idx_min:self.idx_max]


###############################################################################
# Aliases (for compatibility to previous datamdoule versions) #################
data_2d = data_line
