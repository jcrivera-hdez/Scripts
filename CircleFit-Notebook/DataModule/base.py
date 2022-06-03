# -*- coding: utf-8 -*-
"""Base functions of every datamodule type"""

import os
import time
import pickle
import copy
import numpy as np
from .version import __version__


class data_module_base(object):
    """Base class of DataModule

    These functions is used by every datamodule.
    """
    def __init__(self):
        self.__version__ = __version__
        self.comments = ''
        self.par = {}  # Parameters for collected data (e.g. VNA settings)
        self.temp_start = None
        self.temp_stop = None
        self.temp_start_time = None
        self.temp_stop_time = None
        self.time_start = None
        self.time_stop = None
        self.date_format = '%Y-%m-%d-%H:%M:%S'
        self.save_date_format = '%Y-%m-%d'
        self.idx_min = 0
        self.idx_max = None

    def insert_par(self, **kwargs):
        """Add parameters to the data module given by keywords.

        Example
        ---------
            >>> data.insert_par(temp= 25e-3, comment='test', foo='bar')
        """
        self.par.update(**kwargs)

    def remove_par(self, key):
        """Remove parameter by key from data module parameters

        Parameters
        -----------
        key : str
            Key of parameter dictionary
        """
        try:
            self.par.pop(key)
        except KeyError:
            raise Exception('Parameters empty or key not found')

    def save(self, fname, useDate=True, force=False):
        """Save DataModule

        The date will be added in front of the filename and a '.dm' extension
        will be added automatically to fname, if not already given.

        If the file already exists, the existing file will be moved in a
        subfolder named duplicates. If this happens multiple times, numbers
        will be added to the files in the duplicates folder.

        Parameters
        -----------
        fname : str
            Filename to save
        useDate : bool
            Add Date in front of fname
        force: Overwrite existing file.
        """
        # Split fname in folder and filename
        path = os.path.split(fname)

        # Create directory if not already existent
        if not os.path.exists(path[0]) and path[0]:
            os.makedirs(path[0])

        # Add date
        if useDate:
            time_string = time.strftime(self.save_date_format,
                                        time.localtime(time.time()))
            file_name = time_string + '-' + path[1]

        else:
            file_name = path[1]

        # Check for file extension
        if path[1][-3:].lower() != '.dm':
            file_name += '.dm'

        # Append Folder and be adaptive to windows, etc.
        file_name = os.path.normpath(os.path.join(path[0], file_name))

        # Check for Overwrite
        if not force:
            if os.path.isfile(file_name):
                from shutil import copyfile
                # Add a number if force-Overwrite is False
                fpath, fn = os.path.split(file_name)
                # Create duplicates folder
                if not os.path.exists(os.path.join(fpath, 'duplicates')):
                    os.makedirs(os.path.join(fpath, 'duplicates'))
                fpath = os.path.join(fpath, 'duplicates')
                fn, e = os.path.splitext(fn)
                file_name2 = os.path.join(fpath, fn + '%s.dm')
                number = ''
                while os.path.isfile(file_name2 % number):
                    number = int(number or "0") + 1
                file_name2 = file_name2 % number  # Add number
                copyfile(file_name, file_name2)
                print('File already exists.\nTo prevent data loss, the old' +
                      'file - eventually with a number appended - has been ' +
                      'moved into the subfolder duplicates.\n',
                      flush=True, end=" ")

        with open(file_name, "wb") as f:
            pickle.dump(self, f, -1)

    def copy(self):
        """Copy datamodule.

        Returns
        --------
        Datamodule
            A copy of this datamodule
        """
        return copy.deepcopy(self)

    def select(self, xrng=None):
        """Select range of data.

        Plots, fits, etc will then only be applied on this range.
        If nothing is specified all the data will be select

        Parameters
        ----------
        xrng : list, None
            Start and Stop values of the range in a list [start, stop]. Eg. [1.4e9, 6.5e9]
        """
        self.idx_min = 0
        self.idx_max = None

        if xrng is not None:
            idx = np.where((self.x >= xrng[0]) & (self.x <= xrng[1]))[0]
            self.idx_min = idx[0]
            self.idx_max = idx[-1]
