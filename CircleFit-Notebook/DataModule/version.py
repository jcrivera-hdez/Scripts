__version__ = '3.1.0'

"""
Last Updates:

v3.1.0 - OSC:
    - fixed the average_data function, at the moment it works only for data_table
    
v3.0.0. -CHR:
  - Updated to data_table and data_grid structure

v2.6.1 - OSC:
    - corrected the amplitude of the lorentzian_fit function
    
v2.6.0 - OSC:
    - reduced data_complex mem usage, dB and phase are now functions, check the
    help of these functions for more info
    - update the data module conversion function to adapt to the new
    data_complex

v2.5.0 - CHR:
  - Automatically sort data during upgrade for data_line
  - Added function to plot multiple datamodules in one figure dm.plot_multiple
  
v2.4.1 - CHR:
  - Removed Resize Tool in bokeh (depreciated)
  
v2.4.0 - CHR:
  - Added data_IQ class

v2.3.7 - OSC:
    - re-launching the exception when a fit failes (inside data_line.fit() function), one has to have the possibility to handle exceptions. 
    NOTE: This exception handling was added and it is not written in the version.

v2.3.6 - Michael
 - Added extract_max_lorentzianfit_y to data_surface

v2.3.5 - CHR:
 - Added compatibiltiy to matplotlib quick syntax

v2.3.4 - CHR:
 - Some minor improvements and bugfixes

v2.3.3 - CHR:
  - Adapted OSC load folder function to datamodule.

v2.3.2 - OSC:
    - when saving with the 'save' function, if the file exists it will copy the
    old file in the "duplicates" folder and will append a number to it. The new
    file will be saved in the current folder with the given name. In this way
    if a measurement didn't give the correct result, one can measure it again
    using the same name, all the discarded measurements will accumulate in the
    "duplicates" folder.

v2.3.1 - OSC:
    - the return_min/max of data_surface code has been fixed (nnB=1 disables
    interpolation)

v.2.3.0 -CHR:
- Fixed bugs and compatibility of circlefit (thanks to David)

v2.2.1 -CHR:
- Fixed color chosing bug for data_line.plot()

v2.2.0 -CHR:
- Formatted and checked code for Circlefit
- Change structure of circlefit, a data_complex module can now have a subclass
  circuit in which everything about the circlefit is stored

v2.1.2 - CHR:
- Adapted upgrade_dm function for new dB and phase subdatamodule of complex
  data
- took out 'o' from color names, because for style flag o means circles

v2.1.1 - CHR:
- removed hardcoded axes label in surface plot

v2.1.0 - CHR:
- changes in complex data module for convenient working
  - added .dB() and .phase() functions to extract magnitude and phase
  - added plot_dB and plot_phase function for plotting magnitude and phase
  - modulized plotting


v2.0.7 - OSC:
-added the function T2_beating in fit functions

v2.0.6 (CHR)
- Changed to not automatically show plot if figure is given for data_line
- Formatting HoverTool for data_line

v2.0.5 - OSC:
- small update to the function local_min in data_line
- added the function local_max

v2.0.4 - OSC:
- fixed the function avarage data (it was an old version)

v2.0.3 (DAV, CHR)
- added missing return for load_csv functions
- modified return for data_surface, to return correct data
- fixed bug, now data_line works without error
- Fix in update datamodule function: Upgrade message appears now just one time
  if multiple files are upgrade

v2.0.2 (CHR, DAV):
- Bugfix for average function in data_stash_x and combine_data to support
  complex values

v2.0.1 (CHR, OSC, DAV):
- Added function to import csv: dm.load_csv(Filename)
- Bugfixes
- Updated upgrade function

v2.0.0 (CHR):
- Implemented folder structure
- Small code corrections
- Added style configuration for plots
- Changed smooth function to more sophisticated filter
- Removed smooth_vec function
- Added bokeh plot function to data_2d
- Added bokeh plot function to data_3d
- fixed norm in data_3d.imshow()
- update color scheme for data_3d.imshow()
- update smooth functions in data_3d to savitzky golay
- fixes in data_line.fit() function
- changed error to chi_squared in fit function
- renamed retur_fit_values to fit_func
- Renamed data.cplx to data.complex()
- Added plot function to data.complex()
- Added aspect ratio to data_surface.plot() functions
- Changed plot function for circlefit to display dB and phase
- Changed extract function to return just one array (f>= value)
- Added function to import csv

v1.6.0 - OSC:
- modified the functions contourf and imshow in data_3d
  (check the help function for details)
- imshow now plots correctly (it was flipped before)
- The load_datamodule function will now try as default to upgrade an opened
  datamodule, it is possible to toggle this option

v1.5.2 - CHR:
- dm.fit:
  added possibility to change fit function parameters for scipys
  optimize fitfunction

v1.5.1 - OSC:
    - moved the version number inside the class constructor
    - printed a conversion note, but it will work only from now on (see above)
    
v1.5.0 - OSC:
- inserted a common function that tries to convert old data modules in an updated one
- renamed data_2d.__fit_executed in ._fit_executed

v1.4.0 - OSC:
- inserted the functions for errors management in data_2d, plotting errorbars in old datamodule will not work
    
v1.3.3 - DAV:
    - Improved data_cplx to automatically evaluated used parameters when
    using reflection config.
    
v.1.3.2 - CHR:
- Improved .save again
    - Print error message if file already exists
    - Create subfolder duplicates and saves file there to preven data loss

v1.3.1 - DAZ:
- improved .save, such that it:
    - creates directory automatically if it does not exist
    - in case file already exists it appends next free int. number
    - names measurements 'measurement' by default
    
v1.3.0 - Oscar:
- re-organized the classes, now every datamodule class inherit a base class
- improved memory usage
- the datamodule doesn't contain a selection of the copy, to obtain a selection use return_xsel(), return_ysel(), return_zsel() and return_vsel() functions

v1.2.6 - DAZ:
- minor updates in complex datamodule for new bokeh plotting engine
prior pyplot is still available, set plot_engine = 'pyplot' when calling circle fit

v1.2.5 - DAZ:
- minor modifications in complex datamodule

v1.2.4 - DAZ:
- minor modifications in complex datamodule

v1.2.3 - OSC + DAZ
- added a save_date_format, saving will not add the time as default, it is always possible to rewrite the save_date_format parameter
- complex datamodule: dc implemented as option in circle fit notch

v1.2.2 - DAZ:
- additional parameter for circle fit (in complex datamodule )to give additional information  as string

v1.2.1 - Oscar:
- fixed error bars option in data_2d.plot

v1.2.0 - Oscar:
- changed the function save_OBJ, now a date and time will be automatically added at the begin of the file name (the option can be disabled).
- when saving it is not necessary to specify the .dm extension
- it is possible to load the data when creating a new data module
- added error bars plotting as default
- removed the functions load and save (they will be inserted in the UtilitiesLib eventually), the function save_OBJ is now renamed save

v1.1.2 - Oscar:
- added the offset option in the dm.nice_plot()
- fixed the data_3d print options to make it compatible with nice_plot()

v1.1.1
- added option to set initial fr for circle fit in dm_cplx


v1.1.0
- fixed some problem with the select function in data_2d
- modified the function pcolormesh that was not working before in data_3d
- added the functions extrapolate_x, extrapolate_y , extrapolate_min_x, extrapolate_min_y, extrapolate_max_x, exptrapolate_max_y in data_3d
- added the function copy() in each data module, that returns a copy of it


- in the library now we have nice_plot function that can be called BEFORE plotting data, it will make the plot standard and nicer
- the functions data_2d.plot and data_3d.contourf have been slightly modified
- the functions replot and nice_plot have been removed from data_2d
- added the functions to stack data along the x-axis or y-axis for data_2d and data_3d
- added the function to make data averaging for data_2d and data_3d
- add the function to combine data, for example if we decide to put more points in a plot, we can inglobe the new data in the old one

V1.0.3
- added complex data module including circle fit for notch and refl. config.


V1.0.2
- default fontsize in plots is 20 now
"""
