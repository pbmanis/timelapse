# timelapse
Process timelapse of 2P stacks
Convert timelapse+zstack data into a max-filtered video.

This program takes either combined time-lapses of z-stacks, or multiple z-stacks, 
allows the user to select a subset of stacks that contain appropriate registration points, 
and select a z-stack positions in each video that correspond across the time series. 
The first selection corrects for long-term drift by allowing subsetting of the data,
whereas the second corrects for z position drift over time within the stacks.

Input data must be from Acq4, in the format of either ImageSequences that contain an
image_nnn.ma file with concatenated stacks, or from individual sequential directories
of ImageSequences, each containing image_000.ma as the stack for one time point.
Support for other kinds of input sequences can be added, but is not supported. 

Processing is defined by the 'filelist' dictionary at the top of the file.
----------------------

1. 'auto': ImageSequence_nnn has a number if image_nnn.ma files; each of those files is a single
    time point in the sequence. 

2. 'manual': Each ImageSequence_nnn/ directory has a single image_000.ma file.
    The ImageSequence_nnn corresponds to a single time point.
    The slice directory defines the overall structure.

Other parameters are as defined in the source. Currently, the filelist dictionary
must be edited by hand to update the data. For example, after performing a satisfactory
registration, use the "Apply Reference" to list out the reference frames, then copy
the list from the terminal into the dictionary to store the parameters for future use.

Requires:
--------

* acq4 [https://github.com/acq4], make a soft link from the local directory to acq4 to access the DataManager and metaarrays.
* pyqtgraph [https://github.com/pyqtgraph]. Perform an install.
* imreg from [http://www.lfd.uci.edu/~gohlke/code/imreg.py.html] Save source as imreg.py in local directory.
* tifffile from [http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html] Save source as tifffile.py in local directory.

Luke Campagnola and Paul Manis, 4-2015 and March-August-2016.
