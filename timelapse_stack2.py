#!/usr/bin/env python

"""
Convert timelapse+zstack data into a max-filtered video.

This program takes either combined time-lapses of z-stacks, or multiple z-stacks, 
allows the user to select a subset of stacks that contain appropriate registration points, 
and select a z-stack positions in each video that correspond across the time series. 
The first selection corrects for long-term drift by allowing subsetting of the data,
whereas the second corrects for z position drift over time within the stacks.


Luke Campagnola and Paul Manis, 4-2015 and 3, 4-2016.

Input data structures:
1. 'auto': ImageSequence_nnn has a number if image_nnn.ma files; each of those files is a single
    time point in the sequence. 

2. 'manual': Each ImageSequence_nnn/ directory has a single image_000.ma file.
    The ImageSequence_nnn corresponds to a single time point.
    The slice directory defines the overall structure.

-----------------
Requires:
acq4 (github.com/acq4), make a soft link from the local directory to acq4
pyqtgraph (github.com/pyqtgraph) [perform install]
imreg from http://www.lfd.uci.edu/~gohlke/code/imreg.py.html
tifffile from www.lfd.uci.edu/~gohlke/code/tifffile.py.html or pip install tifffile (I think)

"""
import sys
import os
from acq4.util.metaarray import MetaArray
import acq4.util.DataManager as DataManager
import pyqtgraph as pg
from PyQt4 import QtGui
from copy import deepcopy
import imreg
import scipy.stats
import numpy as np
from collections import OrderedDict
import argparse
import tifffile as tf

from pyqtgraph.parametertree import Parameter, ParameterTree

basedir = '/Volumes/Backup2B/Sullivan_Chelsea/Chelsea/'
basedir = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/'
basedir = '/Users/pbmanis/Documents/data/'

import warnings

warnings.simplefilter('default')

#
# Analysis is driven by the filelist data structure
#
# 'filelist' is a dictionary, which contains a dict of parameters to guide the analysis.
# 'reference_frames' is a list of the matching frames from each of the z-stacks in the successive
#   time points. These are selected by the sliders.
# 'mode' is either 'auto', or 'manual'. If the data are collected as a time-lapse seuqence of 
#   Z stacks, and appear as a set of "ImageSequence_000" directories, then the mode should be
#   'auto'. If the time-lapse points were manually collected, but the stacks are automatic,
#   then the mode should be 'manual'.
# 'NStacks' indicates the total number of stackes in the file structure
# 'included_stacks' is a list of the first and last records to include.


filelist = OrderedDict([
            #
            #0
            (('2015.04.17_000/slice_001/ImageSequence_000', 0),   # 0
                {'reference_frames': [40, 37, 33, 30, 28, 26, 23, 21, 19, 17, 16, 14],
                 'mode': 'auto',  'Nstacks': 39, 'included_stacks': [0, 39]}),
            #1
            (('2016.03.22_000/slice_000', 1),                            #1
                {'reference_frames': [0]*13,
                 'mode': 'manual', 'Nstacks': 13,'included_stacks': [0, 13]}),
            #2
            (('2016.03.23_000', 2),
                {'reference_frames': [0]*39,
                 'mode': 'manual', 'Nstacks': 39, 'included_stacks': [0, 39]}),
            #3
            (('2016.03.28_000/slice_000', 3),
                {'reference_frames': [0]*len(range(0, 9)),
                 'mode': 'manual', 'Nstacks': 9, 'included_stacks': [0, 9]}),
            #4
            (('2016.04.11_000/slice_000', 4),
                {'reference_frames': [0, 0, 8, 3, 3, 5, 0, 5, 2, 4, 0, 0, ],
                 'mode': 'manual', 'Nstacks': 12, 'included_stacks': [0, 12]}),
            #5
            (('2016.04.13_000/slice_000', 5),
                {'reference_frames': [0]*len(range(2, 16)),
                 'mode': 'manual', 'Nstacks': 16, 'included_stacks': [2, 16]}),
            #6
            (('2016.04.15_000/slice_000', 6),
                {'reference_frames': [0]*len(range(14, 34)),
                 'mode': 'manual', 'Nstacks': 34, 'included_stacks': [14, 34]}),
            #7
            (('2016.08.11_000/slice_000', 7),
                {'reference_frames': [0]*len(range(0,30)),
                 'mode': 'manual', 'Nstacks': 30, 'included_stacks': [0,30]}),
            #8
            (('2016.08.11_000/slice_001', 8),
                {'reference_frames': [14, 13, 10, 7, 9, 17, 17, 19, 16],
                 'mode': 'manual', 'Nstacks': 25, 'included_stacks': [0,9]}),  # was Nstacks = 11, included [14,25]
            #9
            (('2016.08.11_000/slice_001', 9),
                {'reference_frames': [2, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                 'mode': 'manual', 'Nstacks': 25, 'included_stacks': [14,24]}),  # was Nstacks = 11, included [14,25]
            #10
            (('2016.08.15_000/slice_000', 10),
                {'reference_frames': [0]*30, #[17, 19, 16, 14, 16, 18, 19, 19, 19, 19, 19, 19, 19],
                 'mode': 'manual', 'Nstacks': 30, 'included_stacks': [10, 23]}),
             ])


class DataAnalyzer():
    def __init__(self, layout, winsize, ptreedata):
        """
        accepted_stacks: a list of strings with one element for each stack in the file. The list is coded as follows:
        'In': include the stack in the analysis. Stack is colored in greyscale
        'Ex': exclude the stack from the analysis. Stach is colored in red
        'Bad': exclude the stack from the analysis because dataset is malformed. Stack is colored blue.
        Bad stacks can never be analyzed; Ex stacks can be adjusted to be In stacks. 
        """
        self.layout = layout
        self.winsize = winsize
        self.ptreedata = ptreedata
        self.data = None
        self.expt = None
        self.file = ''
        self.fullfile = ''
        self.dh = None
        self.filelist = None
        self.thisfile = None
        self.maxValue = 1.0
        self.imagePlots = []
        self.Z_stacks = []
        
        # build luts:
        pos = np.array([0.0, 1.0])
        red_color = np.array([[0,0,0,255], [255,0,0,255]], dtype=np.ubyte)
        red_map = pg.ColorMap(pos, red_color)
        self.red_lut = red_map.getLookupTable(start=0.0, stop=1.0, nPts=256)
        grey_color = np.array([[0,0,0,255], [255,255,255,255]], dtype=np.ubyte)
        grey_map = pg.ColorMap(pos, grey_color)
        self.grey_lut = grey_map.getLookupTable(start=0.0, stop=1.0, nPts=256)
        blue_color = np.array([[0,0,0,255], [0,0,255,255]], dtype=np.ubyte)
        blue_map = pg.ColorMap(pos, blue_color)
        self.blue_lut = blue_map.getLookupTable(start=0.0, stop=1.0, nPts=256)
        
        
    def set_file_list(self, filelist):
        """
        Store a filelist dictionary
        
        Parameters
        ----------
        filelist : dict (required)
            A dictionary in the appropriate structure for the file list:
            Ex: OrderedDict([
                    (('filename, integerID),
                        {'reference_frames': [list of frames],
                         'mode': 'auto',  'Nstacks': 39, 'included_stacks': [0, 39]}),
                    ...
                    ])
        
        Returns
        -------
        Nothing
        """
        self.filelist = filelist
        
    def get_data(self, expt):
        """
        Read the data set, parse it out according to the file list information,
        and ultimately display the data
        
        Parameters
        ----------
        expt : int (required)
            The id number that is the second element of the filelist key
        
        """
        
        self.expt = expt
        # get the key that has the matching experiment. 
        # keys are tuple of (filedir, expt)
        self.filekey = None
        for fk in self.filelist.keys():
            if fk[1] == expt:
                self.filekey = fk
                break
        if self.filekey is None:
            raise ValueError('File corresponding to experimnt # %d not found in filellist' % expt)
            
        self.file = self.filekey[0]
        self.fullfile = os.path.join(basedir, self.filekey[0])
        print 'File: ', self.fullfile
        self.dh = DataManager.getDirHandle(self.fullfile, create=False)
        found = False
        for n in self.filelist.keys():
            if n[0] in self.dh.name():
                found = True
                break
        if not found:
            raise ValueError('Unknown file: %s' % self.dh.name())
        self.thisfile = self.filelist[self.filekey]

        # adjust the reference frames for the full data set
        # reference frames are listed relative to the included stack numbers in the dict above
        new_ref = [0]*self.thisfile['Nstacks']
        new_ref[self.thisfile['included_stacks'][0]:self.thisfile['included_stacks'][1]] = self.thisfile['reference_frames']
        self.thisfile['reference_frames'] = new_ref
        
        ap = self.ptreedata.child('Acquisition Parameters')
        self.maxValue = ap['MaxValue']
        ap['Include Stack Start'] = self.thisfile['included_stacks'][0]
        ap['Include Stack End'] = self.thisfile['included_stacks'][1]
        
        if self.thisfile['mode'] == 'auto':
            self.get_auto()
        elif self.thisfile['mode'] == 'manual':
            self.get_manual()
        else:
            raise ValueError('Unknown data mode: %s' % self.self.thisfile['mode'])
        self.show_data()
        self.set_stack_limits()
        
    def get_auto(self):
        """
        Read files in which the data is all within one file - e.g., 
        z-stacks embedded in a time series
        """
        z_length = len(self.dh.info()['zStackValues'])
        indexes = self.thisfile['reference_frames']
#        offsets = [-min(indexes), z_length - max(indexes)]
        offsets = [0, z_length]
        indexes = [0]*len(indexes)
        print 'Analyzing in Auto mode'
        print '\tTimes in timelapse: ', z_length
        print '\tIndexes: ', indexes
        print '\tOffsets: ', offsets

        # data = [dh['image_%03d.ma'%i].read()[indexes[i]+offsets[0]:indexes[i]+offsets[1]].
        #         asarray()[np.newaxis, ...] for i in range(len(indexes))]

        self.data = [self.dh['image_%03d.ma'%i].read() [indexes[i]+offsets[0]:indexes[i]+offsets[1]].
                asarray()[np.newaxis, ...] for i in range(len(indexes))]
        self.data = np.concatenate(self.data, axis=0)
        self.n_time_stacks = z_length
        self.Z_stacks = ['In']*self.n_time_stacks # Z stacks with current tagging
        self.Z_stacks_master = deepcopy(self.Z_stacks)  # Z stacks with original tagging (including "bad" stacks)

    def get_manual(self):
        """
        Read files in which the data is stored in separate files.
        Each file is a z-stack at a separate time point, and should 
        have the same length as the first file in the sequence.
        """
        ref_frames = self.thisfile['reference_frames']  # reference frames within a stack
        nframes = self.dh['ImageSequence_%03d' % self.thisfile['included_stacks'][0] + '/image_000.ma'].read().shape[0]
        self.Z_stacks = []
        print self.thisfile['Nstacks']
        for i in range(0, self.thisfile['Nstacks']):
            th = self.dh['ImageSequence_%03d' % i]['image_000.ma']
            if th.exists() and th.read().shape[0] == nframes:
                self.Z_stacks.append('In')
            else:
                self.Z_stacks.append('Bad')
        self.Z_stacks_master = deepcopy(self.Z_stacks)  # Z stacks with original tagging (including "bad" stacks)

        self.n_time_stacks = len(self.Z_stacks)  # number of available stacks over time
        ts = self.included_indices()
        offsets = [-min(ref_frames), nframes - max(ref_frames)]
        print 'Analyzing in Manual mode-------------------------------'
        print '\t# of stacks timelapse: ', self.n_time_stacks
        print '\t# of frames in each:   ', nframes
        print '\tReference Frames:      ', ref_frames
        print '\tOffsets:               ', offsets
        print '\tUsable Frames:         ', offsets[1]-offsets[0]
        print '\tUsable Stacks:         ', ts
        print '\tAccepted Stacks:       ', self.Z_stacks
        #tsr = [t-ts[0] for t in ts]
        try:
            #print 'list of indexes reading: ',  [[ref_frames[i]+offsets[0],ref_frames[i]+offsets[1]] for i in tsr]
            self.data = [self.dh['ImageSequence_%03d'%i]['image_000.ma'].read()[ref_frames[i-ts[0]]+offsets[0]:ref_frames[i-ts[0]]+offsets[1]].
                asarray()[np.newaxis, ...] for i in ts]  # note that some stacks may be missing or incomplete...
        except:
            print 'Error Reading file in Manual Mode'
            print 'len ts: ', len(ts)
            print 'ts - ts[0]: ', [t-ts[0] for t in ts]
            print 'len ref frames: ', len(ref_frames)
            print 'ts: ', ts
            print 'i: ', i
            print 'index[i], o: ', ref_frames[i], offsets[0], offsets[1]
            raise ValueError('Indexing error for ImageSequence image data set %d' % i)
        self.data = np.concatenate(self.data, axis=0)
        
        
    def print_data_info(self):
        print 'data array lengths axis [0]: ', [len(k) for k in self.data]
        print 'data shape after concatenate: ', self.data.shape
        print 'accepted stacks: ', self.Z_stacks

    def show_data(self):
        """
        Build an image plot and display the data, with time scroll bars
        and coloring for includes, excludes and bad series.
        """
        self.build_image_plots()
        self.minmax = (0, 1.)
        for k in range(self.data.shape[0]):
            if k >= len(self.imagePlots):
                continue
            self.imagePlots[k].setImage(self.data[k])
        minint = np.min(self.data, axis=(0,1,2,3))
        maxint = np.max(self.data, axis=(0,1,2,3))
        self.set_view_scale(minint, maxint)
        self.set_reference_frames()

    def included_indices(self):
        indx = [i for i, tf in enumerate(self.Z_stacks) if tf == 'In']
        return indx

    def excluded_indices(self):
        indx = [i for i, tf in enumerate(self.Z_stacks) if tf == 'Ex']
        return indx

    def bad_indices(self):
        indx = [i for i, tf in enumerate(self.Z_stacks) if tf == 'Bad']
        return indx

    def list_included_frames(self):
        #print 'image plot settings: ', dir(self.imagePlots[0])
        print self.thisfile['included_stacks']
        frlist = []
        for i in self.included_indices(): # range(0, self.thisfile['included_stacks'][1]-self.thisfile['included_stacks'][0]):
            frlist.append(self.imagePlots[i].currentIndex)
        print 'included stacks: ', self.thisfile['included_stacks']
        print 'Matching Z position Frames: ', repr(frlist)
        print self.included_indices()
    
    def set_reference_frames(self):
        # get current referenced frame list
        for i in range(len(self.imagePlots)):
            self.imagePlots[i].setCurrentIndex(self.thisfile['reference_frames'][i])
#        pass
        
    def get_reference_frames(self):
        """
        Get the reference frames as selected by the user from the 
        slider values for each image in the display
        """
        # get current referenced frame list
        frlist = []
        for i in range(len(self.imagePlots)):
            frlist.append(self.imagePlots[i].currentIndex)
        self.thisfile['reference_frames'] = frlist
    
    def set_stack_limits(self):
        """
        Define the beginning and end of the times (stacks) to use for an analysis
        """
        for i, x in enumerate(self.Z_stacks):
            # start by including all stacks that are not marked "Bad"
            if self.Z_stacks_master[i] == 'Bad':
                continue
            if (i < self.thisfile['included_stacks'][0]) or (i >= self.thisfile['included_stacks'][1]):
                self.Z_stacks[i] = 'Ex' # exclude outside range
            else: # be inclusive
                self.Z_stacks[i] = 'In'  # only allow to set if main already says that.
#        print self.thisfile['included_stacks']
#        print 'zstack status: ', self.Z_stacks
        self.set_img_colors()
    
    def set_img_colors(self):
        """
        Color the images according to their type - grey for stacks
        that will be included in the analysis, red for those that are
        excluded and blue for those that are bad (short, missing, etc)
        """
        includes = self.included_indices()
        bad = self.bad_indices()
        excludes = self.excluded_indices()
        for i, p in enumerate(self.imagePlots):
            if i in includes:
                p.getImageItem().setLookupTable(self.grey_lut) 
            elif i in excludes:
                p.getImageItem().setLookupTable(self.red_lut)
            elif i in bad:
                p.getImageItem().setLookupTable(self.blue_lut)
            else:
                raise ValueError('Image plot %d not mapped' % i)
        self.set_view_scale(0., self.maxValue)
            
    def set_view_scale(self, minint, maxint):
        for p in self.imagePlots:
            p.setLevels(minint, maxint)


    def register(self):
        """
        Perform registration of the image stacks.
        This is done by flattening the stacks using the maximum projection,
        then using the imreg method to perform alignment
        The resulting images are reframed onto a larger canvas that
        encloses all of the outer bounds of all of the images. 
        This is displayed for reference, and may be written to a tiff file
        using write_tiff.
        """

        valid_indices = self.included_indices()
        mrange = range(len(valid_indices))
        # dim edges to avoid artifacts at the edges of depth range
        dim = self.data[valid_indices].copy()
        dim[:,0] *= 0.33
        dim[:,1] *= 0.66
        dim[:,-1] *= 0.33
        dim[:,-2] *= 0.66

        # flatten stacks
        m = dim.max(axis=1)
        nreg = m.shape[0]
        ireg = int(nreg/2)  # get one near the middle of the sequence.
        # correct for lateral motion
        # print 'ireg: ', ireg
        # print 'data shape: ', m.shape[0]
        
        
       # off = [imreg_dft.translation(m[ireg], m[i])[0] for i in range(0, m.shape[0])]
        # print 'accepted stacks: ', self.Z_stacks
        # print 'ok indices: ', valid_indices
        off = [imreg.translation(m[ireg], m[i]) for i in mrange]
        print 'Registration Offsets: ', off
        offt = np.array(off).T

        # find boundaries of outer rectangle including all images as registered
        minx = np.min(offt[0])
        maxx = np.max(offt[0])
        miny = np.min(offt[1])
        maxy = np.max(offt[1])
        
        print 'minx, maxx, miny, maxy: ', minx, maxx, miny, maxy
        # build canvas
        canvas = np.zeros(shape=(len(valid_indices), m.shape[1]-minx+maxx,
            m.shape[2]-miny+maxy), dtype=m.dtype)
        
        # set initial image (offsets were computed relative to this, so it has no offset)
        # canvas[0, -minx:-minx+m.shape[1], -miny:-miny+m.shape[2]] = m[0]
        for i in mrange:
            ox = offt[0][i] - minx
            oy = offt[1][i] - miny
            canvas[i, ox:(ox+m.shape[1]), oy:(oy+m.shape[2])] = m[i]
        #    print 'canvas %d set' % i

        # correct for bleaching
        levels = np.array([np.median(m[m>scipy.stats.scoreatpercentile(m[i], 95)]) for i in mrange]) # m.shape[0])])
        self.norm = canvas / levels[:, np.newaxis, np.newaxis]
        w = pg.image()
        w.setImage(self.norm)
        
        # find the minimum area that is present in all images in the stack

    def write_metaarray(self):
        """
        write the resulting compressed z-stacks to a file in the original directory.
        """
        ma = MetaArray(self.norm, info=[{'name': 'Time'}, {'name': 'X'}, {'name': 'Y'}, {}])
        ma.write(self.dh.name() + '/max_stack.ma')

    def write_tiff(self):
        """
        convert aligned stack to tiff file stack
        """
        print self.dh.name()
        fn = self.dh.name() + '/max_stack.tiff'
        fh = open(fn, 'w')
        tf.imsave(fh, self.norm.astype('float32'), imagej=True, software='acq4')
        fh.close()



    def process_changes(self, param, changes):
        """
        Respond to changes in the parametertree and update class variables
    
        Parameters
        ----------
        param : parameter list
    
        changes : changes as returned from the parameter tree object
    
        Returns
        -------
        Nothing
    
        """
        for param, change, data in changes:
            path = ptreedata.childPath(param)
            # Parameters and user-supplied information
            #
            if path[1] == 'MaxValue':
                self.set_view_scale(0., data)
                self.maxValue = data
            if path[1] == 'Include Stack Start':
                self.thisfile['included_stacks'][0] = data
                self.set_stack_limits()
            if path[1] == 'Include Stack End':
                self.thisfile['included_stacks'][1] = data
                self.set_stack_limits()
            #
            # Actions:
            #
            if path[1] == 'List Included Frames':
                self.list_included_frames()
            if path[1] == 'Register':
                self.register()
            # if path[1] == 'Apply Max':
            #     self.set_view_scale(0, self.maxValue)
            if path[1] == 'Apply Reference':
                self.set_reference_frames()
            if path[1] == 'Write Registered Stack':
                self.write_tiff()

    def get_layout_dimensions(self, n, pref='height'):
        """
        Return a tuple of optimized layout dimensions for n axes
    
        Parameters
        ----------
        n : int (no default):
            Number of plots needed
        pref : string (default : 'height')
            preferred73
         way to organized the plots (height, or width)
    
        Returns
        -------
        (h, w) : tuple
            height (rows) and width (columns)
    
        """
        nopt = np.sqrt(n)
        inoptw = int(nopt)
        inopth = int(nopt)
        while inoptw*inopth < n:
            if pref == 'width':
                inoptw += 1
                if inoptw * inopth > (n-inopth):
                    inoptw -= 1
                    inopth += 1
            else:
                inopth += 1
                if inoptw * inopth > (n-inoptw):
                    inopth -= 1
                    inoptw += 1
            
        return(inopth, inoptw)

    def build_image_plots(self):
        """
        Build the layout for all of the images in the dataset.
        """
        if len(self.imagePlots) > 0: # remove prior plots
            for k in range(len(self.imagePlots)):
                if self.imagePlots[k] is not None:
                    self.imagePlots[k].close()
                    self.imagePlots[k] = None
                
        totalPlots = self.n_time_stacks
        self.imagePlots = [None]*totalPlots
        self.image_layout = [None]*totalPlots

        (rows, columns) = self.get_layout_dimensions(totalPlots)
        ptreewidth = 120
        # build layout for plots and parameters
        self.layout.addWidget(ptree, 0, 0, rows, 1) # add the Parameter Tree on left
        self.layout.setColumnMinimumWidth(0, ptreewidth)
        dx = (self.winsize[1] - ptreewidth)/columns
        for i in range(0, columns):
            self.layout.setColumnMinimumWidth(i+1,dx)
        k = 0
        for i in range(rows):
            for j in range(columns):
                if k >= self.n_time_stacks:
                    break
                self.imagePlots[k] = pg.ImageView()
                self.imagePlots[k].ui.roiBtn.hide()
                self.imagePlots[k].ui.menuBtn.hide()
                self.imagePlots[k].ui.histogram.hide()
                self.layout.addWidget(self.imagePlots[k], i, j+1, 1, 1)
                k = k + 1

        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze time lapse stacks')
    parser.add_argument('Experiment',  type=int,
                       help='Select an Experiment number')

    args = parser.parse_args()
    expt = args.Experiment
    #
    # app = pg.mkQApp()

    app = pg.mkQApp()
    win = QtGui.QWidget()  # top level widget
    layout = QtGui.QGridLayout()  # make a grid layout
    win.setLayout(layout) # makes the grid layout to manage the widgets
    win.setWindowTitle('TimeLapse Analysis')
    resolution = QtGui.QDesktopWidget().screenGeometry()
    winsize = (1024, 800)
    win.resize(winsize[0], winsize[1])
    # roughly center window on screen so it does not have to be dragged
    pos = [(resolution.width() / 2) - (win.frameSize().width() / 2),
              (resolution.height() / 2) - (win.frameSize().height() / 2)]
    win.move(pos[0], pos[1])
    win.show()
    
    # Define parameters that control aquisition and buttons...
    params = [
         {'name': 'Acquisition Parameters', 'type': 'group', 'children': [
             {'name': 'MaxValue', 'type': 'float', 'value': 0.3, 'limits': [0.01, 10.], 
                 'step': 0.05, 'suffix': 'V', 'default': 0.3},
             {'name': 'Include Stack Start', 'type': 'int', 'value': 0, 'limits': [0, 40], 'step': 1, 'default': 0},
             {'name': 'Include Stack End', 'type': 'int', 'value': 20, 'limits': [0, 40], 'step': 1, 'default': 20},
            ]},
        {'name': 'Actions', 'type': 'group', 'children': [
            {'name': 'List Included Frames', 'type': 'action'},
            {'name': 'Apply Reference', 'type': 'action'},
            {'name': 'Register', 'type': 'action'},
            {'name': 'Write Registered', 'type': 'action'},
            ]},
        ]
    ptree = ParameterTree()
    ptreedata = Parameter.create(name='params', type='group', children=params)
    ptree.setParameters(ptreedata)

    DA = DataAnalyzer(layout, winsize, ptreedata)
    DA.set_file_list(filelist)
    DA.get_data(expt)
    DA.print_data_info()

    ptreedata.sigTreeStateChanged.connect(DA.process_changes)  # connect parameters to their updates

    if sys.flags.interactive == 0:
        app.exec_()