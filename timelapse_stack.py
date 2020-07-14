"""
Script used to convert timelapse+zstack data into a max-filtered video.

Luke Campagnola and Paul Manis, 4-2015 and 3, 4-2016.

Input data structures:
1. 'auto': ImageSequence_nnn has a number if image_nnn.ma files; each of those files is a single
    time point in the sequence. 

2. 'manual': Each ImageSequence_nnn has a single image_000.ma file; the ImageSequence itself is 
    the individual time point; the slice directory handles the 

"""

from acq4.util.metaarray import MetaArray
import acq4.util.DataManager as DataManager
import imreg_dft
import scipy.stats
import re
import os
import numpy as np
import pyqtgraph as pg
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='Analyze time lapse stacks')
parser.add_argument('Experiment',  type=int,
                   help='Select an Experiment number')

args = parser.parse_args()
expt = args.Experiment

app = pg.mkQApp()
basedir = '/Volumes/Backup2B/Sullivan_Chelsea/Chelsea/'
basedir = '/Volumes/Promise Pegasus/ManisLab_Data3/Sullivan_Chelsea/'
# man.setBaseDir(basedir)

#
# Analysis is driven by the filelist data structure
#
# 'filelist' is a dictionary, which contains a dict of parameters to guide the analysis.
# 'refframes' is a list of the matching frames from each of the z-stacks in the successive
#   time points
# 'mode' is either 'auto', or 'manual'. If the data are collected as a time-lapse seuqence of 
#   z stacks, and appear as a set of "ImageSequence_000" directories, then the mode should be
#   'auto'. If the time-lapse points were manually collected, but the stacks are automatic,
#   then the mode should be 'manual'.
# 'datalist' is a list of the records to include. If 'datalist' is set to None, then all 
#   recordings will be included. Note that if mode is "auto", then datalist should be None.
#

filelist = OrderedDict([('2015.04.17_000/slice_001/ImageSequence_000',
                {'refframes': [40, 37, 33, 30, 28, 26, 23, 21, 19, 17, 16, 14],
                 'mode': 'auto', 'datalist': None}),
            ('2016.03.22_000/slice_000',
                {'refframes': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'mode': 'manual', 'datalist': None}),
            ('2016.03.23_000',
                {'refframes': [0]*39,
                 'mode': 'manual', 'datalist': None}),
            ('2016.03.28_000/slice_000',
                {'refframes': [0]*len(range(0, 9)),
                 'mode': 'manual', 'datalist': range(0, 9)}),
            ('2016.04.11_000/slice_000',
                {'refframes': [0]*len(range(0, 13)),
                 'mode': 'manual', 'datalist': range(0, 13)}),
            ('2016.04.13_000/slice_000',
                {'refframes': [0]*len(range(2, 16)),
                 'mode': 'manual', 'datalist': range(2, 16)}),
            ('2016.04.15_000/slice_000',
                {'refframes': [0]*len(range(14, 34)),
                 'mode': 'manual', 'datalist': range(14, 34)}),

             ])
             
# select a dataset to analyze:

ff = filelist.keys()[expt]  # gets the dataset name
fullpath = os.path.join(basedir, ff)

print 'File: ', fullpath

dh = DataManager.getDirHandle(fullpath, create=False)

# collect all data with depth corrected
#dh = man.currentFile
found = False
for n in filelist.keys():
    if n in dh.name():
        found = True
        break
if not found:
    raise ValueError('Unknown file: %s' % dh.name())
    
print 'Dataset found.'
indexes = filelist[n]['refframes']
    
if filelist[n]['mode'] == 'auto':
    z_length = len(dh.info()['zStackValues'])
    offsets = [-min(indexes), z_length - max(indexes)]

    print 'Analyzing in Auto mode'
    print '\tTimes in timelapse: ', z_length
    print '\tIndexes: ', indexes
    print '\tOffsets: ', offsets

    data = [dh['image_%03d.ma'%i].read()[indexes[i]+offsets[0]:indexes[i]+offsets[1]].
            asarray()[np.newaxis, ...] for i in range(len(indexes))]

elif filelist[n]['mode'] == 'manual':
    nframes = dh['ImageSequence_%03d' % filelist[n]['datalist'][0]]['image_000.ma'].read().shape[0]
    ts = []
    if filelist[n]['datalist'] != None :
        sequence = filelist[n]['datalist']
    else:
        sequence = range(len(indexes))
    for i in sequence:
        th = dh['ImageSequence_%03d'%i]['image_000.ma']
        if th.exists() and th.read().shape[0] == nframes:
            ts.append(i)

    z_length = len(ts)
    offsets = [-min(indexes), z_length - max(indexes)]

    print 'Analyzing in Manual mode'
    print '\t# of depths in timelapse: ', z_length
    print '\t# of frames in each: ', nframes
    print '\tIndexes: ', indexes
    print '\tOffsets: ', offsets
    try:
        print indexes
        print offsets
        print 'list of indexes reading: ',  [[indexes[i]+offsets[0],indexes[i]+offsets[1]] for i in ts]
        data = [dh['ImageSequence_%03d'%i]['image_000.ma'].read()[indexes[i]+offsets[0]:indexes[i]+offsets[1]].
            asarray()[np.newaxis, ...] for i in range(len(ts[:-2]))]
    except:
        print 'error'
        print 'len ts: ', len(ts)
        print 'ts: ', ts
        print 'i: ', i
        print 'index[i], o: ', indexes[i], offsets[0], offsets[1]
        raise ValueError('Indexing error for ImageSequence image data set %d' % i)

else:
    raise ValueError('Unknown data mode: %s' % filelist[n]['mode'])

print 'data shape: ', [len(k) for k in data]
data = np.concatenate(data, axis=0)
# print 'data shape (t, z, x, y): ', data.shape

# dim edges to avoid artifacts at the edges of depth range
dim = data.copy()
dim[:,0] *= 0.33
dim[:,1] *= 0.66
dim[:,-1] *= 0.33
dim[:,-2] *= 0.66

# flatten stacks
m = dim.max(axis=1)
nreg = m.shape[0]
ireg = int(nreg/2)  # get one near the middle of the sequence.
# correct for lateral motion
off = [imreg_dft.translation(m[ireg], m[i])[0] for i in range(0, m.shape[0])]
offt = np.array(off).T

# find boundaries of outer rectangle including all images as registered
minx = np.min(offt[0])
maxx = np.max(offt[0])
miny = np.min(offt[1])
maxy = np.max(offt[1])

# build canvas
canvas = np.zeros(shape=(m.shape[0], m.shape[1]-minx+maxx,
    m.shape[2]-miny+maxy), dtype=m.dtype)

# set initial image (offsets were computed relative to this, so it has no offset)
# canvas[0, -minx:-minx+m.shape[1], -miny:-miny+m.shape[2]] = m[0]
for i in range(0, m.shape[0]):
    ox = offt[0][i] - minx
    oy = offt[1][i] - miny
    canvas[i, ox:(ox+m.shape[1]), oy:(oy+m.shape[2])] = m[i]
#    print 'canvas %d set' % i

# correct for bleaching
levels = np.array([np.median(m[m>scipy.stats.scoreatpercentile(m[i], 95)]) for i in range(m.shape[0])])
norm = canvas / levels[:, np.newaxis, np.newaxis]
w = pg.image()
w.setImage(norm)

# write the resulting compressed z-stacks to a file in the original directory.
ma = MetaArray(norm, info=[{'name': 'Time'}, {'name': 'X'}, {'name': 'Y'}, {}])
ma.write(dh.name() + '/max_stack.ma')
pg.show()
import sys
if sys.flags.interactive == 0:
    app.exec_()