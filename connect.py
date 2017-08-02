#!/usr/bin/env python

import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import datetime as dt

ATLAS_PATH = 'dat/atlas/shen_resamp_boldcomparison.nii.gz'
SUBJECTS_PATH = "dat/lists/bold_list_filt.txt"
OUTPUT_MATRICES_PATH = "dat/corr/corr_matrices_bold_power " + str(dt.datetime.now())
INPUT_ROIS_PATH = "dat/rois/good_rois_bold_power 2017-06-23 12:55:48.986250"
#INPUT_REL_PATH = '/pipeline/subtract_postsmooth.nii.gz'
INPUT_REL_PATH = '/REST/SESS01/func_volsmooth.test.01.nii.gz'

#Load atlas file
atlas = nib.load(ATLAS_PATH)
atlas = atlas.get_data()

#Load list of subject paths, and Python list of precomputed "good" ROIs (inside the brain for all subjects)
paths_file = open(SUBJECTS_PATH, "r")
rois_file = open(INPUT_ROIS_PATH, "r")
good_rois = pickle.load(rois_file)
good_rois = list(good_rois)
paths = paths_file.readlines()
paths_file.close()
rois_file.close()

#Prepare output matrix file for writing
matrices_file = open(OUTPUT_MATRICES_PATH, "wb")
corr_matrices = {}

#For each subject path
for line in paths:
    #Extract SUBJID from path (note: paths must be given with SUBJID as last directory, with no trailing forward slash)
    subj_id = line.split("/")[-1].strip()
    print "Working on {0}...".format(subj_id)

    try:
        #Try loading processed functional data; should be in "pipeline" folder
        func = nib.load(line.strip() + INPUT_REL_PATH)

        #Flatten atlas
        dims = func.shape
        tmp_atlas = np.reshape(atlas, (dims[0] * dims[1] * dims[2], 1))

        #Flatten functional data
        func = func.get_data()
        func = np.reshape(func, (dims[0] * dims[1] * dims[2], dims[3]))

        #timeseries = np.zeros( ( len(np.unique(tmp_atlas))-1, dims[3] ) )
        timeseries = np.zeros((len(good_rois), dims[3]))

        # for i, roi in enumerate(np.unique(tmp_atlas)[1:]):
        for i, roi in enumerate(good_rois):
            idx = np.where(tmp_atlas == roi)[0]
            ts = np.mean(func[idx, :], axis=0)
            timeseries[i, :] = ts

            # Remove any ROIs that have a 0 average intensity (causes NaNs after correlation)
            #f_timeseries = [x for x in timeseries if any(x)]

            #Calculate connectivity matrix, and add to dictionary (with SUBJID as key)
            connectivity = np.corrcoef(timeseries)
            corr_matrices[subj_id] = connectivity

    except:
        print "A problem occurred with this subject. Skipping..."

#Save matrices to disk
pickle.dump(corr_matrices, matrices_file)
matrices_file.close()
