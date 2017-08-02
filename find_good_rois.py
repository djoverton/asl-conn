#!/usr/bin/env python

import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import datetime as dt


ATLAS_PATH = 'dat/atlas/power_atlas_bold.nii'
SUBJECTS_PATH = "dat/lists/bold_list_filt.txt"
OUTPUT_ROIS_PATH = "dat/rois/good_rois_bold_power " + str(dt.datetime.now())
#INPUT_REL_PATH = '/pipeline/subtract_postsmooth.nii.gz'
INPUT_REL_PATH = '/REST/SESS01/func_volsmooth.test.01.nii.gz'


atlas = nib.load(ATLAS_PATH)
atlas = atlas.get_data()

paths_file = open(SUBJECTS_PATH, "r")
good_rois_file = open(OUTPUT_ROIS_PATH, "w")
corr_matrices = {}
paths = paths_file.readlines()
bad_rois = set()

for line in paths:
    subj_id = line.split("/")[-1].strip()
    print "Working on {0}...".format(subj_id)

    try:
        func = nib.load(line.strip() + INPUT_REL_PATH)
        dims = func.shape
        tmp_atlas = np.reshape(atlas, (dims[0] * dims[1] * dims[2], 1))
        func = func.get_data()
        func = np.reshape(func, (dims[0] * dims[1] * dims[2], dims[3]))

        timeseries = np.zeros((len(np.unique(tmp_atlas)) - 1, dims[3]))

        for i, roi in enumerate(np.unique(tmp_atlas)[1:]):
            idx = np.where(tmp_atlas == roi)[0]
            ts = np.mean(func[idx, :], axis=0)
            if not any(ts):
                bad_rois.add(roi)
                print bad_rois
    except:
        print "Bad subject"

all_rois = set(range(269)[1:])
good_rois = all_rois.difference(bad_rois)
pickle.dump(good_rois, good_rois_file)
