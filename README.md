Dawson Overton, 2017

Notes about PNC data:

* 931 subjects with ASL in release 1, 552 subjects with ASL in release 2 (555 and 3 duplicates), 1483 in total. These numbers have been confirmed by looking at raw zips directly from Philadelphia group.
* 881 with BOLD in release 1, 514 with BOLD in release 2, 1395 in total. These numbers have been confirmed by looking at raw zips directly from Philadelphia group.
* 1202 with *usable* BOLD data. 193 failed at the TR scrubbing stage (likely too much head motion).

----

The most important scripts in this directory are (run in this order if starting from scratch):
* preprocess_asl.sh
* subtr_tag_control.py
* find_good_rois.py
* connect.py
* classifier.py

----

`preprocess_asl.sh` will take a subject (SUBJ_ID) directory and fully preprocess the corresponding ASL data using a known location for the T1 and ASL NIfTI files. It performs deskulling, transformation of the functional data to MNI space (after registration to the T1), spatial smoothing and detrending. The fully processed file is named ASL_MNI_detrend_smoothed.nii.gz and is placed in a subdirectory called "pipeline", which itself is placed in the SUBJ_ID directory.

----

`subtr_tag_control.py` takes a list of absolute paths to subject directories, and subtracts control from tag volumes for all subjects in the list. This requires a fully processed NIfTI (ASL_MNI_detrend_smoothed.nii.gz) for each subject. It subtracts odd volumes from even volumes (for the raw PNC data, control volumes are odd and tag volumes are even). It will output a perfusion signal volume in the "pipeline" folder for each subject.

----

Note on atlas: before running the subsequent scripts, ensure you have an atlas that has been resampled to your functional data (e.g., using AFNI). Example:
`3dresample -prefix shen_resamp.nii.gz -master SPN01_CMH_0001_01_01_RST_07_Ax-RestingState_MNI-nonlin.nii.gz -inset shen_2mm_268_parcellation.nii.gz`

----

`find_good_rois.py` requires the following variables to be defined:
* ATLAS_PATH: Relative path to the atlas to be used for the analysis (resample it first).
* SUBJECTS_PATH: Relative path to a text file which has absolute paths to the subject directories to be analyzed, one path per line.
* OUTPUT_ROIS_PATH: Relative path to write the output "good ROIs" file.
* INPUT_REL_PATH: Relative path to the functional data NIfTI file from the subject directory.

This script calculates the mean time series signal in each ROI of the given atlas, for each subject. If it is 0 for a subject, this likely means that the ROI lies entirely outside of the brain for that subject (and the 0 value causes problems for downstream analyses). Any ROI that has this problem for at least one subject is added to a set, and this set of ROIs are ignored in future analyses.

----

`connect.py` requires the following variables to be defined:
* ATLAS_PATH: Relative path to the atlas to be used for the analysis (resample it first).
* SUBJECTS_PATH: Relative path to a text file which has absolute paths to the subject directories to be analyzed, one path per line.
* OUTPUT_MATRICES_PATH: Relative path to write the output matrices file.
* INPUT_ROIS_PATH: Relative path to the "good ROIs" file.
* INPUT_REL_PATH: Relative path to the functional data NIfTI file from the subject directory.

This script uses the "good ROIs" set from the previous step and calculates a correlation matrix for each subject (taking into account only the ROIs in this set). Each of these correlation matrices is added to a Python dictionary, with the key being the subject ID. This dictionary is saved to disk and used in classification.

----

`classifier.py` requires the following:
* INPUT_MATRICES_PATH: Relative path to the matrices file.
* INPUT_CLINICAL_PATH: Relative path to the demographic information CSV (used mainly for diagnosis).
* X_ARRAY_NAME: Relative path to the "temp" (ready for classification) connectivity data (used when "firstload" not specified).
* Y_ARRAY_NAME: Relative path to the "temp" class labels (0 for HC, 1 for PS) -- order must be the same as the X array.

The input data needs to be in the format of a dictionary of connectivity matrices, where the key for each matrix is the PNC subject ID. The script will process this dictionary by flattening and taking the bottom triangle of each matrix, and looking up the psychosis diagnosis value (0 for non-PS, 1 for PS) for the corresponding subject ID. 

`classifier.py` contains a "pca_reduce" function, which is optional. This function can be used on the input data to either specify a desired number of dimensions (letting % variance explained vary), or vice versa. This is especially useful for reducing training time for the model.

Several options can be specified to change the model type and parameters, number of cross validation folds, and feature selection (if any). The script will output a variety of performance metrics and save a figure to disk which includes an AUC curve for each fold and the average AUC curve.

----

dat directory:
- Contains connectivity matrices, subject lists, X and y "temporary" matrices, "good roi" lists, atlases, MNI152 brain, demographic tables.