#!/bin/bash

#Usage: Provide the absolute path to the subject's directory as the first commandline argument (SUBJ_DIR).
#T1: Relative path from SUBJ_DIR to T1 NIfTI file.
#ASL: Relative path from SUBJ_DIR to ASL NIfTI file.
#PROJ_DIR: Absolute path to directory that contains reference volumes.

#Note that many optional steps are commented out; remove these comments to include them.

#Based on Loggia et al. 2013 - "Default mode network connectivity encodes clinical pain: an arterial spin labeling study"

SUBJ_DIR=$1
T1=$SUBJ_DIR/T1/*.nii.gz
ASL=$SUBJ_DIR/FMRI_ASL_pcasl_moco/Dicoms/*.nii.gz
PROJ_DIR=/projects/doverton/ASL_PNC

#**Deskulling**

#Generate mask for ASL data
bet \
    ${ASL} \
    ${SUBJ_DIR}/pipeline/ASL_bet.nii.gz \
    -R -n -m \
    #-f 0.3 \
    #-g 0

#Dilate mask - optional
#fslmaths \
#    ${PROJ_DIR}/${SUBJ_ID}/ASL_bet_mask.nii.gz \
#    -dilD \
#    ${PROJ_DIR}/${SUBJ_ID}/ASL_bet_mask_dil.nii.gz

#Apply mask
3dcalc \
    -prefix ${SUBJ_DIR}/pipeline/ASL_masked.nii.gz \
    -a ${ASL} \
    -b ${SUBJ_DIR}/pipeline/ASL_bet_mask.nii.gz \
    -expr 'a*b'

#Generate mask for T1
bet \
    ${T1} \
    ${SUBJ_DIR}/pipeline/T1_bet.nii.gz \
    -R -n -m \
    #-f 0.3 \
    #-g 0

#Dilate mask - optional
#fslmaths \
#    ${PROJ_DIR}/${SUBJ_ID}/T1_bet_mask.nii.gz \
#    -dilD \
#    ${PROJ_DIR}/${SUBJ_ID}/T1_bet_mask_dil.nii.gz

#Apply mask
3dcalc \
    -prefix ${SUBJ_DIR}/pipeline/T1_masked.nii.gz \
    -a ${T1} \
    -b ${SUBJ_DIR}/pipeline/T1_bet_mask.nii.gz \
    -expr 'a*b'

#Calculate registration of ASL to T1
flirt \
  -in ${SUBJ_DIR}/pipeline/ASL_masked.nii.gz \
  -ref ${SUBJ_DIR}/pipeline/T1_masked.nii.gz \
  -out ${SUBJ_DIR}/pipeline/reg_ASL_to_T1.nii.gz \
  -omat ${SUBJ_DIR}/pipeline/mat_ASL_to_T1.mat

  #-dof ${reg_dof} \
  #-cost ${cost} \
  #-searchcost ${cost} \
  #-searchrx -180 180 -searchry -180 180 -searchrz -180 180

#Calculate registration of T1 to MNI
flirt \
  -in ${SUBJ_DIR}/pipeline/T1_masked.nii.gz \
  -ref ${PROJ_DIR}/MNI152_T1_2mm_brain.nii.gz \
  -out ${SUBJ_DIR}/pipeline/reg_T1_to_MNI.nii.gz \
  -omat ${SUBJ_DIR}/pipeline/mat_T1_to_MNI.mat \
  -dof 12 \
  -searchcost corratio \
  -cost corratio

#Concatenate transformations
convert_xfm \
  -omat ${SUBJ_DIR}/pipeline/mat_ASL_to_MNI.mat \
  -concat ${SUBJ_DIR}/pipeline/mat_T1_to_MNI.mat \
    ${SUBJ_DIR}/pipeline/mat_ASL_to_T1.mat

#Register native space ASL data with MNI
flirt \
  -in ${SUBJ_DIR}/pipeline/ASL_masked.nii.gz \
  -ref ${PROJ_DIR}/MNI152_T1_2mm_brain.nii.gz \
  -applyxfm -init ${SUBJ_DIR}/pipeline/mat_ASL_to_MNI.mat \
  -out ${SUBJ_DIR}/pipeline/ASL_MNI-lin.nii.gz \
  -interp sinc \
  -sincwidth 7 \
  -sincwindow blackman

#Detrend data ('polort' is order of polynomial) -- essentially high-pass filtering
3dDetrend \
  -prefix ${SUBJ_DIR}/pipeline/ASL_MNI_detrend_tmp.nii.gz \
  -polort 4 ${SUBJ_DIR}/pipeline/ASL_MNI-lin.nii.gz

#Add mean back into detrended data
3dcalc \
  -prefix ${SUBJ_DIR}/pipeline/ASL_MNI_detrend.nii.gz \
  -a ${SUBJ_DIR}/pipeline/ASL_MNI_detrend_tmp.nii.gz \
  -b ${SUBJ_DIR}/pipeline/ASL_MNI_mean_tmp.nii.gz \
  -expr 'a+b'

#rm ${SUBJ_DIR}/pipeline/ASL_MNI_detrend_tmp.nii.gz
#rm ${SUBJ_DIR}/pipeline/ASL_MNI_mean_tmp.nii.gz

#Spatially smooth (full width at half maximum = 5 mm)
#NTS: blurmaster and mask are optional -- ensure masking is appropriate
3dBlurToFWHM \
    -quiet \
    -prefix ${SUBJ_DIR}/pipeline/ASL_MNI_detrend_smoothed.nii.gz \
    -FWHM 5 \
    -input ${SUBJ_DIR}/pipeline/ASL_MNI_detrend.nii.gz \
    -mask ${PROJ_DIR}/MNI152lin_T1_2mm_brain_mask.nii.gz


