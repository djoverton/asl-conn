import numpy as np
import nibabel as nib

#Define list of subjects to process, relative location of processed data, and
#relative output path of subtracted data
SUBJECTS_PATH = "/projects/doverton/ASL_PNC/dat/lists/subject_list.txt"
INPUT_REL_PATH = "/pipeline/ASL_MNI_detrend_smoothed.nii.gz"
OUTPUT_REL_PATH = "/pipeline/subtract_postsmooth.nii.gz"

dir_file = open(SUBJECTS_PATH, "r")
dirs = dir_file.readlines()

#For each subject in subject list file
for dir in dirs:
    print "Now working on {0}".format(dir.strip())
    try:
        func = nib.load(dir.strip() + INPUT_REL_PATH)
        dims = func.shape
        func = func.get_data()

        #Create new matrix with half as many timepoints
        subfunc = np.zeros((dims[0], dims[1], dims[2], dims[3] / 2))

        #For each timepoint
        for row in range(dims[3]):
            if row % 2 == 0:
                # Even timepoints (including first) are tag volumes
                subfunc[:, :, :, row / 2] += func[:, :, :, row]
            else:
                # Odd timepoints are control volumes; subtract from tag
                subfunc[:, :, :, row / 2] -= func[:, :, :, row]

        #Transform to NIfTI image and save
        nifti_subfunc = nib.Nifti1Image(subfunc, affine=np.eye(4))
        nib.save(nifti_subfunc, dir.strip() + OUTPUT_REL_PATH)
        
    except KeyboardInterrupt:
        break
    except:
        print "Error processing {0}: does the file exist?".format(dir.strip())
