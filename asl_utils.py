import numpy as np

#Given an index from the feature vector (a flattened bottom triangle of the connectivity matrix), return the row and column from the original matrix (i.e., the two relevant ROIs)
#E.g., for ASL, n = 214
#Returns: (row, column)
def get_rois_from_vector(index, n):
	row = np.tril_indices(n, -1)[0][index] + 1
	column = np.tril_indices(n, -1)[1][index] + 1
	return (row, column)

#Given array of feature vectors, return indices of top n connections (averaged across subjects)
def top_conns(arr_vec, n):
	return np.argsort(np.mean(arr_vec, axis=0))[:-n]

def good_roi_to_abs_roi(good_rois, roi_num):
	return list(good_rois)[roi_num]