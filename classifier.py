#!/usr/bin/env python

from scipy import interp
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import pickle
import pandas
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn import linear_model
from sklearn import decomposition
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


#Specify input correlation matrices file and demographics table
INPUT_MATRICES_PATH = "dat/conn/bold_conn_14older"
#INPUT_CLINICAL_PATH = "/external/PNC/data/clinical_data/psychosis_spectrum/pnc_clinical+spectrum+fs.csv"
INPUT_CLINICAL_PATH = "dat/demog/pnc_phenotypes_v02_spectrum_fixed.csv"
OUTPUT_PLOT_NAME = "auc_weighted_classes_asl_female.png"

#If using a different dataset (e.g., BOLD instead of ASL), change these
X_ARRAY_NAME = "dat/tmp_input/X_array_asl_female"
Y_ARRAY_NAME = "dat/tmp_input/y_array_asl_female"

#If loading new correlation matrices for the first time, specify "firstload" on the command line
#Future runs with the same matrices (e.g., to tweak learning parameters) will not regenerate X and y arrays (much faster)
if len(sys.argv) > 1 and str(sys.argv[1]) == "firstload":

    fp = open(INPUT_MATRICES_PATH, "r")
    mat = pickle.load(fp)
    fp = open(INPUT_CLINICAL_PATH, "r")
    diags = pandas.read_csv(fp)
    diag_dict = {}
    X = []
    y = []

    for subj in diags.iterrows():
        if subj[1]["spectrum"] in [0, 1]:
            diag_dict[subj[1]["SUBJID"]] = subj[1]["spectrum"]

    for subj_id, connmat in mat.items():
        if int(subj_id) in diag_dict.keys():
            #Append flattened bottom triangle of matrix to X array, diagnosis to y array
            #Important that these arrays are in corresponding order
            X.append(connmat[np.tril_indices(len(connmat), k=-1)])
            y.append(diag_dict[int(subj_id.strip())])

    #Convert to numpy arrays
    nX = np.array(X)
    ny = np.array(y)

    #Save X and y to disk
    X_fp = open(X_ARRAY_NAME, "w")
    y_fp = open(Y_ARRAY_NAME, "w")
    pickle.dump(nX, X_fp)
    pickle.dump(ny, y_fp)
    X_fp.close()
    y_fp.close()
    print "Saved input files."

#If "firstload" not specified, it's assumed that processed input files already exist
else:
    try:
        X_fp = open(X_ARRAY_NAME, "r")
        y_fp = open(Y_ARRAY_NAME, "r")
        nX = pickle.load(X_fp)
        ny = pickle.load(y_fp)
        nX = np.array(nX)
        ny = np.array(ny)
    except:
        print "Problem reading input files (did you run the script with the 'firstload' argument at least once?)"


# Random labels for test purposes
#ny = np.random.randint(2, size=len(ny))
#ny = np.random.choice([0,0,0,0,0,0,0,1,1,1], size=len(ny), replace=True)

#Remove low variance features
#print nX.shape
#sel = VarianceThreshold(threshold=0.075)
#nX = sel.fit_transform(nX)
#print nX.shape

#Automated feature selection
#print nX.shape
#nX = SelectKBest(f_classif, k=10000).fit_transform(nX, ny)
#print nX.shape

#PCA
#print nX.shape
#nX = PCA(n_components=0.99, svd_solver="full").fit_transform(nX)
#print nX.shape

#Filter out NaN values
indices = np.array([not np.any(np.isnan(vec)) for vec in nX])
nX = nX[indices]
ny = ny[indices]

#Begin k-fold cross validation
kf = StratifiedKFold(n_splits=10)

#Optional - parameter evaluation using grid search
"""
param_grid = [
  {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
{'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['sigmoid']},
{'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'degree': [2,3], 'kernel': ['poly']}
 ]

grid = GridSearchCV(svm.SVC(class_weight="balanced"), param_grid=param_grid, cv=kf, scoring="f1_macro", verbose=10)
grid.fit(nX,ny)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
"""

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

#colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2
i = 0

targets = ["HC", "PS"]

#Begin cross-validation loop
for train, test in kf.split(nX, ny):

    #Per-fold feature selection
    #best_xform = SelectKBest(mutual_info_classif, k=500).fit(nX[train], ny[train])
    #nX_sel = best_xform.transform(nX)
    #pca_xform = PCA(n_components=10000).fit(nX[train], ny[train])
    #nX_sel = pca_xform.transform(nX)
    #ica_xform = decomposition.FastICA(n_components=1000, max_iter=200).fit(nX[train], ny[train])
    #nX_sel = ica_xform.transform(nX)

    nX_sel = nX
    print nX_sel.shape

    #Different model types. class_weight = "balanced" is very important!

    #clf = linear_model.LogisticRegression(class_weight="balanced",penalty="l2", C=1).fit(nX_sel[train], ny[train])
    #clf = svm.SVC(kernel='sigmoid', probability=True, C=100, gamma=0.0001, class_weight="balanced").fit(nX_sel[train], ny[train])
    clf = svm.SVC(kernel='linear', probability=True, C=100, class_weight="balanced").fit(nX_sel[train], ny[train])
    #clf = RFE(estimator=clf2, n_features_to_select=10000, step=1000).fit(nX_sel[train], ny[train])
    #clf = DecisionTreeClassifier().fit(nX[train], ny[train])
    #clf = MLPClassifier(alpha=1, max_iter=1000).fit(nX_sel[train], ny[train])

    #Calculation of metrics of interest, and AUC curves for plot
    probs = clf.predict_proba(nX_sel[test])
    fpr, tpr, thresholds = roc_curve(ny[test], probs[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

    #Add current fold ROC curve to plot
    plt.plot(fpr, tpr, lw=lw,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1

    pred = clf.predict(nX_sel[test])
    true = ny[test]
    print pred
    print true
    print classification_report(true, pred, target_names=targets)
    print f1_score(true, pred, average="macro")
    print clf.score(nX_sel[test], ny[test])
    print "Confusion matrix:"
    print confusion_matrix(true, pred)
    print "Feature weights:"
    #print np.amax(clf.coef_)

#Other plot parameters and mean ROC curve
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kf.get_n_splits(nX_sel, ny)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PS (+) versus non-PS (-) SVM classification using BOLD connectivity')
plt.legend(loc="lower right")
plt.savefig("dat/plot/" + OUTPUT_PLOT_NAME)
plt.show()
