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
INPUT_ASL_PATH = "dat/conn/corr_matrices_asl 2017-06-22 10:22:11.044669"
INPUT_BOLD_PATH = "dat/conn/corr_matrices_bold 2017-06-23 11:55:06.365908"
#INPUT_CLINICAL_PATH = "/external/PNC/data/clinical_data/psychosis_spectrum/pnc_clinical+spectrum+fs.csv"
INPUT_CLINICAL_PATH = "dat/demog/pnc_phenotypes_v02_spectrum_fixed.csv"
INPUT_TD_PATH = "dat/demog/PNC_typically_developing-withmed.csv"
OUTPUT_PLOT_NAME = "auc_weighted_classes_asl.png"

#If using a different dataset (e.g., BOLD instead of ASL), change these
X_BOLD_ARRAY_NAME = "dat/tmp_input/X_array_bold"
X_ASL_ARRAY_NAME = "dat/tmp_input/X_array_asl"
Y_ARRAY_NAME = "dat/tmp_input/y_array"
SUBJID_ARRAY_NAME = "dat/tmp_input/subjids"

#If loading new correlation matrices for the first time, specify "firstload" on the command line
#Future runs with the same matrices (e.g., to tweak learning parameters) will not regenerate X and y arrays (much faster)
if len(sys.argv) > 1 and str(sys.argv[1]) == "firstload":

    fp = open(INPUT_ASL_PATH, "r")
    mat_asl = pickle.load(fp)
    fp.close()
    fp = open(INPUT_BOLD_PATH, "r")
    mat_bold = pickle.load(fp)
    fp.close()
    fp = open(INPUT_CLINICAL_PATH, "r")
    diags = pandas.read_csv(fp)
    fp.close()

    diag_dict = {}
    X_asl = []
    X_bold = []
    y = []
    subjids = []

    for subj in diags.iterrows():
        if subj[1]["spectrum"] in [0, 1]:
            diag_dict[subj[1]["SUBJID"]] = subj[1]["spectrum"]

    for subj_id in mat_asl.keys():
        if int(subj_id) in diag_dict.keys() and subj_id in mat_bold.keys():
            #Append flattened bottom triangle of matrix to X array, diagnosis to y array
            #Important that these arrays are in corresponding order
            conn_asl = mat_asl[subj_id]
            X_asl.append(conn_asl[np.tril_indices(len(conn_asl), k=-1)])
            conn_bold = mat_bold[subj_id]
            X_bold.append(conn_bold[np.tril_indices(len(conn_bold), k=-1)])

            y.append(diag_dict[int(subj_id.strip())])

            subjids.append(subj_id)

    #Convert to numpy arrays
    nX_asl = np.array(X_asl)
    nX_bold = np.array(X_bold)
    ny = np.array(y)
    nsubjids = np.array(subjids)

    #Save X and y to disk
    X_asl_fp = open(X_ASL_ARRAY_NAME, "w")
    X_bold_fp = open(X_BOLD_ARRAY_NAME, "w")
    y_fp = open(Y_ARRAY_NAME, "w")
    sid_fp = open(SUBJID_ARRAY_NAME, "w")
    pickle.dump(nX_asl, X_asl_fp)
    pickle.dump(nX_bold, X_bold_fp)
    pickle.dump(ny, y_fp)
    pickle.dump(nsubjids, sid_fp)
    X_asl_fp.close()
    X_bold_fp.close()
    y_fp.close()
    sid_fp.close()
    print "Saved input files."

#If "firstload" not specified, it's assumed that processed input files already exist
else:
    try:
        X_asl_fp = open(X_ASL_ARRAY_NAME, "r")
        X_bold_fp = open(X_BOLD_ARRAY_NAME, "r")
        y_fp = open(Y_ARRAY_NAME, "r")
        sid_fp = open(SUBJID_ARRAY_NAME, "r")
        nX_asl = pickle.load(X_asl_fp)
        nX_bold = pickle.load(X_bold_fp)
        ny = pickle.load(y_fp)
        subjids = pickle.load(sid_fp)
        nX_asl = np.array(nX_asl)
        nX_bold = np.array(nX_bold)
        ny = np.array(ny)
        nsubjids = np.array(subjids)
    except:
        print "Problem reading input files (did you run the script with the 'firstload' argument at least once?)"

print nX_asl.shape
print nX_bold.shape
print ny.shape
print nsubjids.shape
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
indices_asl = np.array([not np.any(np.isnan(vec)) for vec in nX_asl])
indices_bold = np.array([not np.any(np.isnan(vec)) for vec in nX_bold])
indices = indices_asl & indices_bold

nX_asl = nX_asl[indices]
nX_bold = nX_bold[indices]
ny = ny[indices]
nsubjids = nsubjids[indices]

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
summary_fp = open("classify_summary.csv", "w")
summary_fp.write("subject,diagnosis,asl_pred,asl_correct,bold_pred,bold_correct\n")

#Begin cross-validation loop
for train, test in kf.split(nX_asl, ny):

    #Per-fold feature selection
    """
    best_asl_xform = SelectKBest(mutual_info_classif, k=500).fit(nX_asl[train], ny[train])
    nX_asl_sel = best_asl_xform.transform(nX_asl)
    best_bold_xform = SelectKBest(mutual_info_classif, k=500).fit(nX_bold[train], ny[train])
    nX_bold_sel = best_bold_xform.transform(nX_bold)
    #pca_xform = PCA(n_components=10000).fit(nX[train], ny[train])
    #nX_sel = pca_xform.transform(nX)
    #ica_xform = decomposition.FastICA(n_components=1000, max_iter=200).fit(nX[train], ny[train])
    #nX_sel = ica_xform.transform(nX)
    """
    nX_asl_sel = nX_asl
    print nX_asl_sel.shape
    nX_bold_sel = nX_bold
    print nX_bold_sel.shape

    #Different model types. class_weight = "balanced" is very important!

    #clf = linear_model.LogisticRegression(class_weight="balanced",penalty="l2", C=1).fit(nX_sel[train], ny[train])
    #clf = svm.SVC(kernel='sigmoid', probability=True, C=100, gamma=0.0001, class_weight="balanced").fit(nX_sel[train], ny[train])
    clf_asl = svm.SVC(kernel='linear', probability=True, C=100, class_weight="balanced").fit(nX_asl_sel[train], ny[train])
    clf_bold = svm.SVC(kernel='linear', probability=True, C=100, class_weight="balanced").fit(nX_bold_sel[train], ny[train])
    #clf = RFE(estimator=clf2, n_features_to_select=10000, step=1000).fit(nX_sel[train], ny[train])
    #clf = DecisionTreeClassifier().fit(nX[train], ny[train])
    #clf = MLPClassifier(alpha=1, max_iter=1000).fit(nX_sel[train], ny[train])

    #Calculation of metrics of interest, and AUC curves for plot
    """
    probs = clf.predict_proba(nX_sel[test])
    fpr, tpr, thresholds = roc_curve(ny[test], probs[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)


    #Add current fold ROC curve to plot
    plt.plot(fpr, tpr, lw=lw,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1
    """

    pred_asl = clf_asl.predict(nX_asl_sel[test])
    pred_bold = clf_bold.predict(nX_bold_sel[test])
    true = ny[test]
    testsubjs = nsubjids[test]
    for ind in range(len(pred_asl)):
        #summary_fp.write(testsubjs[ind]+","+true[ind]+","+pred_asl[ind]+","+true[ind]==pred_asl[ind]+","+pred_bold[ind]+","+true[ind]==pred_bold[ind]+"\n")
        summary_fp.write("{},{},{},{},{},{}\n".format(testsubjs[ind], true[ind], pred_asl[ind], true[ind]==pred_asl[ind], pred_bold[ind], true[ind]==pred_bold[ind]))

    print pred_asl
    print pred_bold
    print true
    print testsubjs
    """
    print classification_report(true, pred, target_names=targets)
    print f1_score(true, pred, average="macro")
    print clf.score(nX_sel[test], ny[test])
    print "Confusion matrix:"
    print confusion_matrix(true, pred)
    print "Feature weights:"
    #print np.amax(clf.coef_)
    """

summary_fp.close()
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
