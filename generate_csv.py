#import bct
import pickle
import numpy as np
import pandas
import csv

INPUT_MATRICES_PATH = "dat/conn/corr_matrices_asl 2017-06-22 10:22:11.044669"
INPUT_CLINICAL_PATH = "dat/demog/pnc_phenotypes_v02_spectrum_fixed.csv"

fp = open(INPUT_MATRICES_PATH, "r")
allconn = pickle.load(fp)
fp.close()

fp = open(INPUT_CLINICAL_PATH, "r")
diags = pandas.read_csv(fp)
fp.close()

diag_dict = {}
sex_dict = {}
age_dict = {}
Xm = []
ym = []
Xf = []
yf = []

for subj in diags.iterrows():
    if subj[1]["spectrum"] in [0, 1]:
        diag_dict[subj[1]["SUBJID"]] = subj[1]["spectrum"]
    if subj[1]["Sex"] in ["M", "F"]:
        sex_dict[subj[1]["SUBJID"]] = subj[1]["Sex"]
    if str(subj[1]["age"]).isdigit() and int(subj[1]["age"]) in range(8, 25):
        age_dict[subj[1]["SUBJID"]] = int(subj[1]["age"])
"""
cfile1 = open("strengths_M_hc.csv","w")
cwriter1 = csv.writer(cfile1, delimiter=',')

cfile2 = open("strengths_F_hc.csv","w")
cwriter2 = csv.writer(cfile2, delimiter=',')

cfile3 = open("strengths_M_ps.csv","w")
cwriter3 = csv.writer(cfile3, delimiter=',')

cfile4 = open("strengths_F_ps.csv","w")
cwriter4 = csv.writer(cfile4, delimiter=',')
"""
for subj_id, conn in allconn.items():
    sid = int(subj_id)
    vec = conn[np.tril_indices(len(conn), k=-1)]
    if sid in diag_dict.keys() and sid in sex_dict.keys():
    	"""
        if sex_dict[sid] == "M" and diag_dict[sid] == 0:
            cwriter1.writerow(vec)
        if sex_dict[sid] == "F" and diag_dict[sid] == 0:
            cwriter2.writerow(vec)
        if sex_dict[sid] == "M" and diag_dict[sid] == 1:
            cwriter3.writerow(vec)
        if sex_dict[sid] == "F" and diag_dict[sid] == 1:
            cwriter4.writerow(vec)
        """
        if sex_dict[sid] == "M" and diag_dict[sid] == 0:
            Xm.append(vec)
            ym.append(0)
        if sex_dict[sid] == "F" and diag_dict[sid] == 0:
            Xf.append(vec)
            yf.append(0)
        if sex_dict[sid] == "M" and diag_dict[sid] == 1:
            Xm.append(vec)
            ym.append(1)
        if sex_dict[sid] == "F" and diag_dict[sid] == 1:
            Xf.append(vec)
            yf.append(1)
"""
cfile1.close()
cfile2.close()
cfile3.close()
cfile4.close()
"""
Xmfile = open("X_array_asl_male","w")
ymfile = open("y_array_asl_male","w")
Xffile = open("X_array_asl_female","w")
yffile = open("y_array_asl_female","w")
pickle.dump(Xm,Xmfile)
pickle.dump(ym,ymfile)
pickle.dump(Xf,Xffile)
pickle.dump(yf,yffile)
Xmfile.close()
ymfile.close()
Xffile.close()
yffile.close()