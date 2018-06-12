# June 11, 2018

#import bct
import pickle
import numpy as np
import pandas
import csv
import math

INPUT_ASL_PATH = "dat/conn/corr_matrices_asl 2017-06-22 10:22:11.044669"
INPUT_BOLD_PATH = "dat/conn/corr_matrices_bold 2017-06-23 11:55:06.365908"
INPUT_CLINICAL_PATH = "dat/demog/pnc_phenotypes_v02_spectrum_fixed.csv"
INPUT_TDLIST_PATH = "dat/demog/PNC_typically_developing-withmed.csv"

fp = open(INPUT_ASL_PATH, "r")
allaslconn = pickle.load(fp)
fp.close()

fp = open(INPUT_BOLD_PATH, "r")
allboldconn = pickle.load(fp)
fp.close()

fp = open(INPUT_CLINICAL_PATH, "r")
diags = pandas.read_csv(fp)
fp.close()

fp = open(INPUT_TDLIST_PATH, "r")
tdlist = pandas.read_csv(fp)
fp.close()

class Subject:
    def __init__(self, sid, aslconn, boldconn, sex, ps, td, age):
        self.sid = sid
        self.aslconn = aslconn
        self.boldconn = boldconn
        self.sex = sex
        self.ps = ps
        self.td = td
        self.age = age

diag_dict = {}
sex_dict = {}
age_dict = {}
"""
Xm = []
ym = []
Xf = []
yf = []
"""
for subj in diags.iterrows():
    if subj[1]["spectrum"] in [0, 1]:
        diag_dict[subj[1]["SUBJID"]] = subj[1]["spectrum"]
    if subj[1]["Sex"] in ["M", "F"]:
        sex_dict[subj[1]["SUBJID"]] = subj[1]["Sex"]
    if not math.isnan(subj[1]["age"]):
        age_dict[subj[1]["SUBJID"]] = int(subj[1]["age"])

bothdata_sids = set(allaslconn.keys()).intersection(set(allboldconn.keys()))
tds = list(tdlist["ID"])
tds = map(str, tds)
allsubj_outdict = {}

for subj_id in bothdata_sids:
    sid = int(subj_id)
    td = 0
    if sid in diag_dict.keys() and sid in sex_dict.keys() and sid in age_dict.keys():
        aslconn = allaslconn[subj_id]
        aslvec = aslconn[np.tril_indices(len(aslconn), k=-1)]
        boldconn = allboldconn[subj_id]
        boldvec = boldconn[np.tril_indices(len(boldconn), k=-1)]
        if subj_id in tds:
            td = 1
        subj_instance = Subject(subj_id,aslvec,boldvec,sex_dict[sid],diag_dict[sid],td,age_dict[sid])
        allsubj_outdict[subj_id] = subj_instance

allsubjfile = open("processed_subjects_dictionary_june11_2018","w")
pickle.dump(allsubj_outdict,allsubjfile)
allsubjfile.close()
