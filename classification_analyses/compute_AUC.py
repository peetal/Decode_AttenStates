"""This script computes AUC for each model"""
##
## Author: Y.Peeta Li; 
## Email: peetal@uoregon.edu
##
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import sem
from pingouin import rm_anova, anova, ttest, pairwise_tukey

# compute fpr ad tpr for each subject for each condition 
# for drawing ROC and computing AUC
class decision_func():

    def __init__(self, clf_conf_3000):
        self.clf_conf_3000 = clf_conf_3000
        
    def compute_fpr_tpr_sub(self, sub_id, condition):
        temp_df = self.clf_conf_3000.loc[(self.clf_conf_3000['condition'] == condition) & (self.clf_conf_3000['subject_id'] == sub_id)]
        fpr, tpr, _ = roc_curve(temp_df['label'].tolist(), temp_df['epoch_conf'].tolist())
        return fpr, tpr

    def compute_fpr_tpr_cond(self, condition):
        fpr_cond, tpr_cond = [],[]
        base_fpr = np.linspace(0,1,20)
        for sub_id in list(range(24)):
            fpr, tpr = self.compute_fpr_tpr_sub(sub_id, condition)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            fpr_cond.append(base_fpr)
            tpr_cond.append(tpr)
        return fpr_cond, tpr_cond

    def compute_auc_sub(self, sub_id, condition):
        temp_df = self.clf_conf_3000.loc[(self.clf_conf_3000['condition'] == condition) & (self.clf_conf_3000['subject_id'] == sub_id)]
        auc = roc_auc_score(temp_df['label'].tolist(), temp_df['epoch_conf'].tolist())
        return auc

    def compute_auc_cond(self, condition):
        auc = []
        for sub_id in list(range(24)):
            auc.append(self.compute_auc_sub(sub_id, condition))
        return auc


if __name__ == "__main__":

    # ------------------------
    # Analyses for figure 2a
    # ------------------------
    #compute AUC for each model, across different size of masks. 
    bg_fcma_clf_conf = pd.read_csv("clf_results/bg_fcma_clf_conf.csv", index_col=0)
    df = []

    for size in ["top_100", "top_1000", "top_3000", "top_5000", "top_7000", "top_10000", "top_15000"]:
        size_clf_conf = bg_fcma_clf_conf.loc[bg_fcma_clf_conf["top_n_mask"]==size].copy(deep = True)
        size_clf_conf["pred_label"] = np.where(size_clf_conf["epoch_conf"] < 0, -1, 1)
        size_clf_conf["label"] = np.tile(np.tile(np.repeat([-1,1],16),24), 3)
        size_clf_conf["correct"] = np.where(size_clf_conf["pred_label"]==size_clf_conf["label"], 1, 0)
        bg_fcma_decision_func = decision_func(size_clf_conf)

        auc_retper_fcma = bg_fcma_decision_func.compute_auc_cond("ret_per_clf")
        auc_scrret_fcma = bg_fcma_decision_func.compute_auc_cond("scramble_ret_clf")
        auc_scrper_fcma = bg_fcma_decision_func.compute_auc_cond("scramble_per_clf")

        size_df = pd.DataFrame({"sub_id": np.tile(range(1,25), 3).tolist(),
                            "mask_size": np.repeat(size, 72).tolist(),
                            "condition": np.repeat(["retper", "scrret", "scrper"], 24).tolist(),
                            "auc": auc_retper_fcma + auc_scrret_fcma + auc_scrper_fcma})
        df.append(size_df)
        df_output = pd.concat(df).reset_index()
        df_output.to_csv('clf_results/fcma_regular_AUC_full.csv', index = False)
        

    # ------------------------
    # Analyses for figure 2b
    # ------------------------
    # background FCMA k = 3000
    bg_fcma_clf_conf_3000 = bg_fcma_clf_conf.loc[bg_fcma_clf_conf["top_n_mask"]=="top_3000"].copy(deep = True)
    bg_fcma_clf_conf_3000["pred_label"] = np.where(bg_fcma_clf_conf_3000["epoch_conf"] < 0, -1, 1)
    bg_fcma_clf_conf_3000["label"] = np.tile(np.tile(np.repeat([-1,1],16),24), 3)
    bg_fcma_clf_conf_3000["correct"] = np.where(bg_fcma_clf_conf_3000["pred_label"]==bg_fcma_clf_conf_3000["label"], 1, 0)
    bg_fcma_decision_func = decision_func(bg_fcma_clf_conf_3000)

    auc_retper_fcma = bg_fcma_decision_func.compute_auc_cond("ret_per_clf")
    auc_scrret_fcma = bg_fcma_decision_func.compute_auc_cond("scramble_ret_clf")
    auc_scrper_fcma = bg_fcma_decision_func.compute_auc_cond("scramble_per_clf")

    # stimulus-evoked MVPA k = 3000
    evoked_mvpa_clf_conf = pd.read_csv("clf_results/evoked_mvpa_clf_conf.csv", index_col=0)
    evoked_mvpa_clf_conf_3000 = evoked_mvpa_clf_conf.loc[evoked_mvpa_clf_conf["top_n_mask"]=="top_3000"].copy(deep = True)
    evoked_mvpa_clf_conf_3000["pred_label"] = np.where(evoked_mvpa_clf_conf_3000["epoch_conf"] < 0, -1, 1)
    evoked_mvpa_clf_conf_3000["label"] = np.tile(np.tile(np.repeat([-1,1],16),24), 3)
    evoked_mvpa_clf_conf_3000["correct"] = np.where(evoked_mvpa_clf_conf_3000["pred_label"]==evoked_mvpa_clf_conf_3000["label"], 1, 0)
    evoked_mvpa_decision_func = decision_func(evoked_mvpa_clf_conf_3000)

    # compute AUC
    auc_retper_mvpa = evoked_mvpa_decision_func.compute_auc_cond("retper")
    auc_scrret_mvpa = evoked_mvpa_decision_func.compute_auc_cond("scrret")
    auc_scrper_mvpa = evoked_mvpa_decision_func.compute_auc_cond("scrper")

    # hybrid classifier by averaging decision function outputs from MVPA and FCMA classifiers:
    bg_fcma_clf_conf_3000 = bg_fcma_clf_conf_3000.rename(columns = {'epoch_conf':'bg_epoch_conf'})
    bg_fcma_clf_conf_3000['evoked_epoch_conf'] = evoked_mvpa_clf_conf_3000['epoch_conf'].tolist()
    bg_fcma_clf_conf_3000['epoch_conf'] = (bg_fcma_clf_conf_3000['bg_epoch_conf'] + bg_fcma_clf_conf_3000['evoked_epoch_conf'])/2
    clf_conf_3000 = bg_fcma_clf_conf_3000.copy(deep = True)
    clf_conf_3000["pred_label"] = np.where(clf_conf_3000["epoch_conf"] < 0, -1, 1)
    clf_conf_3000["label"] = np.tile(np.tile(np.repeat([-1,1],16),24), 3)
    clf_conf_3000["correct"] = np.where(clf_conf_3000["pred_label"]==clf_conf_3000["label"], 1, 0)
    hybrid_mvpa_decision_func = decision_func(clf_conf_3000)

    # compute AUC
    auc_retper_hybrid = hybrid_mvpa_decision_func.compute_auc_cond("ret_per_clf")
    auc_scrret_hybrid = hybrid_mvpa_decision_func.compute_auc_cond("scramble_ret_clf")
    auc_scrper_hybrid = hybrid_mvpa_decision_func.compute_auc_cond("scramble_per_clf")

    # output and stats: 
    df = pd.DataFrame({"sub_id": np.tile(range(1,25), 9),
                   "measure": np.repeat(["bg_FCMA","evoked_MVPA","hybrid"], 72).tolist(), 
                   "condition": np.tile(np.repeat(["retper", "scrret", "scrper"], 24), 3).tolist(),
                   "auc": auc_retper_fcma + auc_scrret_fcma + auc_scrper_fcma + auc_retper_mvpa + auc_scrret_mvpa + auc_scrper_mvpa + auc_retper_hybrid + auc_scrret_hybrid + auc_scrper_hybrid})
    df.to_csv("clf_results/fcma_mvpa_hybrid_3000AUC2.csv", index = False)
    print("-----ANOVA-----")
    print(anova(df, dv="auc", between=["condition","measure"]))
    print("pairwise tuckey")
    print(pairwise_tukey(df, dv="auc", between="measure", effsize= "cohen"))

    df_measure = df.groupby(['measure',"sub_id"]).mean().reset_index()
    print(ttest(df_measure.loc[df_measure['measure'] == "bg_FCMA", "auc"], df_measure.loc[df_measure['measure'] == "hybrid", "auc"]))
    print(ttest(df_measure.loc[df_measure['measure'] == "evoked_MVPA", "auc"], df_measure.loc[df_measure['measure'] == "hybrid", "auc"]))

