"""
Evaluation metrics

"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    ml_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ap_list = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
            ap_list.append(ap)
        else:
            ap_list.append(0.0)
    ml_map = np.mean(ap_list)
    auc_list = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                auc_list.append(auc)
            except:
                auc_list.append(0.5)
        else:
            auc_list.append(0.5)
    ml_auc = np.mean(auc_list)
    ml_score = (ml_map + ml_auc) / 2
    bin_auc = roc_auc_score(y_true[:, 1], y_pred_proba[:, 1]) if len(np.unique(y_true[:, 1])) > 1 else 0.5
    bin_f1 = f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0)
    model_score = (ml_score + bin_auc) / 2
    return {
        'ML_F1': ml_f1,
        'ML_mAP': ml_map,
        'ML_AUC': ml_auc,
        'ML_Score': ml_score,
        'Bin_AUC': bin_auc,
        'Bin_F1': bin_f1,
        'Model_Score': model_score
    }


def optimize_thresholds(y_true, y_prob):
    thresholds = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        best_f1 = 0
        best_t = 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(y_true[:, i], (y_prob[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
    return thresholds
