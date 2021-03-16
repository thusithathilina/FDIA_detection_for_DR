import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support as prf


def print_results(method, trueAttack, falseAttack, falseNonAttack, trueNoneAttack, end = "\n", output=True):
    accuracy = (trueAttack + trueNoneAttack) / (trueAttack + trueNoneAttack + falseAttack + falseNonAttack)
    precision = trueAttack / (trueAttack + falseAttack) if trueAttack + falseAttack != 0 else 0
    recall = trueAttack / (trueAttack + falseNonAttack) if trueAttack + falseNonAttack != 0 else 0
    f1score = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
    fpr = falseAttack / (trueNoneAttack + falseAttack)
    if precision == 0 or recall == 0 or f1score == 0:
        msg = f"Not proper results,{method},{trueAttack},{falseAttack},{falseNonAttack},{trueNoneAttack}"
    else:
        msg = f"{method},{accuracy},{precision},{recall},{f1score},{fpr}"
    if output:
        print(msg, end=end)
    else:
        return msg


def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
    if precision == 0 and recall == 0:
        f01_score = 0
    else:
        f01_score = fbeta_score(y_true, y_pred, average='binary', beta=0.1)
    return accuracy, precision, recall, f_score, f01_score


def get_threshold(score):
    return np.nanmean(score) + 2 * np.nanstd(score)


def binarize(score, threshold=None):
    threshold = threshold if threshold is not None else get_threshold(score)
    score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
    return np.where(score >= threshold, 1, 0)

def invert_binarize(score, threshold=None):
    threshold = threshold if threshold is not None else get_threshold(score)
    score = np.where(np.isnan(score), np.nanmin(score) - sys.float_info.epsilon, score)
    return np.where(score >= threshold, 0, 1)


def get_metrics_by_thresholds(y_test: list, score: list, thresholds: list):
    for threshold in thresholds:
        anomaly = binarize(score, threshold=threshold)
        metrics = get_accuracy_precision_recall_fscore(y_test, anomaly)
        invert_anomaly = invert_binarize(score, threshold=threshold)
        invert_metrics = get_accuracy_precision_recall_fscore(y_test, invert_anomaly)
        if metrics[3] < invert_metrics [3]:
            metrics = invert_metrics
        yield (anomaly.sum(), *metrics)


def get_optimal_threshold(y_test, score, steps=100, return_metrics=False):
    maximum = np.nanmax(score)
    minimum = np.nanmin(score)

    threshold = np.linspace(minimum, maximum, steps)
    metrics = list(get_metrics_by_thresholds(y_test, score, threshold))
    metrics = np.array(metrics).T
    anomalies, acc, prec, rec, f_score, f01_score = metrics
    if return_metrics:
        return anomalies, acc, prec, rec, f_score, f01_score, threshold
    else:
        return threshold[np.argmax(f_score)]

