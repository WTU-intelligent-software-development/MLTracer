# -*- coding: utf-8 -*-
# @Time : 2023/5/28 20:43
# @Author : lxf
import logging
import os

from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

# from imblearn.metrics import geometric_mean_score, specificity_score
from lib.mgbdt.utils.log_utils import logger

# 获取当前脚本的路径
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = logger(os.path.join(root, "outputs/logs/measure_exception.log"), name="measure", loggerLevel=logging.ERROR, consoleLevel=logging.ERROR,
             dirLevel=logging.ERROR)


def get_precision(y_test, y_predict):
    p_final = precision_score(y_test, y_predict, average='binary')
    return p_final


def get_recall(y_test, y_predict):
    r_final = recall_score(y_test, y_predict, average='binary')
    return r_final


def get_f1score(y_test, y_predict):
    f_final = f1_score(y_test, y_predict, average='binary')
    return f_final


def get_f2score(y_test, y_predict):
    f_final = fbeta_score(y_test, y_predict, beta=2, average='binary')
    return f_final


# def get_gmean(y_test, y_predict):
#     g_mean = geometric_mean_score(y_test, y_predict, average='binary')
#     return g_mean
#
#
# def get_specificity(y_test, y_predict):
#     specificity = specificity_score(y_test, y_predict, average='binary')
#     return specificity


def get_all_scores(y_test, y_predict):
    try:
        p_final = get_precision(y_test, y_predict)
        r_final = get_recall(y_test, y_predict)
        f1_final = get_f1score(y_test, y_predict)
        f2_final = get_f2score(y_test, y_predict)
        # g_mean = get_gmean(y_test, y_predict)
        # specificity = get_specificity(y_test, y_predict)
        # return p_final, r_final, f_final, f2_final, g_mean, specificity
        return {"precision": p_final, "recall": r_final, "f1": f1_final, "f2": f2_final}
    except Exception as e:
        log.error("cal scores error")
        log.error(str(y_test))
        log.error(str(y_predict))
        raise print(e)


if __name__ == '__main__':
    get_all_scores([1, 1, 0, 0, 1], [0, 1, 0, 1, 0])
    # tp = 1, fp = 1, fn = 2, tn = 1
    # tp = 2, fp = 3, fn =
