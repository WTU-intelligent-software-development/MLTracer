# -*- coding: utf-8 -*-
# @Time : 2023/7/15 9:22
# @Author : lxf

import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# sys.path.insert(0, "lib")
from lib.mgbdt import MGBDT, MultiXGBModel
from lib.mgbdt import metrics
from lib.mgbdt.utils.log_utils import logger
from scripts import jie_xi__log

measure_name_list = ["precision", "recall", "f1score", "f2score"]
result = pd.DataFrame(index=measure_name_list)

# 获取当前脚本的路径
root = os.path.dirname(os.path.abspath(__file__))

def runexp():
    # 欠采样3 + 过采样3 + 综合采样2
    # data_banlace_methods = [db.No_Balanced, db.undersmapling, db.TomekLink, db.nearmiss, db.Randomos, db.smote,
    #                         db.Adasyn, db.smotenn, db.borderline_smote, db.Smote_Tomek]
    # data_banlace_methods_name = ["None", "RUS", "Tomek_Link", "Near_Miss", "ROS", "SMOTE", "ADASYN", "SMOTE_ENN",
    #                              "Borderline_Smote", "SMOTE_Tomek"]
    data_banlace_methods = [None]
    data_banlace_methods_name = ["None"]

    idx = 0
    for method in data_banlace_methods:
        # ************************----EasyClinic-----************************
        feature_path = os.path.join(root, "outputs/features/EasyClinic/UC_CC/")
        cv("UC_CC-" + data_banlace_methods_name[idx], feature_path, method)
        print("EasyClinic_UC_CC complete!")

        feature_path = os.path.join(root, "outputs/features/EasyClinic/UC_ID/")
        cv("UC_ID-" + data_banlace_methods_name[idx], feature_path, method)
        print("EasyClinic_UC_ID complete!")

        feature_path = os.path.join(root, "outputs/features/EasyClinic/UC_TC/")
        cv("UC_TC-" + data_banlace_methods_name[idx], feature_path, method)
        print("EasyClinic_UC_TC complete!")

        # ************************----CM1-NASA-----************************
        feature_path = os.path.join(root, "outputs/features/CM1-NASA/")
        cv("CM1_NASA-" + data_banlace_methods_name[idx], feature_path, method)
        print("CM1-NASA_HRQ_LRQ complete!")

        # ************************----GANNT-----************************
        feature_path = os.path.join(root, "outputs/features/GANNT/")
        cv("GANNT-" + data_banlace_methods_name[idx], feature_path, method)
        print("GANNT_H_L complete!")

        # ************************----eTour-----************************
        feature_path = os.path.join(root, "outputs/features/eTOUR/")
        cv("eTOUR-" + data_banlace_methods_name[idx], feature_path, method)
        print("eTOUR_UC_CC complete!")

        # ************************----iTrust-----************************
        feature_path = os.path.join(root, "outputs/features/iTrust/")
        cv("iTrust-" + data_banlace_methods_name[idx], feature_path, method)
        print("iTrust_UC_CC complete!")

        idx = idx + 1


def cv(name, feature_path, dbm=None):
    """
    函数说明：进行一折交叉验证(既没有交叉验证)
    输入参数：
    name：文件名
    X：特征
    y：标签
    dbm：数据平衡方法
    """
    # n_samples = 15000
    # x_all, y_all = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.04, random_state=0)

    df = pd.read_excel(os.path.join(feature_path, "all_features.xlsx"), index_col=0)
    X = df.iloc[:, :-1]
    y = df["label"]

    min_max_scaler = preprocessing.MinMaxScaler()  # 归一化
    X = pd.DataFrame(min_max_scaler.fit_transform(X))  # 全部集归一化

    X_banance, X_test, y_banance, y_test = split_data(X, y, dbm)  # 划分数据集 + 数据平衡

    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_banance = min_max_scaler.fit_transform(X_banance)  # 训练集归一化
    # X_test = min_max_scaler.transform(X_test)  # 测试集归一化

    X_banance = X_banance.values
    y_banance = y_banance.values
    X_test = X_test.values
    y_test = y_test.values

    logdir = os.path.join(root, "outputs/logs/", name.split("-")[0],
                          name.split("-")[1] + ".log")  # ./outputs/logs/UC_CC/None.log
    log = logger(logdir)

    net = MGBDT(log, datasetName=name, loss="CrossEntropyLoss", target_lr=1.0,
                epsilon=0.1)  # Create a multi-layerd GBDTs
    init_dim = 32
    global layer_num
    if layer_num < 2:
        raise "layer num mast >= 2"
    net.add_layer("tp_layer",
                  F=MultiXGBModel(input_size=X_banance.shape[1], output_size=init_dim, learning_rate=0.01, max_depth=5,
                                  num_boost_round=5),
                  G=None)
    change_layer = int((init_dim-2) / (layer_num-1))
    for i in range(layer_num-1):
        if i == layer_num-2:
            net.add_layer("tp_layer",
                          F=MultiXGBModel(input_size=init_dim, output_size=2, learning_rate=0.01, max_depth=5,
                                          num_boost_round=5),
                          G=MultiXGBModel(input_size=2, output_size=init_dim, learning_rate=0.01, max_depth=5,
                                          num_boost_round=5))
            break
        net.add_layer("tp_layer",
                      F=MultiXGBModel(input_size=init_dim, output_size=init_dim-change_layer, learning_rate=0.01, max_depth=5,
                                      num_boost_round=5),
                      G=MultiXGBModel(input_size=init_dim-change_layer, output_size=init_dim, learning_rate=0.01, max_depth=5,
                                      num_boost_round=5))
        init_dim = init_dim-change_layer

    net.init(X_banance, n_rounds=5)
    net.fit(X_banance, y_banance, n_epochs=100, eval_sets=[(X_test, y_test)], eval_metric="all")  # fit the dataset
    y_pred = net.forward(X_test)


    scores = metrics.get_all_scores(y_test, y_pred.argmax(axis=1))
    global result
    result[name] = [scores["precision"], scores["recall"], scores["f1"], scores["f2"]]

    log.__del__()
    del net


def write_to_excel(output_fname=os.path.join(root, "outputs/performance/")):
    if output_fname is not None:
        if not os.path.exists(output_fname):
            os.makedirs(output_fname)
        result.to_excel(os.path.join(output_fname + "scores.xlsx"))


def setup_main():
    runexp()
    # write_to_excel()

