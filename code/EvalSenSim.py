"""
计算句子相似度值
"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from YWVecUtil import BERTSentenceEncoder
from typing import List, Tuple, Union
from VecsWhiteningUtil import VecsWhiteningUtil
from sklearn.preprocessing import normalize
from os.path import join
import logging

# logging.basicConfig(level=logging.INFO)
random.seed(2021)


def eval_sim(model_or_path: Union[BERTSentenceEncoder, str], data: List[Tuple[str, str, int]],
             kernel: np.ndarray = None,
             bias: np.ndarray = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model_or_path, str):
        model = BERTSentenceEncoder(model_or_path, device=device, pooling_modes=["cls"], silence=False)
    else:
        model = model_or_path
    sens_a, sens_b, labels = [], [], []
    for sen1, sen2, label in data:
        sens_a.append(sen1)
        sens_b.append(sen2)
        labels.append(float(label))

    sens_vec_a = model.get_sens_vec(sens_a)
    sens_vec_b = model.get_sens_vec(sens_b)
    if kernel is not None:
        sens_vec_a = VecsWhiteningUtil.transform_and_normalize(sens_vec_a, kernel, bias)
        sens_vec_b = VecsWhiteningUtil.transform_and_normalize(sens_vec_b, kernel, bias)
    else:
        sens_vec_a = normalize(sens_vec_a)
        sens_vec_b = normalize(sens_vec_b)
    sims = np.sum(sens_vec_a * sens_vec_b, axis=1, keepdims=False).tolist()
    auc_val = roc_auc_score(labels, sims)

    # 测试三大系数
    kend_val = kendalltau(labels, sims)[0]
    spear_val = spearmanr(labels, sims)[0]
    pearsonr_val = pearsonr(labels, sims)[0]
    return auc_val, kend_val, spear_val, pearsonr_val


if __name__ == "__main__":
    # 全局变量
    path_info = {
        # "adat": ("adat_train.txt", "adat_dev.txt"),
        # "atec": ("atec_train.txt", "atec_dev.txt"),
        "ccks": ("ccks_train.txt", "ccks_dev.txt"),
        "chip": ("chip_train.txt", "chip_dev.txt"),
        "covid": ("covid_train.txt", "covid_dev.txt"),
        "lcqmc": ("lcqmc_train.txt", "lcqmc_dev.txt"),
    }
    data_dir = r"G:\Codes\LUSE\eval\data"
    model_path = r"G:\Data\simbert_torch"
    # model_path = r"G:\Data\RBT3"
    # model_path = r"G:\Codes\LUSE\output\bert_train\step-60000"
    # model_path = r"G:\Codes\LUSE\output\rbt3_train\step-55000-V2"
    train_count, test_count = 20000, 10000
    use_whiten = False
    # 开始检测
    print("DataSet,AUC,Kendall,Spearman,Pearson")
    for k, v in path_info.items():
        train_data_path, test_data_path = join(data_dir, v[0]), join(data_dir, v[1])
        with open(train_data_path, "r", encoding="utf8") as fr:
            train_data = [line.strip().split("\t") for line in fr]
            train_data = [i for i in train_data if len(i) == 3]
        with open(test_data_path, "r", encoding="utf8") as fr:
            test_data = [line.strip().split("\t") for line in fr]
            test_data = [i for i in test_data if len(i) == 3]
            test_data = test_data[:test_count]
        # 训练集所有句子
        train_sens = []
        for i in train_data:
            train_sens.append(i[0])
            train_sens.append(i[1])
        train_sens = list(set(train_sens))
        random.shuffle(train_sens)
        train_sens = train_sens[:train_count]
        kernel, bias = None, None
        if use_whiten:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kernel, bias = VecsWhiteningUtil.get_model_kernel_bias(
                model=BERTSentenceEncoder(model_path, device=device, pooling_modes=["cls"], silence=False),
                sens=train_sens)
        res = eval_sim(model_path, test_data, kernel, bias)
        print(k, res[0], res[1], res[2], res[3], sep=",")
