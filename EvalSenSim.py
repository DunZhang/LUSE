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
import logging

logging.basicConfig(level=logging.INFO)
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
    data_path = "yunwen_general_dev.txt"
    model_path = r"G:\Data\simbert_torch"
    with open(data_path, "r", encoding="utf8") as fr:
        data = [line.strip().split("\t") for line in fr]
    random.shuffle(data)
    train_data, dev_data = data[0:100000], data[100000:110000]
    train_sens = []
    for i in train_data:
        train_sens.append(i[0])
        train_sens.append(i[1])
    kernel, bias = None, None
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # kernel, bias = VecsWhiteningUtil.get_model_kernel_bias(
    #     model=BERTSentenceEncoder(model_path, device=device, pooling_modes=["cls"], silence=False), sens=train_sens)
    res = eval_sim(model_path, dev_data, kernel, bias)
    print(res)
#  general
# (0.9682072128288514, 0.6621768221157278, 0.8109570632426164, 0.701932601534903)
# (0.9657445829783319, 0.65869392404344, 0.8066916717137326, 0.7938405689805593)

# yw
# (0.9648718194872781, 0.6574596368547729, 0.8051800040369308, 0.6941307330223077)
# (0.9709778439113756, 0.6660952279029209, 0.8157559274557852, 0.8109330457244636)
