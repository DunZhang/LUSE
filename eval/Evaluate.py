import sys

sys.path.append("./models")
from sklearn.metrics import auc, roc_curve, classification_report, accuracy_score, f1_score, hamming_loss, log_loss, \
    jaccard_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch
import numpy as np
from DataIter import DataIter, CLFDataIter
import os
from TrainConfig import TrainConfig
from os.path import join
from scipy.special import expit
import random
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)


def evaluate(model: torch.nn.Module, data_iter: DataIter, device: torch.device, task_type: str = "clf"):
    """
    基于非标签掩码的模型评估
    """
    model.eval()
    data_iter.reset()
    if task_type.lower() == "clf":
        labels = []
        proba = []
        for ipt in data_iter:
            ipt = {k: v.to(device) for k, v in ipt.items()}
            labels.extend(ipt["labels"].cpu().numpy().tolist())
            with torch.no_grad():
                logits = model(**ipt)
            proba.extend(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())

        # 获取预测值
        y_pred = np.argmax(np.vstack(proba), axis=1)
        # 开始计算指标
        acc = accuracy_score(labels, y_pred)  # 严格准确率
        f1 = f1_score(labels, y_pred, average="macro")  # 着重与每一个类别的F1值
        model.train()
        return acc, f1, -1
    else:  # 评测句向量
        labels = []
        proba = []
        for ipt in data_iter:
            ipt_a, ipt_b = ipt["ipt_a"], ipt["ipt_b"]
            ipt_a = {k: v.to(device) for k, v in ipt_a.items()}
            ipt_b = {k: v.to(device) for k, v in ipt_b.items()}
            labels.extend([i[0] for i in ipt["labels"]])
            with torch.no_grad():
                vec_a, vec_b = model(**ipt_a), model(**ipt_b)
            batch_proba = (((vec_a * vec_b).sum(dim=1, keepdim=False) + 1) / 2).cpu().detach().numpy().tolist()
            proba.extend(batch_proba)
        # 开始计算指标
        spear_val = spearmanr(labels, proba)[0]
        model.train()
        return -1, -1, spear_val


if __name__ == "__main__":
    model_dir = "../user_data/trained_models/bert_base/mlog_best_model"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = TrainConfig()
    conf.load(join(model_dir, "train_conf.json"))
    if hasattr(conf, "label_mask_type") and conf.label_mask_type in ["part", "all"]:
        model = LableMaskModel(model_dir).to(device)
    else:
        model = SigmoidModel(model_dir).to(device)
    print(isinstance(model, LableMaskModel))
    data_iter = CLFDataIter(data_path="../user_data/data/hold_out/dev.txt", tokenizer=model.tokenizer,
                            batch_size=32, shuffle=False, max_len=220, label2id=model.get_label2id(),
                            label_mask_type=conf.label_mask_type)
    res = evaluate(model=model, device=device, data_iter=data_iter)
    print(res)
