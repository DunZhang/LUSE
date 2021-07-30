"""
处理各种数据集，转为规范数据集
"""
import json
from os.path import join
import re
import random
import pandas as pd


def process_tnews(train_path: str = "./ori_data/tnews_public/train.json",
                  dev_path: str = "./ori_data/tnews_public/dev.json",
                  label_path: str = "./ori_data/tnews_public/labels.json",
                  save_dir: str = "./data/"
                  ):
    """
    处理今日头条数据集
    :return:
    """
    # 获取label2id
    with open(label_path, "r", encoding="utf8") as fr:
        labels = [json.loads(line)["label"] for line in fr]
    label2id = {label: idx for idx, label in enumerate(labels)}
    # 格式化训练集和开发集
    train_data, dev_data = [], []
    with open(train_path, "r", encoding="utf8") as fr:
        for line in fr:
            example = json.loads(line)
            train_data.append(re.sub("\s", "", example["sentence"]) + "\t" + str(label2id[example["label"]]) + "\n")
    with open(dev_path, "r", encoding="utf8") as fr:
        for line in fr:
            example = json.loads(line)
            dev_data.append(re.sub("\s", "", example["sentence"]) + "\t" + str(label2id[example["label"]]) + "\n")
    # 保存
    with open(join(save_dir, "tnews_train.txt"), "w", encoding="utf8") as fw:
        fw.writelines(train_data)
    with open(join(save_dir, "tnews_dev.txt"), "w", encoding="utf8") as fw:
        fw.writelines(dev_data)


def process_nlpcc14_sc(data_path="./ori_data/NLPCC14-SC/train.tsv", save_dir="./data/"):
    data = []
    with open(data_path, "r", encoding="utf8") as fr:
        next(fr)
        for line in fr:
            ss = line.split("\t")
            if len(ss) == 2:
                label, sen = ss
                data.append("{}\t{}\n".format(sen.strip(), label.strip()))
    random.shuffle(data)
    train_data, dev_data = data[2000:], data[:2000]
    with open(join(save_dir, "nlpcc14_sc_train.txt"), "w", encoding="utf8") as fw:
        fw.writelines(train_data)
    with open(join(save_dir, "nlpcc14_sc_dev.txt"), "w", encoding="utf8") as fw:
        fw.writelines(dev_data)


def process_online_shopping_senti(data_path="./ori_data/online_shopping_senti/online_shopping_10_cats.csv",
                                  save_dir="./data/"):
    df = pd.read_csv(data_path, encoding="utf8", index_col=False)
    labels = [str(i).strip() for i in df["label"]]
    sens = [re.sub("\s", "", str(i)) for i in df["review"]]
    assert len(labels) == len(sens)
    data = ["{}\t{}\n".format(i, j) for i, j in zip(sens, labels)]
    random.shuffle(data)
    train_data, dev_data = data[10000:], data[:10000]
    with open(join(save_dir, "online_shopping_senti_train.txt"), "w", encoding="utf8") as fw:
        fw.writelines(train_data)
    with open(join(save_dir, "online_shopping_senti_dev.txt"), "w", encoding="utf8") as fw:
        fw.writelines(dev_data)


def process_weibo_senti(data_path="./ori_data/WeiBoSenti/weibo_senti_100k.csv",
                        save_dir="./data/"):
    df = pd.read_csv(data_path, encoding="utf8", index_col=False, sep=",")
    labels = [str(i).strip() for i in df["label"]]
    sens = [re.sub("\s", "", str(i)) for i in df["review"]]
    assert len(labels) == len(sens)
    data = ["{}\t{}\n".format(i, j) for i, j in zip(sens, labels)]
    random.shuffle(data)
    train_data, dev_data = data[20000:], data[:20000]
    with open(join(save_dir, "weibo_senti_train.txt"), "w", encoding="utf8") as fw:
        fw.writelines(train_data)
    with open(join(save_dir, "weibo_senti_dev.txt"), "w", encoding="utf8") as fw:
        fw.writelines(dev_data)


if __name__ == "__main__":
    # process_tnews()
    # process_nlpcc14_sc()
    # process_online_shopping_senti()
    process_weibo_senti()
