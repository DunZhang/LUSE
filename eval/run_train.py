import sys

sys.path.append("./models")
from TrainConfig import TrainConfig
from Train import train
import logging
import os

logging.basicConfig(level=logging.INFO)
# other_info字典不需要改动
other_info = {
    "LCQMC": {
        "num_labels": None,
        "task_type": "vec",
        "train_data_path": "data/lcqmc_train.txt",
        "dev_data_path": "data/lcqmc_dev.txt",
        "eval_step": 1000,
        "max_len": 64,
    },
    "ADAT": {
        "num_labels": None,
        "task_type": "vec",
        "train_data_path": "data/adat_train.txt",
        "dev_data_path": "data/adat_dev.txt",
        "eval_step": 1000,
        "max_len": 64,
    },
    "ATEC": {
        "num_labels": None,
        "task_type": "vec",
        "train_data_path": "data/atec_train.txt",
        "dev_data_path": "data/atec_dev.txt",
        "eval_step": 1000,
        "max_len": 64,
    },
    "CCKS": {
        "num_labels": None,
        "task_type": "vec",
        "train_data_path": "data/ccks_train.txt",
        "dev_data_path": "data/ccks_dev.txt",
        "eval_step": 1000,
        "max_len": 64,
    },
    "CHIP": {
        "num_labels": None,
        "task_type": "vec",
        "train_data_path": "data/chip_train.txt",
        "dev_data_path": "data/chip_dev.txt",
        "eval_step": 300,
        "max_len": 64,
    },
    "COVID": {
        "num_labels": None,
        "task_type": "vec",
        "train_data_path": "data/covid_train.txt",
        "dev_data_path": "data/covid_dev.txt",
        "eval_step": 300,
        "max_len": 64,
    },
    "TNEWS": {
        "num_labels": 15,
        "task_type": "clf",
        "train_data_path": "data/tnews_train.txt",
        "dev_data_path": "data/tnews_dev.txt",
        "eval_step": 1000,
        "max_len": 64,
    },
    "NLPCC14SC": {
        "num_labels": 2,
        "task_type": "clf",
        "train_data_path": "data/nlpcc14_sc_train.txt",
        "dev_data_path": "data/nlpcc14_sc_dev.txt",
        "eval_step": 100,
        "max_len": 128,
    },
    "ShoppingSenti": {
        "num_labels": 2,
        "task_type": "clf",
        "train_data_path": "data/online_shopping_senti_train.txt",
        "dev_data_path": "data/online_shopping_senti_dev.txt",
        "eval_step": 1000,
        "max_len": 128,
    },
    "WeiBoSenti": {
        "num_labels": 2,
        "task_type": "clf",
        "train_data_path": "data/weibo_senti_train.txt",
        "dev_data_path": "data/weibo_senti_dev.txt",
        "eval_step": 1000,
        "max_len": 128,
    }
}
# 只需要改下面的字典
# "../output/rbt3_train/step-95000"
# "../output/rbt3_train/step-55000-V2"
train_info = {
    "LCQMC": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "ADAT": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "ATEC": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "CCKS": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "CHIP": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "COVID": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "TNEWS": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "NLPCC14SC": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "ShoppingSenti": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    },
    "WeiBoSenti": {
        "sen_encoder_dir": "../output/rbt3_train/step-55000-V2",
        "freeze_bert": True,
    }
}

if __name__ == "__main__":
    # 只改这两项即可
    name = "TNEWS"  # 任务名
    preffix = ""  # 前缀
    suffix = "_55000V2"  # 后缀
    for name in train_info.keys():
        conf = TrainConfig()
        # 模型结构
        conf.sen_encoder_dir = train_info[name]["sen_encoder_dir"]
        conf.freeze_bert = train_info[name]["freeze_bert"]
        conf.lr = 5e-4 if train_info[name]["freeze_bert"] else 5e-5
        conf.num_labels = other_info[name]["num_labels"]
        conf.task_type = other_info[name]["task_type"]
        # 相关路径
        conf.train_data_path = other_info[name]["train_data_path"]
        conf.dev_data_path = other_info[name]["dev_data_path"]
        # LUSE_LCQMC_FreezeBERT_cls
        if "BERT" in conf.sen_encoder_dir:
            out_name = "{}_BERT_".format(name)
        elif "RBT3" in conf.sen_encoder_dir:
            out_name = "{}_RBT3_".format(name)
        else:
            out_name = "{}_LUSE_".format(name)
        out_name += "FreezeBERT" if conf.freeze_bert else "TrainBERT"
        out_name = preffix + out_name + suffix
        conf.out_dir = "output/{}".format(out_name)

        conf.max_len = other_info[name]["max_len"]
        conf.pooling_mode = "cls"
        # 训练相关
        conf.seed = 2021
        conf.device = "0"
        conf.batch_size = 32
        conf.num_epochs = 10
        conf.warmup_proportion = 0.1
        # 输出信息
        conf.log_step = 50
        conf.eval_step = other_info[name]["eval_step"]
        # 输出路径
        conf.desc = ""
        train(conf)
