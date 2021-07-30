from os.path import join
import json
import logging


class TrainConfig():
    def __init__(self):
        # 模型结构
        self.sen_encoder_dir = "../public_pretrained_models/roberta_wwm_ext_base"
        self.max_len = 200
        self.freeze_bert = False
        self.pooling_mode = False
        # 训练相关
        self.device = "0"
        self.lr = 2e-5
        self.warmup_proportion = 2e-5
        self.batch_size = 32
        self.seed = 2021
        self.num_epochs = 40
        self.task_type = "clf"
        self.use_fc = False
        self.num_labels = False
        # 输出信息
        self.log_step = 40
        self.eval_step = 40
        self.train_data_path = "../user_data/data_20210228/train.txt"
        self.dev_data_path = "../user_data/data_20210228/dev.txt"
        # 输出路径
        self.out_dir = "../user_data/output_dir/bert_model_v1"
        # 训练解释
        self.desc = ""

    def save(self, save_dir):
        with open(join(save_dir, "train_conf.json"), "w", encoding="utf8") as fw:
            json.dump(self.__dict__, fw, ensure_ascii=False, indent=1)

    def load(self, conf_path: str):
        with open(conf_path, "r", encoding="utf8") as fr:
            kwargs = json.load(fr)
        for key, value in kwargs.items():
            try:
                if key not in self.__dict__:
                    logging.error("key:{} 不在类定义中, 请根据配置文件重新生成类".format(key))
                    continue
                if isinstance(value, dict):
                    continue
                setattr(self, key, value)
            except AttributeError as err:
                logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err
