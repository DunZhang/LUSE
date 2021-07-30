from os.path import join
import json
import logging


class TrainConfig():
    def __init__(self):
        # 模型结构
        self.pretrained_bert_dir = None
        self.synonym_path = None
        self.stop_words_path = None
        self.eda_count = None
        self.dropout_count = None
        self.max_len = None
        self.seed = None
        self.device = None
        self.lr = None
        self.batch_size = None
        self.num_epochs = None
        self.warmup_proportion_or_steps = None
        self.log_step = None
        self.save_step = None
        self.data_path_or_list = None
        self.out_dir = None
        self.desc = None
        self.loss_type = None
        self.pooling_mode = None
        self.neg_type = None
        self.score_scale = None
        self.max_num_sens = None
        self.max_num_file = None
        self.steps = None
        self.optim_type = None

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
