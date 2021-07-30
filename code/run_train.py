from TrainConfig import TrainConfig
from Train import train
import logging
import os

logging.basicConfig(level=logging.INFO)
CLUE_CORPUS_NAMES = ["baike_qa2019", "comments2019", "new2016zh", "translation2019zh", "webtext2019zh", "wiki_zh_2019"]
DATA_DIR = r"G:\Data\CLUECorpusSmall\format_corpus" # 数据目录
NUM_LINES_PER_FILE = 50000 # 每个文件的最大行数，用于统计total-steps
all_paths = [] # 获取所有路径
for name in CLUE_CORPUS_NAMES:
    corpus_dir = os.path.join(DATA_DIR, name)
    all_paths.extend([os.path.join(corpus_dir, i) for i in os.listdir(corpus_dir) if i.endswith(".txt")])
all_paths = all_paths
if __name__ == "__main__":
    conf = TrainConfig()
    conf.pretrained_bert_dir = "../PublicData/RBT3" # 加载预训练模型
    conf.synonym_path = "../PublicData/word_synonyms.json" # 缓存的同义词词典
    conf.stop_words_path = "../PublicData/hit_stopwords.txt" # 停用词位置
    conf.data_path_or_list = all_paths
    conf.out_dir = "../output/rbt3_train" # 存储目录
    conf.optim_type = "adam" # 优化器
    conf.batch_size = 128
    conf.steps = len(all_paths) * NUM_LINES_PER_FILE // conf.batch_size
    conf.eda_count = 4 # eda增强数量
    conf.dropout_count = 4 # dropout增强数量
    conf.max_len = 64 # 最大长度
    conf.seed = 2021
    conf.device = "0"
    conf.lr = 1e-5 # 学习率

    conf.num_epochs = 10
    conf.warmup_proportion_or_steps = 0.1
    conf.log_step = 30
    conf.save_step = 5000

    conf.desc = "None"
    conf.loss_type = "mpmn" # 损失函数，目前只支持这个
    conf.pooling_mode = "cls"
    conf.neg_type = "batch" # 负例类型，目前只支持这个
    conf.score_scale = 15.0
    conf.max_num_sens = 10000000  # 一次读多少句子到内存 ，只针对单文件
    conf.max_num_file = 100  # 一次读多少文件到内存，只针对多文件

    train(conf=conf)
