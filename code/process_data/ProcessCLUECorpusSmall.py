import sys

sys.path.append("..")
import jieba
import json
import re
import random
import os
from NLPAugmenter import NLPAugmenter
import numpy as np
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors

try:
    import faiss
except:
    pass
CUT_SEN_PATTERN = re.compile("[\n\r。？！.?!]")
CLEAN_SEN_PATTERN = re.compile("\s")
NUM_LINES_PER_FILE = 50000
CLUE_CORPUS_NAMES = ["baike_qa2019", "comments2019", "new2016zh", "translation2019zh", "webtext2019zh", "wiki_zh_2019"]


def cut_clean_text(text: str):
    sens = [re.sub(CLEAN_SEN_PATTERN, "", sen) for sen in re.split(CUT_SEN_PATTERN, text)]
    sens = [sen + "\n" for sen in sens if len(sen) > 1]
    return sens


def baike_qa2019_to_subfiles(trian_path=r"G:\Data\CLUECorpusSmall\baike_qa2019\baike_qa_train.json",
                             dev_path=r"G:\Data\CLUECorpusSmall\baike_qa2019\baike_qa_valid.json",
                             save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\baike_qa2019"):
    """
    清洗，变为小文件，一行一句话，句子不连续完全打散
    :param trian_path:
    :param dev_path:
    :param save_dir:
    :return:
    """
    all_sens = []
    for path in [dev_path, trian_path]:
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                data = json.loads(line)
                all_sens.extend(cut_clean_text(data["title"]))
                all_sens.extend(cut_clean_text(data["desc"]))
                all_sens.extend(cut_clean_text(data["answer"]))
    print("去重")
    all_sens = list(set(all_sens))
    print("随机化")
    random.shuffle(all_sens)
    print("写入文件")
    idx = 1
    while len(all_sens) > 0:
        sub_sens = all_sens[:NUM_LINES_PER_FILE]
        with open(os.path.join(save_dir, "baike_qa2019_{}.txt".format(idx)), "w", encoding="utf8") as fw:
            fw.writelines(sub_sens)
        all_sens = all_sens[NUM_LINES_PER_FILE:]
        sub_sens.clear()
        idx += 1


def comments2019_to_subfiles(data_dir=r"G:\Data\CLUECorpusSmall\comments2019",
                             save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\comments2019"):
    """
    清洗，变为小文件，一行一句话，句子不连续完全打散
    :param data_dir:
    :param save_dir:
    :return:
    """
    path_list = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith(".txt")]
    all_sens = []
    for path in path_list:
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                if len(line) > 1:
                    all_sens.extend(cut_clean_text(line))
    print("去重")
    all_sens = list(set(all_sens))
    print("随机化")
    random.shuffle(all_sens)
    print("写入文件")
    idx = 1
    while len(all_sens) > 0:
        sub_sens = all_sens[:NUM_LINES_PER_FILE]
        with open(os.path.join(save_dir, "comments2019_{}.txt".format(idx)), "w", encoding="utf8") as fw:
            fw.writelines(sub_sens)
        all_sens = all_sens[NUM_LINES_PER_FILE:]
        sub_sens.clear()
        idx += 1


def new2016zh_to_subfiles(trian_path=r"G:\Data\CLUECorpusSmall\new2016zh\news2016zh_train.json",
                          dev_path=r"G:\Data\CLUECorpusSmall\new2016zh\news2016zh_valid.json",
                          save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\new2016zh"):
    """
    清洗，变为小文件，一行一句话，句子不连续完全打散
    :param trian_path:
    :param dev_path:
    :param save_dir:
    :return:
    """
    all_sens = []
    for path in [dev_path, trian_path]:
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                data = json.loads(line)
                all_sens.extend(cut_clean_text(data["title"]))
                all_sens.extend(cut_clean_text(data["desc"]))
                all_sens.extend(cut_clean_text(data["content"]))
    print("去重")
    all_sens = list(set(all_sens))
    print("随机化")
    random.shuffle(all_sens)
    print("写入文件")
    idx = 1
    while len(all_sens) > 0:
        sub_sens = all_sens[:NUM_LINES_PER_FILE]
        with open(os.path.join(save_dir, "new2016zh_{}.txt".format(idx)), "w", encoding="utf8") as fw:
            fw.writelines(sub_sens)
        all_sens = all_sens[NUM_LINES_PER_FILE:]
        sub_sens.clear()
        idx += 1


def translation2019zh_to_subfiles(trian_path=r"G:\Data\CLUECorpusSmall\translation2019zh\translation2019zh_train.json",
                                  dev_path=r"G:\Data\CLUECorpusSmall\translation2019zh\translation2019zh_valid.json",
                                  save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\translation2019zh"):
    """
    清洗，变为小文件，一行一句话，句子不连续完全打散
    :param trian_path:
    :param dev_path:
    :param save_dir:
    :return:
    """
    all_sens = []
    for path in [dev_path, trian_path]:
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                data = json.loads(line)
                all_sens.extend(cut_clean_text(data["chinese"]))
    print("去重")
    all_sens = list(set(all_sens))
    print("随机化")
    random.shuffle(all_sens)
    print("写入文件")
    idx = 1
    while len(all_sens) > 0:
        sub_sens = all_sens[:NUM_LINES_PER_FILE]
        with open(os.path.join(save_dir, "translation2019zh_{}.txt".format(idx)), "w", encoding="utf8") as fw:
            fw.writelines(sub_sens)
        all_sens = all_sens[NUM_LINES_PER_FILE:]
        sub_sens.clear()
        idx += 1


def webtext2019zh_to_subfiles(trian_path=r"G:\Data\CLUECorpusSmall\webtext2019zh\web_text_zh_train.json",
                              dev_path=r"G:\Data\CLUECorpusSmall\webtext2019zh\web_text_zh_valid.json",
                              test_path=r"G:\Data\CLUECorpusSmall\webtext2019zh\web_text_zh_testa.json",
                              save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\webtext2019zh"):
    """
    清洗，变为小文件，一行一句话，句子不连续完全打散
    :param trian_path:
    :param dev_path:
    :param test_path:
    :param save_dir:
    :return:
    """
    all_sens = []
    for path in [dev_path, trian_path, test_path]:
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                data = json.loads(line)
                all_sens.extend(cut_clean_text(data["title"]))
                all_sens.extend(cut_clean_text(data["desc"]))
                all_sens.extend(cut_clean_text(data["content"]))
    print("去重")
    all_sens = list(set(all_sens))
    print("随机化")
    random.shuffle(all_sens)
    print("写入文件")
    idx = 1
    while len(all_sens) > 0:
        sub_sens = all_sens[:NUM_LINES_PER_FILE]
        with open(os.path.join(save_dir, "webtext2019zh_{}.txt".format(idx)), "w", encoding="utf8") as fw:
            fw.writelines(sub_sens)
        all_sens = all_sens[NUM_LINES_PER_FILE:]
        sub_sens.clear()
        idx += 1


def wiki_zh_2019_to_subfiles(data_dir=r"G:\Data\CLUECorpusSmall\wiki_zh_2019",
                             save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\wiki_zh_2019"):
    """
    清洗，变为小文件，一行一句话，句子不连续完全打散
    :param data_dir:
    :param save_dir:
    :return:
    """
    path_list = []
    for dir_name in os.listdir(data_dir):
        for name in os.listdir(os.path.join(data_dir, dir_name)):
            if name.startswith("wiki"):
                path_list.append(os.path.join(data_dir, dir_name, name))
    all_sens = []
    for path in path_list:
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                data = json.loads(line)
                all_sens.extend(cut_clean_text(data["title"]))
                all_sens.extend(cut_clean_text(data["text"]))
    print("去重")
    all_sens = list(set(all_sens))
    print("随机化")
    random.shuffle(all_sens)
    print("写入文件")
    idx = 1
    while len(all_sens) > 0:
        sub_sens = all_sens[:NUM_LINES_PER_FILE]
        with open(os.path.join(save_dir, "wiki_zh_2019_{}.txt".format(idx)), "w", encoding="utf8") as fw:
            fw.writelines(sub_sens)
        all_sens = all_sens[NUM_LINES_PER_FILE:]
        sub_sens.clear()
        idx += 1


def get_all_words(data_dir, save_path):
    """ 获取所有文件的单词并存储起来 """
    # get all paths
    all_paths = []
    for name in CLUE_CORPUS_NAMES:
        corpus_dir = os.path.join(data_dir, name)
        all_paths.extend([os.path.join(corpus_dir, i) for i in os.listdir(corpus_dir) if i.endswith(".txt")])
    print(len(all_paths))
    all_words = []
    for path in all_paths:
        print("处理数据:{}...".format(path))
        with open(path, "r", encoding="utf8") as fr:
            for line in fr:
                sen = line.strip()
                words = list(jieba.cut(sen))
                all_words.extend(words)
        all_words = list(set(all_words))
    with open(save_path, "w", encoding="utf8") as fw:
        json.dump(all_words, fw, ensure_ascii=False, indent=1)


def get_synonyms(words_path, words_vec_path, save_path):
    """
    获取所有单词的相似词，10个
    :param words_path:
    :param words_vec_path:
    :param save_path:
    :return:
    """
    # 全局变量
    MAX_NUM = 150 * 10000  # 读取最大向量个数
    VEC_DIM = 200  # 向量维度
    TOPK = 8  # 相似词数量
    BATCH_SIZE = 2000  # 一次为多少个相似词查询
    VEC_BINARY = False
    word2synonyms = {}
    with open(words_path, "r", encoding="utf8") as fr:
        query_words = json.load(fr)
    print("读取句向量", MAX_NUM)
    wv = KeyedVectors.load_word2vec_format(words_vec_path, binary=VEC_BINARY, limit=MAX_NUM)
    vec_words = [wv.index2word[i] for i in range(len(wv.vocab))]
    vecs = wv.vectors
    vecs = normalize(vecs, axis=1)
    # 创建索引
    print("创建索引")
    faiss_index = faiss.IndexFlatIP(VEC_DIM)  # 使用欧式距离作为度量
    # 添加数据
    print("添加向量")
    faiss_index.add(vecs)
    # 去除不在向量词的查询词
    vec_words_set = set(vec_words)
    print("原始带查询词汇数量", len(query_words))
    query_words = [w for w in query_words if w in vec_words_set]
    print("去除不在向量库后的查询词汇数量", len(query_words))
    # 开始查询
    w2id = {w: idx for idx, w in enumerate(vec_words)}
    start = 0
    while start < len(query_words):
        print("查询进度：{}/{}, {}%".format(start, len(query_words), round(start / len(query_words) * 100, 4)))
        sub_query_words = [word for word in query_words[start:start + BATCH_SIZE]]
        idxs = [w2id[word] for word in sub_query_words]
        query_vectors = vecs[idxs, :]
        res_distance, res_index = faiss_index.search(query_vectors, TOPK + 1)
        res_index = res_index[:, 1:]
        # 更新word2synonyms
        for i in range(res_index.shape[0]):
            word2synonyms[sub_query_words[i]] = [vec_words[int(res_index[i, j])] for j in range(res_index.shape[1])]

        start += BATCH_SIZE
    with open(save_path, "w", encoding="utf8") as fw:
        json.dump(word2synonyms, fw, ensure_ascii=False, indent=1)


def get_eda_data(nlp_augmenter: NLPAugmenter, data_dir, save_dir, start_pos=0, eda_count: int = 40):
    """
    为所有句子获取eda数据
    :param nlp_augmenter:
    :param data_dir:
    :param save_dir:
    :param start_pos:
    :param eda_count:
    :return:
    """
    # get all paths
    all_paths = []
    for name in CLUE_CORPUS_NAMES:
        corpus_dir = os.path.join(data_dir, name)
        all_paths.extend([os.path.join(corpus_dir, i) for i in os.listdir(corpus_dir) if i.endswith(".txt")])
    print(len(all_paths))
    all_paths.sort()
    for idx, path in enumerate(all_paths[start_pos:], start=start_pos):
        print("{}/{}:\t对数据:{}做缓存...".format(idx, len(all_paths), path))
        with open(path, "r", encoding="utf8") as fr:
            sens = [line.strip() for line in fr]
        cache_data = {sen: nlp_augmenter.eda(sentence=sen, num_aug=eda_count) for sen in sens}
        file_name = os.path.split(path)[-1].replace(".txt", "_eda.json")
        with open(os.path.join(save_dir, file_name), "w", encoding="utf8") as fw:
            json.dump(cache_data, fw, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ##################################### 清洗大文件变成小文件 #####################################
    # baike_qa2019_to_subfiles()
    # comments2019_to_subfiles()
    # new2016zh_to_subfiles()
    # translation2019zh_to_subfiles()
    # webtext2019zh_to_subfiles()
    # wiki_zh_2019_to_subfiles()
    ##################################### 对所有的数据做一次数据增强 #####################################
    aug = NLPAugmenter(synonym_path=r"G:\Data\CLUECorpusSmall\format_corpus\word_synonyms.json",
                       stop_words_path=r"G:\Codes\LUSE\PublicData\hit_stopwords.txt",
                       )
    get_eda_data(nlp_augmenter=aug,
                 data_dir=r"G:\Data\CLUECorpusSmall\format_corpus",
                 save_dir=r"G:\Data\CLUECorpusSmall\format_corpus\eda_data",
                 start_pos=0,
                 eda_count=6)
    # get_all_words(data_dir=r"G:\Data\CLUECorpusSmall\format_corpus",
    #               save_path=r"G:\Data\CLUECorpusSmall\format_corpus\all_words.json")

    # this function should be used in docker or linux
    # docker run -it --name get_synonyms -v G:\Data\CLUECorpusSmall\format_corpus:/format_corpus -v G:\Data\word_vecs:/word_vecs -v G:\Codes\LUSE:/LUSE centos:centos7 /bin/bash
    # get_synonyms(words_path="/format_corpus/all_words.json",
    #              words_vec_path="/word_vecs/Tencent_AILab_ChineseEmbedding.txt",
    #              save_path="/format_corpus/word_synonyms.json")
    # get_synonyms(words_path="G:/Data/CLUECorpusSmall/format_corpus/all_words.json",
    #              words_vec_path="G:/Data/word_vecs/Tencent_AILab_ChineseEmbedding.txt",
    #              save_path="/format_corpus/word_synonyms.json")
