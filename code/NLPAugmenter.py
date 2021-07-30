import random
from typing import List, Callable, Union
from gensim.models.keyedvectors import KeyedVectors
import jieba
import logging
# from SentenceEncoder import SentenceEncoder
from TrainConfig import TrainConfig
import json

logging.basicConfig(level=logging.INFO)


class NLPAugmenter():
    def __init__(self, stop_words_path: str,
                 synonym_path: str,
                 eda_count: int = 4,
                 dropout_count: int = 4,
                 eda_cache_path_list: List[str] = None):
        self.stop_words_path = stop_words_path
        self.eda_count = eda_count
        self.dropout_count = dropout_count
        self.eda_cache_path_list = eda_cache_path_list
        self.synonym_path = synonym_path
        self.init_data_model()

    def init_data_model(self):
        with open(self.stop_words_path, "r", encoding="utf8") as fr:
            self.stop_words = set([line.strip() for line in fr])
        # 初始化eda cache
        if isinstance(self.eda_cache_path_list, list):
            self.eda_cache = {}
            for p in self.eda_cache_path_list:
                with open(p, "r", encoding="utf8") as fr:
                    self.eda_cache.update(json.load(fr))
        else:
            self.eda_cache = None
        # 初始化同义词
        with open(self.synonym_path, "r", encoding="utf8") as fr:
            self.word_synonyms = json.load(fr)

    def augment(self, model, sens: List[str], conf: TrainConfig, inference_fn: Callable):
        total_count = self.eda_count + self.dropout_count
        # get aug sens
        aug_sens = [[] for _ in range(total_count)]
        for sen in sens:
            if total_count > 0:
                aug_sens_for_single_sen = self.eda(sentence=sen, num_aug=self.eda_count) + [sen] * self.dropout_count
            elif random.random() > 0.5:
                aug_sens_for_single_sen = [sen]
            else:
                aug_sens_for_single_sen = self.eda(sentence=sen, num_aug=1)
            random.shuffle(aug_sens_for_single_sen)
            for idx, aug_sen in enumerate(aug_sens_for_single_sen):
                aug_sens[idx].append(aug_sen)
        return [inference_fn(model, batch_sens, conf) for batch_sens in aug_sens]
        # get logits
        # aug_logits = [inference_fn(model, batch_sens, conf).unsqueeze(0) for batch_sens in aug_sens]
        # aug_logits = torch.cat(aug_logits, dim=0)  # num_aug * bsz * hidden_size
        # return aug_logits

    def _get_synonyms(self, word):
        """
        get synonyms words
        :param word: str
        :return: List[str]
        """
        if word in self.word_synonyms:
            return self.word_synonyms[word]
        else:
            return []

    ########################################################################
    # 同义词替换
    # 替换一个语句中的n个单词为其同义词
    ########################################################################
    def _synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        # sentence = ' '.join(new_words)
        # new_words = sentence.split(' ')
        return new_words

    ########################################################################
    # 随机插入
    # 随机在语句中插入n个词
    ########################################################################
    def _random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self._add_word(new_words)
        return new_words

    def _add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self._get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    ########################################################################
    # Random swap
    # Randomly swap two words in the sentence n times
    ########################################################################

    def _random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self._swap_word(new_words)
        return new_words

    def _swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    ########################################################################
    # 随机删除
    # 以概率p删除语句中的词
    ########################################################################
    def _random_deletion(self, words, p):

        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words

    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        if num_aug < 1:
            return []
        if self.eda_cache is not None and sentence in self.eda_cache:
            return random.sample(self.eda_cache[sentence], num_aug)
        words = list(jieba.cut(sentence))
        num_words = len(words)
        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # print(words, "\n")

        # 同义词替换sr
        for _ in range(num_new_per_technique):
            a_words = self._synonym_replacement(words, n_sr)
            augmented_sentences.append(''.join(a_words))

        # 随机插入ri
        for _ in range(num_new_per_technique):
            a_words = self._random_insertion(words, n_ri)
            augmented_sentences.append(''.join(a_words))

        # 随机交换rs
        for _ in range(num_new_per_technique):
            a_words = self._random_swap(words, n_rs)
            augmented_sentences.append(''.join(a_words))

        # 随机删除rd
        for _ in range(num_new_per_technique):
            a_words = self._random_deletion(words, p_rd)
            augmented_sentences.append(''.join(a_words))

        # print(augmented_sentences)
        random.shuffle(augmented_sentences)
        augmented_sentences = augmented_sentences[:num_aug]
        # augmented_sentences.append(seg_list)
        return augmented_sentences


if __name__ == "__main__":
    # tencent words vec, 800W words
    # aug = NLPAugmenter(word_vec_path="H:/迅雷下载/Tencent_AILab_ChineseEmbedding.txt", stop_words_path="hit_stopwords.txt")
    # for _ in range(100):
    #     print("111111111111111111111111111")
    #     res = aug.eda("经济学是一门对产品和服务的生产、分配以及消费进行研究的社会科学", num_aug=4)
    # for idx, i in enumerate(res, start=1):
    #     print(idx, i)
    wv = KeyedVectors.load_word2vec_format("H:/迅雷下载/Tencent_AILab_ChineseEmbedding.txt", limit=200 * 10000, )
