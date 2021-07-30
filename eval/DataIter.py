import random
import torch
from transformers import BertTokenizer
from collections import Iterable
import logging

logger = logging.getLogger("OPPOSTS")


class DataIter(Iterable):
    def __init__(self):
        self.swap_pair = None

    def reset(self):
        pass


class VecDataIter(DataIter):
    """ """

    def __init__(self, data_path: str, tokenizer: BertTokenizer, batch_size: int = 64, max_len: int = 128):
        """
        labe2id不为空代表使用完形填空模型
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_path = data_path
        self.data = []

    def reset(self):
        logger.info("dataiter reset, 读取数据")
        self.data.clear()
        with open(self.data_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) == 3:
                    self.data.append([ss[0].strip(), ss[1].strip(), int(ss[2])])
        logger.info("共读取数据:{}条".format(len(self.data)))
        random.shuffle(self.data)
        self.data_iter = iter(self.data)

    def get_steps(self):
        return len(self.data) // self.batch_size

    def get_batch_data(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.batch_size:
                break
        # 判断
        if len(batch_data) == self.batch_size:
            batch_sens_a = [i[0] for i in batch_data]
            batch_sens_b = [i[1] for i in batch_data]
            batch_labels = [[i[2]] for i in batch_data]
            ipt_a = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_sens_a, padding="longest",
                                                     return_tensors="pt", max_length=self.max_len, truncation=True)
            ipt_b = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_sens_b, padding="longest",
                                                     return_tensors="pt", max_length=self.max_len, truncation=True)
            ipt = {
                "labels": torch.tensor(batch_labels),
                "ipt_a": ipt_a,
                "ipt_b": ipt_b
            }
            return ipt
        return None

    def __iter__(self):
        return self

    def __next__(self):
        ipts = self.get_batch_data()
        if ipts is None:
            raise StopIteration
        else:
            return ipts


class CLFDataIter(DataIter):
    """ """

    def __init__(self, data_path: str, tokenizer: BertTokenizer, batch_size: int = 64, max_len: int = 128):
        """
        labe2id不为空代表使用完形填空模型
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_path = data_path
        self.data = []

    def reset(self):
        logger.info("dataiter reset, 读取数据")
        self.data.clear()
        with open(self.data_path, "r", encoding="utf8") as fr:
            for line in fr:
                ss = line.strip().split("\t")
                if len(ss) == 2:
                    self.data.append([ss[0].strip(), int(ss[1])])
        logger.info("共读取数据:{}条".format(len(self.data)))
        random.shuffle(self.data)
        self.data_iter = iter(self.data)

    def get_steps(self):
        return len(self.data) // self.batch_size

    def get_batch_data(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == self.batch_size:
                break
        # 判断
        if len(batch_data) == self.batch_size:
            batch_sens = [i[0] for i in batch_data]
            batch_labels = [i[1] for i in batch_data]
            ipt = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_sens, padding="longest",
                                                   return_tensors="pt", max_length=self.max_len, truncation=True)
            ipt["labels"] = torch.tensor(batch_labels)
            return ipt
        return None

    def __iter__(self):
        return self

    def __next__(self):
        ipts = self.get_batch_data()
        if ipts is None:
            raise StopIteration
        else:
            return ipts


if __name__ == "__main__":
    pass
