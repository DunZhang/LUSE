import random
import logging
from typing import List

logger = logging.getLogger("LUSE")


class SingleFileDataIter():
    """ 对于bert的数据加载器 只获得area或type """

    def __init__(self, data_path: str, max_num_sens: int, batch_size: int):
        self.data_path = data_path
        self.max_num_sens = max_num_sens
        self.batch_size = batch_size
        self.steps = None
        self.mem_sens = []
        self.fr = None

    def __del__(self):
        try:
            self.fr.close()
        except:
            pass

    def reset(self):
        self.mem_sens.clear()
        try:
            self.fr.close()
        except:
            pass
        self.fr = open(self.data_path, "r", encoding="utf8")

    def get_batch_sens(self):
        if len(self.mem_sens) < self.batch_size:  # sens in memory is not enough
            if self.fr is not None:
                try:
                    for _ in range(self.max_num_sens):
                        self.mem_sens.append(next(self.fr).strip())
                except StopIteration:
                    self.fr.close()
                    self.fr = None
                random.shuffle(self.mem_sens)
            else:  # finish one epoch
                return None
        batch_sens = self.mem_sens[:self.batch_size]
        self.mem_sens = self.mem_sens[self.batch_size:]
        if len(batch_sens) == self.batch_size:
            return batch_sens
        else:
            return None

    def get_steps(self):
        if self.steps is None:
            steps = 0
            with open(self.data_path, "r", encoding="utf8") as fr:
                for _ in fr: steps += 1
            self.steps = steps
        return self.steps // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batch_sens = self.get_batch_sens()
        if batch_sens is None:
            raise StopIteration
        else:
            return batch_sens


class MultiFileDataIter():
    """  """

    def __init__(self, data_path_list: List[str], max_num_file: int, batch_size: int, max_len: int):
        self.data_path_list = data_path_list
        self.file_id = 0
        self.max_num_file = max_num_file
        self.batch_size = batch_size
        self.steps = None
        self.max_len = max_len
        self.mem_sens = []

    def reset(self):
        self.mem_sens.clear()
        self.file_id = 0
        random.shuffle(self.data_path_list)

    def read_sens(self):
        if self.file_id >= len(self.data_path_list):
            return False
        for path in self.data_path_list[self.file_id:self.file_id + self.max_num_file]:
            with open(path, "r", encoding="utf8") as fr:
                self.mem_sens.extend([line.strip() for line in fr if self.max_len > len(line.strip()) > 1])
                # self.mem_sens.extend([line.strip() for line in fr])
        self.file_id += self.max_num_file
        return True

    def get_batch_sens(self):
        if len(self.mem_sens) < self.batch_size:  # sens in memory is not enough
            flag = self.read_sens()
            if flag:  # 读取成功
                random.shuffle(self.mem_sens)
            else:
                return None  # 无文件可读

        batch_sens = self.mem_sens[:self.batch_size]
        self.mem_sens = self.mem_sens[self.batch_size:]
        if len(batch_sens) == self.batch_size:
            return batch_sens
        else:
            return None  # 剩余的文件过少

    def get_steps(self):
        return self.steps

    def __iter__(self):
        return self

    def __next__(self):
        batch_sens = self.get_batch_sens()
        if batch_sens is None:
            raise StopIteration
        else:
            return batch_sens


if __name__ == "__main__":
    data_iter = SingleFileDataIter(data_path=r"H:\我的坚果云\文本相似度数据集\SimilarityData\lcqmc_train.txt",
                                   max_num_sens=10000,
                                   batch_size=32)

    for epoch in range(4):
        data_iter.reset()
        for sens in data_iter:
            pass
