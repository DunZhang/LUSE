import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from transformers import BertTokenizer, BertForMaskedLM
from typing import List
import logging
import random
from os.path import join, isfile, isdir
from os import listdir
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

logger = logging.getLogger("OPPOSTS")


class AllUtil():
    @staticmethod
    def init_logger(log_name: str = "dianfei", log_file=None, log_file_level=logging.NOTSET):
        log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')

        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.handlers = [console_handler]
        file_handler = logging.FileHandler(log_file, encoding="utf8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        return logger
