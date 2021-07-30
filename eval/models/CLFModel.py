""" 基于sigmoid的一套模型 无脑全连接 一次计算所有 """
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
from os.path import join, exists
from typing import Union, List
import logging

logger = logging.getLogger("dianfei")

HIDDEN_SIZE = 768


class CLFModel(nn.Module):
    def __init__(self, model_dir: str, num_labels: int, use_fc: bool = True):
        super().__init__()
        logger.info("load model from:{}".format(model_dir))
        self.bert = BertModel.from_pretrained(model_dir)
        if use_fc:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                torch.nn.ReLU())
            if exists(join(model_dir, "vec_fc.bin")):
                logger.info("加载全连接层")
                self.fc.load_state_dict(torch.load(join(model_dir, "vec_fc.bin")))
        else:
            self.fc = None
        self.clf = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=num_labels)
        if exists(join(model_dir, "clf.bin")):
            logger.info("加载分类层")
            self.clf.load_state_dict(torch.load(join(model_dir, "clf.bin")))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

    def _get_mean_embed(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sen_vec = sum_embeddings / sum_mask
        return sen_vec

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, pooling_mode="cls", *args, **kwargs):
        token_embeddings, pooler_output, hidden_states = self.bert(input_ids=input_ids,
                                                                   attention_mask=attention_mask,
                                                                   token_type_ids=token_type_ids,
                                                                   output_hidden_states=True)[0:3]
        if pooling_mode == "cls":
            sen_vec = pooler_output
        elif pooling_mode == "mean":
            # get mean token sen vec
            sen_vec = self._get_mean_embed(token_embeddings, attention_mask)
        elif pooling_mode == 'first_last_mean':
            sen_vec = (self._get_mean_embed(hidden_states[-1], attention_mask) + self._get_mean_embed(hidden_states[1],
                                                                                                      attention_mask)) / 2
        elif pooling_mode == 'last2mean':
            sen_vec = (self._get_mean_embed(hidden_states[-1], attention_mask) + self._get_mean_embed(hidden_states[-2],
                                                                                                      attention_mask)) / 2
        elif pooling_mode == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sen_vec = torch.max(token_embeddings, 1)[0]
        if self.fc is not None:
            sen_vec = self.fc(sen_vec)
        # sen_vec = nn.functional.normalize(sen_vec, p=2, dim=1)
        logits = self.clf(sen_vec)
        return logits

    def save(self, save_dir: str):
        self.bert.save_pretrained(save_dir)
        if self.fc is not None:
            torch.save(self.fc.state_dict(), join(save_dir, "vec_fc.bin"))
        torch.save(self.clf.state_dict(), join(save_dir, "clf.bin"))
        self.tokenizer.save_pretrained(save_dir)
