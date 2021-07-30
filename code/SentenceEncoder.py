""" 基于sigmoid的一套模型 无脑全连接 一次计算所有 """
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
from os.path import join, exists
from typing import Union, List
import logging

logger = logging.getLogger("dianfei")


class SentenceEncoder(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        logger.info("load model from:{}".format(model_dir))
        self.bert = BertModel.from_pretrained(model_dir)
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
        return nn.functional.normalize(sen_vec, p=2, dim=1)

    def save(self, save_dir: str):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def get_sens_vec(self, sens: List[str], pooling_mode="cls"):
        self.eval()
        device = self.bert.device
        ### get sen vec
        all_sen_vec = []
        start = 0
        with torch.no_grad():
            while start < len(sens):
                if not self.silence:
                    logger.info("get sentences vector: {}/{}".format(start, len(sens)))
                batch_data = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=sens[start:start + self.batch_size], padding="longest",
                    return_tensors="pt", max_length=self.max_length,
                    truncation=True)
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                batch_data["pooling_mode"] = pooling_mode
                sen_vec = self(**batch_data)
                all_sen_vec.append(sen_vec.cpu().numpy())
                start += self.batch_size
        self.train()
        return np.vstack(all_sen_vec)
