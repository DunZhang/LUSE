from typing import List
from SentenceEncoder import SentenceEncoder
from TrainConfig import TrainConfig
from NLPAugmenter import NLPAugmenter
from MoCoQueue import MoCoQueue
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from AllUtil import AllUtil
from DataIter import SingleFileDataIter, MultiFileDataIter
from math import log
import torch_optimizer as optim


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def inference_fn(model: SentenceEncoder, batch_sens: List[str], conf: TrainConfig):
    device = model.bert.device
    # Step1 original sentences vectors
    batch_data = model.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_sens, padding="longest",
                                                   return_tensors="pt", max_length=conf.max_len, truncation=True)
    batch_data = {k: v.to(device) for k, v in batch_data.items()}
    batch_data["pooling_mode"] = conf.pooling_mode
    vecs = model(**batch_data)  # bsz * hiddensize
    return vecs


# TODO use MOCO
def train_once(model: SentenceEncoder, sens: List[str], nlp_augmenter: NLPAugmenter,
               conf: TrainConfig, mask_for_batch_pos: torch.Tensor = None, mask_for_diag: torch.Tensor = None,
               moco_queue: MoCoQueue = None):
    vecs = inference_fn(model=model, batch_sens=sens, conf=conf)
    aug_count = conf.eda_count + conf.dropout_count
    aug_vecs = nlp_augmenter.augment(model=model, sens=sens, inference_fn=inference_fn, conf=conf)
    aug_vecs.append(vecs)
    aug_vecs = torch.cat(aug_vecs, dim=0)  # bsz*n, hiddensize
    scores = aug_vecs.mm(aug_vecs.t())
    if conf.loss_type == "mpmn":
        scores = scores * conf.score_scale
        log_prob = torch.nn.functional.log_softmax(scores, dim=1)
        loss = -1 * (aug_count + 1) * log(aug_count + 1) - log_prob.masked_select(
            mask_for_batch_pos).sum() / (conf.batch_size * aug_count + conf.batch_size)
    elif conf.loss_type == "bce":
        scores.masked_fill_(mask_for_diag, 0.99999)
        scores = (1 + scores) / 2
        loss = F.binary_cross_entropy(scores, mask_for_batch_pos)
    return loss


def train(conf: TrainConfig):
    # 设置随机数种子
    seed_everything(conf.seed)
    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # directory
    out_dir = conf.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # log file
    log_file_path = os.path.join(out_dir, "logs.txt")
    logger = AllUtil.init_logger(log_name="LUSE", log_file=log_file_path)
    # readme
    with open(os.path.join(out_dir, "readme.txt"), "w", encoding="utf8") as fw:
        for k, v in conf.__dict__.items():
            fw.write("{}\t=\t{}\n".format(k, v))
    # model
    model = SentenceEncoder(conf.pretrained_bert_dir)
    model = model.to(device)
    model.train()

    # train data
    if isinstance(conf.data_path_or_list, list):
        data_iter = MultiFileDataIter(data_path_list=conf.data_path_or_list, max_num_file=conf.max_num_file,
                                      batch_size=conf.batch_size, max_len=conf.max_len)
        data_iter.steps = conf.steps
    else:
        data_iter = SingleFileDataIter(data_path=conf.data_path_or_list, batch_size=conf.batch_size,
                                       max_num_sens=conf.max_num_sens)
    epoch_steps = data_iter.get_steps()

    # some data
    aug_count = conf.eda_count + conf.dropout_count
    mask_for_batch_pos = torch.zeros((conf.batch_size * (aug_count + 1), conf.batch_size * (aug_count + 1)),
                                     requires_grad=False)
    mask_for_diag = torch.zeros((conf.batch_size * (aug_count + 1), conf.batch_size * (aug_count + 1)),
                                requires_grad=False)
    for i in range(mask_for_batch_pos.shape[0]):
        mask_for_diag[i, i] = 1
        for j in range(1 + aug_count):
            mask_for_batch_pos[i, i % conf.batch_size + j * conf.batch_size] = 1
    mask_for_diag = mask_for_diag.bool().to(device)
    if conf.loss_type == "mpmn":
        mask_for_batch_pos = mask_for_batch_pos.bool().to(device)
    elif conf.loss_type == "bce":
        mask_for_batch_pos = mask_for_batch_pos.float().to(device)
    # data augmennter
    nlp_augmenter = NLPAugmenter(stop_words_path=conf.stop_words_path, synonym_path=conf.synonym_path,
                                 eda_count=conf.eda_count, dropout_count=conf.dropout_count)
    # optimizer
    logger.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    optimizer_grouped_parameters = [{
        "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    if conf.optim_type == "lamb":
        logger.info("使用lamb优化器")
        optimizer = optim.Lamb(optimizer_grouped_parameters, lr=conf.lr, eps=1e-8)
    elif conf.optim_type == "adam":
        logger.info("使用adam优化器")
        optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr, eps=1e-8)
    elif conf.optim_type == "lookahead":
        logger.info("使用lookahead优化器")
        optimizer = optim.Lookahead(AdamW(optimizer_grouped_parameters, lr=conf.lr, eps=1e-8))

    total_steps = epoch_steps * conf.num_epochs
    if conf.warmup_proportion_or_steps < 1:
        num_warmup_steps = int(conf.warmup_proportion_or_steps * total_steps)
    else:
        num_warmup_steps = int(conf.warmup_proportion_or_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    # train
    global_step = 1
    logger.info("start train")
    for epoch in range(conf.num_epochs):
        data_iter.reset()
        for step, batch_sens in enumerate(data_iter):
            step += 1
            loss = train_once(model=model, sens=batch_sens, nlp_augmenter=nlp_augmenter, conf=conf,
                              mask_for_batch_pos=mask_for_batch_pos, mask_for_diag=mask_for_diag, moco_queue=None)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # 梯度下降，更新参数
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            global_step += 1
            if step % conf.log_step == 0:
                logger.info("epoch-{}, step-{}/{}, loss:{}".format(epoch + 1, step, epoch_steps, loss.data))
            if global_step % conf.save_step == 0:
                save_dir = os.path.join(out_dir, "step-{}".format(global_step))
                os.makedirs(save_dir)
                logger.info("save model to :{}".format(save_dir))
                model.save(save_dir)
