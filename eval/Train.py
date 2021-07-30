import os
import torch
import random
import numpy as np
from TrainConfig import TrainConfig
from DataIter import CLFDataIter, VecDataIter
from Evaluate import evaluate
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from os.path import join
import torch_optimizer as optim
from CLFModel import CLFModel
from SimVecModel import SimVecModel
import logging


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
    acc_best_model_dir = join(out_dir, "acc_best_model")
    os.makedirs(acc_best_model_dir)
    f1_best_model_dir = join(out_dir, "f1_best_model")
    os.makedirs(f1_best_model_dir)
    sp_best_model_dir = join(out_dir, "sp_best_model")
    os.makedirs(sp_best_model_dir)
    # log file
    log_file_path = os.path.join(out_dir, "logs.txt")
    logger = init_logger(log_name="dianfei", log_file=log_file_path)
    # readme
    with open(os.path.join(out_dir, "readme.txt"), "w", encoding="utf8") as fw:
        for k, v in conf.__dict__.items():
            fw.write("{}\t=\t{}\n".format(k, v))
    # tokenizer
    tokenizer = BertTokenizer(join(conf.sen_encoder_dir, "vocab.txt"))
    # models
    if conf.task_type.lower() == "clf":
        model = CLFModel(model_dir=conf.sen_encoder_dir, use_fc=conf.freeze_bert, num_labels=conf.num_labels)
    else:
        model = SimVecModel(model_dir=conf.sen_encoder_dir, use_fc=conf.freeze_bert)
    model = model.to(device)
    model.train()
    if conf.task_type == "clf":
        # train data
        train_data_iter = CLFDataIter(data_path=conf.train_data_path, tokenizer=tokenizer,
                                      batch_size=conf.batch_size, max_len=conf.max_len)
        # dev data
        dev_data_iter = CLFDataIter(data_path=conf.dev_data_path, tokenizer=tokenizer,
                                    batch_size=conf.batch_size, max_len=conf.max_len)
    else:
        # train data
        train_data_iter = VecDataIter(data_path=conf.train_data_path, tokenizer=tokenizer,
                                      batch_size=conf.batch_size, max_len=conf.max_len)
        # dev data
        dev_data_iter = VecDataIter(data_path=conf.dev_data_path, tokenizer=tokenizer,
                                    batch_size=conf.batch_size, max_len=conf.max_len)
    # loss models
    train_data_iter.reset()
    if conf.task_type.lower() == "clf":
        logger.info("使用交叉熵损失函数")
        loss_model = torch.nn.CrossEntropyLoss()
    else:
        logger.info("使用BCELoss损失函数")
        loss_model = torch.nn.BCELoss()
    # optimizer
    logger.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    optimizer_grouped_parameters = [{
        "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
        {"params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.lr, eps=1e-8)
    if conf.freeze_bert:
        for n, p in model.bert.named_parameters():
            p.requires_grad = False
    total_steps = train_data_iter.get_steps() * conf.num_epochs
    epoch_steps = train_data_iter.get_steps()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(conf.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # train
    global_step = 1
    logger.info("start train")
    accs, f1s, sps = [], [], []
    for epoch in range(conf.num_epochs):
        train_data_iter.reset()
        for step, ipt in enumerate(train_data_iter):
            step += 1
            if conf.task_type == "clf":
                ipt = {k: v.to(device) for k, v in ipt.items()}
                labels = ipt["labels"].long()
                ipt["pooling_mode"] = conf.pooling_mode
                logits = model(**ipt)
                loss = loss_model(logits, labels)
            else:
                ipt_a, ipt_b, labels = ipt["ipt_a"], ipt["ipt_b"], ipt["labels"]
                ipt_a = {k: v.to(device) for k, v in ipt_a.items()}
                ipt_b = {k: v.to(device) for k, v in ipt_b.items()}
                labels = labels.to(device).float()
                ipt_a["pooling_mode"] = conf.pooling_mode
                ipt_b["pooling_mode"] = conf.pooling_mode
                vec_a, vec_b = model(**ipt_a), model(**ipt_b)
                proba = ((vec_a * vec_b).sum(dim=1, keepdim=True) + 1) / 2
                loss = loss_model(proba, labels)
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
            if global_step % conf.eval_step == 0 or global_step == 10:
                # 做个测试
                acc, f1, sp = evaluate(model=model, data_iter=dev_data_iter, device=device, task_type=conf.task_type)
                logger.info(
                    "epoch-{}, step-{}/{}, global_step-{}, acc:{}, f1:{}, sp:{}".format(epoch + 1, step, epoch_steps,
                                                                                        global_step, acc, f1, sp))
                accs.append(acc)
                f1s.append(f1)
                sps.append(sp)
                logger.info("best accs:{}, best f1:{}, best sps:{}".format(max(accs), max(f1s), max(sps)))
                if len(accs) == 1 or accs[-1] > max(accs[:-1]):
                    logger.info("save acc best model")
                    model.save(acc_best_model_dir)
                if len(f1s) == 1 or f1s[-1] > max(f1s[:-1]):
                    logger.info("save f1 best model")
                    model.save(f1_best_model_dir)
                if len(sps) == 1 or sps[-1] > max(sps[:-1]):
                    logger.info("save sp best model")
                    model.save(sp_best_model_dir)
