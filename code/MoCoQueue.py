# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCoQueue():
    pass
# class MoCoQueue(nn.Module):
#     """
#     Build a MoCo model with: a query encoder, a key encoder, and a queue
#     https://arxiv.org/abs/1911.05722
#     """
#
#     def __init__(self, encoder_q, encoder_k, dim=128, queue_length=65536, m=0.999):
#         """
#         dim: feature dimension (default: 128)
#         K: queue size; number of negative keys (default: 65536)
#         m: moco momentum of updating key encoder (default: 0.999)
#         T: softmax temperature (default: 0.07)
#         """
#         super().__init__()
#
#         self.queue_length = queue_length
#         self.m = m
#
#         # create the encoders
#         # num_classes is the output fc dimension
#         self.encoder_q = encoder_q
#         self.encoder_k = encoder_k
#
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient
#
#         # create the queue
#         self.queue = nn.functional.normalize(torch.randn(dim, queue_length, requires_grad=False), dim=0)
#         self.ptr = 0
#
#     def get_queue(self):
#         return self.queue
#
#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         Momentum update of the key encoder
#         """
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
#
#     def enqueue(self, vecs: torch.Tensor):
#         # gather keys before updating queue
#         batch_size = vecs.shape[0]
#         assert self.queue_length % batch_size == 0  # for simplicity
#         self.queue[:, self.ptr:self.ptr + batch_size] = vecs.detach().t()
#         self.ptr = (self.ptr + batch_size) % self.queue_length  # move pointer
#
#     def forward(self, im_q, im_k):
#         """
#         Input:
#             im_q: a batch of query images
#             im_k: a batch of key images
#         Output:
#             logits, targets
#         """
#
#         # compute query features
#         q = self.encoder_q(im_q)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)
#
#         # compute key features
#         with torch.no_grad():  # no gradient to keys
#             self._momentum_update_key_encoder()  # update the key encoder
#             k = self.encoder_k(im_k)  # keys: NxC
#             k = nn.functional.normalize(k, dim=1)
#
#         # compute logits
#         # Einstein sum is more intuitive
#         # positive logits: Nx1
#         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#         # negative logits: NxK
#         l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
#
#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1)
#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#
#         # dequeue and enqueue
#         self.enqueue(k)
#
#         return logits, labels
