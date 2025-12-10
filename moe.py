# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_model import Mamba_dflow_MAE
import numpy as np
import random
from parallel_experts import ParallelExperts
from torch.cuda.amp import autocast, GradScaler

# from .parallel_experts import ParallelExperts

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0.01, switchloss=0, zloss=0,
                 bias=False, gating_activation=None,
                 activation=nn.GELU(), noisy_gating=True, usage_mem = 10000,
                 acc_aux_loss=False):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.activation = activation
        # self.usage = np.random.randint(num_experts, size=(usage_mem, k))
        # self.cur = 0


        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()

        if True:
            if gating_activation is None:
                gating_activation = nn.ReLU()
            self.f_gate = nn.Sequential(
                # nn.Linear(input_size, input_size),
                # gating_activation,
                nn.Linear(input_size,
                          2 * num_experts if noisy_gating else num_experts,
                          bias=False)
            )
            nn.init.zeros_(self.f_gate[-1].weight)
        else:
            self.f_gate = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(self.f_gate.weight)


    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        # self._gates = []
        # self._probs = []
        # self._logits = []
        # self._expert_sizes = []

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        # cvloss = self.acc_gates.mean() / 10000.0
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        # loss = (self.cvloss * cvloss)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)

        # print("cvloss")
        # true_cvloss = self.compute_cvloss(torch.cat(self._gates, dim=0))
        # print(self.cvloss, cvloss, true_cvloss)

        # print("switchloss")
        # cat_probs = torch.cat(self._probs, dim=0)
        # true_switchloss = self.compute_switchloss(cat_probs, sum(self._expert_sizes))
        # print(self.switchloss, switchloss, true_switchloss)

        # print("zloss")
        # true_zloss = self.compute_zloss(torch.cat(self._logits, dim=0))
        # print(self.zloss, zloss, true_zloss)

        # assert torch.allclose(cvloss, true_cvloss)
        # assert torch.allclose(switchloss, true_switchloss)
        # assert torch.allclose(zloss, true_zloss)

        self.init_aux_statistics()
        return loss

    # def compute_topk_loss(self, probs):


    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate(x)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        assert sample_topk == 0
        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]


        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # print('probs: ', probs)
        # print('top_k_gates: ', top_k_gates)
        # print('top_k_indices: ', top_k_indices)
        # print('expert_size: ', expert_size)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            # if self.training:
            self.update_aux_statistics(logits, probs, gates)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate(x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y

    def map(self, x, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y

class TaskMoE(MoE):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,  input_size, head_size, num_experts, k, w_MI=0, limit_k=0, w_topk_loss=0.0, task_num=9, noisy_gating=True, gating_activation=None, **kwargs):
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI

        self.limit_k = max(k, limit_k)

        super(TaskMoE, self).__init__(input_size, head_size, num_experts, k, noisy_gating=noisy_gating, gating_activation=gating_activation, **kwargs)
        
        if gating_activation is None:
            gating_activation = nn.ReLU()

        self.f_gate = nn.ModuleList([nn.Sequential(
                                        # nn.Linear(input_size, input_size),
                                        # gating_activation,
                                        nn.Linear(input_size,
                                                  2 * num_experts if noisy_gating else num_experts,
                                                  bias=False)
                                    ) for i in range(task_num)])
        for i in range(task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)
    
    def init_aux_statistics(self, clear=True):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        if clear:
            self.task_gate_freq = [0] * self.task_num
            self.topk_acc_probs = 0.

        self.MI_task_gate = torch.zeros(self.task_num, self.num_experts).cuda()

    def update_aux_statistics(self, logits, probs, gates, task_bh):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.0001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

        self.topk_acc_probs = self.topk_acc_probs + probs.mean(0)

        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + ((gates > 0).float().sum(0)).detach()*0.05

        # self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + gates.sum(0)
        self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + probs.sum(0)

    def get_topk_loss_and_clear(self):
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.limit_k, dim=0)
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()

        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss # 0.004 * 12 * 2 = 0.09

    def get_aux_loss_and_clear(self):
        '''
            acc_gates: sum of topk soft score
            acc_freq: the number of being chosen
            acc_probs: sum of probs (probs = softmax(score))
        '''

        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)

        tot = self.acc_freq.sum() / self.k
        self.MI_task_gate = self.MI_task_gate / (tot+0.0001)
        P_TI = torch.sum(self.MI_task_gate, dim=1, keepdim=True) + 0.0001
        P_EI = torch.sum(self.MI_task_gate, dim=0, keepdim=True) + 0.0001

        MI_loss = -(self.MI_task_gate * torch.log(self.MI_task_gate / P_TI / P_EI + 0.0001)).sum()
        
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss +
                self.w_MI * MI_loss
                )

        self.init_aux_statistics(clear=False)
        return loss

    def top_k_gating(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate[task_bh](x)
        if self.noisy_gating and self.training:
        # if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        probs = torch.softmax(logits, dim=1) + 1e-4
        
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]
       

        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, task_bh)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, task_bh=0, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate[task_bh](x)
        
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, task_bh, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y

    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        # print('batch_index: ', batch_index)
        # print('expert_inputs: ', expert_inputs)
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y

class TaskMoE2(MoE):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,  input_size, head_size, num_experts, k, w_MI=0, limit_k=0, w_topk_loss=0.0, task_num=9, noisy_gating=True, gating_activation=None, **kwargs):
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI

        self.limit_k = max(k, limit_k)

        super(TaskMoE2, self).__init__(input_size, head_size, num_experts, k, noisy_gating=noisy_gating, gating_activation=gating_activation, **kwargs)
        
        if gating_activation is None:
            gating_activation = nn.ReLU()

        self.f_gate = nn.ModuleList([nn.Sequential(
                                        # nn.Linear(input_size, input_size),
                                        # gating_activation,
                                        nn.Linear(input_size*2,
                                                  2 * num_experts if noisy_gating else num_experts,
                                                  bias=False)
                                    ) for i in range(task_num)])
        
        # self.gate = ParallelExperts(num_experts, input_size*2, 2, True)#
        self.gate = ParallelExperts(num_experts, input_size*2, input_size*2, True)#

        for i in range(task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)
    
    def init_aux_statistics(self, clear=True):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        if clear:
            self.task_gate_freq = [0] * self.task_num
            self.topk_acc_probs = 0.

        self.MI_task_gate = torch.zeros(self.task_num, self.num_experts).cuda()

    def update_aux_statistics(self, logits, probs, gates, task_bh):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.0001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

        self.topk_acc_probs = self.topk_acc_probs + probs.mean(0)

        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + ((gates > 0).float().sum(0)).detach()*0.05

        # self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + gates.sum(0)
        self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + probs.sum(0)

    def get_topk_loss_and_clear(self):
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.limit_k, dim=0)
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()

        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss # 0.004 * 12 * 2 = 0.09

    def get_aux_loss_and_clear(self):
        '''
            acc_gates: sum of topk soft score
            acc_freq: the number of being chosen
            acc_probs: sum of probs (probs = softmax(score))
        '''

        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)

        tot = self.acc_freq.sum() / self.k
        self.MI_task_gate = self.MI_task_gate / (tot+0.0001)
        P_TI = torch.sum(self.MI_task_gate, dim=1, keepdim=True) + 0.0001
        P_EI = torch.sum(self.MI_task_gate, dim=0, keepdim=True) + 0.0001

        MI_loss = -(self.MI_task_gate * torch.log(self.MI_task_gate / P_TI / P_EI + 0.0001)).sum()
        
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss +
                self.w_MI * MI_loss
                )

        self.init_aux_statistics(clear=False)
        return loss

    def top_k_gating(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate[task_bh](x)
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1) + 1e-4

        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]
       

        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, task_bh)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, task_bh=0, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        
        # h = F.softmax(self.gate(expert_inputs, self.expert_size)).unsqueeze(2) *  expert_inputs.view(expert_inputs.shape[0], 2, -1)
        h_gate  = self.gate(expert_inputs, self.expert_size).unsqueeze(1)
        h_gate = F.softmax(torch.cat([h_gate[:,:,0:emb_size//2],h_gate[:,:,emb_size//2:]],dim=1),dim=1) 
        h = h_gate *  expert_inputs.view(expert_inputs.shape[0], 2, -1)
        
        h = torch.sum(h,dim=1)
        # h  = h.view(expert_inputs.shape[0],-1)

        h = self.experts(h, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate[task_bh](x)
        
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, task_bh, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y

    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        # print('batch_index: ', batch_index)
        # print('expert_inputs: ', expert_inputs)
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y

class RandomMoE(MoE):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, *args, **kwargs):
        super(RandomMoE, self).__init__(*args, **kwargs)
        del self.f_gate
        del self.w_noise

    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = torch.randn(x.shape[0], self.num_experts).to(x.device)
        clean_logits = clean_logits * 0.001 + 1 # make the weight similar
        logits = clean_logits

        if skip_mask is not None:
            probs = torch.masked_fill(
                torch.softmax(logits, dim=1), skip_mask, 0)
        else:
            probs = torch.softmax(logits, dim=1)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
            
        # top_k_gates = top_k_gates / \
        #     (top_k_gates.sum(dim=1, keepdim=True) + 1e-6).detach()
        
        zeros = torch.zeros_like(probs, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        self.expert_size = (gates > 0).long().sum(0)

        top_k_gates = top_k_gates.flatten()
        top_k_experts = top_k_indices.flatten()
        
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]

        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
        self.index_sorted_experts = nonzeros[_index_sorted_experts]
        self.batch_index = self.index_sorted_experts.div(self.k, rounding_mode='trunc') 
        self.batch_gates = top_k_gates[self.index_sorted_experts]

        loss = 0
        # loss += self.cvloss * self.compute_cvloss(gates)
        # loss += self.switchloss * \
        #     self.compute_switchloss(probs, self.expert_size)
        # loss += self.zloss * self.compute_zloss(logits)
        return loss

class TaskTower(nn.Module):
    def __init__(self, input_dim, task_type):
        super().__init__()
        if task_type == "cls": 
            self.mlp = nn.Sequential(
                # nn.Linear(input_dim, hidden_dim),
                # nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(input_dim, 2)
            )
        else:
            self.mlp = nn.Sequential(
                            # nn.Linear(input_dim, hidden_dim),
                            # nn.ReLU(),
                            nn.Linear(input_dim, 1)
                        )
        # self.proj = nn.Linear(input_dim, input_dim)
    def forward(self, h):
        h = h.squeeze(1)
        y = self.mlp(h)
        return y
    
class MoEMTL(nn.Module):
    def __init__(self, config,task_types=['cls','cls','rec','rec','rec','rec','rec','rec'],noisy_gating=True):
        super().__init__()
        self.task_types = task_types
        self.dropout =  nn.Dropout(config.classifier_dropout)
        self.dropout2 =  nn.Dropout(config.classifier_dropout)
        self.towers = nn.ModuleDict({
            str(index): TaskTower(config.hidden_size, task) for index, task in enumerate(task_types)
        })
        self.towers1 = nn.ModuleDict({
            str(index): TaskTower(config.hidden_size, task) for index, task in enumerate(task_types)
        })
        self.towers2 = nn.ModuleDict({
            str(index): TaskTower(config.hidden_size, task) for index, task in enumerate(task_types)
        })
        self.num_experts = 2
        self.noisy_gating = noisy_gating
        # self.f_gate = nn.ModuleList([nn.Sequential(
        #             nn.Linear(config.hidden_size*2, config.hidden_size),
        #             nn.GELU(),
        #             nn.Linear(config.hidden_size,
        #                         2 * self.num_experts if self.noisy_gating else self.num_experts,
        #                         bias=False)
        #         ) for i in range(len(task_types))])
        self.f_gate = nn.ModuleList([nn.Sequential(
                nn.Linear(config.hidden_size*2, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size,
                            2 * self.num_experts*config.hidden_size if self.noisy_gating else self.num_experts*config.hidden_size,
                            bias=False)
            ) for i in range(len(task_types))])
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, mri_h, snp_h, task,mode):
        mri_h = self.dropout(mri_h)
        snp_h =  self.dropout2(snp_h)

        y1 = self.towers1[str(task)](snp_h)

        y2 = self.towers2[str(task)](mri_h)
      
        P_t = self.expert_gating(torch.cat([snp_h,mri_h],dim=1),task)
        # h = P_t[:,0].unsqueeze(1) * snp_h + P_t[:,1].unsqueeze(1) * mri_h
        h = P_t[:,:,0] * snp_h + P_t[:,:,1] * mri_h

        y = self.towers[str(task)](h)

        loss = torch.zeros(1).to(h.device)
        return y1,y2,y, loss

    def expert_gating(self, x, task_bh, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        B,C = x.shape
        clean_logits = self.f_gate[task_bh](x)
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        probs = torch.softmax(logits.reshape(B,C//2,2), dim=-1) + 1e-4
        
        return probs

class MoEMTL2(nn.Module):
    def __init__(self, config, num_experts=16,keep_experts=2,task_types=['cls','cls','rec','rec','rec','rec','rec','rec']):
        super().__init__()
        self.task_types = task_types
        self.moe = TaskMoE2(config.hidden_size, config.hidden_size, num_experts, keep_experts)
        self.dropout =  nn.Dropout(config.classifier_dropout)
        self.dropout2 =  nn.Dropout(config.classifier_dropout)
        self.towers = nn.ModuleDict({
            str(index): TaskTower(config.hidden_size, task) for index, task in enumerate(task_types)
        })
        self.towers1 = nn.ModuleDict({
            str(index): TaskTower(config.hidden_size, task) for index, task in enumerate(task_types)
        })
        self.towers2 = nn.ModuleDict({
            str(index): TaskTower(config.hidden_size, task) for index, task in enumerate(task_types)
        })
        
    def forward(self, mri_h, snp_h, task,mode):
        mri_h = self.dropout(mri_h)
        snp_h =  self.dropout2(snp_h)

        y1 = self.towers1[str(task)](snp_h)

        y2 = self.towers2[str(task)](mri_h)

        # if mode == 1:
        #     zero_target = random.choice(['mri', 'snp'])
        #     if zero_target == 'mri':
        #         mri_h = torch.zeros_like(mri_h).to(mri_h.device)
        #     else:
        #         snp_h = torch.zeros_like(snp_h).to(snp_h.device)

        h = torch.cat([snp_h,mri_h],dim=1).unsqueeze(1)
        
        h, loss = self.moe(h, task)
        
        y = self.towers[str(task)](h)

        loss = torch.zeros(1).to(h.device)
        return y1,y2,y, loss

class MaskcomputeMoE(nn.Module):
    def __init__(self, config=None,num_experts=100,token_num=100,keep_experts=2, noisy_gating=True, noisy_gating2=False,task_types=['cls','cls','rec','rec','rec','rec','rec','rec']):
        super().__init__()
        self.config = config
        self.num_experts = num_experts  #
        self.token_capcity = np.ceil(token_num / num_experts).astype(np.int64) #
        self.k = keep_experts
        self.fc1 = nn.Linear(config.hidden_size, self.num_experts)

        self.task_num = len(task_types)
        # self.f_gate = nn.ModuleList([nn.Sequential(
        #                     nn.Linear(config.hidden_size, config.hidden_size),
        #                     nn.GELU(),
        #                     nn.Linear(config.hidden_size,
        #                                 2 * num_experts if noisy_gating else num_experts,
        #                                 bias=False)
        #                 ) for i in range(len(task_types))])
        self.f_gate = nn.Sequential(
                            nn.Linear(self.task_num, config.hidden_size),
                            nn.GELU(),
                            nn.Linear(config.hidden_size,
                                        2 * self.num_experts if noisy_gating else self.num_experts,
                                        bias=False))
        
        self.f_gate_token = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size,
                      2 * self.num_experts if noisy_gating2 else self.num_experts,
                      bias=False)) 
        self.experts = ParallelExperts(self.num_experts,config.hidden_size, config.hidden_size, False)
        self.output_experts = ParallelExperts(self.num_experts,config.hidden_size, config.hidden_size, False)
        self.activation = nn.GELU()

        self.noisy_gating = noisy_gating
        self.noisy_gating2 = noisy_gating2
        self.acc_aux_loss = False
        self.cvloss = 0.01
        self.switchloss = 0
        self.zloss = 0
        self.epoch = 0

        position = torch.arange(config.max_position_embeddings).unsqueeze(1)    
        div_term = torch.exp(
            torch.arange(0, config.hidden_size, 2) * (-math.log(10000.0) / config.hidden_size)
        )                                                
        pe = torch.zeros(config.max_position_embeddings, config.hidden_size) 
        pe[:, 0::2] = torch.sin(position * div_term)   
        pe[:, 1::2] = torch.cos(position * div_term)          
        self.position_embeddings = pe # nn.Parameter(pe)  #
    
    def apply_rope(self, x):
        bsz, seqlen, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

        # 生成频率张量
        half_dim = head_dim // 2
        theta = 12000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(x.device) #12000
        seq_idx = torch.arange(seqlen, dtype=torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # [seq_len, half_dim]

        # 转换为 cos/sin 编码
        sin = freqs.sin()[None, :, :]  # shape: [1, seq_len, half_dim]
        cos = freqs.cos()[None, :, :]

        # 拆分并旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated

    def forward(self, input_features=None, input_features2=None, G=None, I=None, task_index=0, multiply_by_gates=True, mode="share"):
        B,L,C = input_features.shape
        if mode == "share":
            k = self.token_capcity #np.ceil(L / self.num_experts).astype(np.int64)
            # pos_features = self.apply_rope(torch.ones([1,L,C]).to(input_features))
            pos_features = self.position_embeddings[0:L,:].unsqueeze(0).to(input_features)
            if self.training:
                G,I,loss = self.top_k_gating_token(pos_features,k=k)
                G = G.repeat(B,1,1)
                I = I.repeat(B,1,1)
                return G,I,loss
            else:
                G,I = self.top_k_gating_token(pos_features,k=k)
                G = G.repeat(B,1,1)
                I = I.repeat(B,1,1)
                return G,I
        else:
            loss, P_t = self.top_k_gating(input_features2,task_index)
            expert_index = [[i for j in range(self.expert_size[i].item())] for i in range(self.expert_size.shape[0]) if self.expert_size[i].item() > 0]
            expert_index = torch.Tensor(sum(expert_index,[])).to(self.expert_size)
            
            h_gate =  G[self.batch_index,expert_index,:].unsqueeze(2)
            h = h_gate * input_features[self.batch_index]
            h = h.view(-1,C)
            
            # h = self.experts(h, self.expert_size*L)
            # h = self.activation(h)
            # expert_outputs = self.output_experts(h, self.expert_size*L).view(-1,L,C)
            expert_outputs = h.view(-1,L,C)

            if multiply_by_gates:
                expert_outputs = expert_outputs * self.batch_gates[:, None, None]

            zeros = torch.zeros((B,L,C), 
                                dtype=expert_outputs.dtype, device=expert_outputs.device)
            y = zeros.index_add(0, self.batch_index, expert_outputs)
            
            expert_mask = P_t.sum(dim=0).unsqueeze(0)#
            # expert_mask = (self.expert_size).unsqueeze(0)# 
           
            token_mask = G[self.batch_index,expert_index,:].sum(dim=0).unsqueeze(0)
     
            return y, expert_mask, token_mask

    def top_k_gating(self, x, task_bh, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        # probs = F.one_hot(torch.Tensor([task_bh]).long(), num_classes=self.task_num).float().repeat(x.shape[0],1).to(x)
        # clean_logits = self.f_gate(probs)
        clean_logits = self.f_gate[task_bh](x).view(x.shape[0],-1)
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        probs = torch.softmax(logits, dim=1) + 1e-4

        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        thresh = top_k_gates[..., -1, None]
        mask = (probs >= thresh).to(x.dtype)
        expert_mask = probs * mask

        # top_k_indecis: [batch, K]
       
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, task_bh)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss, expert_mask

    def top_k_gating_token(self, x, noise_epsilon=1e-2,k=100):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate_token(x)
        
        if self.noisy_gating2 and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating2:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        # probs = torch.softmax(logits, dim=-1) + 1e-4 
        
        tau0,tau_min, alpha = 1.0, 0.1, 0.95
        tau = max(tau_min, tau0 * (alpha**self.epoch))
        # tau = 0.1
        if self.training:
            probs = F.gumbel_softmax(logits, dim=-1,hard=False,tau=tau) + 1e-4
        else:        
            probs = F.gumbel_softmax(logits, dim=-1,hard=True,tau=tau) + 1e-4
        
        indices = torch.argmax(logits, dim=-1)  # (B, L)
        probs_oh = F.one_hot(indices, num_classes=logits.size(-1)).float()  # (B, L, N)
        G,I = probs.permute([0,2,1]),probs_oh.permute([0,2,1])

        # G,I = self.topk_keep(probs.permute([0,2,1]), k) #probs.permute([0,2,1])#
        
        if self.training:
            # m = probs.mean(dim=1)
            # u = torch.full_like(m,1.0/probs.size(-1))
            # loss = torch.mean((m-u)**2) * 100
            loss = 1.0 *self.compute_cvloss(probs)
            return G,I,loss
        else:
            return G,I

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def topk_keep(self, x: torch.Tensor, k: int):
        topk_vals, topk_index = torch.topk(x, k, dim=-1, largest=True)
        # mask = torch.zeros_like(x,dtype=torch.bool)
        # mask.scatter_(-1,topk_index,True)
        # mask = mask.to(x.dtype) 
        thresh = topk_vals[..., -1, None]
        mask = (x >= thresh).to(x.dtype)          # (B, L) 中 0 / 1
        return  x * mask, mask + x - x.detach() # 

class MaskcomputeMoE2(nn.Module):
    def __init__(self, config=None,num_experts=100,token_num=100,keep_experts=2,noisy_gating=True,task_types=['cls','cls','rec','rec','rec','rec','rec','rec']):
        super().__init__()
        self.config = config
        self.num_experts = num_experts  #
        self.token_capcity = np.ceil(token_num / num_experts).astype(np.int64) #
        self.k = keep_experts
        self.fc1 = nn.Linear(config.hidden_size, self.num_experts)

        self.task_num = len(task_types)
        # self.f_gate = nn.ModuleList([nn.Sequential(
        #                     nn.Linear(config.hidden_size, config.hidden_size),
        #                     nn.GELU(),
        #                     nn.Linear(config.hidden_size,
        #                                 2 * num_experts if noisy_gating else num_experts,
        #                                 bias=False)
        #                 ) for i in range(len(task_types))])
        self.f_gate = nn.Sequential(
                            nn.Linear(self.task_num, config.hidden_size),
                            nn.GELU(),
                            nn.Linear(config.hidden_size,
                                        2 * self.num_experts if noisy_gating else self.num_experts,
                                        bias=False))
        
        self.f_gate_token = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size,
                      2 * self.num_experts if noisy_gating else self.num_experts,
                      bias=False)) 
        self.experts = ParallelExperts(self.num_experts,config.hidden_size, config.hidden_size, False)
        self.output_experts = ParallelExperts(self.num_experts,config.hidden_size, config.hidden_size, False)
        self.activation = nn.GELU()

        self.noisy_gating = True
        self.acc_aux_loss = False
        self.cvloss = 0.01
        self.switchloss = 0
        self.zloss = 0
        self.epoch = 0

        position = torch.arange(config.max_position_embeddings).unsqueeze(1)    
        div_term = torch.exp(
            torch.arange(0, config.hidden_size, 2) * (-math.log(10000.0) / config.hidden_size)
        )                                                
        pe = torch.zeros(config.max_position_embeddings, config.hidden_size) 
        pe[:, 0::2] = torch.sin(position * div_term)   
        pe[:, 1::2] = torch.cos(position * div_term)          
        self.position_embeddings = pe # nn.Parameter(pe)  #
    
    def apply_rope(self, x):
        bsz, seqlen, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

        # 生成频率张量
        half_dim = head_dim // 2
        theta = 12000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(x.device) #12000
        seq_idx = torch.arange(seqlen, dtype=torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # [seq_len, half_dim]

        # 转换为 cos/sin 编码
        sin = freqs.sin()[None, :, :]  # shape: [1, seq_len, half_dim]
        cos = freqs.cos()[None, :, :]

        # 拆分并旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated

    def forward(self, input_features=None, input_features2=None, P=None, task_index=0, multiply_by_gates=True, mode="share"):
        B,L,C = input_features.shape
        if mode == "share":
            k = self.token_capcity #np.ceil(L / self.num_experts).astype(np.int64)
            pos_features = self.apply_rope(torch.ones([1,L,C]).to(input_features))
            # pos_features = self.position_embeddings[0:L,:].unsqueeze(0).to(input_features)
            P = self.top_k_gating_token(pos_features,k=k).repeat(B,1,1)
            return P
        else:
            loss, P_t = self.top_k_gating(input_features,task_index)
            expert_index = [[i for j in range(self.expert_size[i].item())] for i in range(self.expert_size.shape[0]) if self.expert_size[i].item() > 0]
            expert_index = torch.Tensor(sum(expert_index,[])).to(self.expert_size)
            
            h_gate =  P[self.batch_index,expert_index,:].unsqueeze(2)
            h = h_gate * input_features[self.batch_index]
            h = h.view(-1,C)
            
            # h = self.experts(h, self.expert_size*L)
            # h = self.activation(h)
            # expert_outputs = self.output_experts(h, self.expert_size*L).view(-1,L,C)
            expert_outputs = h.view(-1,L,C)

            if multiply_by_gates:
                expert_outputs = expert_outputs * self.batch_gates[:, None, None]

            zeros = torch.zeros((B,L,C), 
                                dtype=expert_outputs.dtype, device=expert_outputs.device)
            y = zeros.index_add(0, self.batch_index, expert_outputs)
            
            # if multiply_by_gates:
            #     expert_mask = P_t.sum(dim=0).unsqueeze(0)#
            #     token_mask = (P[self.batch_index,expert_index,:] * self.batch_gates.unsqueeze(1)).sum(dim=0).unsqueeze(0)
            # else:
            #     expert_mask = (self.expert_size).unsqueeze(0)# 
            #     token_mask = P[self.batch_index,expert_index,:].sum(dim=0).unsqueeze(0)
            
            expert_mask = (self.expert_size).unsqueeze(0)# 
            token_mask = P[self.batch_index,expert_index,:].sum(dim=0).unsqueeze(0)
     
            return y,expert_mask,token_mask

    def top_k_gating(self, x, task_bh, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        probs = F.one_hot(torch.Tensor([task_bh]).long(), num_classes=self.task_num).float().repeat(x.shape[0],1).to(x)
        clean_logits = self.f_gate(probs)
        # clean_logits = self.f_gate[task_bh](x).view(x.shape[0],-1)
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        probs = torch.softmax(logits, dim=1) + 1e-4

        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        thresh = top_k_gates[..., -1, None]
        mask = (probs >= thresh).to(x.dtype)
        expert_mask = probs * mask

        # top_k_indecis: [batch, K]
       
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, task_bh)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss, expert_mask

    def top_k_gating_token(self, x, noise_epsilon=1e-2,k=100):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate_token(x)
        
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        # probs = torch.softmax(logits, dim=-1) + 1e-4 
        
        # tau0,tau_min, alpha = 1.0, 0.1, 0.95
        # tau = max(tau_min, tau0 * (alpha**self.epoch))
        tau = 0.1
        if self.training:
            probs = F.gumbel_softmax(logits, dim=-1,hard=False,tau=tau) + 1e-4
        else:        
            probs = F.gumbel_softmax(logits, dim=-1,hard=True,tau=tau) + 1e-4

        # indices = torch.argmax(logits, dim=-1)  # (B, L)
        # probs = F.one_hot(indices, num_classes=logits.size(-1)).float()  # (B, L, N)

        G,I = self.topk_keep(probs.permute([0,2,1]), k) #probs.permute([0,2,1])#
        
        return G,I

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def topk_keep(self, x: torch.Tensor, k: int):
        topk_vals, topk_index = torch.topk(x, k, dim=-1, largest=True)
        oh = F.one_hot(topk_index,num_classes=x.size(-1)).to(torch.bool)
        mask = oh.any(dim=-2).to(x)
        # thresh = topk_vals[..., -1, None]
        # mask = (x >= thresh).to(x.dtype)          # (B, L) 中 0 / 1
        return  x * mask, mask + x - x.detach() # 

class MaskcomputeMoE3(nn.Module):
    def __init__(self, config=None,token_num=100,num_experts=100,keep_experts=2,noisy_gating=True,task_types=['cls','cls','rec','rec','rec','rec','rec','rec']):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.k = keep_experts
        self.fc1 =nn.Linear(config.hidden_size, self.num_experts)
        self.f_gate = nn.ModuleList([nn.Sequential(
                            nn.Linear(config.hidden_size, config.hidden_size),
                            nn.GELU(),
                            nn.Linear(config.hidden_size,
                                        2 * num_experts if noisy_gating else num_experts,
                                        bias=False)
                        ) for i in range(len(task_types))])
        self.f_gate_token = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size,
                      2 * num_experts if noisy_gating else num_experts,
                      bias=False)) 
        self.experts = ParallelExperts(num_experts,config.hidden_size, config.hidden_size, False)
        self.output_experts = ParallelExperts(num_experts,config.hidden_size, config.hidden_size, False)
        self.activation = nn.GELU()

        self.noisy_gating = True
        self.acc_aux_loss = False
        self.cvloss = 0.01
        self.switchloss = 0
        self.zloss = 0
        self.training = False
    
    def forward(self, input_features=None, expert_outputs=None, input_features2=None, task_index=0, mode="share", multiply_by_gates=True):
        if mode == "share":
            B,L,C = input_features.shape
            pos_features = self.apply_rope(torch.ones_like(input_features).to(input_features))
            P = self.top_k_gating_token(pos_features,self.k)

            expert_size = torch.Tensor([B]*self.num_experts).to(input_features).long()
            batch_index = torch.Tensor([i for i in range(B)]*self.num_experts).to(input_features).long()
            expert_index = torch.Tensor(sum([[i]*B for i in range(self.num_experts)],[])).to(input_features).long()

            h_gate =  P[batch_index,expert_index,:].unsqueeze(2)
            h = h_gate * input_features[batch_index]
            h = h.view(-1,C)
            
            # expert_size = expert_size*L
            # h = self.experts(h, expert_size)
            # h = self.activation(h)
            # expert_outputs = self.output_experts(h, expert_size).view(-1,L,C)
            expert_outputs = h.view(-1,L,C)
            return expert_outputs, P
        else:
            BB,L,C = expert_outputs.shape
            B = BB // self.num_experts

            if multiply_by_gates:
                # P_t = self.expert_gating(input_features,task_index)
                # batch_index = torch.Tensor([i for i in range(B)]*self.num_experts).to(input_features).long()
                # expert_index = torch.Tensor(sum([[i]*B for i in range(self.num_experts)],[])).to(input_features).long()
                # batch_gates = P_t[batch_index,:,expert_index].unsqueeze(2)
                # expert_outputs = expert_outputs * batch_gates

                P_t = self.expert_gating(input_features2,task_index)
                batch_index = torch.Tensor([i for i in range(B)]*self.num_experts).to(input_features).long()
                expert_index = torch.Tensor(sum([[i]*B for i in range(self.num_experts)],[])).to(input_features).long()
                batch_gates = P_t[batch_index,expert_index]
                expert_outputs = expert_outputs * batch_gates[:, None, None]
            else:
                P_t = torch.ones([B,self.num_experts]).to(expert_outputs)

            batch_index = torch.Tensor([i for i in range(B)]*self.num_experts).to(expert_outputs).long()
            zeros = torch.zeros((B,L,C), 
                                dtype=expert_outputs.dtype, device=expert_outputs.device)
            y = zeros.index_add(0, batch_index, expert_outputs)
            
            return y, P_t

    def expert_gating(self, x, task_bh, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate[task_bh](x)
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        probs = torch.softmax(logits, dim=-1) + 1e-4
        
        return probs

    def top_k_gating_token(self, x, noise_epsilon=1e-2,k=100):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate_token(x)
        if self.noisy_gating and self.training:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        
        # probs = torch.softmax(logits, dim=-1) + 1e-4
        if self.training:
            # probs = torch.softmax(logits, dim=-1) + 1e-4
            probs = F.gumbel_softmax(logits, dim=-1,hard=False,tau=0.5) + 1e-4
        else:
            probs = F.gumbel_softmax(logits, dim=-1,hard=True,tau=0.5) + 1e-4

        mask = self.topk_keep(probs.permute([0,2,1]), k)#probs.permute([0,2,1])#
        
        return mask

    def apply_rope(self, x):
        bsz, seqlen, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

        # 生成频率张量
        half_dim = head_dim // 2
        theta = 12000 ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(x.device) #12000
        seq_idx = torch.arange(seqlen, dtype=torch.float32).to(x.device)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # [seq_len, half_dim]

        # 转换为 cos/sin 编码
        sin = freqs.sin()[None, :, :]  # shape: [1, seq_len, half_dim]
        cos = freqs.cos()[None, :, :]

        # 拆分并旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def topk_keep(self, x: torch.Tensor, k: int):
        topk_vals, topk_index = torch.topk(x, k, dim=-1, largest=True)
        thresh = topk_vals[..., -1, None]
        mask = (x >= thresh).to(x.dtype)          # (B, L) 中 0 / 1
        return x * mask #mask + x - x.detach() # 
    
class CLIPMoE(nn.Module):
    def __init__(self, temperature=0.07,feature_num=512,task_types=['cls','cls','rec','rec','rec','rec','rec','rec']):
        super(CLIPMoE, self).__init__()
        self.image_proj = nn.ModuleDict({
            str(index): nn.Linear(feature_num, 512) for index, task in enumerate(task_types)
        })
        self.snp_proj = nn.ModuleDict({
            str(index): nn.Linear(feature_num, 512) for index, task in enumerate(task_types)
        })
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, image_features=None, snp_features=None, mask=None, age_sex=None,group=False, task_index=None):
        image_features = image_features.view(image_features.size(0), -1)   
        image_features = self.image_proj[str(task_index)](image_features)
       
        snp_features = snp_features.view(snp_features.size(0), -1)
        snp_features = self.snp_proj[str(task_index)](snp_features)

        clip_scores = F.cosine_similarity(image_features, snp_features)
        # normalized features
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        snp_features_norm = snp_features / snp_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features_norm @ snp_features_norm.t()

        if group:
            loss_img = torch.sum(-1.0 * F.log_softmax(logits, dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss_snp = torch.sum(-1.0 * F.log_softmax(logits.t(), dim=0) * mask  / (torch.sum(mask,dim=0,keepdim=True)+1e-12) ) / logits.shape[0]
            loss = loss_img + loss_snp
        else:
            # 计算对角线元素（正样本）的索引
            labels = torch.arange(logits.shape[0], device=logits.device)
            # 计算损失
            loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

        return loss / 2,clip_scores,image_features,snp_features
    
    def forward2(self, image_features=None, snp_features=None,age_sex=None, task_index=None):
        image_features = image_features.view(image_features.size(0), -1)   
        image_features = self.image_proj[str(task_index)](image_features)
       
        snp_features = snp_features.view(snp_features.size(0), -1)
        snp_features = self.snp_proj[str(task_index)](snp_features)
        return image_features,snp_features

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# from options.train_options import TrainOptions   
# if __name__ == "__main__":
#     opt = TrainOptions().parse()
#     model = MaskcomputeMoE(opt,num_experts=opt.mri_num_experts,keep_experts=opt.mri_keep_experts).cuda()
#     input = torch.ones([2,1200,512]).cuda()
#     input2 = torch.ones([2,512]).cuda()
#     mask, expert_gate, diversity_loss = model(input,input2)
