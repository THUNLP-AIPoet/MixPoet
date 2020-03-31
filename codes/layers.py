# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 22:11:42
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import math
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm as SN

from config import device


class BidirEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        super(BidirEncoder, self).__init__()
        self.cell_type = cell
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size


        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)
        elif cell == 'Elman':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers,
                bidirectional=True, batch_first=True)

        self.dropout_layer = nn.Dropout(drop_ratio)


    def forward(self, embed_seq, input_lens=None):
        # embed_seq: (B, L, emb_dim)
        # input_lens: (B)
        embed_inps = self.dropout_layer(embed_seq)

        if input_lens is None:
            outputs, state = self.rnn(embed_inps, None)
        else:
            # Dynamic RNN
            total_len = embed_inps.size(1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embed_inps,
                input_lens, batch_first=True, enforce_sorted=False)
            outputs, state = self.rnn(packed, None)
            # outputs: (B, L, num_directions*H)
            # state: (num_layers*num_directions, B, H)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                batch_first=True, total_length=total_len)

        return outputs, state

    # bi-direction
    def init_state(self, batch_size):
        init_h = torch.zeros( (self.n_layers*2, batch_size,
            self.hidden_size), requires_grad=False, device=device)

        if self.cell_type == 'LSTM':
            init_c = torch.zeros( (self.n_layers*2, batch_size,
                self.hidden_size), requires_grad=False, device=device)
            return (init_h, init_c)
        else:
            return init_h


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        super(Decoder, self).__init__()

        self.dropout_layer = nn.Dropout(drop_ratio)

        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'Elman':
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)


    def forward(self, embed_seq, last_state):
        embed_inps = self.dropout_layer(embed_seq)
        output, state = self.rnn(embed_inps, last_state.unsqueeze(0))
        output = output.squeeze(1)  # (B, 1, N) -> (B,N)
        return output, state.squeeze(0) # (B, H)



class MLP(nn.Module):
    def __init__(self, ori_input_size, layer_sizes, activs=None,
        drop_ratio=0.0, no_drop=False):
        super(MLP, self).__init__()

        layer_num = len(layer_sizes)

        orderedDic = OrderedDict()
        input_size = ori_input_size
        for i, (layer_size, activ) in enumerate(zip(layer_sizes, activs)):
            linear_name = 'linear_' + str(i)
            orderedDic[linear_name] = nn.Linear(input_size, layer_size)
            input_size = layer_size

            if activ is not None:
                assert activ in ['tanh', 'relu', 'leakyrelu']

            active_name = 'activ_' + str(i)
            if activ == 'tanh':
                orderedDic[active_name] = nn.Tanh()
            elif activ == 'relu':
                orderedDic[active_name] = nn.ReLU()
            elif activ == 'leakyrelu':
                orderedDic[active_name] = nn.LeakyReLU(0.2)


            if (drop_ratio > 0) and (i < layer_num-1) and (not no_drop):
                orderedDic["drop_" + str(i)] = nn.Dropout(drop_ratio)

        self.mlp = nn.Sequential(orderedDic)


    def forward(self, inps):
        return self.mlp(inps)


class ContextLayer(nn.Module):
    def __init__(self, inp_size, out_size, kernel_size=3):
        super(ContextLayer, self).__init__()
        # (B, L, H)
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)
        self.linear = nn.Linear(out_size+inp_size, out_size)

    def forward(self, last_context, dec_states):
        # last_context: (B, context_size)
        # dec_states: (B, H, L)
        hidden_feature = self.conv(dec_states).permute(0, 2, 1) # (B, L_out, out_size)
        feature = torch.tanh(hidden_feature).mean(dim=1) # (B, out_size)
        new_context = torch.tanh(self.linear(torch.cat([last_context, feature], dim=1)))
        return new_context


class Discriminator(nn.Module):
    def __init__(self, n_class1, n_class2, factor_emb_size,
        latent_size, drop_ratio):
        super(Discriminator, self).__init__()
        # (B, L, H)
        self.inp2feature = nn.Sequential(
            SN(nn.Linear(latent_size, latent_size)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(latent_size, factor_emb_size*2)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(factor_emb_size*2, factor_emb_size*2)),
            nn.LeakyReLU(0.2))
        self.feature2logits = SN(nn.Linear(factor_emb_size*2, 1))

        # factor embedding for the discriminator
        self.dis_fembed1 = SN(nn.Embedding(n_class1, factor_emb_size))
        self.dis_fembed2 = SN(nn.Embedding(n_class2, factor_emb_size))


    def forward(self, x, labels1, labels2):
        # x, latent variable: (B, latent_size)

        femb1 = self.dis_fembed1(labels1)
        femb2 = self.dis_fembed2(labels2)
        factor_emb = torch.cat([femb1, femb2], dim=1) # (B, factor_emb_size*2)

        feature = self.inp2feature(x) # (B, factor_emb_size*2)
        logits0 = self.feature2logits(feature).squeeze(1) # (B)

        logits = logits0 + torch.sum(feature*factor_emb, dim=1)

        return logits



class CBNLayer(nn.Module):
    """Contitional BatchNorm Transform Layer"""
    def __init__(self, inp_size, out_size, n_classes):
        super(CBNLayer, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.n_classes = n_classes

        self.linear = nn.Linear(self.inp_size, self.out_size)
        self.cbn = CondtionalBatchNorm(self.n_classes, self.out_size)
        self.activ  = nn.LeakyReLU(0.2)


    def forward(self, inps):
        x, y = inps[0], inps[1]
        # x: (B, inp_size)
        h = self.linear(x)
        out = self.cbn(h, y)
        return (self.activ(out), y)


class PriorGenerator(nn.Module):
    def __init__(self, inp_size, latent_size, n_class1, n_class2, factor_emb_size):
        super(PriorGenerator, self).__init__()

        self.factor_embed1 = nn.Embedding(n_class1, factor_emb_size)
        self.factor_embed2 = nn.Embedding(n_class2, factor_emb_size)

        self.slatent_size = int(latent_size//2)
        self.mlp1 = nn.Sequential(
            CBNLayer(inp_size+factor_emb_size, latent_size, n_class1),
            CBNLayer(latent_size, latent_size, n_class1))
        self.bn1 = nn.Sequential(
            nn.Linear(latent_size, self.slatent_size),
            nn.BatchNorm1d(self.slatent_size))


        self.mlp2 = nn.Sequential(
            CBNLayer(inp_size+factor_emb_size, latent_size, n_class2),
            CBNLayer(latent_size, latent_size, n_class2))
        self.bn2 = nn.Sequential(
            nn.Linear(latent_size, self.slatent_size),
            nn.BatchNorm1d(self.slatent_size))


    def forward(self, key_state, labels1, labels2):
        factor1 = self.factor_embed1(labels1)
        factor2 = self.factor_embed2(labels2)


        batch_size = key_state.size(0)
        eps1 = torch.randn((batch_size, self.slatent_size),
            dtype=torch.float, device=device)
        cond1 = torch.cat([eps1, key_state, factor1], dim=1) #
        prior1 = self.mlp1((cond1, labels1))[0]
        #print (prior1.size())
        prior1 = self.bn1(prior1)

        eps2 = torch.randn((batch_size, self.slatent_size),
            dtype=torch.float, device=device)
        cond2 = torch.cat([eps2, key_state, factor2], dim=1) #
        prior2 = self.mlp2((cond2, labels2))[0]
        prior2 = self.bn2(prior2)

        return torch.cat([prior1, prior2], dim=1)


class PosterioriGenerator(nn.Module):
    def __init__(self, inp_size, latent_size, n_class1, n_class2, factor_emb_size):
        super(PosterioriGenerator, self).__init__()

        self.latent_size = latent_size

        self.post_embed1 = nn.Embedding(n_class1, factor_emb_size)
        self.post_embed2 = nn.Embedding(n_class2, factor_emb_size)

        self.mlp = MLP(
            inp_size+factor_emb_size*2+latent_size,
            layer_sizes=[512, latent_size, latent_size],
            activs=['leakyrelu', 'leakyrelu', None], no_drop=True)


    def forward(self, sen_state, key_state, labels1, labels2):
        factor1 = self.post_embed1(labels1)
        factor2 = self.post_embed2(labels2)

        batch_size = key_state.size(0)
        eps = torch.randn((batch_size, self.latent_size),
            dtype=torch.float, device=device)

        cond = torch.cat([sen_state, key_state, factor1, factor2, eps], dim=1) #
        z_post = self.mlp(cond)

        return z_post



class CondtionalBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1):
        super(CondtionalBatchNorm, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.delta_gamma = nn.Embedding(num_classes, num_features)
        self.delta_beta = nn.Embedding(num_classes, num_features)


        self.gamma = nn.Parameter(torch.Tensor(1, num_features), requires_grad = True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features), requires_grad = True)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()


    def reset_parameters(self):
        self.reset_running_stats()

        self.delta_gamma.weight.data.fill_(1)
        self.delta_beta.weight.data.zero_()

        self.gamma.data.fill_(1)
        self.beta.data.zero_()


    def forward(self, inps, labels):
        # inps: (B, D)
        # labels: (B)
        exp_avg_factor = 0.0
        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exp_avg_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exp_avg_factor = self.momentum

        B, D = inps.size(0), inps.size(1)

        mean = inps.mean(dim=0) # (D)
        variance = inps.var(dim=0) # (D)


        if self.training:
            running_mean = self.running_mean*(1-exp_avg_factor) + mean*exp_avg_factor
            running_var = self.running_var*(1-exp_avg_factor) + variance*exp_avg_factor

            mu = mean
            var = variance

            self.running_mean = running_mean
            self.running_var = running_var

        else:

            mu = self.running_mean
            var = self.running_var

        x = (inps - mu.view(1, D).repeat(B, 1)) / torch.sqrt(var.view(1, D).repeat(B, 1) + self.eps)

        delta_weight = self.delta_gamma(labels) # (B, D)
        delta_bias = self.delta_beta(labels)

        weight = self.gamma.repeat(B, 1) + delta_weight
        bias = self.beta.repeat(B, 1) + delta_bias

        out = weight * x + bias

        return out


class GumbelSampler(object):
    '''
    utilize gumbel softmax to return long type sampled labels
        instead of one-hot labels or soft
    '''
    def __init__(self):
        super(GumbelSampler, self).__init__()
        self.__tau =1.0

    def set_tau(self, tau):
        if 0.0 < tau <= 1.0:
            self.__tau = tau

    def get_tau(self):
        return self.__tau


    def __call__(self, logits):
        # this part is just borrowed from the official implementation
        y_soft = F.gumbel_softmax(logits, tau=self.__tau, hard=False)

        y_hard = y_soft.max(dim=-1, keepdim=True)[1] # [B, n_class]

        y_hard = (y_hard.float() - y_soft).detach() + y_soft

        return y_hard[:, 0]

#-------------------------------------
#-------------------------------------
class LossWrapper(object):
    def __init__(self, pad_idx, sens_num, sen_len):
        self.__sens_num = sens_num
        self.__sen_len = sen_len
        self.___pad_idx = pad_idx

        self.__gen_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.__cl_criterion = torch.nn.CrossEntropyLoss(reduction='none')


    def seq_ce_loss(self, outs, trgs):
        vocab_size = outs.size(2)
        trg_max_len = trgs.size(1)

        output = outs[:, 0:trg_max_len, :].contiguous().view(-1, vocab_size)
        target = trgs.contiguous().view(-1)
        return self.__gen_criterion(output, target)


    def cross_entropy_loss(self, all_outs, all_trgs):
        #print ("cross_entropy_loss!")
        # all_outs: (B, L, V) * sens_num
        # all_trgs: (B, L) * sens_num
        batch_size, vocab_size = all_outs[0].size(0), all_outs[0].size(2)

        all_loss = []
        for step in range(0, self.__sens_num):

            all_loss.append(self.seq_ce_loss(all_outs[step], all_trgs[step]).unsqueeze(0))

        rec_loss = torch.cat(all_loss, dim=0)
        rec_loss = torch.mean(rec_loss) # (sens_num)

        return rec_loss


    def bow_loss(self, bow_logits, all_trgs):
        # bow_logits: (B, V)
        # all_trgs: (B, L) * sens_num
        all_loss = []
        all_loss2 = []
        for step in range(0, self.__sens_num):
            line_loss = []
            trgs = all_trgs[step]
            max_dec_len = trgs.size(1)
            for i in range(0, max_dec_len):
                all_loss.append(self.__gen_criterion(bow_logits, trgs[:, i]).unsqueeze(0))

        all_loss = torch.cat(all_loss, dim=0)
        bow_loss = all_loss.mean() # [B, T, sens_num]
        return bow_loss


    def cl_loss(self, logits_w, logits_xw, combined_label, mask):
        '''
        (1) with labelled poems, both q(y|x,w) and p(y|w) are optimized
        (2) with unlabelled poems, q(y|x,w) is optimized with the entropy loss, H(q(y|x,w)),
            p(y|w) is optimized with the fake labels sampled from q(y|x,w)
        to sum up,
            p(y|w) is optimized with true and fake labels,
            q(y|x, w) is optimized with true labels and the entrop loss
        '''
        cl_loss_w = self.__cl_criterion(logits_w, combined_label).mean() # (B) -> (1)

        entropy_loss_xw = self.__get_entropy(logits_xw, 1-mask)

        #print (mask)

        cl_loss_xw = self.__cl_criterion(logits_xw, combined_label) * mask # (B)
        cl_loss_xw = cl_loss_xw.sum() / (mask.sum()+1e-10)

        return cl_loss_w, cl_loss_xw, entropy_loss_xw


    def __get_entropy(self, logits, mask):
        # logits: (B, n_class)
        # mask: (B)
        # the entrop loss is only applied to unlabelled factors
        probs = F.softmax(logits, dim=-1)
        entropy = torch.log(probs+1e-10) * probs # (B, n_class)

        # we need to  maximize the entropy term, that is,
        #   minimize the entropy loss (negative entropy)
        entropy_loss = entropy.mean(dim=-1) * mask # (B)

        entropy_loss = entropy_loss.sum() / (mask.sum()+1e-10)

        return entropy_loss

#-----------------------------------------------------------------
#-----------------------------------------------------------------
class ScheduledOptim(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, warmup_steps, max_lr=5e-4, min_lr=3e-5, beta=0.55):
        self.__optimizer = optimizer

        self._step = 0
        self._rate = 0

        self.__warmup_steps = warmup_steps
        self.__max_lr = max_lr
        self.__min_lr = min_lr

        self.__alpha = warmup_steps**(-beta-1.0)
        self.__beta = -beta

        self.__scale = 1.0 / (self.__alpha*warmup_steps)


    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.__optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.__optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        lr = self.__max_lr*self.__scale*min(step*self.__alpha, step**(self.__beta))
        if step > self.__warmup_steps:
            lr = max(lr, self.__min_lr)
        return lr

    def zero_grad(self):
        self.__optimizer.zero_grad()

    def state_dict(self):
        return self.__optimizer.state_dict()

    def load_state_dict(self, dic):
        self.__optimizer.load_state_dict(dic)


#---------------------------------------------------
class RateDecay(object):
    '''Basic class for different types of rate decay,
        e.g., teach forcing ratio, gumbel temperature,
        KL annealing.
    '''
    def __init__(self, burn_down_steps, decay_steps, limit_v):

        self.step = 0
        self.rate = 1.0

        self.burn_down_steps = burn_down_steps
        self.decay_steps = decay_steps

        self.limit_v = limit_v


    def decay_funtion(self):
        # to be reconstructed
        return self.rate


    def do_step(self):
        # update rate
        self.step += 1
        if self.step > self.burn_down_steps:
            self.rate = self.decay_funtion()

        return self.rate


    def get_rate(self):
        return self.rate


class ExponentialDecay(RateDecay):
    def __init__(self, burn_down_steps, decay_steps, min_v):
        super(ExponentialDecay, self).__init__(
            burn_down_steps, decay_steps, min_v)

        self.__alpha = np.log(self.limit_v)/ (-decay_steps)

    def decay_funtion(self):
        new_rate = max(np.exp(-self.__alpha*self.step), self.limit_v)
        return new_rate


class InverseLinearDecay(RateDecay):
    def __init__(self, burn_down_steps, decay_steps, max_v):
        super(InverseLinearDecay, self).__init__(
            burn_down_steps, decay_steps, max_v)

        self.__alpha = (max_v-0.0) / decay_steps

    def decay_funtion(self):
        new_rate = min(self.__alpha * self.step, self.limit_v)
        return new_rate


class LinearDecay(RateDecay):
    def __init__(self, burn_down_steps, decay_steps, max_v, min_v):
        super(LinearDecay, self).__init__(
            burn_down_steps, decay_steps, min_v)

        self.__max_v = max_v
        self.__alpha = (self.__max_v-min_v) / decay_steps

    def decay_funtion(self):
        new_rate = max(self.__max_v-self.__alpha * self.step, self.limit_v)
        return new_rate
