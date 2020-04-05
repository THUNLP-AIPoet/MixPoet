# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-04-05 09:33:36
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import math
import random
from itertools import chain
import torch
from torch import nn
import torch.nn.functional as F

from layers import BidirEncoder, Decoder, MLP, ContextLayer, GumbelSampler
from layers import Discriminator, PriorGenerator, PosterioriGenerator

from config import device

def get_non_pad_mask(seq, pad_idx):
    # seq: [B, L]
    assert seq.dim() == 2
    # [B, L]
    mask = seq.ne(pad_idx).type(torch.float)
    return mask.to(device)


def get_seq_length(seq, pad_idx):
    mask = get_non_pad_mask(seq, pad_idx)
    # mask: [B, T]
    lengths = mask.sum(dim=-1)
    lengths = lengths.type(torch.long)
    return lengths


class MixPoetAUS(nn.Module):
    def __init__(self, hps):
        super(MixPoetAUS, self).__init__()
        self.hps = hps

        self.vocab_size = hps.vocab_size
        self.n_class1 = hps.n_class1
        self.n_class2 = hps.n_class2
        self.emb_size = hps.emb_size
        self.hidden_size = hps.hidden_size
        self.factor_emb_size = hps.factor_emb_size
        self.latent_size = hps.latent_size
        self.context_size = hps.context_size
        self.poem_len = hps.poem_len
        self.sens_num = hps.sens_num
        self.sen_len = hps.sen_len

        self.pad_idx = hps.pad_idx
        self.bos_idx = hps.bos_idx

        self.bos_tensor = torch.tensor(hps.bos_idx, dtype=torch.long, device=device).view(1, 1)

        self.gumbel_tool = GumbelSampler()

        # build postional inputs to distinguish lines at different positions
        # [sens_num, sens_num], each line is a one-hot input
        self.pos_inps = F.one_hot(torch.arange(0, self.sens_num), self.sens_num)
        self.pos_inps = self.pos_inps.type(torch.FloatTensor).to(device)


        # ----------------------------
        # build componets
        self.layers = nn.ModuleDict()
        self.layers['embed'] = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.pad_idx)

        self.layers['encoder'] = BidirEncoder(self.emb_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        # p(x|z, w, y)
        self.layers['decoder'] = Decoder(self.hidden_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        # RNN to combine characters to form the representation of a word
        self.layers['word_encoder'] = BidirEncoder(self.emb_size, self.emb_size, cell='Elman',
            drop_ratio=hps.drop_ratio)

        # p(y_1|x,w), p(y_2|x,w)
        self.layers['cl_xw1'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size, 128, self.n_class1], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)
        self.layers['cl_xw2'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size, 128, self.n_class2], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)

        # p(y_1|w), p(y_2|w)
        self.layers['cl_w1'] = MLP(self.emb_size*2,
            layer_sizes=[self.emb_size, 64, self.n_class1], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)
        self.layers['cl_w2'] = MLP(self.emb_size*2,
            layer_sizes=[self.emb_size, 64, self.n_class2], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)

        # factor embedding
        self.layers['factor_embed1'] = nn.Embedding(self.n_class1, self.factor_emb_size)
        self.layers['factor_embed2'] = nn.Embedding(self.n_class2, self.factor_emb_size)

        # posteriori and prior
        self.layers['prior'] = PriorGenerator(
            self.emb_size*2+int(self.latent_size//2),
            self.latent_size, self.n_class1, self.n_class2, self.factor_emb_size)

        self.layers['posteriori'] = PosterioriGenerator(
            self.hidden_size*2+self.emb_size*2, self.latent_size,
            self.n_class1, self.n_class2, self.factor_emb_size)


        # for adversarial training
        self.layers['discriminator'] = Discriminator(self.n_class1, self.n_class2,
            self.factor_emb_size, self.latent_size, drop_ratio=hps.drop_ratio)

        #--------------
        # project the decoder hidden state to a vocanbulary-size output logit
        self.layers['out_proj'] = nn.Linear(hps.hidden_size, hps.vocab_size)

        # MLP for calculate initial decoder state
        # NOTE: Here we use a two-dimension one-hot vector as the input length embedding o_i,
        #   since there are only two kinds of line length, 5 chars and 7 chars, for Chinese
        #   classical quatrains.
        self.layers['dec_init'] = MLP(self.latent_size+self.emb_size*2+self.factor_emb_size*2,
            layer_sizes=[self.hidden_size-6],
            activs=['tanh'], drop_ratio=hps.drop_ratio)



        self.layers['map_x'] = MLP(self.context_size+self.emb_size,
            layer_sizes=[self.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        # update the context vector
        self.layers['context'] = ContextLayer(self.hidden_size, self.context_size)


        # two annealing parameters
        self.__tau = 1.0
        self.__teach_ratio = 1.0

        # only for pre-training
        self.layers['dec_init_pre'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size-6],
            activs=['tanh'], drop_ratio=hps.drop_ratio)


    #---------------------------------
    def set_tau(self, tau):
        self.gumbel_tool.set_tau(tau)

    def get_tau(self):
        return gumbel_tool.get_tau()

    def set_teach_ratio(self, teach_ratio):
        if 0.0 < teach_ratio <= 1.0:
            self.__teach_ratio = teach_ratio

    def get_teach_ratio(self):
        return self.__teach_ratio

    #---------------------------------
    def dec_step(self, inp, state, context):
        emb_inp = self.layers['embed'](inp)

        x = self.layers['map_x'](torch.cat([emb_inp, context.unsqueeze(1)], dim=2))

        cell_out, new_state = self.layers['decoder'](x, state)
        out = self.layers['out_proj'](cell_out)
        return out, new_state


    def generator(self, dec_init_state, dec_inps, lengths, specified_teach=None):
        # the decoder p(x|z, w, y)
        # initialize the context vector
        batch_size = dec_init_state.size(0)
        context = torch.zeros((batch_size, self.context_size),
            dtype=torch.float, device=device) # (B, context_size)

        all_outs = []
        if specified_teach is None:
            teach_ratio = self.__teach_ratio
        else:
            teach_ratio = specified_teach

        for step in range(0, self.sens_num):
            pos_inps = self.pos_inps[step, :].unsqueeze(0).repeat(batch_size, 1)

            state = torch.cat([dec_init_state, lengths, pos_inps], dim=-1) # (B, H)
            max_dec_len = dec_inps[step].size(1)

            outs = torch.zeros(batch_size, max_dec_len, self.vocab_size, device=device)
            dec_states = []

            # generate each line
            inp = self.bos_tensor.expand(batch_size, 1)
            for t in range(0, max_dec_len):
                out, state = self.dec_step(inp, state, context)
                outs[:, t, :] = out

                # teach force with a probability
                is_teach = random.random() < teach_ratio
                if is_teach or (not self.training):
                    inp = dec_inps[step][:, t].unsqueeze(1)
                else:
                    normed_out = F.softmax(out, dim=-1)
                    top1 = normed_out.data.max(1)[1]
                    inp  = top1.unsqueeze(1)

                dec_states.append(state.unsqueeze(2)) # (B, H, 1)

            # save each generated line
            all_outs.append(outs)

            # update the context vector
            # (B, 1, L)
            dec_mask = get_non_pad_mask(dec_inps[step], self.pad_idx).unsqueeze(1)
            states = torch.cat(dec_states, dim=2) # (B, H, L)
            context = self.layers['context'](context, states*dec_mask)

        return all_outs


    def computer_enc(self, inps, encoder):
        lengths = get_seq_length(inps, self.pad_idx)

        emb_inps = self.layers['embed'](inps) # (batch_size, length, emb_size)

        enc_outs, enc_state = encoder(emb_inps, lengths)

        return enc_outs, enc_state


    def get_factor_emb(self, condition, factor_id, label, mask):
        # -----------------------------------
        # sample labels for unlabelled poems from the classifier
        logits_cl = self.layers['cl_xw'+str(factor_id)](condition)
        sampled_label = self.gumbel_tool(logits_cl)

        fin_label = label.float() * mask + (1-mask) * sampled_label
        fin_label = fin_label.long()

        factor_emb = self.layers['factor_embed'+str(factor_id)](fin_label)

        return factor_emb, logits_cl, fin_label


    def get_prior_and_posterior(self, key_inps, vae_inps, factor_labels,
        factor_mask, ret_others=False):
        # get the representation of a whole poem
        _, vae_state = self.computer_enc(vae_inps, self.layers['encoder']) # (2, B, H)
        sen_state = torch.cat([vae_state[0, :, :], vae_state[1, :, :]], dim=-1) # [B, 2*H]

        # get the representation of the keyword
        # TODO: incorporate multiple keywords
        _, key_state0 = self.computer_enc(key_inps, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        condition = torch.cat([sen_state, key_state], dim=1)
        # get embedding of either provided or sampled labels
        factor_emb1, logits_cl_xw1, combined_label1 = self.get_factor_emb(condition, 1,
            factor_labels[:, 0], factor_mask[:, 0])
        factor_emb2, logits_cl_xw2, combined_label2 = self.get_factor_emb(condition, 2,
            factor_labels[:, 1], factor_mask[:, 1])

        factors = torch.cat([factor_emb1, factor_emb2], dim=-1)


        # get posteriori p(z|x,w,y)
        batch_size = key_state.size(0)
        eps = torch.randn((batch_size, self.latent_size), dtype=torch.float, device=device)
        z_post = self.layers['posteriori'](sen_state, key_state, combined_label1, combined_label2)
        z_prior = self.layers['prior'](key_state, combined_label1, combined_label2)

        if ret_others:

            return z_prior, z_post, key_state, factors,\
                logits_cl_xw1, logits_cl_xw2, combined_label1, combined_label2
        else:
            return z_prior, z_post, combined_label1, combined_label2


    def forward(self, key_inps, vae_inps, dec_inps, factor_labels, factor_mask, lengths,
        use_prior=False, specified_teach=None):
        # key_inps: (B, ken_len)
        # vae_inps: (B, poem_len)
        # dec_inps: (B, sen_len) * sens_num
        # factor_labels: (B, number of factors), here (B, 2)
        # factor_mask: (B, 2)
        # lengths: (B, 2): [0, 1] for 5-char quatrains and [1, 0] for 7-char quatrains

        z_prior, z_post, key_state, factors, logits_cl_xw1, logits_cl_xw2, cb_label1, cb_label2\
            = self.get_prior_and_posterior(key_inps, vae_inps, factor_labels, factor_mask, True)

        if use_prior:
            z = z_prior
        else:
            z = z_post

        # generate a poem line by line
        dec_init_state = self.layers['dec_init'](torch.cat([z, key_state, factors], dim=-1)) # (B, H-2)
        all_gen_outs = self.generator(dec_init_state, dec_inps, lengths, specified_teach)

        # for classifier loss
        logits_cl_w1 = self.layers['cl_w1'](key_state)
        logits_cl_w2 = self.layers['cl_w2'](key_state)

        return all_gen_outs, cb_label1, cb_label2, \
            logits_cl_xw1, logits_cl_xw2, logits_cl_w1, logits_cl_w2

    # --------------------------
    # graphs for pre-training
    def classifier_graph(self, keys, poems, factor_id):
        _, poem_state0 = self.computer_enc(poems, self.layers['encoder'])
        poem_state = torch.cat([poem_state0[0, :, :], poem_state0[1, :, :]], dim=-1) # [B, 2*H]

        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        condition = torch.cat([poem_state, key_state], dim=-1)

        logits_w = self.layers['cl_w'+str(factor_id)](key_state)
        logits_xw = self.layers['cl_xw'+str(factor_id)](condition)

        probs_w = F.softmax(logits_w, dim=-1)
        probs_xw = F.softmax(logits_xw, dim=-1)


        return logits_xw, logits_w, probs_xw, probs_w


    def dae_graph(self, keys, poems, dec_inps, lengths):
        # pre-train the encoder and decoder as a denoising AutoEncoder
        _, poem_state0 = self.computer_enc(poems, self.layers['encoder'])
        poem_state = torch.cat([poem_state0[0, :, :], poem_state0[1, :, :]], dim=-1) # [B, 2*H]

        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        dec_init_state = self.layers['dec_init_pre'](torch.cat([poem_state, key_state], dim=-1))

        all_gen_outs = self.generator(dec_init_state, dec_inps, lengths)

        return all_gen_outs



    # ----------------------------------------------
    def dae_parameter_names(self):
        required_names = ['embed', 'encoder', 'word_encoder',
            'dec_init_pre', 'decoder', 'out_proj', 'context', 'map_x']
        return required_names

    def dae_parameters(self):
        names = self.dae_parameter_names()

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)


    # -------------------------------------
    def classifier_parameter_names(self, factor_id):
        assert factor_id == 1 or factor_id == 2
        if factor_id == 1:
            required_names = ['embed', 'encoder', 'word_encoder',
                'cl_w1', 'cl_xw1']
        else:
            required_names = ['embed', 'encoder', 'word_encoder',
            'cl_w2', 'cl_xw2']
        return required_names

    def cl_parameters(self, factor_id):
        names = self.classifier_parameter_names(factor_id)

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)


    # ---------------------------------------------
    # for adversarial training
    def rec_parameters(self):
        # parameters of the classifiers, recognition network and decoder
        names = ['embed', 'encoder', 'decoder', 'word_encoder',
            'cl_xw1', 'cl_xw2', 'cl_w1', 'cl_w2',
            'factor_embed1', 'factor_embed2', 'posteriori',
            'out_proj', 'dec_init', 'context', 'map_x']

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)


    def dis_parameters(self):
        # parameters of the discriminator
        return self.layers['discriminator'].parameters()


    def gen_parameters(self):
        # parameters of the recognition network and prior network
        names = ['prior', 'posteriori', 'encoder', 'word_encoder',
            'embed']

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)


    # ------------------------------------------------
    # functions for generating
    def compute_key_state(self, keys):
        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]
        return key_state

    def compute_inferred_label(self, key_state, factor_id):
        logits = self.layers['cl_w'+str(factor_id)](key_state)
        probs = F.softmax(logits, dim=-1)
        pred = probs.max(dim=-1)[1] # (B)
        return pred

    def compute_dec_init_state(self, key_state, labels1, labels2):
        z_prior = self.layers['prior'](key_state, labels1, labels2)

        factor_emb1 = self.layers['factor_embed1'](labels1)
        factor_emb2 = self.layers['factor_embed2'](labels2)

        dec_init_state = self.layers['dec_init'](
            torch.cat([z_prior, key_state, factor_emb1, factor_emb2], dim=-1)) # (B, H-2)

        return dec_init_state


    def compute_prior(self, keys, labels1, labels2):
        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]


        z_prior = self.layers['prior'](key_state, labels1, labels2)

        return z_prior