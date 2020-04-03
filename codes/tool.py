# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 22:40:26
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import pickle
import numpy as np
import random
import copy
import torch


def readPickle(data_path):
    corpus_file = open(data_path, 'rb')
    corpus = pickle.load(corpus_file)
    corpus_file.close()

    return corpus

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Tool(object):
    '''
    a tool to hold training data and the vocabulary
    '''
    def __init__(self, sens_num, key_len,
        sen_len, poem_len, corrupt_ratio=0):
        # corrupt ratio for dae
        self.sens_num = sens_num
        self.key_len = key_len
        self.sen_len = sen_len
        self.poem_len = poem_len
        self.corrupt_ratio = corrupt_ratio

        self.__vocab = None
        self.__ivocab = None

        self.__PAD_ID = None
        self.__B_ID = None
        self.__E_ID = None
        self.__UNK_ID = None

    # -----------------------------------
    # map functions
    def idxes2line(self, idxes, truncate=True):
        if truncate and self.__E_ID in idxes:
            idxes = idxes[:idxes.index(self.__E_ID)]

        tokens = self.idxes2tokens(idxes, truncate)
        line = self.tokens2line(tokens)
        return line

    def line2idxes(self, line):
        tokens = self.line2tokens(line)
        return self.tokens2idxes(tokens)

    def line2tokens(self, line):
        '''
        in this work, we treat each Chinese character as a token.
        '''
        line = line.strip()
        tokens = [c for c in line]
        return tokens


    def tokens2line(self, tokens):
        return "".join(tokens)


    def tokens2idxes(self, tokens):
        ''' Characters to idx list '''
        idxes = []
        for w in tokens:
            if w in self.__vocab:
                idxes.append(self.__vocab[w])
            else:
                idxes.append(self.__UNK_ID)
        return idxes


    def idxes2tokens(self, idxes, omit_special=True):
        tokens = []
        for idx in idxes:
            if  (idx == self.__PAD_ID or idx == self.__B_ID
                or idx == self.__E_ID) and omit_special:
                continue
            tokens.append(self.__ivocab[idx])

        return tokens

    # -------------------------------------------------
    def greedy_search(self, probs):
        # probs: (B, L, V)
        outidx = [int(np.argmax(prob, axis=-1)) for prob in probs]
        #print (outidx)
        if self.__E_ID in outidx:
            outidx = outidx[:outidx.index(self.__E_ID)]

        tokens = self.idxes2tokens(outidx)
        return self.tokens2line(tokens)

    # ----------------------------
    def get_vocab(self):
        return copy.deepcopy(self.__vocab)

    def get_ivocab(self):
        return copy.deepcopy(self.__ivocab)

    def get_vocab_size(self):
        if self.__vocab:
            return len(self.__vocab)
        else:
            return -1

    def get_PAD_ID(self):
        assert self.__PAD_ID is not None
        return self.__PAD_ID

    def get_B_ID(self):
        assert self.__B_ID is not None
        return self.__B_ID

    def get_E_ID(self):
        assert self.__E_ID is not None
        return self.__E_ID

    def get_UNK_ID(self):
        assert self.__UNK_ID is not None
        return self.__UNK_ID


    # ----------------------------------------------------------------
    def load_dic(self, vocab_path, ivocab_path):
        dic = readPickle(vocab_path)
        idic = readPickle(ivocab_path)

        assert len(dic) == len(idic)


        self.__vocab = dic
        self.__ivocab = idic

        self.__PAD_ID = dic['PAD']
        self.__UNK_ID = dic['UNK']
        self.__E_ID = dic['<E>']
        self.__B_ID = dic['<B>']


    def build_data(self, train_data_path, valid_data_path, batch_size, mode):
        '''
        Build data as batches.
        NOTE: Please run load_dic() at first.
        mode:
            cl1 or cl2: pre-train the classifier for factor 1 or factor 2
            dae: pre-train the encoder and decoder as a denoising AutoEncoder
            mixpoet_pre: train mixpoet with both labelled and unlabelled data
            mixpoet_tune: fine-tune mixpoet with only labelled data
        '''
        assert mode in ['cl1', 'cl2', 'dae', 'mixpoet_pre', 'mixpoet_tune']
        train_data = readPickle(train_data_path)
        valid_data = readPickle(valid_data_path)

        print (len(train_data))
        print (len(valid_data))

        # data limit for debug
        self.train_batches = self.__build_data_core(train_data, batch_size, mode)
        self.valid_batches = self.__build_data_core(valid_data, batch_size, mode)

        self.train_batch_num = len(self.train_batches)
        self.valid_batch_num = len(self.valid_batches)


    def __build_data_core(self, data, batch_size, mode, data_limit=None):
        # data: [keyword, sens, label0, label1] * data_num
        if data_limit is not None:
            data = data[0:data_limit]

        if mode == 'cl1' or mode == 'cl2':
            return self.build_classifier_batches(data, batch_size, mode)
        elif mode == 'dae':
            return self.build_dae_batches(data, batch_size)
        elif mode == 'mixpoet_pre' or mode == 'mixpoet_tune':
            return self.build_mixpoet_batches(data, batch_size, mode)


    def build_classifier_batches(self, ori_data, batch_size, mode):
        # extract data for a specified factor
        data = []
        for instance in ori_data:
            if mode == 'cl1':
                label = instance[2]
            elif mode == 'cl2':
                label = instance[3]
            if label == -1:
                continue
            assert label >= 0
            # [keyword, sens, label]
            data.append((instance[0], instance[1], label))

        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            # build poetry batch
            poems = [instance[1] for instance in instances] # all poems
            sequences = [sum(poem, []) for poem in poems]
            batch_poems = self.__build_batch_seqs(sequences, True, True)

            # build keyword batch
            keys = [instance[0] for instance in instances] # keywords
            batch_keys = self.__build_batch_seqs(keys, True, True)

            # label batch
            labels = [instance[2] for instance in instances]
            batch_labels = torch.tensor(labels, dtype=torch.long)

            batched_data.append((batch_keys, batch_poems, batch_labels))

        return batched_data


    def build_dae_batches(self, data, batch_size):
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            # build poetry batch
            poems = [instance[1] for instance in instances] # all poems
            sequences = [sum(poem, []) for poem in poems]
            batch_poems = self.__build_batch_seqs(sequences, True, True, corrupt=True)

            # build keyword batch
            keys = [instance[0] for instance in instances] # keys
            batch_keys = self.__build_batch_seqs(keys, True, True)

            # build each line
            batch_dec_inps = []
            for step in range(0, self.sens_num):
                lines = [poem[step] for poem in poems]
                # NOTE: for Chinese quatrains, all lines in a poem share the same length
                batch_lengths = self.__build_batch_length(lines)
                batch_lines = self.__build_batch_seqs(lines, False, True)
                batch_dec_inps.append(batch_lines)

            batched_data.append((batch_keys, batch_poems, batch_dec_inps, batch_lengths))


        return batched_data


    def build_mixpoet_batches(self, ori_data, batch_size, mode):
        # extract data for pre-training or fine-tuning
        if mode == 'mixpoet_pre':
            data = ori_data
        else:
            data = []
            # for fine-tuning, we filter out unlabelled poems
            for instance in ori_data:
                label1 = instance[2]
                label2 = instance[3]
                if label1 == -1 and label2 == -1:
                    continue
                data.append(instance)


        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            # build poetry batch
            poems = [instance[1] for instance in instances] # all poems
            sequences = [sum(poem, []) for poem in poems]
            batch_poems = self.__build_batch_seqs(sequences, True, True)

            # build keyword batch
            keys = [instance[0] for instance in instances] # keys
            batch_keys = self.__build_batch_seqs(keys, True, True)

            # build line batch
            batch_dec_inps = []
            for step in range(0, self.sens_num):
                lines = [poem[step] for poem in poems]

                # NOTE: for Chinese quatrains, all lines in a poem share the same length
                batch_lengths = self.__build_batch_length(lines)
                batch_lines = self.__build_batch_seqs(lines, False, True)
                batch_dec_inps.append(batch_lines)

            # build labels
            labels = [(instance[2], instance[3]) for instance in instances]
            label_mask = [(float(pair[0] != -1), float(pair[1] != -1)) for pair in labels]

            batch_labels = torch.tensor(labels, dtype=torch.long)
            batch_label_mask = torch.tensor(label_mask, dtype=torch.float)

            batched_data.append((batch_keys, batch_poems, batch_dec_inps,
                batch_labels, batch_label_mask, batch_lengths))


        return batched_data


    def __build_batch_length(self, lines):
        # TODO: cover more kinds of length
        lengths = []
        for line in lines:
            yan = len(line)
            assert yan == 5 or yan == 7
            if yan == 5:
                lengths.append([0.0, 1.0])
            else:
                lengths.append([1.0, 0.0])

        batch_lengths = torch.tensor(lengths, dtype=torch.float)
        return batch_lengths


    def __build_batch_seqs(self, instances, with_B, with_E, corrupt=False):
        # pack sequences as a tensor
        seqs = self.__get_batch_seq(instances, with_B, with_E, corrupt)
        seqs_tensor =self.__sens2tensor(seqs)
        return seqs_tensor


    def __get_batch_seq(self, seqs, with_B, with_E, corrupt):
        batch_size = len(seqs)
        max_len = max([len(seq) for seq in seqs])
        max_len = max_len + int(with_B) + int(with_E)

        batched_seqs = []
        for i in range(0, batch_size):
            # max length for each sequence
            ori_seq = copy.deepcopy(seqs[i])

            if corrupt:
                seq = self.__do_corruption(ori_seq)
            else:
                seq = ori_seq
            # ----------------------------------

            pad_size = max_len - len(seq) - int(with_B) - int(with_E)
            pads = [self.__PAD_ID] * pad_size

            new_seq = [self.__B_ID] * int(with_B) + seq\
                + [self.__E_ID] * int(with_E) + pads

            batched_seqs.append(new_seq)

        return batched_seqs


    def __sens2tensor(self, sens):
        batch_size = len(sens)
        sen_len = max([len(sen) for sen in sens])
        tensor = torch.zeros(batch_size, sen_len, dtype=torch.long)
        for i, sen in enumerate(sens):
            for j, token in enumerate(sen):
                tensor[i][j] = token
        return tensor


    def __do_corruption(self, inp):
        # corrupt the sequence by setting some tokens as UNK
        m = int(np.ceil(len(inp) * self.corrupt_ratio))
        m = min(m, len(inp))

        unk_id = self.get_UNK_ID()

        corrupted_inp = copy.deepcopy(inp)
        pos = random.sample(list(range(0, len(inp))), m)
        for p in pos:
            corrupted_inp[p] = unk_id

        return corrupted_inp



    def shuffle_train_data(self):
        random.shuffle(self.train_batches)

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # Tools for beam search
    def keys2tensor(self, keys):
        key_idxes = []
        for key in keys:
            tokens = self.line2tokens(key)
            idxes = self.tokens2idxes(tokens)
            key_idxes.append([self.__B_ID] + idxes + [self.__E_ID])
        return self.__sens2tensor(key_idxes)


    def lengths2tensor(self, lengths):
        vec = []
        for length in lengths:

            assert length == 5 or length == 7
            if length == 5:
                vec.append([0.0, 1.0])
            else:
                vec.append([1.0, 0.0])

        batch_lengths = torch.tensor(vec, dtype=torch.float)
        return batch_lengths


    def pos2tensor(self, step):
        assert step in list(range(0, self.sens_num))
        pos = [0.0] * self.sens_num
        pos[step] = 1.0


        pos_tensor = torch.tensor([pos], dtype=torch.float)
        return pos_tensor
