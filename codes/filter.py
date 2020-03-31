# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi and Jiannan Liang
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-30 21:29:37
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import pickle
import random
import copy
import os

from rhythm_tool import RhythmRecognizer

class PoetryFilter(object):

    def __init__(self, vocab, ivocab, data_dir):
        self.__vocab = vocab
        self.__ivocab = ivocab

        '''
        rhythm patterns.
        for Chinese quatrains, we generalize four main poem-level patterns
        '''
        self.__RHYTHM_TYPES = [[0, 1, 3, 2], [1, 2, 0, 1], [2, 1, 3, 2], [3, 2, 0, 1]]

        '''
        we genelize four main line-level rhythm patterns for 5-char line and 7-char line respectively.
        0: level tone (ping); 1: oblique tone (ze); -1: either
        '''
        self.__RHYTHM_PATTERNS = {7: [[-1, 1, -1, 0, 0, 1, 1], [-1, 0, -1, 1, 1, 0, 0],
            [-1, 1, 0, 0, 1, 1, 0], [-1, 0, -1, 1, 0, 0, 1]],
            5: [[-1, 0, 0, 1, 1], [-1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [-1, 1, 0, 0, 1]]}

        self.__rhythm_tool = RhythmRecognizer(data_dir+"pingsheng.txt", data_dir+"zesheng.txt")

        self.__load_rhythm_dic(data_dir+"pingsheng.txt", data_dir+"zesheng.txt")
        self.__load_rhyme_dic(data_dir+"pingshui.txt", data_dir+"pingshui_amb.pkl")
        self.__load_line_lib(data_dir+"training_lines.txt")


    def __load_line_lib(self, data_path):
        self.__line_lib = {}

        with open(data_path, 'r') as fin:
            lines = fin.readlines()

        for line in lines:
            line = line.strip()
            self.__line_lib[line] = 1

        print ("  line lib loaded, %d lines" % (len(self.__line_lib)))


    def __load_rhythm_dic(self, level_path, oblique_path):
        with open(level_path, 'r') as fin:
            level_chars = fin.read()

        with open(oblique_path, 'r') as fin:
            oblique_chars = fin.read()

        self.__level_list = []
        self.__oblique_list = []
        # convert char to id
        for char, idx in self.__vocab.items():
            if char in level_chars:
                self.__level_list.append(idx)

            if char in oblique_chars:
                self.__oblique_list.append(idx)

        print ("  rhythm dic loaded, level tone chars: %d, oblique tone chars: %d" %\
            (len(self.__level_list), len(self.__oblique_list)))


    #------------------------------------------
    def __load_rhyme_dic(self, rhyme_dic_path, rhyme_disamb_path):

        self.__rhyme_dic = {} # char id to rhyme category ids
        self.__rhyme_idic = {} # rhyme category id to char ids

        with open(rhyme_dic_path, 'r') as fin:
            lines = fin.readlines()

        amb_count = 0
        for line in lines:
            (char, rhyme_id) = line.strip().split(' ')
            if char not in self.__vocab:
                continue
            char_id = self.__vocab[char]
            rhyme_id = int(rhyme_id)

            if not char_id in self.__rhyme_dic:
                self.__rhyme_dic.update({char_id:[rhyme_id]})
            elif not rhyme_id in self.__rhyme_dic[char_id]:
                self.__rhyme_dic[char_id].append(rhyme_id)
                amb_count += 1

            if not rhyme_id in self.__rhyme_idic:
                self.__rhyme_idic.update({rhyme_id:[char_id]})
            else:
                self.__rhyme_idic[rhyme_id].append(char_id)

        print ("  rhyme dic loaded, ambiguous rhyme chars: %d" % (amb_count))

        # load data for rhyme disambiguation
        self.__ngram_rhyme_map = {} # rhyme id list of each bigram or trigram
        self.__char_rhyme_map = {} # the most likely rhyme id for each char
        # load the calculated data, if there is any
        #print (rhyme_disamb_path)
        assert rhyme_disamb_path is not None and os.path.exists(rhyme_disamb_path)

        with open(rhyme_disamb_path, 'rb') as fin:
            self.__char_rhyme_map = pickle.load(fin)
            self.__ngram_rhyme_map = pickle.load(fin)

            print ("  rhyme disamb data loaded, cached chars: %d, ngrams: %d"
                % (len(self.__char_rhyme_map), len(self.__ngram_rhyme_map)))


    def get_line_rhyme(self, line):
        """ we use statistics of ngram to disambiguate the rhyme category,
        but there is still risk of mismatching and ambiguity
        """
        tail_char = line[-1]

        if tail_char in self.__char_rhyme_map:
            bigram = line[-2] + line[-1]
            if bigram in self.__ngram_rhyme_map:
                return self.__ngram_rhyme_map[bigram]

            trigram = line[-3] + line[-2] + line[-1]
            if trigram in self.__ngram_rhyme_map:
                return self.__ngram_rhyme_map[trigram]

            return self.__char_rhyme_map[tail_char]

        if not tail_char in self.__vocab:
            return -1
        else:
            tail_id = self.__vocab[tail_char]


        if tail_id in self.__rhyme_dic:
            return self.__rhyme_dic[tail_id][0]

        return -1

    # ------------------------------
    def reset(self, length, verbose):
        assert length == 5 or length == 7
        self.__length = length
        self.__rhyme = -1
        self.__rhythm_vec = []
        self.__repetitive_ids = []
        self.__verbose = verbose


    def set_pattern(self, line):
        # set a rhythm pattern in terms of the first generated line
        assert len(line) == 5 or len(line) == 7
        rhythm_l1 = self.__rhythm_tool.get_rhythm(line)

        if self.__verbose >= 2:
            print ("set rhythm_id of l1: %d" % (rhythm_l1))

        # when the first line doesn't match any pattern,
        #   randomly select one
        if rhythm_l1 < 0:
            rhythm_l1 = random.sample([0,1,2,3], 1)[0]
            if self.__verbose >= 2:
                print ("sample rhythm_id of l1: %d" % (rhythm_l1))

        # pattern id of each line
        self.__rhythm_vec = self.__RHYTHM_TYPES[rhythm_l1]
        if self.__verbose >= 2:
            rhythm_str = " ".join([str(r) for r in self.__rhythm_vec])
            print ("set rhythm ids of all lines: %s" % (rhythm_str))


        # set rhyme in terms of the first line
        self.set_rhyme(line)


    def set_rhyme(self, line):
        rhyme = self.get_line_rhyme(line)
        if isinstance(rhyme, list) or isinstance(rhyme, tuple):
            rhyme = int(rhyme[0])
        else:
            rhyme = int(rhyme)
        if self.__verbose >= 2:
            print ("set rhyme id: %s" % (rhyme))
        self.__rhyme = rhyme


    def add_repetitive(self, ids):
        self.__repetitive_ids = list(set(ids+self.__repetitive_ids))


    # -------------------------------
    def get_pattern(self, step, pure=False):
        # before the first line is generated, rerutn
        #   empty patterns
        if len(self.__rhythm_vec) == 0:
            return -1, [], -1

        # return the pattern of the current line and the rhyme
        l_rhythm = self.__rhythm_vec[step]
        l_rhythm_pattern \
            = self.__RHYTHM_PATTERNS[self.__length][l_rhythm]

        # for Chinese classical quatrains, the third line doesn't rhyme
        rhyme = -1 if step == 2 else self.__rhyme

        #print (step, l_rhythm, rhyme)
        #print (type(step), type(l_rhythm), type(rhyme))

        if self.__verbose >= 2 and not pure:
            print ("step: %d, line rhythm id: %d, rhyme: %d" %
                (step, l_rhythm, rhyme))

        return l_rhythm, l_rhythm_pattern, rhyme


    def get_rhyme(self):
        return self.__rhyme


    def get_level_cids(self):
        return copy.deepcopy(self.__level_list)

    def get_oblique_cids(self):
        return copy.deepcopy(self.__oblique_list)

    def get_rhyme_cids(self, rhyme_id):
        if rhyme_id not in self.__rhyme_idic:
            return []
        else:
            return copy.deepcopy(self.__rhyme_idic[rhyme_id])

    def get_repetitive_ids(self):
        return copy.deepcopy(self.__repetitive_ids)


    def filter_illformed(self, lines, costs, states, step):
        if len(lines) == 0:
            return [], [], []

        new_lines = []
        new_costs = []
        new_states = []

        required_l_rhythm, _, _ = self.get_pattern(step, True)

        len_error = 0
        lib_count = 0
        rhythm_error = 0
        rhythm_mismatch = 0

        for i in range(len(lines)):
            #print (lines[i])
            if len(lines[i]) < self.__length:
                len_error += 1
                continue
            line = lines[i][0:self.__length]

            # we filter out the lines that already exist in the
            #   training set, to guarantee the novelty of generated poems
            if line in self.__line_lib:
                lib_count += 1
                continue

            rhythm_id = self.__rhythm_tool.get_rhythm(line)

            if rhythm_id < 0:
                rhythm_error += 1
                continue

            if required_l_rhythm != -1 and rhythm_id != required_l_rhythm:
                rhythm_mismatch += 1
                continue

            new_lines.append(line)
            new_costs.append(costs[i])
            new_states.append(states[i])


        if self.__verbose >= 3:
            print ("input lines: %d, ilter out %d illformed lines, %d remain"
                % (len(lines), len(lines)-len(new_lines), len(new_lines)))
            print ("%d len error, %d exist in lib, %d rhythm error, %d rhythm mismatch"
                % (len_error, lib_count, rhythm_error, rhythm_mismatch))

        return new_lines, new_costs, new_states
