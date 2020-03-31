# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 22:09:37
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import torch
import torch.nn.functional as F

from graphs import MixPoetAUS

from tool import Tool
from beam import PoetryBeam
from filter import PoetryFilter
from config import hparams, device
import utils

class Generator(object):
    '''
    generator for testing
    '''

    def __init__(self):
        self.tool = Tool(hparams.sens_num, hparams.key_len,
            hparams.sen_len, hparams.poem_len, 0.0)
        self.tool.load_dic(hparams.vocab_path, hparams.ivocab_path)
        vocab_size = self.tool.get_vocab_size()
        print ("vocabulary size: %d" % (vocab_size))
        PAD_ID = self.tool.get_PAD_ID()
        B_ID = self.tool.get_B_ID()
        assert vocab_size > 0 and PAD_ID >=0 and B_ID >= 0
        self.hps = hparams._replace(vocab_size=vocab_size, pad_idx=PAD_ID, bos_idx=B_ID)


        # load model
        model = MixPoetAUS(self.hps)

        # load trained model
        utils.restore_checkpoint(self.hps.model_dir, device, model)
        self.model = model.to(device)
        self.model.eval()

        #utils.print_parameter_list(self.model)

        # load poetry filter
        print ("loading poetry filter...")
        self.filter = PoetryFilter(self.tool.get_vocab(),
            self.tool.get_ivocab(), self.hps.data_dir)
        print("--------------------------")



    def generate_one(self, keyword, length, factor_label1, factor_label2,
        beam_size=20, verbose=1, manu=False):
        '''
        generate one poem according to the inputs:
            keyword: a topic word
            length: the length of each line, 5 or 7
            factor_label: label for the two factors, when factor_label = -1,
                the model infers an appropriate class in terms of the keyword
            verbose: 0, 1, 2, 3
        '''
        assert length == 5 or length == 7
        key_state = self.get_key_state(keyword)

        # infer labels when factor_label = -1
        if factor_label1 == -1:
            factor_label1 = self.get_inferred_factor(key_state, 1)
            if verbose >= 1:
                print ("inferred label1: %d" % (factor_label1[0].item()))
        else:
            factor_label1 = torch.tensor([factor_label1],
                dtype=torch.long, device=device)

        if factor_label2 == -1:
            factor_label2 = self.get_inferred_factor(key_state, 2)
            if verbose >= 1:
                print ("inferred label2: %d" % (factor_label2[0].item()))
        else:
            factor_label2 = torch.tensor([factor_label2],
                dtype=torch.long, device=device)

        # get decoder initial state

        dec_init_state = self.get_dec_init_state(
            key_state, factor_label1, factor_label2, length)


        context = torch.zeros((1, self.hps.context_size),
            dtype=torch.float, device=device) # (B, context_size)

        # initialize beam pool
        beam_pool = PoetryBeam(beam_size, length,
            self.tool.get_B_ID(), self.tool.get_E_ID(), self.tool.get_UNK_ID(),
            self.filter.get_level_cids(), self.filter.get_oblique_cids())


        # beam search
        poem = []
        self.filter.reset(length, verbose)
        for step in range(0, self.hps.sens_num):
            # generate each line
            if verbose >= 1:
                print ("\ngenerating step: %d" % (step))

            # get the rhythm pattern and rhyme id of te current line
            _, rhythms, rhyme = self.filter.get_pattern(step)

            #
            pos_tensor = self.tool.pos2tensor(step)
            pos_tensor = pos_tensor.to(device)

            init_state = torch.cat([dec_init_state.clone().detach(), pos_tensor], dim=-1)

            # reset beam pool
            beam_pool.reset(init_state, rhythms, rhyme,
                self.filter.get_rhyme_cids(rhyme), self.filter.get_repetitive_ids())

            candidates, costs, states = self.beam_search(beam_pool, length, context)

            lines = [self.tool.idxes2line(idxes) for idxes in candidates]

            lines, costs, states = self.filter.filter_illformed(lines, costs, states, step)

            if len(lines) == 0:
                return "", "generation failed!"

            which = 0
            if manu:
                for i, (line, cost) in enumerate(zip(lines, costs)):
                    print ("%d, %s, %.2f" % (i, line, cost))
                which = int(input("select sentence>"))

            line = lines[which]
            poem.append(line)

            # the first line determin the rhythm pattern of the poem
            if step == 0:
                self.filter.set_pattern(line)

            # when the first line doesn't rhyme, we can determin the rhyme
            #   in terms of the second line
            if step == 1 and self.filter.get_rhyme() == -1:
                self.filter.set_rhyme(line)

            # set repetitive chars
            self.filter.add_repetitive(self.tool.line2idxes(line))

            # update the context vector
            context = self.update_context(context, states[which], beam_size, length)


        return poem, "ok"


    # ------------------------------------
    def beam_search(self, beam_pool, trg_len, ori_context):

        # current size of beam candidates in the beam pool
        n_samples = beam_pool.uncompleted_num()
        # (1, context_size) -> (B, context_size)
        context = ori_context.repeat(n_samples, 1)

        for k in range(0, trg_len+10):
            #print ("beam search position %d" % (k))
            inps, states = beam_pool.get_beam_tails()

            logits, new_states = self.do_dec_step(inps, states, context[:n_samples, :])

            beam_pool.advance(logits, new_states, k)

            n_samples = beam_pool.uncompleted_num()

            if n_samples == 0:
                break

        candidates, costs, dec_states = beam_pool.get_search_results()
        return candidates, costs, dec_states

    # ---------------------------
    def get_key_state(self, keyword):
        #print (keyword)
        key_tensor = self.tool.keys2tensor([keyword])
        #print (key_tensor)
        return self.model.compute_key_state(key_tensor.to(device))


    def get_inferred_factor(self, key_state, factor_id):
        assert factor_id == 1 or factor_id == 2
        return self.model.compute_inferred_label(key_state, factor_id)


    def get_dec_init_state(self, key_state, label1, label2, length):
        state = self.model.compute_dec_init_state(key_state, label1, label2)

        # length tensor
        len_tensor = self.tool.lengths2tensor([length]).to(device)
        dec_init_state = torch.cat([state, len_tensor], dim=-1) # (1, H)
        return dec_init_state

    def do_dec_step(self, inps, states, context):
        logits, new_state = self.model.dec_step(inps, states, context)
        return logits, new_state


    def update_context(self, old_context, ori_states, beam_size, length):
        # update the context vector
        # old_context: (1, context_size)
        # states: (1, H) * L
        H = ori_states[0].size(1)

        states = ori_states[1:length+2] + [torch.zeros_like(ori_states[0], device=device)] * (7-length)

        states = [state.view(H, 1) for state in states]

        states = torch.cat(states, dim=1).unsqueeze(0) # (1, H, L)

        context = self.model.layers['context'](old_context, states)


        return context