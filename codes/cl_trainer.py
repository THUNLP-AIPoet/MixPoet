# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-06 23:35:15
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import numpy as np
import random

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from layers import ScheduledOptim
from config import device
from logger import ClassifierLogger
import utils

class ClassifierTrainer(object):

    def __init__(self, hps):
        self.hps = hps

    def run_validation(self, epoch, mixpoet, tool, lr):
        logger = ClassifierLogger('valid')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_path(self.hps.cl_valid_log_path)

        for step in range(0, tool.valid_batch_num):

            batch = tool.valid_batches[step]
            batch_keys = batch[0].to(device)
            batch_poems = batch[1].to(device)
            batch_labels = batch[2].to(device)

            loss_xw, loss_w, preds_xw, preds_w  = \
                self.run_step(mixpoet, None,
                    batch_keys, batch_poems, batch_labels, True)
            logger.add_losses(loss_xw, loss_w, preds_xw, preds_w,
                batch_labels.cpu().numpy())

        logger.print_log(epoch)


    def run_step(self, mixpoet, optimizer, keys, poems, labels, valid=False):
        if not valid:
            optimizer.zero_grad()
        logits_xw, logits_w, probs_xw, probs_w = \
            mixpoet.classifier_graph(keys, poems, self.__factor_id)

        loss_xw = self.criterion(logits_xw, labels)
        loss_w = self.criterion(logits_w, labels)

        loss = (loss_xw+loss_w).mean()

        preds_xw = probs_xw.argmax(dim=-1)
        preds_w = probs_w.argmax(dim=-1)

        if not valid:
            loss.backward()
            clip_grad_norm_(mixpoet.cl_parameters(self.__factor_id),
                self.hps.clip_grad_norm)
            optimizer.step()

        return loss_xw.mean().item(), loss_w.mean().item(),\
            preds_xw.detach().cpu(), preds_w.detach().cpu()


    def run_train(self, mixpoet, tool, optimizer, logger):
        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            batch = tool.train_batches[step]
            batch_keys = batch[0].to(device)
            batch_poems = batch[1].to(device)
            batch_labels = batch[2].to(device)

            loss_xw, loss_w, preds_xw, preds_w  = \
                self.run_step(mixpoet, optimizer,
                    batch_keys, batch_poems, batch_labels)

            logger.add_losses(loss_xw, loss_w, preds_xw, preds_w, batch_labels.cpu().numpy())
            logger.set_rate("learning_rate", optimizer.rate())
            if step % self.hps.cl_log_steps == 0:
                logger.set_end_time()
                logger.print_log()
                logger.set_start_time()



    def train(self, mixpoet, tool, factor_id):
        self.__factor_id = factor_id

        utils.print_parameter_list(mixpoet, mixpoet.classifier_parameter_names(factor_id))
        #input("please check the parameters, and then press any key to continue >")
        # load data for pre-training
        print ("building data for classifier, factor_id: %d ..." % (factor_id))
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.cl_batch_size, mode='cl'+str(factor_id))

        print ("train batch num: %d" % (tool.train_batch_num))
        print ("valid batch num: %d" % (tool.valid_batch_num))


        # training logger
        logger = ClassifierLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.cl_log_steps)
        logger.set_log_path(self.hps.cl_train_log_path)
        logger.set_rate('learning_rate', 0.0)


        # build optimizer
        opt = torch.optim.AdamW(mixpoet.cl_parameters(self.__factor_id),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizer = ScheduledOptim(optimizer=opt, warmup_steps=self.hps.cl_warmup_steps,
            max_lr=self.hps.cl_max_lr, min_lr=self.hps.cl_min_lr)


        mixpoet.train()


        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')


        # train
        for epoch in range(1, self.hps.cl_epoches+1):

            self.run_train(mixpoet, tool, optimizer, logger)

            if epoch % self.hps.cl_validate_epoches == 0:
                print("run validation...")
                mixpoet.eval()
                print ("in training mode: %d" % (mixpoet.training))
                self.run_validation(epoch, mixpoet, tool, optimizer.rate())
                mixpoet.train()
                print ("validation Done: %d" % (mixpoet.training))


            if (self.hps.cl_save_epoches >= 0) and \
                (epoch % self.hps.cl_save_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, mixpoet,
                    prefix="cl"+str(self.__factor_id))


            logger.add_epoch()

            print("shuffle data...")
            tool.shuffle_train_data()