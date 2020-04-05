# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 22:50:06
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
from dae_trainer import DAETrainer
from cl_trainer import ClassifierTrainer
from mix_trainer import MixTrainer

from graphs import MixPoetAUS
from tool import Tool
from config import device, hparams
import utils


def pretrain(mixpoet, tool, hps):
    dae_trainer = DAETrainer(hps)
    cl_trainer = ClassifierTrainer(hps)


    # --------------------------------------
    print ("dae pretraining...")
    dae_trainer.train(mixpoet, tool)
    print ("dae pretraining done!")

    #---------------------------------------
    print ("classifier1 pretraining...")
    cl_trainer.train(mixpoet, tool, factor_id=1)
    print ("classifier1 pretraining done!")

    print ("classifier2 pretraining...")
    cl_trainer.train(mixpoet, tool, factor_id=2)
    print ("classifier2 pretraining done!")
    # --------------------------------------


def train(mixpoet, tool, hps):
    last_epoch = utils.restore_checkpoint(hps.model_dir, device, mixpoet)

    if last_epoch is not None:
         print ("checkpoint exsits! directly recover!")
    else:
         print ("checkpoint not exsits! train from scratch!")

    mix_trainer = MixTrainer(hps)
    mix_trainer.train(mixpoet, tool)


def main():
    hps = hparams
    tool = Tool(hps.sens_num, hps.key_len,
        hps.sen_len, hps.poem_len, hps.corrupt_ratio)
    tool.load_dic(hps.vocab_path, hps.ivocab_path)
    vocab_size = tool.get_vocab_size()
    PAD_ID = tool.get_PAD_ID()
    B_ID = tool.get_B_ID()
    assert vocab_size > 0 and PAD_ID >=0 and B_ID >= 0
    hps = hps._replace(vocab_size=vocab_size, pad_idx=PAD_ID, bos_idx=B_ID)

    print ("hyper-patameters:")
    print (hps)
    input("please check the hyper-parameters, and then press any key to continue >")

    mixpoet = MixPoetAUS(hps)
    mixpoet = mixpoet.to(device)

    pretrain(mixpoet, tool, hps)
    train(mixpoet, tool, hps)


if __name__ == "__main__":
    main()
