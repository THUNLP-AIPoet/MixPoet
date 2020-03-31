# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-30 18:19:26
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
from matplotlib import pyplot as pylab
import math
import numpy as np
import time
from sklearn.metrics import accuracy_score as accuracy_tool
from sklearn.metrics import f1_score as f1_tool
from sklearn import manifold

pylab.switch_backend('agg')

class InfoLogger(object):
    """docstring for LogInfo"""
    def __init__(self, mode):
        super(InfoLogger).__init__()
        self.__mode = mode # string, 'train' or 'valid'
        self.__total_steps = 0
        self.__batch_num = 0
        self.__log_steps = 0
        self.__cur_step = 0
        self.__cur_epoch = 1

        self.__start_time = 0
        self.__end_time = 0

        #--------------------------
        self.__log_path = "" # path to save the log file
        self.__fig_path = ""

        # -------------------------
        self.__decay_rates = {'learning_rate':1.0, 'teach_ratio':1.0,
            'temperature':1.0, 'noise_weight':1.0}


    def set_batch_num(self, batch_num):
        self.__batch_num = batch_num
    def set_log_steps(self, log_steps):
        self.__log_steps = log_steps
    def set_log_path(self, log_path, fig_path=""):
        self.__log_path = log_path
        self.__fig_path = fig_path

    def set_rate(self, name, value):
        self.__decay_rates[name] = value


    def set_start_time(self):
        self.__start_time = time.time()

    def set_end_time(self):
        self.__end_time = time.time()


    def add_step(self):
        self.__total_steps += 1
        self.__cur_step += 1

    def add_epoch(self):
        self.__cur_step = 0
        self.__cur_epoch += 1



    # # ------------------------------
    @property
    def cur_process(self):
        ratio = float(self.__cur_step) / self.__batch_num * 100
        process_str = "%d/%d %.1f%%" % (self.__cur_step, self.__batch_num, ratio)
        return process_str

    @property
    def time_cost(self):
        return (self.__end_time-self.__start_time) / self.__log_steps

    @property
    def total_steps(self):
        return self.__total_steps

    @property
    def epoch(self):
        return self.__cur_epoch

    @property
    def mode(self):
        return self.__mode

    @property
    def log_path(self):
        return self.__log_path

    @property
    def fig_path(self):
        return self.__fig_path


    @property
    def learning_rate(self):
        return self.__decay_rates['learning_rate']

    @property
    def teach_ratio(self):
        return self.__decay_rates['teach_ratio']

    @property
    def temperature(self):
        return self.__decay_rates['temperature']

    @property
    def noise_weight(self):
        return self.__decay_rates['noise_weight']



#------------------------------------
class DAELogger(InfoLogger):
    def __init__(self, mode):
        super(DAELogger, self).__init__(mode)
        self.__gen_loss = 0.0

    def add_losses(self, gen_loss):
        self.add_step()
        self.__gen_loss += gen_loss


    def get_cur_losses(self):
        cur_gen_loss = self.__gen_loss / self.total_steps
        cur_ppl = math.exp(cur_gen_loss)

        return cur_gen_loss, cur_ppl


    def print_log(self, epoch=None):
        if self.mode == 'train':
            time_cost = self.time_cost
        process_str = self.cur_process
        gen_loss, ppl = self.get_cur_losses()


        if self.mode == 'train':
            process_info = "epoch: %d, %s, %.2fs per iter, lr: %.4f, tr: %.2f" % (self.epoch, process_str,
                time_cost, self.learning_rate, self.teach_ratio)
        else:
            process_info = "epoch: %d, lr: %.4f, tr: %.2f" % (epoch, self.learning_rate, self.teach_ratio)

        train_info = "  gen loss: %.3f  ppl:%.2f" \
            % (gen_loss, ppl)

        print (process_info)
        print (train_info)
        print ("______________________")

        info = process_info + "\n" + train_info
        fout = open(self.log_path, 'a')
        fout.write(info + "\n\n")
        fout.close()


#------------------------------------------------
class ClassifierLogger(InfoLogger):
    def __init__(self, mode):
        super(ClassifierLogger, self).__init__(mode)
        self.__loss_xw = 0.0
        self.__loss_w = 0.0

        self.__accu_xw = 0.0
        self.__accu_w = 0.0

        self.__f1_xw = 0.0
        self.__f1_w = 0.0

    def add_losses(self, loss_xw, loss_w, preds_xw, preds_w, labels):
        self.add_step()
        self.__loss_xw += loss_xw
        self.__loss_w += loss_w

        accu_xw = accuracy_tool(labels, preds_xw)
        accu_w = accuracy_tool(labels, preds_w)

        f1_xw = f1_tool(labels, preds_xw, average="macro")
        f1_w = f1_tool(labels, preds_w, average="macro")

        self.__accu_xw += accu_xw
        self.__accu_w += accu_w

        self.__f1_xw += f1_xw
        self.__f1_w += f1_w


    def get_cur_losses(self):
        cur_loss_xw = self.__loss_xw / self.total_steps
        cur_loss_w = self.__loss_w / self.total_steps

        cur_accu_xw = self.__accu_xw / self.total_steps
        cur_accu_w = self.__accu_w / self.total_steps

        cur_f_xw = self.__f1_xw / self.total_steps
        cur_f_w = self.__f1_w / self.total_steps


        return cur_loss_xw, cur_loss_w, cur_accu_xw*100, cur_accu_w*100, cur_f_xw*100, cur_f_w*100


    def print_log(self, epoch=None):
        if self.mode == 'train':
            time_cost = self.time_cost
        process_str = self.cur_process
        loss_xw, loss_w, accu_xw, accu_w, f1_xw, f1_w = self.get_cur_losses()

        if self.mode == 'train':

            process_info = "epoch: %d, %s, %.3f s per iter, lr: %.4f" % (self.epoch, process_str,
                time_cost, self.learning_rate)

        else:
            process_info = "epoch: %d, lr: %.4f" % (epoch, self.learning_rate)

        train_info1 = "  cl_xw loss: %.3f, accu: %.1f, f1: %.1f; " \
            % (loss_xw, accu_xw, f1_xw)
        train_info2 = "  cl_w loss: %.3f, accu: %.1f, f1: %.1f; " \
            % (loss_w, accu_w, f1_w)

        print (process_info)
        print (train_info1)
        print (train_info2)
        print ("______________________")

        info = process_info + "\n" + train_info1 + "\n" + train_info2
        fout = open(self.log_path, 'a')
        fout.write(info + "\n\n")
        fout.close()


#----------------------------------------------
class MixAUSLogger(InfoLogger):
    def __init__(self, mode):
        super(MixAUSLogger, self).__init__(mode)

        self.__rec_loss = 0.0 # reconstruction loss
        self.__entro_loss = 0.0 # entropy loss
        self.__cl_loss_w = 0.0 # classifier loss
        self.__cl_loss_xw = 0.0
        self.__dis_loss = [] # discriminator loss
        self.__adv_loss = [] # adversarial loss of prior generator
        self.__latent_distance = [] # distance of prior and posteriori sampled points
        self.__factor_distance = [] # distance of prior conditioned on different mixtures


    def add_rec_losses(self, rec_loss, cl_loss_w, cl_loss_xw, entro_loss):
        self.add_step()
        self.__rec_loss += rec_loss
        self.__entro_loss += entro_loss
        self.__cl_loss_w += cl_loss_w
        self.__cl_loss_xw += cl_loss_xw

    def add_dis_loss(self, dis_loss):
        self.__dis_loss.append((dis_loss, self.total_steps))


    def add_adv_loss(self, adv_loss):
        self.__adv_loss.append((adv_loss, self.total_steps))


    def add_distance(self, dist):
        self.__latent_distance.append((dist, self.total_steps))

    def add_factor_distance(self, fadist):
        self.__factor_distance.append((fadist, self.total_steps))


    def get_cur_losses(self):
        # get current accumulative loss
        cur_rec_loss = self.__rec_loss / self.total_steps
        cur_ppl = math.exp(cur_rec_loss)

        cur_entro_loss = self.__entro_loss / self.total_steps
        cur_cl_loss_w = self.__cl_loss_w / self.total_steps
        cur_cl_loss_xw = self.__cl_loss_xw / self.total_steps

        cur_dis_loss = np.mean([pair[0] for pair in self.__dis_loss]) \
            if len(self.__dis_loss) > 0 else 0
        cur_adv_loss = np.mean([pair[0] for pair in self.__adv_loss]) \
            if len(self.__adv_loss) > 0 else 0

        cur_distance = np.mean([pair[0] for pair in self.__latent_distance]) \
            if len(self.__latent_distance) > 0 else 0


        cur_fa_distance = np.mean([pair[0] for pair in self.__factor_distance]) \
            if len(self.__latent_distance) > 0 else 0



        return cur_rec_loss, cur_ppl, cur_entro_loss,cur_cl_loss_w, cur_cl_loss_xw,\
            cur_dis_loss, cur_adv_loss, cur_distance, cur_fa_distance


    def print_log(self, epoch=None):
        if self.mode == 'train':
            time_cost = self.time_cost
        process_str = self.cur_process
        rec_loss, ppl, entro_loss, cl_loss_w, cl_loss_xw,\
            dis_loss, adv_loss, ladist, fadist = self.get_cur_losses()



        train_info1 = "  ppl:%.1f, rec loss: %.3f, entropy loss: %.3f, cl loss w: %.3f, cl loss xw: %.3f" \
            % (ppl, rec_loss, entro_loss, cl_loss_w, cl_loss_xw)


        if self.mode == 'train':
            process_info = "epoch: %d, %s, %.3f s per iter, lr: %.4f, tr: %.2f, tau: %.3f, noise: %.3f" % (
                self.epoch, process_str, time_cost, self.learning_rate,
                self.teach_ratio, self.temperature, self.noise_weight)

            train_info2 = "  dis loss: %.3f, adv loss: %.3f, latent dist: %.3f, factors dist: %.3f" \
            % (dis_loss, adv_loss, ladist, fadist)
        else:
            process_info = "epoch: %d, lr: %.3f, tr: %.2f, tau: %.3f, noise: %.3f" % (epoch,
            self.learning_rate, self.teach_ratio, self.temperature, self.noise_weight)

            train_info2 = "  dis loss: %.3f, adv loss: %.3f" % (dis_loss, adv_loss)


        print (process_info)
        print (train_info1)
        print (train_info2)
        print ("______________________")

        info = process_info + "\n" + train_info1 + "\n" + train_info2
        fout = open(self.log_path, 'a')
        fout.write(info + "\n\n")
        fout.close()


    def __build_accumulative_mean(self, vals):
        vec = []
        for i in range(1, len(vals)+1):
            vec.append(np.mean(vals[0:i]))
        return vec


    def draw_curves(self):

        fontsize = 12
        lw = 1.8

        pylab.figure(figsize=(18, 8))
        pylab.subplot(131)
        # ------------------
        # mahalanobis distance
        y_ma = [pair[0] for pair in self.__latent_distance]
        #print (y_ma)
        y_ma = self.__build_accumulative_mean(y_ma)
        #print (y_ma)

        x_ma = [pair[1] for pair in self.__latent_distance]
        #print (x_ma)
        pylab.title("Distance of Prior and Posteriori", fontsize=fontsize)

        pylab.plot(x_ma, y_ma, linewidth=lw, linestyle='-', c='magenta')
        pylab.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

        pylab.xlabel("Training Step", fontsize=fontsize)
        pylab.ylabel("Mean Accumulative Mahalanobis", fontsize=fontsize)

        # ------------------
        # mahalanobis distance of factor-priors
        pylab.subplot(132)
        y_fa = [pair[0] for pair in self.__factor_distance]
        #print (y_ma)
        y_fa = self.__build_accumulative_mean(y_fa)
        #print (y_ma)

        x_fa = [pair[1] for pair in self.__factor_distance]
        #print (x_ma)
        pylab.title("Distance of Prior on Different Mixtures", fontsize=fontsize)

        pylab.plot(x_fa, y_fa, linewidth=lw, linestyle='-', c='magenta')
        pylab.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

        pylab.xlabel("Training Step", fontsize=fontsize)
        pylab.ylabel("Mean Accumulative Mahalanobis", fontsize=fontsize)



        # --------------------------
        # discriminator and generator loss
        pylab.subplot(133)
        y_dis = [pair[0] for pair in self.__dis_loss]
        y_dis = self.__build_accumulative_mean(y_dis)
        x_dis = [pair[1] for pair in self.__dis_loss]

        y_adv = [pair[0] for pair in self.__adv_loss]
        y_adv = self.__build_accumulative_mean(y_adv)
        x_adv = [pair[1] for pair in self.__adv_loss]

        pylab.title("Loss", fontsize=fontsize)

        pylab.plot(x_dis, y_dis, linewidth=lw, linestyle='-', c='magenta', label='dis')
        pylab.plot(x_adv, y_adv, linewidth=lw, linestyle='-', c='steelblue', label='adv')
        pylab.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        pylab.legend(loc="best", fontsize=10)

        pylab.xlabel("Training Step", fontsize=fontsize)
        pylab.ylabel("Mean Accumulative Loss", fontsize=fontsize)


        # ------------------------------
        fig = pylab.gcf()
        #pylab.show()
        fig.savefig(self.fig_path+"/latent_distance.png", dpi=300, quality=100, bbox_inches="tight")
        pylab.close()
