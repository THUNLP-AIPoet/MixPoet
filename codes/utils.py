# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-31 22:22:40
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
import os
import torch
import torch.nn.functional as F

import random
import math
import numpy as np

def save_checkpoint(model_dir, epoch, model, prefix='',
    optimizer1=None, optimizer2=None, optimizer3=None):
    # save model state dict
    checkpoint_name = "model_ckpt_{}_{}e.tar".format(prefix, epoch)
    model_state_path = os.path.join(model_dir, checkpoint_name)

    saved_dic = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }

    if optimizer1 is not None:
        saved_dic['optimizer1'] = optimizer1.state_dict()

    if optimizer2 is not None:
        saved_dic['optimizer2'] = optimizer2.state_dict()

    if optimizer3 is not None:
        saved_dic['optimizer3'] = optimizer3.state_dict()


    torch.save(saved_dic, model_state_path)

    # write checkpoint information
    log_path = os.path.join(model_dir, "ckpt_list.txt")
    fout = open(log_path, 'a')
    fout.write(checkpoint_name+"\n")
    fout.close()


def restore_checkpoint(model_dir, device, model):
    ckpt_list_path = os.path.join(model_dir, "ckpt_list.txt")
    if not os.path.exists(ckpt_list_path):
        print ("checkpoint list not exists, creat new one!")
        return None

    # get latest ckpt name
    fin = open(ckpt_list_path, 'r')
    latest_ckpt_path = fin.readlines()[-1].strip()
    fin.close()

    latest_ckpt_path = os.path.join(model_dir, latest_ckpt_path)
    if not os.path.exists(latest_ckpt_path):
        print ("latest checkpoint not exists!")
        return None


    print ("restore checkpoint from %s" % (latest_ckpt_path))
    print ("loading...")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    #checkpoint = torch.load(latest_ckpt_path)
    print ("load state dic, params: %d..." % (len(checkpoint['model_state_dict'])))
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']


    return epoch


def sample_dae(ori_keys, ori_poems, all_trgs, all_logits, sample_num, tool):
    # enc_inps [batch size, sen len]
    # all_logits [batch size, trg len, vocab size] * sens_num
    batch_size = ori_poems.size(0)
    enc_len = ori_poems.size(1)
    key_len = ori_keys.size(1)

    sample_num = min(sample_num, batch_size)

    # random select some examples
    sample_ids = random.sample(list(range(0, batch_size)), sample_num)

    for sid in sample_ids:
        # Build lines
        inps = [ori_poems[sid, t].item() for t in range(0, enc_len)]
        sline = tool.tokens2line(tool.idxes2tokens(inps))

        key_inps = [ori_keys[sid, t].item() for t in range(0, key_len)]
        key_line = tool.tokens2line(tool.idxes2tokens(key_inps))

        # ------------------------------------------
        out_lines = []
        for logits in all_logits:
            trg_len = logits.size(1)
            probs = F.softmax(logits, dim=-1)
            outline = [probs[sid, t, :].cpu().data.numpy() for t in range(0, trg_len)]
            outline = tool.greedy_search(outline)
            out_lines.append(outline)

        # -------------------------------------------
        trg_lines = []
        for trgs in all_trgs:
            trg_len = trgs.size(1)
            trg_idxes = [trgs[sid, t].item() for t in range(0, trg_len)]
            tline = tool.tokens2line(tool.idxes2tokens(trg_idxes))
            trg_lines.append(tline)


        print("key: " + key_line)
        print("inp: " + sline)
        print("trg: " + "|".join(trg_lines))
        print("out: " + "|".join(out_lines))
        print ("")

#------------------------------
def sample_mix(ori_keys, ori_dec_inps, ori_labels, clabels1, clabels2,
    all_logits_post, all_logits_prior, sample_num, tool):
    # enc_inps [batch size, sen len]
    # all_logits [batch size, trg len, vocab size] * sens_num
    batch_size = ori_keys.size(0)
    key_len = ori_keys.size(1)

    sample_num = min(sample_num, batch_size)

    # random select some examples
    sample_ids = random.sample(list(range(0, batch_size)), sample_num)

    for sid in sample_ids:
        # Build lines
        key_inps = [ori_keys[sid, t].item() for t in range(0, key_len)]
        key_line = tool.tokens2line(tool.idxes2tokens(key_inps))

        trg_lines = []
        for dec_inp in ori_dec_inps:
            dec_len = dec_inp.size(1)
            trgs = [dec_inp[sid, t].item() for t in range(0, dec_len)]
            trg_line = tool.tokens2line(tool.idxes2tokens(trgs))
            trg_lines.append(trg_line)

        out_lines_post = []
        for logits in all_logits_post:
            trg_len = logits.size(1)
            probs = F.softmax(logits, dim=-1)
            outline = [probs[sid, t, :].cpu().data.numpy() for t in range(0, trg_len)]
            outline = tool.greedy_search(outline)
            out_lines_post.append(outline)
        #--------------------------------------------
        out_lines_prior = []
        for logits in all_logits_prior:
            trg_len = logits.size(1)
            probs = F.softmax(logits, dim=-1)
            outline = [probs[sid, t, :].cpu().data.numpy() for t in range(0, trg_len)]
            outline = tool.greedy_search(outline)
            out_lines_prior.append(outline)

        #--------------------------------------------
        label1, label2 = ori_labels[sid, 0].item(), ori_labels[sid, 1].item()
        flabel1, flabel2 = clabels1[sid].item(), clabels2[sid].item()

        label_str1 = "factor1 label: %d" % (label1)
        if label1 == -1:
            label_str1 += ", inferred: %d" % (flabel1)

        label_str2 = "factor2 label: %d" % (label2)
        if label2 == -1:
            label_str2 += ", inferred: %d" % (flabel2)

        print (label_str1)
        print (label_str2)
        print("key: " + key_line)
        print("trg: " + "|".join(trg_lines))
        print("out_post: " + "|".join(out_lines_post))
        print("out_prior: " + "|".join(out_lines_prior))
        print ("")


def print_parameter_list(model, prefix=None):
    params = model.named_parameters()

    param_num = 0
    for name, param in params:
        if prefix is not None:
            seg = name.split(".")[1]
            if seg in prefix:
                print(name, param.size())
                param_num += 1
        else:
            print(name, param.size())
            param_num += 1

    print ("params num: %d" % (param_num))



#-------------------------------------------------------------------
# for observation
def get_dist_distance_ma(X1, X2):
    #print ("get_dist_distance_ma_torch")
    K = X1.size(0)

    # cov
    mu1 = torch.mean(X1, dim=0).unsqueeze(0) # (1, H)

    s0 = X1-mu1
    S = s0.t().contiguous().mm(s0) / (K-1)
    SI = torch.inverse(S)
    Delta = X2 - mu1

    dis = torch.sqrt(torch.sum(torch.mul(torch.matmul(Delta, SI), Delta), dim=1))
    #print (dis)
    dis = dis.mean()

    return dis.item()

def cal_distance(mixpoet, ori_keys, ori_poems, ori_labels, ori_label_mask):
    n_samples = 128

    vec_prior, vec_post = [], []
    with torch.no_grad():
        batch_size = ori_keys.size(0)
        idx = random.sample(list(range(0, batch_size)), 1)[0]

        poems = ori_poems[idx, :].unsqueeze(0).repeat(n_samples, 1)
        keys = ori_keys[idx, :].unsqueeze(0).repeat(n_samples, 1)
        labels = ori_labels[idx, :].unsqueeze(0).repeat(n_samples, 1)
        label_mask = ori_label_mask[idx, :].unsqueeze(0).repeat(n_samples, 1)

        for i in range(0, 4):
            z_prior, z_post, _, _ \
                = mixpoet.get_prior_and_posterior(keys, poems, labels, label_mask)

            vec_prior.append(z_prior)
            vec_post.append(z_post)

    z1 = torch.cat(vec_prior, dim=0)
    z2 = torch.cat(vec_post, dim=0)

    distance = get_dist_distance_ma(z1, z2)


    return distance



def factor_distance(mixpoet, ori_keys, n_class1, n_class2, device):
    N = 64
    K = 8
    M = n_class1 * n_class2
    #print (M)

    idx = random.sample(list(range(0, ori_keys.size(0))), 1)[0]
    keys = ori_keys[idx, :].unsqueeze(0).repeat(N, 1) # (N, key_len)

    vec = [[] for _ in range(0, M)]
    label_count = 0
    with torch.no_grad():
        for l1 in range(0, n_class1):
            for l2 in range(0, n_class2):
                for k in range(0, K):
                    labels1 = torch.tensor([l1]*N, dtype=torch.long, device=device)
                    labels2 = torch.tensor([l2]*N, dtype=torch.long, device=device)
                    points = mixpoet.compute_prior(keys, labels1, labels2)
                    vec[label_count].append(points.detach())
                label_count += 1


    total_distance = []
    for i in range(0, M):
        for j in range(i+1, M):
            z1 = torch.cat(vec[i], dim=0)
            z2 = torch.cat(vec[j], dim=0)


            dis1 = get_dist_distance_ma(z1, z2)
            dis2 = get_dist_distance_ma(z2, z1)
            distance = (dis1+dis2) / 2.0
            total_distance.append(distance)

    total_distance = np.mean(total_distance)
    return total_distance