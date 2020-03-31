# -*- coding: utf-8 -*-
# @Author: Xiaoyuan Yi
# @Last Modified by:   Xiaoyuan Yi
# @Last Modified time: 2020-03-30 19:59:47
# @Email: yi-xy16@mails.tsinghua.edu.cn
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
from generator import Generator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="The parametrs for the generator.")
    parser.add_argument("-m", "--mode", type=str, choices=['interact', 'file'], default='interact',
        help='The mode of generation. interact: generate in a interactive mode.\
        file: take an input file and generate poems for each input in the file.')
    parser.add_argument("-b", "--bsize",  type=int, default=20, help="beam size, 20 by default.")
    parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0, 1, 2, 3],
        help="Show other information during the generation, False by default.")
    parser.add_argument("-s", "--select", type=int, default=0,
        help="If manually select each generated line from beam candidates? False by default.\
        It works only in the interact mode.")
    parser.add_argument("-l", "--length", type=int, choices=[5, 7],
        help="The length of lines of generated quatrains. 5 or 7.\
        It works only in the file mode.")
    parser.add_argument("-i", "--inp", type=str,
        help="input file path. it works only in the file mode.")
    parser.add_argument("-o", "--out", type=str,
        help="output file path. it works only in the file mode")
    return parser.parse_args()


def generate_manu(args):
    generator = Generator()
    beam_size = args.bsize
    verbose = args.verbose
    manu = True if args.select ==1 else False

    while True:
        keyword = input("input a keyword:>")
        length = int(input("specify the length, 5 or 7:>"))
        label1 = int(input("specify the living experience label\n\
            0: military career, 1: countryside life, 2: other:, -1: not specified>"))
        label2 = int(input("specify the historical background label\n\
            0: prosperous times, 1: troubled times, -1: not specified>"))

        lines, info = generator.generate_one(keyword, length, label1, label2,
            beam_size, verbose, manu)

        if len(lines) != 4:
            print("generation failed!")
            continue
        else:
            print("\n".join(lines))


def generate_file(args):
    generator = Generator()
    beam_size = args.bsize
    verbose = args.verbose
    manu = True if args.select ==1 else False

    assert args.inp is not None
    assert args.out is not None
    assert args.length is not None

    length = args.length

    with open(args.inp, 'r') as fin:
        inps = fin.readlines()

    poems = []
    N = len(inps)
    log_step = max(int(N/100), 2)
    for i, inp in enumerate(inps):
        para = inp.strip().split(" ")
        keyword = para[0]

        if len(para) >= 2:
            length = int(para[1])
        if len(para) >= 3:
            label1 = int(para[2])
        else:
            label1 = -1

        if len(para) >= 4:
            label2 = int(para[3])
        else:
            label2 = -1

        lines, info = generator.generate_one(keyword, length,
            label1, label2, beam_size, verbose, manu)

        if len(lines) != 4:
            ans = info
        else:
            ans = "|".join(lines)

        poems.append(ans)

        if i % log_step == 0:
            print ("generating, %d/%d" % (i, N))

    with open(args.out, 'w') as fout:
        for poem in poems:
            fout.write(poem+"\n")


def main():
    args = parse_args()
    if args.mode == 'interact':
        generate_manu(args)
    else:
        generate_file(args)


if __name__ == "__main__":
    main()
