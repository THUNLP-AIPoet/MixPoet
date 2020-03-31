# -*- coding: utf-8 -*-
# @Author: Ruoyu Li
# @Description:
'''
Copyright 2019 THUNLP Lab. All Rights Reserved.
This code is part of the online Chinese poetry generation system, Jiuge.
System URL: https://jiuge.thunlp.cn/.
Github: https://github.com/THUNLP-AIPoet.
'''
class RhythmRecognizer(object):
    """Get the rhythm id of a input line
    This tool can be applied to Chinese classical quatrains only
    """

    def __init__(self, ping_file, ze_file):

        # read level tone char list
        with open(ping_file, 'r') as fin:
            self.__ping = fin.read()
            #print (type(self.__ping))

        with open(ze_file, 'r') as fin:
            self.__ze = fin.read()
            #print (type(self.__ze))

    def get_rhythm(self, sentence):
        # print "#" + sentence + "#"
        if(len(sentence) == 5):
			#1
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #2
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #3
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze):
                return 0
            #4
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 0
            #5
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 0
            #6
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping):
                return 1
            #7
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping):
                return 1
            #8
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #9
            if(sentence[0] in self.__ping and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #10
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #11
            if(sentence[0] in self.__ze and sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze):
                return 3
            #12
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2
            #13
            if(sentence[0] in self.__ze and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2
            #14
            if(sentence[0] in self.__ping and sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping):
                return 2


        elif (len(sentence) == 7):
            #1
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #2
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #3
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ze):
                return 0
            #4
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            #5
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 0
            #6
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            #7
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ping):
                return 1
            #8
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ping and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #9
            if(sentence[1] in self.__ping and sentence[2] in self.__ping and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #10
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ping and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #11
            if(sentence[1] in self.__ping and sentence[2] in self.__ze and sentence[3] in self.__ze and sentence[4] in self.__ze and sentence[5] in self.__ping and sentence[6] in self.__ze):
                return 3
            #12
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ze and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            #13
            if(sentence[1] in self.__ze and sentence[2] in self.__ze and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
            #14
            if(sentence[1] in self.__ze and sentence[2] in self.__ping and sentence[3] in self.__ping and sentence[4] in self.__ping and sentence[5] in self.__ze and sentence[6] in self.__ping):
                return 2
        else:
            return -2
        return -1