# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: utils.py
# $Date: Sat Dec 23 02:0854 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

def list2tuple(list):
    try:
        return tuple(list2tuple(sub) for sub in list)
    except TypeError:
        return list


def tuple2list(tuple):
    try:
        return list(tuple2list(sub) for sub in tuple)
    except TypeError:
        return tuple


