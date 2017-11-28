# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: utils.py
# $Date: Fri Nov 17 10:2407 2017 +0800
# $Author: renyong15 © <mails.tsinghua.edu.cn>
#

WHITE = -1
BLACK = +1
EMPTY = 0

PASS = (0, 0)
RESIGN = "resign"

from collections import defaultdict
import functools
import itertools
import operator
import random
import re
import time

import gtp
import go

KGS_COLUMNS = 'ABCDEFGHJKLMNOPQRST'
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_sgf_to_flat(sgf):
    return flatten_coords(parse_sgf_coords(sgf))


def flatten_coords(c):
    return go.N * c[0] + c[1]


def unflatten_coords(f):
    return divmod(f, go.N)


def parse_sgf_coords(s):
    'Interprets coords. aa is top left corner; sa is top right corner'
    if s is None or s == '':
        return None
    return SGF_COLUMNS.index(s[1]), SGF_COLUMNS.index(s[0])


def unparse_sgf_coords(c):
    if c is None:
        return ''
    return SGF_COLUMNS[c[1]] + SGF_COLUMNS[c[0]]


def parse_kgs_coords(s):
    'Interprets coords. A1 is bottom left; A9 is top left.'
    if s == 'pass':
        return None
    s = s.upper()
    col = KGS_COLUMNS.index(s[0])
    row_from_bottom = int(s[1:]) - 1
    return go.N - row_from_bottom - 1, col


def parse_pygtp_coords(vertex):
    'Interprets coords. (1, 1) is bottom left; (1, 9) is top left.'
    if vertex in (gtp.PASS, gtp.RESIGN):
        return None
    return go.N - vertex[1], vertex[0] - 1


def unparse_pygtp_coords(c):
    if c is None:
        return gtp.PASS
    return c[1] + 1, go.N - c[0]


def parse_game_result(result):
    if re.match(r'[bB]\+', result):
        return go.BLACK
    elif re.match(r'[wW]\+', result):
        return go.WHITE
    else:
        return None


def product(numbers):
    return functools.reduce(operator.mul, numbers)


def take_n(n, iterable):
    return list(itertools.islice(iterable, n))


def iter_chunks(chunk_size, iterator):
    while True:
        next_chunk = take_n(chunk_size, iterator)
        # If len(iterable) % chunk_size == 0, don't return an empty chunk.
        if next_chunk:
            yield next_chunk
        else:
            break


def shuffler(iterator, pool_size=10 ** 5, refill_threshold=0.9):
    yields_between_refills = round(pool_size * (1 - refill_threshold))
    # initialize pool; this step may or may not exhaust the iterator.
    pool = take_n(pool_size, iterator)
    while True:
        random.shuffle(pool)
        for i in range(yields_between_refills):
            yield pool.pop()
        next_batch = take_n(yields_between_refills, iterator)
        if not next_batch:
            break
        pool.extend(next_batch)
    # finish consuming whatever's left - no need for further randomization.
    yield pool


class timer(object):
    all_times = defaultdict(float)

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.tick = time.time()

    def __exit__(self, type, value, traceback):
        self.tock = time.time()
        self.all_times[self.label] += self.tock - self.tick

    @classmethod
    def print_times(cls):
        for k, v in cls.all_times.items():
            print("%s: %.3f" % (k, v))
