#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import linecache

class MyDataSet(chainer.dataset.DatasetMixin):
    def __init__(self, filename):
        self._len = None
        self.file = filename

    def __len__(self):
        if self._len is None:
            self._len = sum(1 for line in open(self.file))
        return self._len

    def get_example(self, i):
        ary = np.fromstring(linecache.getline(self.file, i+1), dtype=np.float32, sep=",")
        xt = ary[:-1]
        yt = np.int32(ary[-1])
        return xt, yt
