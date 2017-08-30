#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""このモジュールの説明
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import linecache

from MyNet import MyNet


def test(filename):
    with chainer.using_config('train', False):
        model = L.Classifier(MyNet())
        serializers.load_npz("mynet.model", model)    

        len = sum(1 for line in open(filename))

        ok = 0
        for i in range(len):
            ary = np.fromstring(linecache.getline(filename, i+1), dtype=np.float32, sep=",")
            x = ary[:-1]
            y = np.int32(ary[-1])

            a = model.predictor(Variable(x.reshape(1, -1)))
            clas = np.argmax(a.data)

            if clas == y:
                ok += 1
            else:
                pass

            temp = F.softmax(a).data.flatten()

            print("{0} : [{1:1.3f}, {2:1.3f}, {3:1.3f}] {4} {5}".format(i, temp[0], temp[1], temp[2], clas, y))

        print(ok, "/", len, "   ", (ok*1.0)/len)
        

if __name__ == '__main__':
    test("iris_all.dat")
