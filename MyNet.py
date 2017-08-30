#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MyNet(Chain):
    def __init__(self):
        super(MyNet, self).__init__(
            l1 = L.Linear(None, 16),
            l2 = L.Linear(None, 50),
            l3 = L.Linear(None, 50),
            l4 = L.Linear(None, 10),
            l5 = L.Linear(None, 3),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1( x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.l5(h4)

