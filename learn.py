#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.datasets as D
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from MyNet import MyNet
from MyDataSet import  MyDataSet

batch_size = 15
epoch = 1000

def study():

    dataset = MyDataSet("iris_all.dat")
    train, test = chainer.datasets.split_dataset_random(dataset, int(dataset.__len__() * 0.7))

    model = L.Classifier(MyNet())

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter  = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer)

    trainer = training.Trainer(updater, (epoch, 'epoch'))

    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(extensions.PlotReport(['main/loss',     'validation/main/loss'],     'epoch', file_name="loss.png"))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name="accuracy.png"))

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
#    trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))
#    trainer.extend(extensions.ProgressBar())

    trainer.run()

    chainer.serializers.save_npz("mynet.model", model)

if __name__ == '__main__':
    study()
