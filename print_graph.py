from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time
import operator
import io
import array
import datetime
import pickle

import os
import sys

import itertools

with open('lang_loss', 'rb') as fp:     # get the loss graph from previous session
    dump_list = pickle.load(fp)
    lang_loss_list = dump_list[:-1]
    counter_lang = dump_list[-1]
    print("number of epochs for language training: ", counter_lang)
    print("last recorded language loss: ", lang_loss_list[-1])

#ax = plt.subplot(1,1,1)

#plt.semilogy(lang_loss_list, 'b')
plt.plot(lang_loss_list, 'b')

plt.show()

#time.sleep(30)

with open('motor_loss', 'rb') as fp:    # get the loss graph from previous session
    dump_list = pickle.load(fp)
    motor_loss_list = dump_list[:-1]
    counter_motor = dump_list[-1]
    print("number of epochs for motor training: ", counter_motor)
    print("last recorded motor loss: ", motor_loss_list[-1])



plt.semilogy(motor_loss_list, 'r')
plt.show()

#def plot(language_loss, action_loss, fig, ax):

