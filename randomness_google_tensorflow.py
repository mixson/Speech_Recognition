#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:36:55 2018

@author: mixson
"""

# adding something
import tensorflow as tf
# change 5
# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([1, 3], mean=-1, stddev=4)

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

# Each time we run these ops, different results are generated
sess = tf.Session()
print(sess.run(norm))
print('\n')
print(sess.run(norm))

print( tf.cast(sess.run(tf.reduce_mean(norm) ), tf.string)
# Set an op-level seed to generate repeatable sequences across sessions.
norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print('\n')
print(sess.run(norm))
print(sess.run(norm))
sess = tf.Session()
print('\n')
print(sess.run(norm))
print(sess.run(norm))



# Use random uniform values in [0, 1) as the initializer for a variable of shape
# [2, 3]. The default type is float32.
var = tf.Variable(tf.random_uniform([2, 3]), name="var")
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(var))
print('/n')
a = tf.reduce_max(var)
print(sess.run(a))