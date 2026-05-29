"""
Examples from https://www.tensorflow.org/api_docs/python/tf/random/set_seed
"""
import os
import sys

# https://www.codegrepper.com/code-examples/python/suppres+tensorflow+warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
assert tf.__version__ >= "2.0"

# from tensorflow import keras

# If neither the global seed nor the operation seed is set, we get different
# results for every call to the random op and every re-run of the program:
print("No random seed")
print(tf.random.uniform([1]))  # generates 'A1'
print(tf.random.uniform([1]))  # generates 'A2'

# If the global seed is set but the operation seed is not set, we get
# different results for every call to the random op, but the same sequence
# for every re-run of the program:
print("\nOnly global seed")
tf.random.set_seed(1234)
print(tf.random.uniform([1]))  # generates 'A3'
print(tf.random.uniform([1]))  # generates 'A4'

# Note that tf.function acts like a re-run of a program in this case.
print("\nOnly global seed")
tf.random.set_seed(1234)

@tf.function
def f():
  a = tf.random.uniform([1])
  b = tf.random.uniform([1])
  return a, b

@tf.function
def g():
  a = tf.random.uniform([1])
  b = tf.random.uniform([1])
  return a, b

tf.print(f())  # prints '(A1, A2)'
tf.print(g())  # prints '(A1, A2)'

# If the operation seed is set, we get different results for every call to the
# random op, but the same sequence for every re-run of the program:
print("\nOnly op seed")
tf.print(tf.random.uniform([1], seed=1))  # generates 'A1'
tf.print(tf.random.uniform([1], seed=1))  # generates 'A2'

print("\nBoth global and op seeds with reseting")
tf.random.set_seed(1234)
tf.print(tf.random.uniform([1], seed=1))  # generates 'A1'
tf.random.set_seed(1234)
tf.print(tf.random.uniform([1], seed=1))  # generates 'A1'
