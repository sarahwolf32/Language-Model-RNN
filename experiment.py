import tensorflow as tf
import numpy as np

'''
Can we iterate through a placeholder of [None, c] size?
Maybe this should be an experiment in eager execution?

Normal RNN style uses padding to force lengths to be predictable.
This is distasteful, but also a separate issue from sequentiality. How is this normally done?

Options:
    - eager execution (some refactoring required)
    - tf.nn.dynamic_rnn (already done the work, don't want to do this)
    - tf.While (the sequential heart of tf.nn.dynamic_rnn)
'''



tf.enable_eager_execution()