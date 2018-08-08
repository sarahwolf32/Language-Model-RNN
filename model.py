import numpy as np
import tensorflow as tf
from random import shuffle

'''
What can this model do?
    1. Generate new Elvish names.
    2. Score words on how 'Elvish' they are.
'''

# hyperparameters

filename = 'elvish_words.txt'

# data handling

def words_list(filename):
    '''
    filename: Path to a .txt file containing a list of words. One word per line, no commas.
    returns: A shuffled list of the words in that file.
    '''
    with open(filename) as f:
        words = f.readlines()
    words = [w.rstrip() for w in words]
    shuffle(words)
    return words

def create_dataset(words_list):
    '''
    words_list: A python list of strings.
    returns: A Tensorflow dataset of those strings.
    '''
    words = tf.constant(words_list)
    dataset = tf.data.Dataset.from_tensor_slices(words)
    return dataset

def input_letter(prev_y, prev_a, var):
    '''
    prev_y: A placeholder for y<t-1>, the previous character. A one-hot vector of shape [1, c]
    prev_a: A tensor for a<t-1>, the previous activation. Shaped [1, n]
    var: Dictionary of trainable weights and biases
        Wa: [n+c, n]
        ba: [1, n]
        Wy: [n, c]
        by: [1, c]
    returns: y<t> and a<t>
    '''
    # unwrap vars
    Wa, ba, Wy, by = var['Wa'], var['ba'], var['Wy'], var['by']

    # concatenate prev_a and prev_y to create a tensor of shape [1, c + n]
    inputs = tf.concat([prev_y, prev_a], axis=1)

    # a = tanh( Wa[ prev_a, prev_y ] + ba )
    a = tf.nn.tanh(tf.matmul(inputs, Wa) + ba)

    # y_pred = sigmoid( dot(Wy, a) + by )
    y_pred = tf.nn.sigmoid(tf.matmul(a, Wy) + by)

    return (y_pred, a)



def train():

    # load words
    words = words_list(filename)
    print("Found " + str(len(words)) + " words!")

    # create dataset
    dataset = create_dataset(words)

    # test dataset
    iterator = dataset.make_one_shot_iterator()
    next_item = iterator.get_next()
    sess = tf.Session()

    # one epoch
    while True:
        try:
            word = sess.run(next_item)
            print(word)
        except Exception:
            print("Done!")
            break


