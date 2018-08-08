import numpy as np
import tensorflow as tf
from random import shuffle

'''
What can this model do?
    1. Generate new Elvish names.
    2. Score words on how 'Elvish' they are.
'''

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


# load words
filename = 'elvish_words.txt'
words = words_list(filename)
print("Found " + str(len(words)) + " words!")

# create dataset
dataset = create_dataset(words)

# test dataset
iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()
sess = tf.Session()

while True:
    try:
        word = sess.run(next_item)
        print(word)
    except Exception:
        print("Done!")
        break
