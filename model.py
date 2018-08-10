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
nodes = 10
learning_rate = 0.001
weights_init_stddev = 0.2

# load words

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

# build character maps

def character_maps(words_list):
    i = 0
    character_map = {}

    # assign code to each unique character
    for word in words_list:
        for char in word:
            if not char in character_map:
                character_map[char] = i
                i += 1

    # add any special codes
    character_map['<END>'] = i

    # reverse dictionary to look up in other direction
    code_map = {v: k for k, v in character_map.iteritems()}

    return character_map, code_map

def one_hot(code, C):
    '''A single one-hot vector for one letter-code.'''
    x = np.zeros(C)
    x[code] = 1.
    return x

def vectorize_word(word, character_map, C):
    codes = [character_map[letter] for letter in word]
    vectors = np.array([one_hot(code, C) for code in codes])
    return vectors

# build graph

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

def input_word(y, var, C):
    '''
    y: All the one-hot vectors for the letters in the word, shaped [word_len, C]
    var: Dictionary of trainable weights and biases
    returns: y_pred, shaped [word_len + 1, C]  
    '''

    # x = [0, y<1>, y<2>,...,y<t>]
    zeros = tf.zeros([1, C])
    x = tf.concat([zeros, y], axis=0)

    # initialize a<0>
    a_t = np.zeros((1, nodes))

    # sequentially input letters
    y_pred = []
    for t in range(x.shape[0]):
        x_t = tf.reshape(x[t],[1, -1])
        y_pred_t, a_t =  input_letter(x_t, a_t, var)
        y_pred.append(y_pred_t[0]) 

    y_pred = tf.stack(y_pred)
    return y_pred

def compute_loss(y, y_pred):
    '''
    y: One hot vectors for correct letter values. Shaped [word_len + 1, C].
    y_pred: Predicted probability distribution for letter values. Shaped [word_len + 1, C].
    '''
    loss = -tf.reduce_sum(y * tf.log(y_pred))
    return loss

def create_trainer(C, character_map):

    # create variables
    Wa = tf.Variable(tf.random_normal([nodes + C, nodes], stddev=weights_init_stddev), name='Wa')
    ba = tf.Variable(tf.zeros([1, nodes]), name='ba')
    Wy = tf.Variable(tf.random_normal([nodes, C], stddev=weights_init_stddev), name='Wy')
    by = tf.Variable(tf.zeros([1, C]), name='by')
    var = {'Wa': Wa, 'ba': ba, 'Wy': Wy, 'by': by}

    y_holder = tf.placeholder(tf.float32, shape=[None, C])
    y_pred = input_word(y_holder, var, C)
    
    # add <END> tag to y_holder
    end_code_vec = one_hot(character_map['<END>'], C)
    y_extended = tf.concat([y_holder, end_code_vec], axis=0)

    # train
    loss = compute_loss(y_extended, y_pred)
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainer = adam.minimize(loss)

    return trainer


def train():

    # load words
    words = words_list(filename)
    print("Found " + str(len(words)) + " words!")

    # create dataset
    dataset = create_dataset(words)
    character_map, code_map = character_maps(words)
    C = len(character_map)

    # test dataset
    iterator = dataset.make_one_shot_iterator()
    next_item = iterator.get_next()
    sess = tf.Session()

    # one epoch
    while True:
        try:
            word = sess.run(next_item)
            y = vectorize_word(word, character_map, C)

            print(word)
        except Exception:
            print("Done!")
            break


