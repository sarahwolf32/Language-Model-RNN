import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import model 

'''
Use eager execution to train my RNN.
'''

tf.enable_eager_execution()

# import words
words = model.words_list(model.filename)
print("Found " + str(len(words)) + " words!")

# create dataset
dataset = model.create_dataset(words)
character_map, code_map = model.character_maps(words)
C = len(character_map)
iterator = dataset.make_one_shot_iterator()

# create variables
nodes = model.nodes
weights_init_stddev = model.weights_init_stddev
Wa = tfe.Variable(tf.random_normal([nodes + C, nodes], stddev=weights_init_stddev), name='Wa')
ba = tfe.Variable(tf.zeros([1, nodes]), name='ba')
Wy = tfe.Variable(tf.random_normal([nodes, C], stddev=weights_init_stddev), name='Wy')
by = tfe.Variable(tf.zeros([1, C]), name='by')
var = {'Wa': Wa, 'ba': ba, 'Wy': Wy, 'by': by}

print("C: " + str(C))
print("Wa: " + str(Wa.shape))
print("trainable? " + str(Wa.trainable))
print("ba: " + str(ba.shape))
print("Wy: " + str(Wy.shape))
print("by: " + str(by.shape))



# train loop
for tensor in iterator:
    word = tensor.numpy()
    print(word)
    y = model.vectorize_word(word, character_map, C)
    print("y: " + str(y.shape))
    y_pred = model.input_word(y, var, C)
    print("y_pred: " + str(y_pred.shape))

    # get loss
    y_with_end = model.add_end_tag(y, C, character_map)
    print("y-with-end: " + str(y_with_end.shape))
    lg = tf.log(y_pred)
    print("log(y-pred) = " + str(lg.shape))
    print("lg type: " + str(lg.dtype))
    print("y type: " + str(y_with_end.dtype))
    print("y type: " + str(y.dtype))

    loss = model.compute_loss(tf.cast(y_with_end, tf.float32), y_pred)
    print("loss: " + str(loss.numpy()))


    break






