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
variables = {'Wa': Wa, 'ba': ba, 'Wy': Wy, 'by': by}
variables_list = variables.values()

# create optimizer
optimizer = tf.train.GradientDescentOptimizer(model.learning_rate)


# train loop
for epoch in range(model.num_epochs):
    for tensor in iterator:
        word = tensor.numpy()
        y = model.vectorize_word(word, character_map, C)
        with tf.GradientTape() as tape:
            y_pred = model.input_word(y, variables, C)

            # get loss
            y_with_end = model.add_end_tag(y, C, character_map)
            loss = model.compute_loss(tf.cast(y_with_end, tf.float32), y_pred)
            print(word + ": loss = " + str(loss.numpy()))

            # get gradients
            grads = tape.gradient(loss, variables_list)
            optimizer.apply_gradients(zip(grads, variables_list))







