'''
Eager execution learnings:
    "Action at a distance" approaches generally don't work.
    Just write normal, do-things-explicitly code.

    You can "flip a switch" to turn valid eager code into a graph.

    Allows "dynamic control flow", where actions depend on the actual values being executed.

    A new way of handling gradients. Gradient tape!

    Can use normal python profiling tools like cProfile.

    Can create custom gradients for things like gradient clipping.

    Object-based saving:
        - current checkpoints depend on tensor names, which are fragile
        - tfe.Checkpoint
        - example in talk, at 10:36
        - can save or load whole model, or any subset of model
            - e.g., in GAN, save D and G separately, 
              then load up only G when you need it for something else


    - Advice on architecture:
        - Good to write a Model class that inherits from tf.keras-model
            - well tested
            - includes utilities for saving & loading the model
            - eager-friendly
    - Things not compatible with eager excution:
        - Variables (use tfe.Variable instead)
            - Is a python object, so if you set it to None, it is no longer in memory
            - Yay, cleans up memory leaks!
        - Placeholders
        - tf.summary (use tf.contrib.summary instead)
        - tf.metrics (use tfe.metrics)

    - Must call tf.enable_eager_execution() at the very beginning of the code, 
      regardless of when you start using it.

    - Call .numpy() on your EagerTensor to print its value as a numpy array
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# old way of doing gradients
'''
w = tf.Variable([[2.0]])
loss = w * w
dw, = tf.gradients(loss, [w])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(grads))
'''


tf.enable_eager_execution()

# gradients with eager

w = tfe.Variable([[2.0]])
with tf.GradientTape() as tape:
    loss = w * w
dw, = tape.gradient(loss, [w])
print("w = " + str(w.numpy()))
print("dw = " + str(dw.numpy()))

print("optimizing...")
optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer.apply_gradients(zip([dw], [w]))

print("w: " + str(w.numpy()))

# custom gradients - for gradient clipping! 
# Prevents exploding gradients!
# Creates a version of tf.identity that clips its gradient on the backward pass
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
    y = tf.identity(x)
    def grad_fn(dresult):
        return [tf.clip_by_norm(dresult, norm), None]
    return y, grad_fn




'''
# import words
words = model.words_list(model.filename)
print("Found " + str(len(words)) + " words!")

# create dataset
dataset = model.create_dataset(words)
character_map, code_map = model.character_maps(words)
C = len(character_map)

iterator = dataset.make_one_shot_iterator()

for word in iterator:
    print(word)
'''
