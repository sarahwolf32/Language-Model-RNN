# Character-Level Language Model with a Vanilla RNN

A letter-based language model for generating new words similar to a trained-on list of words. Includes a trained model for generating Tolkein-style Elvish names.

## Why RNNs?

A great deal of the data we care about - music, language, anything with a time dimension - comes in the form of <i>sequences</i>. Unfortunately, standard fully connected neural networks do not handle sequences well. They expect input and output data to be of a fixed size, which is not well-suited to working with words or sentences, which can vary in length. While one could pad the sequences to a max size, this would still require a larger network than needed, an inefficiency we'd like to avoid.

The more serious limitation is that if you <i>did</i> put a sequence into a standard neural network, it would not be able to generalize its learnings at a given input index to other input positions. For example, if the sentence 'Harry Potter took off his glasses' helped it learn that 'Harry' is a name, this would not necessarily improve its ability to predict that 'Harry' is a name in the sequence 'She looked up at Harry', because it occurs at a different position.

Recurrent Neural Networks (RNNs) are a class of neural networks that <i>share parameters</i> over input positions. This allows learnings to at one time-step to be potentially generalized to others. They also maintain an internal state (which we can think of as "memory") that is passed forward between time-steps.

While there are many variations on RNNs, this is an implementation of a vanilla RNN, the simplest version. 

## Language Models

<!-- a<t> = tanh(Wa[y<t-1>, a<t-1>] + ba) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;a^{<t>}=tanh(W_{a}[a^{t-1},y^{<t-1>}]&space;&plus;&space;b_{a})" title="\huge a^{<t>}=tanh(W_{a}[a^{t-1},y^{<t-1>}] + b_{a})" />

<!-- y_hat<t> = softmax(Wy[a<t>] + by) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\hat{^}^{y}^{<t>}=softmax(W_{y}\cdot&space;a^{<t>}&space;&plus;&space;b_{y})" title="\huge \hat{^}^{y}^{<t>}=softmax(W_{y}\cdot a^{<t>} + b_{y})" />

## The Loss Function

To compute the loss for a given word, we first compute the losses at each time-step <i>t</i> (each letter). We use a loss function commonly used for softmax outputs, that considers only the probability assigned to the "correct" letter. You can see this in the equation below. Since y<sup>\<t></sup> is a one-hot letter vector with zero entries in all but one index, and anything times zero is zero, only the index where y<sub>i</sub><sup>\<t></sup> = 1 will count toward the sum. 

<!-- L(y<t>, y_hat<t> = - sum[y<t>log(y_hat<t>)] -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})=-\sum_{i}y_{i}^{<t>}log(\hat{^}^{y}_{i}^{<t>})" title="\huge \mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})=-\sum_{i}y_{i}^{<t>}log(\hat{^}^{y}_{i}^{<t>})" />

Once we have the per-letter losses, we can compute the loss for the word by simply summing them.

<!-- L(y, y_hat) = sum[L(y<t>, y_hat<t>)] -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\mathcal{L}(y,\hat{^}^{y})&space;=&space;\sum_{t}\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})" title="\huge \mathcal{L}(y,\hat{^}^{y}) = \sum_{t}\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})" />

## Acknowledgements

* Inspired by Andrew Ng's lecture on language models in this [Coursera Specialization](https://www.coursera.org/specializations/deep-learning) on Deep Learning.

* I used the excellent [Tolkiendil](http://www.tolkiendil.com/langues/english/i-lam_arth/compound_sindarin_names) website was the source of my training data for Elvish names.
