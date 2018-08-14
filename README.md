# Character-Level Language Model with a Vanilla RNN

A letter-based language model for generating new words similar to a trained-on list of words. Includes a trained model for generating Tolkein-style Elvish names.

## Why RNNs?

A great deal of the data we care about - music, language, anything with a time dimension - comes in the form of <i>sequences</i>. Unfortunately, standard fully connected neural networks do not handle sequences well. They expect input and output data to be of a fixed size, which is not well-suited to working with words or sentences, which can vary in length. While one could pad the sequences to a max size, this would still require a larger network than needed, an inefficiency we'd like to avoid.

The more serious limitation is that if you <i>did</i> put a sequence into a standard neural network, it would not be able to generalize its learnings at a given input index to other input positions. For example, if the sentence 'Harry Potter took off his glasses' helped it learn that 'Harry' is a name, this would not necessarily improve its ability to predict that 'Harry' is a name in the sequence 'She looked up at Harry', because it occurs at a different position.

Recurrent Neural Networks (RNNs) are a class of neural networks that <i>share parameters</i> over input positions. This allows learnings to at one time-step to be potentially generalized to others. They also maintain an internal state (which we can think of as "memory") that is passed forward between time-steps.

While there are many variations on RNNs, this is an implementation of a vanilla RNN, the simplest version. 

## Language Models

Broadly speaking, language models are built to predict the probability of a sequence. This can be used to:
* Generate new plausible sequences 
    * E.g., inventing words, writing sentences, writing music
* Pick the most plausible sequence given a few options 
    * E.g., helping a handwriting recognition system decide that an unreadable letter in a three-letter word between <i>a</i> and <i>d</i> is probably <i>n</i>.
* Suggest likely ways to complete a partial sequence
    * E.g., autocomplete.

In our case, we will train a model to predict P(word), given the sequence of letters within it. At each time step, the RNN attempts to predict the next letter, given the previous letter and its internal memory state. 

For example, in the space of English words, we would expect the P(a|ca) to be low, since "caa" is not a common combination in English. By the same token, P(r|ca) should be higher, because there are many English words that include "car", like "car", "carry", "carnival", etc. This is the sort of thing we want our model to learn.

If we input a word like "cat" into our language model, it will output the probability of each letter given the previous ones. So we will have:

P(c), P(a|c), P(t|ca), P(\<END>|cat)

Here, \<END> is a special tag that tells the language model to stop adding letters. It is important when words can be of variable length. 

Since the basic rule of conditional probability is that:

P(AB) = P(A|B) * P(B)

It follows that:

P(cat\<END>) = P(\<END>|cat) * P(t|ca) * P(a|c) * P(c)

In other words, we can compute P(word) by multiplying together the conditional probabilities of each letter.

## The Architecture

<!-- a<t> = tanh(Wa[y<t-1>, a<t-1>] + ba) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;a^{<t>}=tanh(W_{a}[a^{t-1},y^{<t-1>}]&space;&plus;&space;b_{a})" title="\huge a^{<t>}=tanh(W_{a}[a^{t-1},y^{<t-1>}] + b_{a})" />

<!-- y_hat<t> = softmax(Wy[a<t>] + by) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\hat{^}^{y}^{<t>}=softmax(W_{y}\cdot&space;a^{<t>}&space;&plus;&space;b_{y})" title="\huge \hat{^}^{y}^{<t>}=softmax(W_{y}\cdot a^{<t>} + b_{y})" />

## The Loss Function

To compute the loss for a given word, we first compute the losses at each time-step <i>t</i> (each letter). We use a loss function commonly used for softmax outputs, that considers only the probability assigned to the "correct" letter. You can see this in the equation below. Since y<sup>\<t></sup> is a one-hot letter vector with zero entries in all but one index, and anything times zero is zero, only the index where y<sub>i</sub><sup>\<t></sup> = 1 will count toward the sum. 

<!-- L(y<t>, y_hat<t> = - sum[y<t>log(y_hat<t>)] -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})=-\sum_{i}y_{i}^{<t>}log(\hat{^}^{y}_{i}^{<t>})" title="\huge \mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})=-\sum_{i}y_{i}^{<t>}log(\hat{^}^{y}_{i}^{<t>})" />

We can then compute the loss for a word by simply summing its per-letter losses.

<!-- L(y, y_hat) = sum[L(y<t>, y_hat<t>)] -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\mathcal{L}(y,\hat{^}^{y})&space;=&space;\sum_{t}\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})" title="\huge \mathcal{L}(y,\hat{^}^{y}) = \sum_{t}\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})" />

## Generating Words

Once we have trained the model, our goal is to invent new words of similar style to the training words. 


## Acknowledgements

* Inspired by Andrew Ng's lecture on language models in this [Coursera Specialization](https://www.coursera.org/specializations/deep-learning) on Deep Learning.

* The excellent [Tolkiendil](http://www.tolkiendil.com/langues/english/i-lam_arth/compound_sindarin_names) website was the source of my training data for Elvish names.


