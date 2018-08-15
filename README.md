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

If we input a word like "cat" into our language model, it will output the probability of each letter given the previous ones. So we will have values for: P(c), P(a|c), P(t|ca), P(<i>end</i>|cat). Here, <i>end</i> is a special tag that tells the language model to stop adding letters. It is important when words can be of variable length. 

Since the basic rule of conditional probability is that:

<!--P(AB) = P(A|B) * P(B)-->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;P(AB)&space;=&space;P(A|B)&space;*&space;P(B)" title="\huge P(AB) = P(A|B) * P(B)" />

It follows that:

<!-- P(cat-<i>end</i>>) = P(<i>end</i>>|cat) * P(t|ca) * P(a|c) * P(c) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\LARGE&space;P('cat')&space;=&space;P(c)*P(a|c)*P(t|ca)*P(end|cat)" title="\LARGE P('cat') = P(c)*P(a|c)*P(t|ca)*P(end|cat)" />

In other words, we can compute P(word) by multiplying together the conditional probabilities of each letter.

## The Architecture

Our vanilla RNN consists of one simple "cell". At each time-step <i>t</i> we feed the previous letter <i>y<sup>\<t-1></sup></i> and the previous cell activation <i>a<sup>\<t-1></sup></i> into the cell, and it outputs a probability distribution for the current letter, <i>ŷ<sup>\<t></sup></i>. Since there will be no previous letter or activation for the first letter of a word, we'll simply feed in a vector of zeros for both.

<img height="270" src="language-model-diagram.png" title="language model diagram and equations"/>

## The Loss Function

To compute the loss for a given word, we first compute the losses at each time-step <i>t</i> (each letter). We use a loss function commonly used for softmax outputs, that considers only the probability assigned to the "correct" letter. You can see this in the equation below. Since y<sup>\<t></sup> is a one-hot letter vector with zero entries in all but one index, and anything times zero is zero, only the index where y<sub>i</sub><sup>\<t></sup> = 1 will count toward the sum. 

<!-- L(y<t>, y_hat<t> = - sum[y<t>log(y_hat<t>)] -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})=-\sum_{i}y_{i}^{<t>}log(\hat{^}^{y}_{i}^{<t>})" title="\huge \mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})=-\sum_{i}y_{i}^{<t>}log(\hat{^}^{y}_{i}^{<t>})" />

We can then compute the loss for a word by simply summing its per-letter losses.

<!-- L(y, y_hat) = sum[L(y<t>, y_hat<t>)] -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;\mathcal{L}(y,\hat{^}^{y})&space;=&space;\sum_{t}\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})" title="\huge \mathcal{L}(y,\hat{^}^{y}) = \sum_{t}\mathcal{L}(y^{<t>},\hat{^}^{y}^{<t>})" />

## Generating Words

Once we have trained the model, our goal is to invent new words of similar style to the training words. To sample words from our model, we pick the letter <i>y<sup>\<t></sup></i> randomly, weighting by the probability distribution output for <i>ŷ<sup>\<t></sup></i>. Our chosen <i>y<sup>\<t></sup></i> is then fed into the model at the next time step. This continues until the <i>end</i> tag is chosen. 

## Trained Models

I've included a couple of models trained with this code. 

## Training Your Own

To train your own character-level language model using this code, you only need a list of words stored in a ```.txt``` file. There should be one word per line.

To train:
1. Download this code and navigate to the project directory
2. Run ```python task.py --train True --data-dir [PATH_TO_YOUR_WORD_LIST]```
3. The model will be saved into a ```checkpoints``` directory by default. You can save it somewhere else by adding ```--save-dir [YOUR_SAVE_LOCATION]``` to the train command.

After some experimentation, I found that the following hyperparameters worked well:

```
nodes = 80
learning_rate = 0.001
num_epochs = 200
optimizer = Adam
```

These are set as the defaults, but if you wish to change them, there are command line arguments to set them all, except for the Adam optimizer, which you must switch in the code. 

## Acknowledgements

* Inspired by Andrew Ng's lecture on language models in this [Coursera Specialization](https://www.coursera.org/specializations/deep-learning) on Deep Learning.

* The excellent [Tolkiendil](http://www.tolkiendil.com/langues/english/i-lam_arth/compound_sindarin_names) website was the source of my training data for Elvish names.


