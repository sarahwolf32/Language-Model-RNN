# Character-Level Language Model with a Vanilla RNN

A letter-based language model for generating new words similar to a trained-on list of words. Includes a trained model for generating Tolkein-style Elvish names.

## Why RNNs?

A great deal of the data we care about - music, language, anything with a time dimension - comes in the form of <i>sequences</i>. Unfortunately, standard fully connected neural networks do not handle sequences well. They expect input and output data to be of a fixed size, which is not well-suited to working with words or sentences, which can vary in length. While one could pad the sequences to a max size, this would still require a larger network than needed, an inefficiency we'd like to avoid.

The more serious limitation is that if you <i>did</i> put a sequence into a standard neural network, it would not be able to generalize its learnings at a given input index to other input positions. For example, if the sentence 'Harry Potter took off his glasses' helped it learn that 'Harry' is a name, this would not necessarily improve its ability to predict that 'Harry' is a name in the sequence 'She looked up at Harry', because it occurs at a different position.

Recurrent Neural Networks (RNNs) are a class of neural networks that <i>share parameters</i> over input positions. This allows learnings to at one time-step to be potentially generalized to others. They also maintain an internal state (which we can think of as "memory") that is passed forward between time-steps.

While there are many variations on RNNs, this is an implementation of a vanilla RNN, the simplest version. 

## Language Models



## Acknowledgements

* Inspired by Andrew Ng's lecture on language models in this [Coursera Specialization](https://www.coursera.org/specializations/deep-learning) on Deep Learning.

* I used the excellent [Tolkiendil](http://www.tolkiendil.com/langues/english/i-lam_arth/compound_sindarin_names) website was the source of my training data for Elvish names.
