# Language-Model-RNN

A letter-based language model for generating new words similar to a trained-on list of words. Includes a trained model for generating Tolkein-style Elvish names.

## A Word on Word Lengths

Since we impose no artificial limit on the length of a word, initial word-lengths will be largely up to chance. The model will continue adding letters until it predicts the <EOW> token. At the beginning, before much training has taken place, we would expect the probability of getting <EOW> to be roughly 1/C, where C is the number of possible characters (including tokens like <EOW>). 

Technically, it would be possible for the untrained model to produce a word thousands of characters long, which is not really what we want. Do we need to be concerned about the possibility of giant words causing problems, or this unlikely enough to not be an issue? Under these conditions, how long would we expect words to be? 

Let's think it through:

<!-- P(end) = 1/c -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;P(end)&space;=&space;\frac{1}{C}" title="\huge P(end) = \frac{1}{C}" />

<!-- P(letter) = (c - 1)/c -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;P(letter)&space;=&space;\frac{C-1}{C}" title="\huge P(letter) = \frac{C-1}{C}" />

<!-- P(l) = P(letter)^l * P(end) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;P(l)&space;=&space;P(letter)^{l}&space;*&space;P(end)" title="\huge P(l) = P(letter)^{l} * P(end)" />

<!-- P(l) = ((c - 1)/c)^l * (1/c) -->
<img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;P(l)&space;=&space;\left&space;(&space;\frac{C-1}{C}&space;\right&space;)^{l}&space;*&space;\left&space;(&space;\frac{1}{C}&space;\right&space;)" title="\huge P(l) = \left ( \frac{C-1}{C} \right )^{l} * \left ( \frac{1}{C} \right )" />

## Acknowledgements

* I used the excellent [Tolkiendil](http://www.tolkiendil.com/langues/english/i-lam_arth/compound_sindarin_names) website was the source of my training data for Elvish names.