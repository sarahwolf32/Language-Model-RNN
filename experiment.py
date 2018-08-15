from model import Model
import argparse

parser = argparse.ArgumentParser()
config = parser.parse_args()

# Somewhere along the way, my letters are getting corrupted.
# Find it!

# read in word_list
with open('word_lists/elvish_words.txt') as f:
    words = f.readlines()
words = [w.rstrip().lower() for w in words]

#print words => still prints characters correctly
'''
for word in words:
    print(word)
'''

model = Model(config)
character_map, code_map = model.character_maps(words)
for char in character_map.keys():
    print(char)

# print - the characters are correct, yes?
# grab example
# paste and print
# break it up into chars, put them in a dictionary, then print the keys
# is it still correct?
    # If no (and I suspect no), then this is our bug-in-a-jar