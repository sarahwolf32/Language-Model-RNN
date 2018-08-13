import pickle

# create dictionary of variables
# create code map
# put C, N, code map, and variables in one dictionary
# can I save and restore that dictionary?

filename = 'test_saves/test_config.pickle'

'''
# can I save and restore any arbitrary dictionary?
test_dict = {
    'a': 123,
    'b': 5,
    'c': 10
}

# save
with open(filename, 'wb') as handle:
    pickle.dump(test_dict, handle)
'''

# restore
with open(filename, 'rb') as handle:
    restored_dict = pickle.load(handle)
print(restored_dict)