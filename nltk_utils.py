import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

# Break down sentence into words
def tokenize(sentence):
    return nltk.wordpunct_tokenize(sentence)

# reduce a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma.
def stem(word):
    return stemmer.stem(word.lower())

# Create a binary vector representing sentence using bag of words
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag



