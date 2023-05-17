import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, Activation
from tensorflow.python.keras.optimizer_v1 import SGD

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        classes.append(intent['tag'])

print(documents)
