import re 
import pandas as pd
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
#%matplotlib inline
import pickle
import model


texts = []
labels = []



df = pd.read_csv('../dataset/dataset.csv')
df = df.dropna()
df = df.reset_index(drop=True)
print("Information on the dataset")
print('Shape of dataset ', df.shape)
print(df.columns)
print('No. of unique news types: ', len(set(df['Type'])))
print(df.head())


texts, labels, sorted_type, indexed_type = model.df_to_list(df, texts, labels)
pickle.dump(indexed_type, open('indexed_type.sav', 'wb'))
word_index, embedding_matrix, data, labels, sequences = model.tokenize(texts, labels)
model, history = model.model(word_index, embedding_matrix, sorted_type, data, labels)
model.save_model(model)
model.plot(history)
