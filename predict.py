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

text_to_list = []
pred_class = []
output_class = []
max_seq_len = 1000

loaded_model = pickle.load(open('rnn_model.sav', 'rb'))
indexed_type = pickle.load(open('indexed_type.sav', 'rb'))
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


text = BeautifulSoup(input("Enter the text to be classified..."))
text = model.remove(str(text.get_text().encode()))
text_to_list.append(text)


sequence = tokenizer.texts_to_sequences(text_to_list)
data = pad_sequences(sequence, maxlen=max_seq_len)
print('Shape of Data Tensor:', data.shape)


y_pred = loaded_model.predict(data)
for row in y_pred:
    rows = list(row)
    pred_class.append(rows.index(max(rows)))
for i in pred_class:
    output_class.append(list(indexed_type.keys())[list(indexed_type.values()).index(i)])
    
print(f'The document belongs to \"{output_class[0]}\" class.')
