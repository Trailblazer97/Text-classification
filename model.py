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

max_seq_len = 1000
max_words = 20000
embedding_dim = 100
split_fraction = 0.2


#Removing unnecessary content
def remove(string):
    string = re.sub(r"\'s", "", string)     
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)   
    return string.strip().lower()


def df_to_list(df, texts, labels):
    sorted_type = sorted(set(df['Type']))
    indexed_type = dict((type_name, index) for index, type_name in enumerate(sorted_type))
    
    df = df.replace({"Type": indexed_type})
    for i in range(df.shape[0]):
        text = BeautifulSoup(df.News[i])
        texts.append(remove(str(text.get_text().encode())))
    
    for i in df['Type']:
        labels.append(i)
    
    return texts, labels, sorted_type, indexed_type
        
        
def tokenize(texts, labels):
    global max_seq_len, split_fraction, max_words
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    print('Number of Unique Tokens', len(word_index))


    data = pad_sequences(sequences, maxlen=max_seq_len)
    labels = to_categorical(np.asarray(labels))
    print('Shape of Data Tensor:', data.shape)
    print('Shape of Label Tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(split_fraction * data.shape[0])

    X_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    X_test = data[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]

    embeddings_index = {}
    f = open('glove.6B.100d.txt',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
    return word_index, embedding_matrix, data, labels, sequences
        
def model(word_index, embedding_matrix, sorted_type, data, labels):
    global embedding_dim, max_seq_len
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=True)


    sequence_input = Input(shape=(max_seq_len, ), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    preds = Dense(len(sorted_type), activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    
    print("Bidirectional LSTM")
    model.summary()


    cp = ModelCheckpoint('model_rnn.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
    history = model.fit(data, labels, validation_split=0.3, epochs=10, batch_size=10, callbacks=[cp])

    return model, history

def save_model(model):
    
    filename = 'rnn_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    #prediction part
    '''
    pred_class = []
    output_class = []
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(X_train)
    for row in y_pred:
    rows = list(row)
    pred_class.append(rows.index(max(rows)))
    for i in pred_class:
    output_class.append(list(indexed_type.keys())[list(indexed_type.values()).index(i)])
    '''   
    
def plot(history):
    fig1 = plt.figure()
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves :RNN',fontsize=16)
    fig1.savefig('loss_rnn.png')
    plt.show()


    fig2=plt.figure()
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves : RNN',fontsize=16)
    fig2.savefig('accuracy_rnn.png')
    plt.show()
    
    
    
