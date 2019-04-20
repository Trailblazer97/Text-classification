# Text-classification
The project aims to classify text-based documents into appropriate categories. Today, we are surrounded by documents, news articles, online blogs and other forms of text. It has therefore become necessary to give some kind of order to all this textual information. A good way to do this is to categorize the text into labelled groups of similar articles. Sequence classification is a predictive modeling problem where you have some sequence of inputs over space or time and the task is to predict a category for the sequence. What makes this problem difficult is that the sequences can vary in length, be comprised of a very large vocabulary of input symbols and may require the model to learn the long-term context or dependencies between symbols in the input sequence. A solution to this problem of text categorization would be of great use in organizing documents in offices as well as providing news article feeds of the readers’ preference. The scope of the project comprises of categorizing text articles into five different classes. The dataset used for training is the freely online available dataset - BBC News Dataset. The classification is done with the help of special type of neural networks, called the LSTMs.


The code section is divided into 4 parts: get_data.py, model.py, train.py and predict.py. The get_data.py file is used to convert the raw data files in text format from various documents into a consolidated .csv file format. model.py has various functions for further processing this data. train.py and predict.py, as their names suggest are used for training on the dataset and predicting the class/category of a text article respectively.

 

# Running the codes
Sequence of file execution is get_data.py -> model.py -> train.py -> predict.py

Once train.py file is executed, a tokenizer and an LSTM model are saved locally in a .sav and .pickle file format respectively. They are later retrieved for prediction. 

# Downloading additional files
Dataset can be downloaded here: http://mlg.ucd.ie/datasets/bbc.html
model.py uses  100d GloVe (Global Vectors) word embedding which can be downloaded from https://nlp.stanford.edu/projects/glove/

# Authors
* **Yash Barapatre** - [Trailblazer97](https://github.com/Trailblazer97)
* **Aditya Samant**


# License
This project is licensed under the MIT License - see the LICENSE.md file for details.
