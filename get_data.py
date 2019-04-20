import os  #operating system dependent functionality
import pandas as pd
from sklearn.model_selection import train_test_split

data_folder = "../dataset/data_files"
folders = ["business","entertainment","politics","sport","tech"]

os.chdir(data_folder) # changes the current directory to the given path

x = []
y = []

for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        print("Reading file:", file_path)
        with open(file_path) as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data)
        y.append(i)
   
data = {'News': x, 'Type': y}       
df = pd.DataFrame(data)
print("Writing .csv file ...")
df.to_csv('../dataset.csv', index=False)
