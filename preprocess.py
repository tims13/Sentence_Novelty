import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import trim_string

des_path = 'data/'
data_path = des_path + 'annotation_sentence.xlsx'
data_pos_csv_path = des_path + 'data_pos.csv'
data_neg_csv_path = des_path + 'data_neg.csv'
data_irre_csv_path = des_path + 'data_irre.csv'
data_novel_csv_path = des_path + 'data_novel.csv'

train_test_ratio = 0.90
train_valid_ratio = 0.80

data_paths = [
    data_pos_csv_path,
    data_neg_csv_path,
    data_irre_csv_path,
    data_novel_csv_path
]

headers = [
    ['pos'],
    ['neg'],
    ['irre'],
    ['novel']
]

data = pd.read_excel(data_path)
data = data.iloc[:, 0:4]

for i in range(4):
    data_t = data.iloc[:, i].astype(str)
    data_t = data_t[data_t != 'nan']
    data_t.reset_index(drop=True, inplace=True)
    data_t.to_csv(data_paths[i], header=headers[i], index=0)

data_pos = pd.read_csv(data_pos_csv_path)
data_pos['label'] = 1
data_pos.rename(columns={'pos': 'text'}, inplace=True)
data_pos['text'] = data_pos['text'].apply(trim_string)

data_neg = pd.read_csv(data_neg_csv_path)
data_neg['label'] = 0
data_neg.rename(columns={'neg': 'text'}, inplace=True)
data_neg['text'] = data_neg['text'].apply(trim_string)

# Train - Test
df_pos_full_train, df_pos_test = train_test_split(data_pos, train_size = train_test_ratio, random_state=1)
df_neg_full_train, df_neg_test = train_test_split(data_neg, train_size = train_test_ratio, random_state=1)

# Train - valid
df_pos_train, df_pos_valid = train_test_split(df_pos_full_train, train_size = train_valid_ratio, random_state=1)
df_neg_train, df_neg_valid = train_test_split(df_neg_full_train, train_size = train_valid_ratio, random_state=1)
print("train-valid-test:")
print("pos:", df_pos_train.shape, df_pos_valid.shape, df_pos_test.shape)
print("neg:", df_neg_train.shape, df_neg_valid.shape, df_neg_test.shape)

df_train = pd.concat([df_pos_train, df_neg_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_pos_valid, df_neg_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_pos_test, df_neg_test], ignore_index=True, sort=False)

df_train.to_csv(des_path + 'train.csv', index=False)
df_valid.to_csv(des_path + 'valid.csv', index=False)
df_test.to_csv(des_path + 'test.csv', index=False)

print("Preprocess finished")