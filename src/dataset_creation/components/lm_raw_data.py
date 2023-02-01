import os
import pandas as pd
from src.common.constants import DF_INTERVALS_SETH, COL_SET_TYPE, COL_INTERVAL_ID
from src.common.path_resolvers import resolve_interval_raw_text_path, resolve_dataset_csv_dataframe, resolve_subset_data_dir, resolve_subset_raw_text_path


DATASET_NAME = 'Noah_V2'
path_df = resolve_dataset_csv_dataframe(DATASET_NAME)
df_token_voken = pd.read_csv(path_df)

df_train = df_token_voken[df_token_voken[COL_SET_TYPE] == 'train']
df_test  = df_token_voken[df_token_voken[COL_SET_TYPE] == 'test']

train_intervals = df_train[COL_INTERVAL_ID].unique().tolist()
test_intervals  = df_test[COL_INTERVAL_ID].unique().tolist()

assert not (set(train_intervals) & set(test_intervals))


def get_texts(interval_ids):
    texts = []
    for interval_id in interval_ids:
        raw_text_path = resolve_interval_raw_text_path(interval_id)
        df_raw_text = pd.read_csv(raw_text_path)
        interval_text = ' '.join(df_raw_text['word'].tolist())
        texts.append(interval_text)
    return texts


train_texts = get_texts(train_intervals)
test_texts  = get_texts(test_intervals)


raw_text_train_path = resolve_subset_raw_text_path(DATASET_NAME, 'train')
raw_text_test_path  = resolve_subset_raw_text_path(DATASET_NAME, 'test')

assert not os.path.exists(raw_text_train_path)
assert not os.path.exists(raw_text_test_path)

# Generates /home/stav/Data/Vokenization/Datasets/O_V7_N_V2_S_V2_128/train/raw.txt
with open(raw_text_train_path, 'w') as f:
    full_text = '\n\n\n'.join(train_texts)
    f.write(full_text)

# Generates /home/stav/Data/Vokenization/Datasets/O_V7_N_V2_S_V2_128/test/raw.txt
with open(raw_text_test_path, 'w') as f:
    full_text = '\n\n\n'.join(test_texts)
    f.write(full_text)


train_data_dir = resolve_subset_data_dir(DATASET_NAME, 'train')
test_data_dir  = resolve_subset_data_dir(DATASET_NAME, 'test')

# From Vokenization run:
#pyvok tokenization/tokenize_dataset.py /home/stav/Data/Vokenization/Datasets/Noah_V2/train raw.txt bert-base-uncased
print(f"pyvok tokenization/tokenize_dataset.py {train_data_dir} raw.txt bert-base-uncased")
print(f"pyvok tokenization/tokenize_dataset.py {test_data_dir} raw.txt bert-base-uncased")