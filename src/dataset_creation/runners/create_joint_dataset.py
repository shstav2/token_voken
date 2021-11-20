"""
Create joint dataset of oliver and noah
---------------------------------------
"""
import os
import numpy as np
import pandas as pd
from src.dataset_creation.components._2_save_dataset import save_full_dataset, save_h5_files
from src.common.constants import DATASETS_VOKENIZATION

COL_DATASET   = 'dataset'
COL_VOKEN_ID  = 'voken_id'
COL_SPEAKER   = 'speaker'

# Input Datasets
DATASET_OLIVER = 'Oliver_V3'
DATASET_NOAH   = 'Noah_V1'

# Output Dataset target path
new_dataset_name = f'{DATASET_OLIVER}_{DATASET_NOAH}_SHUFFLE_INTERVAL'
new_dataset_path = os.path.join(DATASETS_VOKENIZATION, new_dataset_name)



def get_speaker_df_token_voken(dataset_name):
    # /home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken_pkl.csv
    data_path = resolve_dataset_dataframe(dataset_name)
    df_speaker = pd.read_pickle(data_path)
    return df_speaker


def get_full_df_token_voken(dataset_oliver, dataset_noah):
    df_oliver = get_speaker_df_token_voken(dataset_oliver)
    df_noah   = get_speaker_df_token_voken(dataset_noah)
    df_oliver[COL_SPEAKER] = 'oliver'
    df_noah[COL_SPEAKER]   = 'noah'
    df_all = pd.concat([df_oliver, df_noah])
    return df_all



df_all[COL_SPEAKER].value_counts()

df_all[COL_VOKEN_ID] = range(0, len(df_all))

save_full_dataset(new_dataset_path, df_all)

from src.common.path_resolvers import resolve_dataset_tokens_path
from src.common.file_utils import read_hdf
DATASET_OLIVER = 'Oliver_V3'

tokens_oliver_train_path = resolve_dataset_tokens_path(DATASET_OLIVER, 'train')
tokens_oliver_test_path = resolve_dataset_tokens_path(DATASET_OLIVER,  'test')

tokens_noah_train_path = resolve_dataset_tokens_path(DATASET_NOAH, 'train')
tokens_noah_test_path = resolve_dataset_tokens_path(DATASET_NOAH,  'test')


tokens_oliver_train_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V3/train/tokens.hdf5'
tokens_oliver_test_path  = '/home/stav/Data/Vokenization/Datasets/Oliver_V3/test/tokens.hdf5'

tokens_noah_train_path = '/home/stav/Data/Vokenization/Datasets/Noah_V1/train/tokens.hdf5'
tokens_noah_test_path  = '/home/stav/Data/Vokenization/Datasets/Noah_V1/test/tokens.hdf5'

# /home/stav/Data/Vokenization/Datasets/Oliver_V1/train/tokens.hdf
df_token_voken_path = resolve_dataset_tokens_path(dataset_name, subset)

tokens_train_oliver = read_hdf(tokens_oliver_train_path, 'tokens')
tokens_test_oliver  = read_hdf(tokens_oliver_test_path, 'tokens')
len(tokens_train_oliver), len(tokens_test_oliver) # (64,713, 16,999) = 81,712

tokens_train_noah = read_hdf(tokens_noah_train_path, 'tokens')
tokens_test_noah  = read_hdf(tokens_noah_test_path,  'tokens')
len(tokens_train_noah), len(tokens_test_noah)     # (130,232, 32,393) = 162,625



split_index_oliver = 64713
df_oliver[COL_DATASET] = None
df_oliver.iloc[:split_index_oliver+1, df_oliver.columns.get_loc(COL_DATASET)] = 'train'
df_oliver.iloc[split_index_oliver:, df_oliver.columns.get_loc(COL_DATASET)]   = 'test'
df_oliver[COL_DATASET].value_counts()
# train    64713
# test     16999


split_index_noah = 130232
df_noah[COL_DATASET] = None
df_noah.iloc[:split_index_noah+1, df_noah.columns.get_loc(COL_DATASET)] = 'train'
df_noah.iloc[split_index_noah:, df_noah.columns.get_loc(COL_DATASET)]   = 'test'
df_noah[COL_DATASET].value_counts()
# train    130232
# test      32393



df_all = pd.concat([df_oliver, df_noah])
df_all['speaker'].value_counts()
df_all[COL_DATASET].value_counts()

df_all.shape   # (244,337, 15)
save_full_dataset(new_dataset_path, df_all)

"""
train
"""

df_all_train = df_all[df_all[COL_DATASET] == 'train']
df_all_train.shape # (194,945, 15)

df_all_train['speaker'].value_counts()
# noah      130232
# oliver     64713

train_path = os.path.join(new_dataset_path, 'train')
train_path
save_h5_files(train_path, df_all_train)

"""
test
"""
df_all_test = df_all[df_all[COL_DATASET] == 'test']
df_all_test.shape # (49,392, 15)

df_all_test['speaker'].value_counts()
# noah      32393
# oliver    16999

test_path = os.path.join(new_dataset_path, 'test')
test_path
save_h5_files(test_path, df_all_test)

vokens_mix_train_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V3_Noah_V1/train/vokens.hdf5'
vokens_mix_test_path  = '/home/stav/Data/Vokenization/Datasets/Oliver_V3_Noah_V1/test/vokens.hdf5'

vokens_mix_train = read_hdf(vokens_mix_train_path, 'vokens')
vokens_mix_test  = read_hdf(vokens_mix_test_path,  'vokens')



"""
np_vokens_path = os.path.join(DATASETS_VOKENIZATION, 'Oliver_V3_Noah_V1', 'vokens.npy')
np_vokens = np.load(np_vokens_path)
np_vokens.shape # (244,337, 16) -> 81,712 (oliver )+ 162,625 (noah)
"""
