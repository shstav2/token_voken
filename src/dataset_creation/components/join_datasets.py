import os
import math
import pandas as pd
from random import shuffle
from src.common.path_resolvers import resolve_dataset_dataframe
from src.common.constants import SPLIT_INDEX, \
    COL_SPEAKER, COL_VOKEN_ID, COL_SET_TYPE


def get_speaker_df_token_voken(dataset_name, speaker_name):
    # /home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken_pkl.csv
    data_path = resolve_dataset_dataframe(dataset_name)
    df_speaker = pd.read_pickle(data_path)
    df_speaker[COL_SPEAKER] = speaker_name
    mark_train_test_split(df_speaker, dataset_name)
    return df_speaker


def get_full_df_token_voken(dataset_oliver, dataset_noah):
    df_oliver = get_speaker_df_token_voken(dataset_oliver, 'Oliver')
    df_noah   = get_speaker_df_token_voken(dataset_noah, 'Noah')
    df_all    = pd.concat([df_oliver, df_noah])
    return df_all


def split_dataframe(df, chunk_size):
    chunks = list()
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def shuffle_blocks(df_all, block_size):
    lst_df_blocks = split_dataframe(df_all, block_size)
    assert len(df_all) == sum(list(map(len, lst_df_blocks)))
    lst_df_blocks_full, lst_df_blocks_reminder = lst_df_blocks[:-1], lst_df_blocks[-1]
    assert len(df_all) == (sum(list(map(len, lst_df_blocks_full))) + len(lst_df_blocks_reminder))
    indices = list(range(len(lst_df_blocks_full)))
    shuffle(indices)
    lst_df_blocks_full_shuffled = [lst_df_blocks_full[i] for i in indices]
    lst_df_blocks_shuffled = lst_df_blocks_full_shuffled + [lst_df_blocks_reminder]
    df_all_shuffled = pd.concat(lst_df_blocks_shuffled)
    assert len(df_all) == len(df_all_shuffled)
    first_idx = indices[0]
    assert df_all[first_idx * block_size: first_idx * block_size + 10].equals(df_all_shuffled[:10])
    return df_all_shuffled, indices


def mark_train_test_split(df_speaker, dataset_name):
    split_index = SPLIT_INDEX[dataset_name]
    df_speaker[COL_SET_TYPE] = None
    df_speaker.iloc[:split_index+1, df_speaker.columns.get_loc(COL_SET_TYPE)] = 'train'
    df_speaker.iloc[split_index:, df_speaker.columns.get_loc(COL_SET_TYPE)]   = 'test'


def get_df_all_train_test(dataset_oliver, dataset_noah):
    df_all = get_full_df_token_voken(dataset_oliver, dataset_noah)
    df_train = df_all[df_all[COL_SET_TYPE] == 'train']
    df_test  = df_all[df_all[COL_SET_TYPE] == 'test']
    return df_all, df_train, df_test


def shuffle_train_test(df_train, df_test, block_size):
    df_train_shuffled, train_indices = shuffle_blocks(df_train, block_size)
    print('Train:')
    print(df_train_shuffled[COL_SPEAKER].value_counts().astype('float32'))
    # noah     162,625
    # oliver    81,712
    print('Test:')
    df_test_shuffled, test_indices = shuffle_blocks(df_test, block_size)
    print(df_test_shuffled[COL_SPEAKER].value_counts().astype('float32'))
    return df_train_shuffled, train_indices, df_test_shuffled, test_indices
