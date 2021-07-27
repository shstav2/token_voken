import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.data_loader import load_valid_intervals
from src.common.path_resolvers import resolve_interval_text_tokenized_path, \
    resolve_interval_facial_embedding_path, resolve_interval_facial_embeddings_dir

COL_BERT_TOKEN_ID       = 'token_id'
COL_WORD_FRAME_SELECTED = 'selected_frame'
COL_VOKEN_ID            = 'voken_id'
COL_VOKEN_PATH          = 'voken_path'


def read_sorted_intervals():
    df_intervals = load_valid_intervals()
    df_intervals['start_time'] = pd.to_timedelta(df_intervals['start_time'])
    df_intervals['end_time']   = pd.to_timedelta(df_intervals['end_time'])
    df_intervals.sort_values(by=['video_id', 'start_time'], ascending=True, inplace=True)
    return df_intervals


def read_embedding(interval_id, frame_id, last_frame):
    if last_frame < frame_id :
        print(f'Interval {interval_id} selected frame fixed {frame_id} -> {last_frame}')
    path = resolve_interval_facial_embedding_path(interval_id, min(frame_id, last_frame))
    try:
        emb = np.load(path)
    except Exception as e:
        print(e)
        print("ERROR!", interval_id, frame_id, last_frame, path)
        raise e
    return emb


def get_token_voken(df_intervals):
    all_df_tokens = []
    for video_id, interval_id in tqdm(df_intervals[['video_id', 'interval_id']].values):
        bert_tokens_path = resolve_interval_text_tokenized_path(interval_id)
        df_tokens = pd.read_csv(bert_tokens_path)
        df_tokens['interval_id'] = interval_id
        df_tokens['video_id'] = video_id
        # todo: move to component
        last_frame, _ = max(os.listdir(resolve_interval_facial_embeddings_dir(interval_id))).split('.')
        df_tokens['voken'] = df_tokens[COL_WORD_FRAME_SELECTED].apply(
            lambda frame_id: read_embedding(interval_id, frame_id, int(last_frame)))
        all_df_tokens.append(df_tokens)
    df_token_voken = pd.concat(all_df_tokens)
    df_token_voken[COL_VOKEN_PATH] = df_token_voken['video_id'] + '_' + df_token_voken['interval_id'] + '_'\
                                 + df_token_voken[COL_WORD_FRAME_SELECTED].astype(str)
    df_token_voken[COL_VOKEN_ID] = range(1, len(df_token_voken) + 1)
    return df_token_voken


df_intervals = read_sorted_intervals()
df_token_voken = get_token_voken(df_intervals)

# >>> df_token_voken.shape
# (81712, 10)

df_token_voken.to_csv('/home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken.csv')
df_token_voken.to_pickle('/home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken_pkl.csv')



def create_token_ids_hdf(df_token_voken):
    token_ids = df_token_voken[COL_BERT_TOKEN_ID].tolist()
    # resolve_dataset_tokens_path
    with h5py.File('/home/stav/Data/Vokenization/Datasets/Oliver_V1/tokens.hdf5', 'w') as hf:
        hf.create_dataset('tokens', data=token_ids)


def create_voken_ids_hdf(df_token_voken):
    # vokens.hdf5 - voken idex
    voken_ids = df_token_voken[COL_VOKEN_ID].tolist()
    with h5py.File('/home/stav/Data/Vokenization/Datasets/Oliver_V1/vokens.hdf5', 'w') as hf:
        hf.create_dataset('vokens', data=voken_ids)
    # vokens.hdf5 - voken idex
    vokens = df_token_voken['voken'].tolist()
    np_vokens = np.stack(vokens)
    np.save('/home/stav/Data/Vokenization/Datasets/Oliver_V1/vokens.npy', np_vokens)

