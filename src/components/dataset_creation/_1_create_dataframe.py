import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.data_loader import load_valid_intervals
from src.common.constants import SPEAKER_NAME, DF_INTERVALS_NOAH
from src.common.path_resolvers import resolve_interval_text_tokenized_path, \
    resolve_interval_facial_embedding_path, resolve_interval_facial_embeddings_dir


COL_BERT_TOKEN_ID       = 'token_id'
COL_WORD_FRAME_SELECTED = 'selected_frame'
COL_VIDEO_ID            = 'video_id'
COL_VOKEN_ID            = 'voken_id'
COL_VOKEN_PATH          = 'voken_path'
COL_VOKEN               = 'voken'


def read_sorted_intervals():
    df_intervals = load_valid_intervals(DF_INTERVALS_NOAH)
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
        return None
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
    df_token_voken[COL_VOKEN_ID] = range(0, len(df_token_voken))
    return df_token_voken


def split_train_test(df_token_voken, split_index):
    # Train/Test split
    df_train = df_token_voken[df_token_voken[COL_VOKEN_ID] <= split_index].copy()
    df_test = df_token_voken[split_index < df_token_voken[COL_VOKEN_ID]].copy()
    ratio = len(df_train) * 100 / len(df_token_voken)
    print(f'Train: {len(df_train):,}, Test: {len(df_test):,} (ratio: {ratio:.1f} %)')
    assert not (set(df_train[COL_VIDEO_ID].unique()) & set(df_test[COL_VIDEO_ID].unique()))
    return df_train, df_test

