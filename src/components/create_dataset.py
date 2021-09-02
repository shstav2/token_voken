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
COL_VOKEN               = 'voken'


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
    df_token_voken[COL_VOKEN_ID] = range(0, len(df_token_voken))
    return df_token_voken


def create_token_ids_hdf(data_dir, df_token_voken):
    token_ids = df_token_voken[COL_BERT_TOKEN_ID].tolist()
    # resolve_dataset_tokens_path
    with h5py.File(os.path.join(data_dir, 'tokens.hdf5'), 'w') as hf:
        hf.create_dataset('tokens', data=token_ids)


def create_voken_ids_hdf(data_dir, df_token_voken):
    # vokens.hdf5 - voken idex
    voken_ids = df_token_voken[COL_VOKEN_ID].tolist()
    with h5py.File(os.path.join(data_dir, 'vokens.hdf5'), 'w') as hf:
        hf.create_dataset('vokens', data=voken_ids)
    # vokens.hdf5 - voken idex
    vokens = df_token_voken['voken'].tolist()
    np_vokens = np.stack(vokens)
    np.save(os.path.join(data_dir, 'vokens.npy'), np_vokens)


"""
TODO: creates too many files.
Only need:
V1/
    df_token_voken.csv (82K token-voken pairs with metadata)
    vokens.npy         (82K voken embeddings)
    train/
        tokens.hd5     (65K bert token ids)
        vokens.hd5     (0-65K voken ids)
    test/
        tokens.hd5     (17K bert token ids)
        vokens.hd5     (65K-82K voken ids)
"""
def save_dataset(data_dir, df_token_voken):
    if not os.path.exists(data_dir):
        print(f'Creating data directory {data_dir}..')
        os.mkdir(data_dir)
    df_token_voken.to_csv(os.path.join(data_dir, 'df_token_voken.csv'))
    df_token_voken.to_pickle(os.path.join(data_dir, 'df_token_voken_pkl.csv'))
    create_token_ids_hdf(data_dir, df_token_voken)
    create_voken_ids_hdf(data_dir, df_token_voken)


df_intervals = read_sorted_intervals()
df_token_voken = get_token_voken(df_intervals)

# Only train/test split
df_token_voken = pd.read_pickle('/home/stav/Data/Vokenization/Datasets/Oliver_V3/df_token_voken_pkl.csv')

# >>> df_token_voken.shape
# (81712, 10)
VERSION = None
VERSIONED_DATASET_DIR = f'/home/stav/Data/Vokenization/Datasets/Oliver_{VERSION}'
TRAIN_DIR             = os.path.join(VERSIONED_DATASET_DIR, 'train')
TEST_DIR              = os.path.join(VERSIONED_DATASET_DIR, 'test')


save_dataset(VERSIONED_DATASET_DIR, df_token_voken)

df_train = df_token_voken[df_token_voken[COL_VOKEN_ID] <= 64712].copy()
df_valid = df_token_voken[64712 < df_token_voken[COL_VOKEN_ID]].copy()


save_dataset(TRAIN_DIR, df_train)
save_dataset(TEST_DIR, df_valid)




# CHECK VOKENS ARE THE SAME IN ALL FILES:
# ========================================
"""
import numpy as np
import pandas as pd
from src.common.path_resolvers import resolve_interval_facial_embedding_path
# ---- np array
vokens_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V1/vokens.npy'
vokens = np.load(vokens_path)
# ---- dataframe
token_voken_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V1/df_token_voken.csv'
df_token_voken = pd.read_csv(token_voken_path)
sample = df_token_voken.sample().iloc[0]
voken_str = sample[COL_VOKEN]
voken_str = voken_str.replace('\n', '').replace('[', '').replace(']', '')
voken_df = np.fromstring(voken_str, sep=' ')
# ---- FECNet - '/home/stav/Data/PATS_DATA/Videos/oliver/iAgKHSNqxa8/214675/FECNet/00111.npy'
fecnet_path = resolve_interval_facial_embedding_path(str(sample.interval_id), sample.selected_frame)
voken_fecnet = np.load(fecnet_path)
np.isclose(voken_fecnet, voken_df)
np.isclose(voken_fecnet, vokens[sample.voken_id - 1])
"""

# CHECK VOKENS ARE THE SAME IN ALL FILES:
# ========================================
"""
vokens_v1_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V1/vokens.npy'
vokens_v1 = np.load(vokens_v1_path)
vokens_v2_path = '/home/stav/Data/Vokenization/Datasets/Oliver_V2/vokens.npy'
vokens_v2 = np.load(vokens_v2_path)
>>> np.array_equal(vokens_v1, vokens_v2)
True
>>> (vokens_v1 == vokens_v2).all(axis=1).sum()
76873
>>> vokens_v1.shape[0] - _
4839 # 6%
"""

"""
df_token_voken[df_token_voken['interval_id'] == int(interval_id)]
vokens[78460]
np.load(resolve_interval_facial_embedding_path(interval_id, 38))
"""

