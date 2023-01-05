import os
import h5py
import simplejson
import numpy as np
from src.common.constants import EMBEDDING_DIM, INDICES_FILENAME, \
    DF_TOKEN_VOKEN_CSV_FILENAME, DF_TOKEN_VOKEN_PKL_FILENAME, \
    COL_VOKEN_ID, COL_SET_TYPE, COL_WORD_FRAME_SELECTED
from src.common.file_utils import ls_alh, tree
from src.common.display_utils import SVE


COL_BERT_TOKEN_ID       = 'token_id'
COL_VOKEN_PATH          = 'voken_path'
COL_VOKEN               = 'voken'


"""
Oliver_V1/
    df_token_voken.csv (82K token-voken pairs with metadata)
    vokens.npy         (82K voken embeddings)          --- MUST (voken_feats = np.load(args.voken_feat_dir=vokens.npy))
    train/
        tokens.hd5     (65K bert token ids)
        vokens.hd5     (0-65K voken ids)
    test/
        tokens.hd5     (17K bert token ids)
        vokens.hd5     (65K-82K voken ids)
Noah_V1/ (162,625)
    df_token_voken.csv (162K token-voken pairs with metadata)
    vokens.npy         (162K voken embeddings)
"""


def save_dataset(data_dir, df_token_voken):
    # 1) Save entire dataset
    save_full_dataset(data_dir, df_token_voken)

    # 2) Save train/test
    df_train = df_token_voken[df_token_voken[COL_SET_TYPE] == 'train']
    df_test  = df_token_voken[df_token_voken[COL_SET_TYPE] == 'test']
    assert (len(df_train) + len(df_test)) == len(df_token_voken)
    save_subset(data_dir, 'train', df_train)
    save_subset(data_dir, 'test',  df_test)

    # 3) Show resulted dataset
    print(f'{SVE}  FINAL:')
    ls_alh(data_dir)
    tree(data_dir)


def save_subset(data_dir, subset_name, df_subset):
    subset_dir = os.path.join(data_dir, subset_name)
    save_h5_files(subset_dir, df_subset)


def save_full_dataset(data_dir, df_token_voken):
    """
    Creating data directory /home/stav/Data/Vokenization/Datasets/Oliver_V3_Noah_V1..
        total 151M
        -rw-rw-r-- 1 stav stav  70M Nov 16 17:57 df_token_voken.csv
        -rw-rw-r-- 1 stav stav  52M Nov 16 17:57 df_token_voken_pkl.csv
        -rw-rw-r-- 1 stav stav  30M Nov 16 17:57 vokens.npy
    """
    if os.path.exists(data_dir):
        raise RuntimeError(f'Already exists: {data_dir}')
    print(f'Creating data directory {data_dir}..')
    os.mkdir(data_dir)
    df_token_voken.drop('voken', axis=1).to_csv(os.path.join(data_dir, DF_TOKEN_VOKEN_CSV_FILENAME))
    df_token_voken.to_pickle(os.path.join(data_dir, DF_TOKEN_VOKEN_PKL_FILENAME))
    special_tokens_mask = df_token_voken[COL_WORD_FRAME_SELECTED] == -1
    vokens = df_token_voken[~special_tokens_mask]['voken'].tolist()
    # replace nan with np.zeros
    vokens_padded = [voken if voken is not None else np.zeros(EMBEDDING_DIM) for voken in vokens]
    assert len(vokens_padded) - 1 == df_token_voken[COL_VOKEN_ID].max()
    np_vokens = np.stack(vokens_padded)
    np.save(os.path.join(data_dir, 'vokens.npy'), np_vokens)
    ls_alh(data_dir)


def save_h5_files(data_dir, df_token_voken):
    if not os.path.exists(data_dir):
        print(f'\tCreating {data_dir}..')
        os.mkdir(data_dir)
    create_token_ids_hdf(data_dir, df_token_voken)
    create_voken_ids_hdf(data_dir, df_token_voken)
    ls_alh(data_dir)


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
    # vokens = df_token_voken[COL_VOKEN].tolist()
    # np_vokens = np.stack(vokens)
    # np.save(os.path.join(data_dir, 'vokens.npy'), np_vokens)


def save_indices(indices_path, indices):
    f = open(indices_path, 'w')
    simplejson.dump(indices, f)
    f.close()


def save_metadata(data_dir, train_indices, test_indices):
    metadata_dir = os.path.join(data_dir, 'metadata') # TODO: path resolvers
    os.mkdir(metadata_dir)
    train_indices_path = os.path.join(metadata_dir, f'train_{INDICES_FILENAME}')
    save_indices(train_indices_path, train_indices)
    test_indices_path = os.path.join(metadata_dir, f'test_{INDICES_FILENAME}')
    save_indices(test_indices_path, test_indices)

