import os
import h5py
import numpy as np


COL_BERT_TOKEN_ID       = 'token_id'
COL_WORD_FRAME_SELECTED = 'selected_frame'
COL_VOKEN_ID            = 'voken_id'
COL_VOKEN_PATH          = 'voken_path'
COL_VOKEN               = 'voken'


"""
TODO: creates too many files.
Only need:
Oliver_V1/
    df_token_voken.csv (82K token-voken pairs with metadata)
    vokens.npy         (82K voken embeddings)
    train/
        tokens.hd5     (65K bert token ids)
        vokens.hd5     (0-65K voken ids)
    test/
        tokens.hd5     (17K bert token ids)
        vokens.hd5     (65K-82K voken ids)
Noah_V1/
    df_token_voken.csv (162K token-voken pairs with metadata)
    vokens.npy         (162K voken embeddings)
"""
def save_dataset(data_dir, df_token_voken, df_train, df_test):
    if not os.path.exists(data_dir):
        print(f'Creating data directory {data_dir}..')
        os.mkdir(data_dir)
    df_token_voken.to_csv(os.path.join(data_dir, 'df_token_voken.csv'))
    df_token_voken.to_pickle(os.path.join(data_dir, 'df_token_voken_pkl.csv'))
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    save_h5_files(train_dir, df_train)
    save_h5_files(test_dir, df_test)


def save_h5_files(data_dir, df_token_voken):
    if not os.path.exists(data_dir):
        print(f'\tCreating {data_dir}..')
        os.mkdir(data_dir)
    create_token_ids_hdf(data_dir, df_token_voken)
    create_voken_ids_hdf(data_dir, df_token_voken)


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

