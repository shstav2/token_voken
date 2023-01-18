import os
import pandas as pd
from src.common.constants import DATASETS_VOKENIZATION, \
    COL_SPEAKER, COL_SET_TYPE, COL_VOKEN_ID, \
    BLOCK_SIZE
from src.dataset_creation.components.join_datasets import get_df_all_train_test, shuffle_train_test
from src.dataset_creation.components.save_dataset import save_dataset, save_metadata

pd.options.display.float_format = '{:,.0f}'.format



BLOCK_SIZE_ALTERNATIVE = 21 # (= 126 / 2)


# Input Datasets
DATASET_OLIVER = 'Oliver_V4'
DATASET_NOAH   = 'Noah_V2'


# Output Dataset target path
new_dataset_name = f'{DATASET_OLIVER}_{DATASET_NOAH}_Shuffle_Sentence_{BLOCK_SIZE_ALTERNATIVE}'
new_dataset_path = os.path.join(DATASETS_VOKENIZATION, new_dataset_name)


df_all, df_train, df_test = get_df_all_train_test(DATASET_OLIVER, DATASET_NOAH)
df_all[COL_SPEAKER].value_counts().astype('float32')
# Oliver_V3_Noah_V1      # Oliver_V4_Noah_V2
# -----------------        -----------------
# noah     162,625         258,654
# oliver    81,712         129,045
len(df_train), len(df_test)

df_train_shuffled, train_indices, df_test_shuffled, test_indices = shuffle_train_test(df_train, df_test, BLOCK_SIZE_ALTERNATIVE)
df_all_shuffled = pd.concat([df_train_shuffled, df_test_shuffled])

assert df_all_shuffled[COL_SET_TYPE].value_counts().equals(df_all[COL_SET_TYPE].value_counts())

df_all_shuffled[COL_VOKEN_ID] = range(0, len(df_all_shuffled))

df_token_voken = df_all_shuffled
special_tokens_mask = df_token_voken[COL_WORD_FRAME_SELECTED] == -1  # originated from padding
df_token_voken[COL_VOKEN_PATH] = \
    df_token_voken['video_id'] + '_' + df_token_voken['interval_id'] + '_' + df_token_voken[
        COL_WORD_FRAME_SELECTED].astype(str)
df_token_voken.loc[special_tokens_mask, COL_VOKEN_PATH] = ''
# set voken id running index, and ignore index for special tokens
df_token_voken[COL_VOKEN_ID] = VOKEN_IGNORE_ID
df_token_voken.loc[~special_tokens_mask, COL_VOKEN_ID] = range(0, len(df_token_voken[~special_tokens_mask]))

new_dataset_path = '/home/stav/Data/Vokenization/Datasets/O_V7_N_V2_S_V2_128'
save_dataset(new_dataset_path, df_all_shuffled)
save_metadata(new_dataset_path, train_indices, test_indices)

