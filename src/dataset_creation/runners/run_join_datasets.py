import os
import pandas as pd
from src.common.constants import DATASETS_VOKENIZATION, \
    COL_SPEAKER, COL_DATASET, COL_VOKEN_ID, \
    BLOCK_SIZE
from src.dataset_creation.components.join_datasets import get_df_all_train_test, shuffle_train_test
from src.dataset_creation.components.save_dataset import save_dataset, save_metadata

pd.options.display.float_format = '{:,.0f}'.format


# Input Datasets
DATASET_OLIVER = 'Oliver_V3'
DATASET_NOAH   = 'Noah_V1'


# Output Dataset target path
new_dataset_name = f'{DATASET_OLIVER}_{DATASET_NOAH}_Shuffle_Sentence_{BLOCK_SIZE}'
new_dataset_path = os.path.join(DATASETS_VOKENIZATION, new_dataset_name)


df_all, df_train, df_test = get_df_all_train_test(DATASET_OLIVER, DATASET_NOAH)
df_all[COL_SPEAKER].value_counts().astype('float32')
# noah     162,625
# oliver    81,712
len(df_train), len(df_test)

df_train_shuffled, train_indices, df_test_shuffled, test_indices = shuffle_train_test(df_train, df_test, BLOCK_SIZE)
df_all_shuffled = pd.concat([df_train_shuffled, df_test_shuffled])

assert df_all_shuffled[COL_DATASET].value_counts().equals(df_all[COL_DATASET].value_counts())

df_all_shuffled[COL_VOKEN_ID] = range(0, len(df_all_shuffled))


save_dataset(new_dataset_path, df_all_shuffled)
save_metadata(new_dataset_path, train_indices, test_indices)

