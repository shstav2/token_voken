from src.common.constants import DF_INTERVALS_OLIVER
from src.dataset_creation.components.create_dataframe import read_sorted_intervals, get_token_voken, split_train_test
from src.dataset_creation.components.save_dataset import save_dataset


# 1) Create dataframe and split train/test
df_intervals = read_sorted_intervals(DF_INTERVALS_OLIVER)  # [noah] v1: (2652, 20), v2: (3735, 20) [oliver] v4: (2942, 20)
df_token_voken = get_token_voken(df_intervals)             # [noah] v2: (258,654, 15)              [oliver] v4: 129045
# SPLIT_INDEX = 130231
# SPLIT_INDEX = 206662 # noah v2
SPLIT_INDEX = 102670 # oliver v4
df_token_voken, df_train, df_test = split_train_test(df_token_voken, SPLIT_INDEX)
"""
>>> df_token_voken, df_train, df_test = split_train_test(df_token_voken, SPLIT_INDEX)
Train: 206,663, Test: 51,991 (ratio: 79.9% train) # noah   v2
Train: 102,671, Test: 26,374 (ratio: 79.6% train) # oliver v4
"""

# 2) Save dataset
VERSION = None
SPEAKER_NAME = None
VERSION = 'V4'
SPEAKER_NAME = 'Oliver'
NEW_DATASET_NAME = f'{SPEAKER_NAME.title()}_{VERSION}'
VERSIONED_DATASET_DIR = f'/home/stav/Data/Vokenization/Datasets/{NEW_DATASET_NAME}'
VERSIONED_DATASET_DIR
save_dataset(VERSIONED_DATASET_DIR, df_token_voken)
