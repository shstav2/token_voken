from src.common.constants import SPEAKER_NAME, \
    COL_SPEAKER, COL_VIDEO_ID, COL_VOKEN_ID
SPEAKER_NAME

from src.dataset_creation.components.create_dataframe import read_sorted_intervals, \
    get_token_voken, split_train_test
from src.dataset_creation.components.save_dataset import save_dataset


df_intervals = read_sorted_intervals()
df_intervals.head()

df_intervals[COL_SPEAKER].value_counts() # noah    2652

df_intervals_org = df_intervals.copy()

df_intervals = df_intervals[df_intervals[COL_VIDEO_ID].isin(['-DoL722yNn4', '-EDP12VmqvM'])]
df_intervals.shape # (10, 20)

df_token_voken = get_token_voken(df_intervals)
df_token_voken.shape # (913, 14)

df_token_voken[COL_VOKEN_ID].tolist() == list(range(913)) # True

SPLIT_INDEX = 343
df_token_voken, df_train, df_test = split_train_test(df_token_voken, SPLIT_INDEX) # Train: 344, Test: 569 (ratio: 37.7%)

NEW_DATASET_NAME = 'Noah_Sample'
VERSIONED_DATASET_DIR = f'/home/stav/Data/Vokenization/Datasets/{NEW_DATASET_NAME}'

save_dataset(VERSIONED_DATASET_DIR, df_token_voken)

