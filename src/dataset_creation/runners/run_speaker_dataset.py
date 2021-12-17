from src.common.constants import SPEAKER_NAME
from src.dataset_creation.components.create_dataframe import read_sorted_intervals, get_token_voken, split_train_test
from src.dataset_creation.components.save_dataset import save_dataset


# 1) Create dataframe and split train/test
df_intervals = read_sorted_intervals() # noah: (2652, 20)
df_token_voken = get_token_voken(df_intervals)
SPLIT_INDEX = 130231
df_token_voken, df_train, df_test = split_train_test(df_token_voken, SPLIT_INDEX)

# 2) Save dataset
VERSION = None
VERSION = 'V1'
NEW_DATASET_NAME = f'{SPEAKER_NAME.title()}_{VERSION}'
VERSIONED_DATASET_DIR = f'/home/stav/Data/Vokenization/Datasets/{NEW_DATASET_NAME}'

save_dataset(VERSIONED_DATASET_DIR, df_token_voken)
