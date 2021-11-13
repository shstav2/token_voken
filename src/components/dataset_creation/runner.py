from src.common.constants import SPEAKER_NAME
from src.components.dataset_creation._1_create_dataframe import read_sorted_intervals, get_token_voken, split_train_test
from src.components.dataset_creation._2_save_dataset import save_dataset

# 1) Create dataframe and split train/test
df_intervals = read_sorted_intervals()
df_token_voken = get_token_voken(df_intervals)
SPLIT_INDEX = 130231
df_train, df_test = split_train_test(df_token_voken, SPLIT_INDEX)

# 2) Save dataset
VERSION = None
VERSION = 'V1'
VERSIONED_DATASET_DIR = f'/home/stav/Data/Vokenization/Datasets/{SPEAKER_NAME.title()}_{VERSION}'
save_dataset(VERSIONED_DATASET_DIR, df_token_voken, df_train, df_test)