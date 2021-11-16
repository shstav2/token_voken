"""
Create joint dataset of oliver and noah
---------------------------------------
"""
import os
import pandas as pd
from src.dataset_creation.components._2_save_dataset import save_full_dataset
from src.common.constants import DATASETS_VOKENIZATION

# Input Datasets
DATASET_OLIVER = 'Oliver_V3'
DATASET_NOAH   = 'Noah_V1'

# Output Dataset target path
new_dataset_name = f'{DATASET_OLIVER}_{DATASET_NOAH}'
new_dataset_path = os.path.join(DATASETS_VOKENIZATION, new_dataset_name)

# Input Paths
oliver_feat_path = os.path.join(DATASETS_VOKENIZATION, DATASET_OLIVER, 'vokens.npy')
noah_feat_path   = os.path.join(DATASETS_VOKENIZATION, DATASET_NOAH,   'vokens.npy')
# /home/stav/Data/Vokenization/Datasets/Oliver_V3/df_token_voken_pkl.csv
oliver_df_path   = os.path.join(DATASETS_VOKENIZATION, DATASET_OLIVER, 'df_token_voken_pkl.csv')
noah_df_path     = os.path.join(DATASETS_VOKENIZATION, DATASET_NOAH,   'df_token_voken_pkl.csv')

df_oliver = pd.read_pickle(oliver_df_path)  # shape: ( 81,712  13)
df_noah   = pd.read_pickle(noah_df_path)    # shape: (162,625  13)

df_oliver['speaker'] = 'oliver'
df_noah['speaker']   = 'noah'

df_all = pd.concat([df_oliver, df_noah])
df_all['speaker'].value_counts()

save_full_dataset(new_dataset_path, df_all)