import os
import pathlib
from src.common.conf import STAV_ROOT, STAV_LOCAL_ROOT

# Base
PATS_DATA_ROOT = os.path.join(STAV_ROOT, 'PATS_DATA')
LOCAL_PATS_DATA_ROOT = STAV_LOCAL_ROOT
SPEAKER_NAME = 'oliver'

# Source Channels
VIDEOS                    =  'Videos'    # 105810.mp4
VIDEO_FRAMES_DIR_NAME     =  'Frames'    # 00029.jpg
ALL_FACES_IMAGE_DIR_NAME  =  'FacesAll'  # face_0.jpg
FACES_IMAGE_DIR_NAME      =  'Faces'     # 00029.jpg (224x224)
RESNET_EMBEDDING_DIR_NAME =  'ResNet'    # 512 vector
FECNET_EMBEDDING_DIR_NAME =  'FECNet'    # 8 vector
TEXT_DIR_NAME             =  'Text'
TEXT_RAW_FILENAME         =  'Raw'       # DataFrame [word|start_frame|end_frame|frames_count]
TEXT_TOKENS_FILENAME      =  'Tokens'    # DataFrame

# Interval Parsing
FRAME_EXTENSION           = 'jpg'
EMBEDDING_EXTENSION       = 'npy'
TEXT_EXTENSION            = 'csv'
TOKEN_VOKEN_EXTENSTION    = 'hdf5'


# Vokenization Datasets
# /home/stav/Data/Vokenization/Datasets/Oliver_V1
DATASETS_VOKENIZATION        = os.path.join(STAV_ROOT, 'Vokenization/Datasets')
DF_TOKEN_VOKEN_PKL_FILENAME  = 'df_token_voken_pkl.csv'
DF_TOKEN_VOKEN_CSV_FILENAME  = 'df_token_voken_partial_cols.csv'
TOKENS_DATA_FILENAME         = 'tokens.hdf5'
VOKENS_DATA_FILENAME         = 'vokens.hdf5'
INDICES_FILENAME             = 'indices.txt'

# Magic Numbers
VIDEO_ID_LEN = 11
FRAME_RATE = 15
FACE_IMAGE_SIZE = 224
EMBEDDING_DIM = 16
BLOCK_SIZE = 126 # Set by vokenization as the size of each row in the batch


# dataframe columns names
COL_SPEAKER                   = 'speaker'
COL_SET_TYPE                  = 'set_type'
COL_VIDEO_ID                  = 'video_id'
COL_INTERVAL_ID               = 'interval_id'
COL_BERT_TOKEN_ID             = 'token_id'
COL_WORD                      = 'word'
COL_VOKEN                     = 'voken'
COL_VOKEN_ID                  = 'voken_id'
COL_WORD_FRAME_SELECTED       = 'selected_frame'
COL_WORD_FRAME_SELECTED_FIXED = 'selected_frame_fix' # incase the selected frame id does not exist
COL_VOKEN_PATH                = 'voken_path'

# Train/Test split
SPLIT_INDEX = {
    'Oliver_V3':  64713,
    'Oliver_V4':  102670,
    'Noah_V1':    130232,
    'Noah_V2':    102670
}


# Tree
PATS_VIDEOS_DIR     = os.path.join(PATS_DATA_ROOT, VIDEOS)
PATS_DF_DIR = os.path.join(PATS_DATA_ROOT, 'DataFrames')


# Projects
PROJECT_TOKEN_VOKEN = pathlib.Path(__file__).parent.parent.parent.absolute()
PROJECT_FECNET = '/home/stav/Projects/FECNet'


OLIVER_FACE_PATH = '/home/stav/Data/Sample/oliver/face.jpg'
OLIVER_FACE_RESNET_EMBEDDING_PATH = '/home/stav/Data/Sample/oliver/face.npy'

# /home/stav/Data/PATS_DATA/DataFrames/original/cmu_intervals_df.csv
DF_INTERVALS_ORG         = os.path.join(PATS_DF_DIR, 'original/cmu_intervals_df.csv')     # shape: (84,289,  8) ['dataset', 'delta_time', 'end_time', 'interval_id', 'speaker', 'start_time', 'video_fn', 'video_link']
DF_INTERVALS_ALL         = os.path.join(PATS_DF_DIR, 'all/df_intervals_all.csv')          # shape: (84,289,  20)
DF_INTERVALS_OLIVER_ALL  = os.path.join(PATS_DF_DIR, 'oliver/df_intervals_oliver_valid_text.csv') # shape: (4629, 19)
DF_INTERVALS_OLIVER_V2   = os.path.join(PATS_DF_DIR, 'oliver/df_intervals_oliver_v2.csv') # shape: (4629, 20)     2,942 valid (1687 not)
DF_INTERVALS_NOAH_V1     = os.path.join(PATS_DF_DIR, 'noah/df_intervals_noah_v1.csv')     # shape: (4367, 20)     2,657 valid (1715 not)
DF_INTERVALS_NOAH_V2     = os.path.join(PATS_DF_DIR, 'noah/df_intervals_noah_v2.csv')     # shape: (4367, 20)     3,734 valid ( 623 not)
# set current context
DF_INTERVALS_NOAH        = DF_INTERVALS_NOAH_V2
DF_INTERVALS_OLIVER      = DF_INTERVALS_OLIVER_V2


VOKENS_VOCAB_ROOT_DIR    = '/home/stav/Data/Vokenization/Vokens'
VOKENS_VOCAB_NOAH_V1_DIR = os.path.join(VOKENS_VOCAB_ROOT_DIR, 'Noah_V1')
